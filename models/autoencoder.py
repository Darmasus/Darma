"""Conditional autoencoder with temporal prior.

 Frame flow:
   I-frame: g_a(x) -> y_I -> rANS
   P-frame: x_mc = motion(x_prev, x_curr)
            y_P = g_a(x_curr) conditioned on g_a(x_mc) (temporal prior)
            residual = x_curr - g_s(y_P)  -> masked-diffusion tokens

 The *synthesis* transform `g_s`'s last few conv layers are AdaptableConv2d,
 so they participate in Weight-Adaptation. `g_a` is held fixed at decode time
 (it's only used on the encoder).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptation import AdaptableConv2d, AdaptationConfig
from .hyperprior import ScaleHyperprior
from .motion import MotionCompensationNet
from .diffusion_residual import MaskedDiffusionResidual
from .numerics import safe_act

try:
    from compressai.layers import GDN
except ImportError:                   # pragma: no cover
    GDN = None


def _gdn(N: int, inverse: bool = False):
    """GDN (Generalized Divisive Normalization) — the activation used in
    CompressAI's reference codecs. Empirically much better than ReLU/LeakyReLU
    for image-coding networks (Ballé et al. 2017). Falls back to LeakyReLU
    only if compressai isn't installed."""
    if GDN is None:
        return nn.LeakyReLU(0.1, inplace=True)
    return GDN(N, inverse=inverse)


class Analysis(nn.Module):
    """g_a: x -> y, 16x spatial downsample."""
    def __init__(self, N: int = 192, M: int = 320):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, N, 5, stride=2, padding=2),
            _gdn(N),
            nn.Conv2d(N, N, 5, stride=2, padding=2),
            _gdn(N),
            nn.Conv2d(N, N, 5, stride=2, padding=2),
            _gdn(N),
            nn.Conv2d(N, M, 5, stride=2, padding=2),
        )

    def forward(self, x): return safe_act(self.net(x))


class Synthesis(nn.Module):
    """g_s: y -> x_hat, with the final two conv stages adaptable.

    Uses IGDN (inverse GDN) as the synthesis-side activation. Adaptable convs
    are followed by IGDN(N) so the rank-r delta participates in normalisation,
    not a separate post-norm stage.
    """
    def __init__(self, N: int = 192, M: int = 320, cfg: AdaptationConfig | None = None):
        super().__init__()
        cfg = cfg or AdaptationConfig(rank=4)
        self.up1 = nn.ConvTranspose2d(M, N, 5, stride=2, padding=2, output_padding=1)
        self.igdn1 = _gdn(N, inverse=True)
        self.up2 = nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1)
        self.igdn2 = _gdn(N, inverse=True)

        # Adaptation tail: per-sequence texture / chroma.
        self.adapt_tail_1 = AdaptableConv2d(N, N, 3, padding=1, cfg=cfg)
        self.igdn3 = _gdn(N, inverse=True)
        self.adapt_up = nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1)
        self.igdn4 = _gdn(N, inverse=True)
        self.adapt_tail_2 = AdaptableConv2d(N, N, 3, padding=1, cfg=cfg)
        self.igdn5 = _gdn(N, inverse=True)
        self.adapt_up2 = nn.ConvTranspose2d(N, 3, 5, stride=2, padding=2, output_padding=1)

    def forward(self, y):
        h = safe_act(self.igdn1(self.up1(y)))
        h = safe_act(self.igdn2(self.up2(h)))
        h = safe_act(self.igdn3(self.adapt_tail_1(h)))
        h = safe_act(self.igdn4(self.adapt_up(h)))
        h = safe_act(self.igdn5(self.adapt_tail_2(h)))
        # Targets are in [0, 1]; ±2 is a generous margin for training.
        # Anything wider lets the model drift to huge outputs under rare
        # gradient spikes and gets stuck in a bad equilibrium.
        return safe_act(self.adapt_up2(h), bound=2.0)


class WANVCAutoencoder(nn.Module):
    """Full conditional autoencoder: I-frames + P-frames + residual coder."""

    def __init__(self, N: int = 192, M: int = 320):
        super().__init__()
        self.g_a = Analysis(N, M)
        self.g_s = Synthesis(N, M)
        self.hyper = ScaleHyperprior(N, M)
        self.motion = MotionCompensationNet(base=32)
        self.residual = MaskedDiffusionResidual(codebook_size=8192, dim=128, patch=8)
        self.N, self.M = N, M

    # -------- I-frame path -------- #
    def encode_iframe(self, x: torch.Tensor):
        y = self.g_a(x)
        hp = self.hyper(y)
        x_hat = self.g_s(hp["y_hat"])
        return {
            "x_hat": x_hat,
            "likelihoods": hp["likelihoods"],
            "y_hat": hp["y_hat"],
            "z_hat": hp["z_hat"],
        }

    # -------- P-frame path -------- #
    def encode_pframe(self, x_prev: torch.Tensor, x_curr: torch.Tensor):
        mc = self.motion(x_prev, x_curr)
        y_curr = self.g_a(x_curr)
        y_mc = self.g_a(mc["x_mc"])
        # Temporal prior: encode the *delta* in latent space. Saves rate.
        dy = y_curr - y_mc
        hp = self.hyper(dy)
        dy_hat = hp["y_hat"]
        x_from_latent = self.g_s(y_mc + dy_hat)

        residual = x_curr - x_from_latent
        idx, keep_prob = self.residual.encode_residual(residual, mc["x_mc"])
        return {
            "x_from_latent": x_from_latent,
            "residual_idx": idx,
            "residual_keep_prob": keep_prob,
            "mc": mc,
            "likelihoods": hp["likelihoods"],
            "y_hat": dy_hat,
            "y_mc": y_mc,
        }

    # -------- bitstream compress/decompress -------- #
    # These are the real rANS-based methods (not just likelihood estimates).

    @torch.no_grad()
    def compress_iframe(self, x: torch.Tensor) -> dict:
        """Returns the bytes needed to reconstruct x (minus PUP)."""
        y = self.g_a(x)
        streams = self.hyper.compress(y)
        return streams

    @torch.no_grad()
    def decompress_iframe(self, streams: dict, hw: tuple[int, int]) -> torch.Tensor:
        y_hat = self.hyper.decompress(
            streams["y_strings"], streams["z_strings"], streams["z_shape"],
        )
        x_hat = self.g_s(y_hat).clamp(0, 1)
        return x_hat

    # Flow coder: fixed-scale Gaussian rANS over quantized flow values.
    # Flow has 2 channels, typically |values| < ~64 px at 1080p -> int16 suffices.
    _FLOW_STEP = 0.25        # sub-pixel quantization
    _FLOW_BOUND = 256        # clamp range in quantized units

    @torch.no_grad()
    def _compress_flow(self, flow: torch.Tensor) -> tuple[bytes, tuple[int, ...]]:
        from entropy_coder.ans import encode_gaussian
        import numpy as np
        q = torch.round(flow / self._FLOW_STEP).clamp(-self._FLOW_BOUND, self._FLOW_BOUND - 1)
        sym = q.to(torch.int32).cpu().numpy()
        shape = sym.shape
        # Empirical std ~8 quant units covers most panning/handheld motion.
        sigma = np.full(shape, 8.0, dtype=np.float64)
        mu = np.zeros(shape, dtype=np.float64)
        payload = encode_gaussian(sym, mu, sigma, bound=self._FLOW_BOUND)
        return payload, shape

    @torch.no_grad()
    def _decompress_flow(self, payload: bytes, shape: tuple[int, ...],
                         device: str | torch.device) -> torch.Tensor:
        from entropy_coder.ans import decode_gaussian
        import numpy as np
        sigma = np.full(shape, 8.0, dtype=np.float64)
        mu = np.zeros(shape, dtype=np.float64)
        sym = decode_gaussian(payload, shape, mu, sigma, bound=self._FLOW_BOUND)
        q = torch.from_numpy(sym).float().to(device)
        return q * self._FLOW_STEP

    @torch.no_grad()
    def compress_pframe(self, x_prev: torch.Tensor, x_curr: torch.Tensor) -> dict:
        """Encode a P-frame against a reconstructed previous frame.

        Transmits:
          * flow payload (rANS over int-quantized flow)
          * dy hyperprior streams (y_strings, z_strings, z_shape)
          * VQ residual indices + keep mask (packed numpy)
        """
        import numpy as np

        mc = self.motion(x_prev, x_curr)
        flow_payload, flow_shape = self._compress_flow(mc["flow"])

        # Reproduce the decoder's x_mc from the quantized flow so encoder and
        # decoder use matching y_mc (avoids drift).
        flow_hat = self._decompress_flow(flow_payload, flow_shape, x_prev.device)
        from models.motion import _bilinear_warp
        x_warp_hat = _bilinear_warp(x_prev, flow_hat)
        x_mc_hat = self.motion.refine(torch.cat([x_warp_hat, x_prev, flow_hat], dim=1))
        y_mc = self.g_a(x_mc_hat)
        y_curr = self.g_a(x_curr)
        dy = y_curr - y_mc
        dy_streams = self.hyper.compress(dy)

        # Residual VQ indices and keep mask.
        x_from_latent = self.g_s(y_mc + self.hyper.decompress(
            dy_streams["y_strings"], dy_streams["z_strings"], dy_streams["z_shape"]
        ))
        idx, keep_prob = self.residual.encode_residual(x_curr - x_from_latent, x_mc_hat)
        keep = (keep_prob < 0.5).cpu().numpy()
        idx_np = idx.cpu().numpy().astype(np.int32)

        return {
            "flow_payload": flow_payload,
            "flow_shape": flow_shape,
            "dy_streams": dy_streams,
            "residual_idx": idx_np,
            "residual_keep": keep,
        }

    @torch.no_grad()
    def decompress_pframe(self, x_prev: torch.Tensor, packet: dict) -> torch.Tensor:
        from models.motion import _bilinear_warp
        device = x_prev.device
        flow_hat = self._decompress_flow(packet["flow_payload"], packet["flow_shape"], device)
        x_warp = _bilinear_warp(x_prev, flow_hat)
        x_mc = self.motion.refine(torch.cat([x_warp, x_prev, flow_hat], dim=1))
        y_mc = self.g_a(x_mc)
        dy_hat = self.hyper.decompress(
            packet["dy_streams"]["y_strings"],
            packet["dy_streams"]["z_strings"],
            packet["dy_streams"]["z_shape"],
        )
        x_from_latent = self.g_s(y_mc + dy_hat)
        idx = torch.from_numpy(packet["residual_idx"]).long().to(device)
        keep = torch.from_numpy(packet["residual_keep"]).bool().to(device)
        r_hat = self.residual.decode_residual(idx, keep, x_mc)
        return (x_from_latent + r_hat).clamp(0, 1)

    # -------- utilities -------- #
    def freeze_base_for_adaptation(self) -> None:
        """Freeze *everything* except AdaptableConv2d A/B deltas and their
        log_scale/log_sigma. Used by the per-GOP overfitting loop."""
        for p in self.parameters():
            p.requires_grad_(False)
        for m in self.modules():
            if isinstance(m, AdaptableConv2d):
                for p in m.adaptable_parameters():
                    p.requires_grad_(True)

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)

    def update_entropy_tables(self, force: bool = False) -> bool:
        return self.hyper.update(force=force)
