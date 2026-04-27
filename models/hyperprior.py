"""Scale-hyperprior entropy model (Balle et al. 2018), thin wrapper on CompressAI.

Used for latent z of both I-frames and P-frame residuals. The hyper-latent
lets us transmit per-element scales that sharpen the rANS distribution.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from compressai.entropy_models import EntropyBottleneck, GaussianConditional
except Exception as e:  # pragma: no cover
    EntropyBottleneck = None
    GaussianConditional = None
    _COMPRESSAI_ERR = e

from .numerics import safe_act, safe_scales, safe_means


class ScaleHyperprior(nn.Module):
    def __init__(self, N: int = 192, M: int = 320):
        super().__init__()
        if EntropyBottleneck is None:
            raise ImportError(
                "CompressAI is required for ScaleHyperprior. "
                f"Install `compressai` (underlying import error: {_COMPRESSAI_ERR})"
            )
        self.N = N
        self.M = M

        # Hyper-analysis: y -> z
        self.h_a = nn.Sequential(
            nn.Conv2d(M, N, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(N, N, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(N, N, 5, stride=2, padding=2),
        )
        # Hyper-synthesis: z -> scales (and means for P-frame residual)
        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(N, 2 * M, 3, stride=1, padding=1),   # means + scales
        )
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, y: torch.Tensor):
        # Bound y so the entropy estimates can't explode under a single
        # extreme batch.
        y = safe_act(y, bound=128.0)
        z = safe_act(self.h_a(y), bound=128.0)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        mu, scales = gaussian_params.chunk(2, dim=1)
        # Numerical guards on the Gaussian params — small scales make the
        # likelihood blow up; huge means do too.
        mu = safe_means(mu)
        scales = safe_scales(scales)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales, means=mu)
        return {
            "y_hat": y_hat,
            "z_hat": z_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "params": (mu, scales),
        }

    def update(self, force: bool = False) -> bool:
        """Populate CDF tables for rANS. Call once before encoding."""
        updated = self.entropy_bottleneck.update(force=force)
        updated |= self.gaussian_conditional.update_scale_table(
            torch.exp(torch.linspace(torch.log(torch.tensor(0.11)),
                                     torch.log(torch.tensor(256.0)), 64)),
            force=force,
        )
        return updated

    # ------------------------------------------------------------------ #
    # real-bitstream compress/decompress (CompressAI range coder under the hood)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compress(self, y: torch.Tensor) -> dict:
        """Encode latent y to bytes.

        Returns a dict with:
          y_strings : list[bytes]   per-batch rANS streams for y
          z_strings : list[bytes]   per-batch rANS streams for z
          z_shape   : (H', W')      spatial shape of the hyper-latent, needed
                                    by the decoder to reconstruct z_hat
        """
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.shape[-2:])
        gaussian_params = self.h_s(z_hat)
        mu, scales = gaussian_params.chunk(2, dim=1)
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=mu)
        return {
            "y_strings": y_strings,
            "z_strings": z_strings,
            "z_shape": tuple(z.shape[-2:]),
        }

    @torch.no_grad()
    def decompress(self, y_strings: list, z_strings: list,
                   z_shape: tuple[int, int]) -> torch.Tensor:
        z_hat = self.entropy_bottleneck.decompress(z_strings, z_shape)
        gaussian_params = self.h_s(z_hat)
        mu, scales = gaussian_params.chunk(2, dim=1)
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes, means=mu)
        return y_hat
