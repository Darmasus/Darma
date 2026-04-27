"""Per-GOP overfitting loop — the encoder-side Weight-Adaptation routine.

Given a base-trained WANVC model and a GOP (T frames, [0,1] RGB), this runs
K Adam steps over *only* the AdaptableConv2d A/B parameters and their
log_scale/log_sigma knobs, minimizing:

    L = D(x, x_hat) + lambda * R(y_hat) + beta * R(PUP)

where R(PUP) is the differentiable Gaussian-prior upper bound from
`AdaptableConv2d.rate_bits()`. Beta is annealed: start low (so the model
explores the solution space), end high (compressible packet). This keeps
small-delta solutions from being pruned too early.

Usage
-----
  adapted = adapt_to_gop(model, gop_frames, lambda_rd=0.013, beta=1e-3,
                        steps=120, lr=1e-3, device="cuda")
  pup_bytes = encode_pup([(n, m) for n, m in collect_adaptable_layers(model)])
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F
from torch.optim import Adam

from models import (
    WANVCAutoencoder,
    AdaptableConv2d,
    collect_adaptable_layers,
)


@dataclass
class AdaptConfig:
    steps: int = 120
    lr: float = 1.0e-3
    lambda_rd: float = 0.013        # rate–distortion weight on latent rate
    beta_start: float = 1.0e-5      # PUP rate weight at step 0
    beta_end: float = 5.0e-4        # PUP rate weight at final step
    grad_clip: float = 1.0
    use_ms_ssim: bool = True
    msssim_weight: float = 0.5      # D = w * (1 - MS-SSIM) + (1-w) * MSE
    device: str = "cuda"
    verbose: bool = False
    # Hook called every step with (step, loss_dict). For wandb/tb.
    on_step: Callable[[int, dict], None] | None = None


def _distortion(x: torch.Tensor, x_hat: torch.Tensor, cfg: AdaptConfig) -> torch.Tensor:
    mse = F.mse_loss(x_hat.clamp(0, 1), x)
    if not cfg.use_ms_ssim:
        return mse
    try:
        from pytorch_msssim import ms_ssim
        mssim = ms_ssim(x_hat.clamp(0, 1), x, data_range=1.0)
        return cfg.msssim_weight * (1.0 - mssim) + (1.0 - cfg.msssim_weight) * mse
    except ImportError:
        return mse


def _latent_rate(likelihoods: dict[str, torch.Tensor], num_pixels: int) -> torch.Tensor:
    """Bits per pixel under the hyperprior."""
    total = sum((-torch.log2(l.clamp(min=1e-9)).sum() for l in likelihoods.values()))
    return total / num_pixels


def _pup_rate_bits(model: WANVCAutoencoder) -> torch.Tensor:
    total = torch.zeros(1, device=next(model.parameters()).device)
    for _, layer in collect_adaptable_layers(model):
        if layer._adapted:
            total = total + layer.rate_bits()
    return total.squeeze()


def _encode_pass(model: WANVCAutoencoder, frames: torch.Tensor):
    """Run the full RD forward on a GOP. Returns (x_hat stack, likelihoods list)."""
    T = frames.shape[0]
    x0 = frames[0:1]
    out_i = model.encode_iframe(x0)
    x_hats = [out_i["x_hat"]]
    liks = [out_i["likelihoods"]]

    x_prev = out_i["x_hat"].clamp(0, 1)
    for t in range(1, T):
        xt = frames[t:t + 1]
        out_p = model.encode_pframe(x_prev, xt)
        # Reconstruction for the loss: x_from_latent + residual_from_VQ.
        # For differentiability we use the decoded residual from VQ tokens
        # run through the tokenizer decode (STE through quantize).
        zq, _ = model.residual.tok.quantize(model.residual.tok.encode(xt - out_p["x_from_latent"]))
        r_hat = model.residual.tok.dec(zq)
        x_hat = (out_p["x_from_latent"] + r_hat).clamp(0, 1)
        x_hats.append(x_hat)
        liks.append(out_p["likelihoods"])
        x_prev = x_hat.detach()

    return torch.cat(x_hats, dim=0), liks


def adapt_to_gop(model: WANVCAutoencoder,
                 gop_frames: torch.Tensor,
                 cfg: AdaptConfig | None = None) -> dict:
    """Overfit the model's A/B deltas to a GOP. Returns loss history."""
    cfg = cfg or AdaptConfig()
    device = cfg.device
    model = model.to(device)
    frames = gop_frames.to(device)

    # Initialize adaptable layers (fresh deltas, base frozen).
    model.freeze_base_for_adaptation()
    for _, layer in collect_adaptable_layers(model):
        layer.reset_delta()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = Adam(params, lr=cfg.lr)

    T, _, H, W = frames.shape
    num_pixels = T * H * W
    history = {"loss": [], "D": [], "R_latent": [], "R_pup_bits": []}

    for step in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        x_hat, liks_list = _encode_pass(model, frames)
        D = _distortion(frames, x_hat, cfg)
        R_lat = sum(_latent_rate(l, num_pixels) for l in liks_list) / len(liks_list)
        R_pup = _pup_rate_bits(model)
        # Convert PUP bits to bpp so it's on the same scale as R_lat.
        R_pup_bpp = R_pup / num_pixels

        beta = cfg.beta_start + (cfg.beta_end - cfg.beta_start) * (step / max(cfg.steps - 1, 1))
        loss = D + cfg.lambda_rd * R_lat + beta * R_pup_bpp
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()

        history["loss"].append(float(loss))
        history["D"].append(float(D))
        history["R_latent"].append(float(R_lat))
        history["R_pup_bits"].append(float(R_pup))

        if cfg.verbose and step % max(cfg.steps // 10, 1) == 0:
            print(f"[adapt {step:4d}] L={float(loss):.4f}  D={float(D):.4f}"
                  f"  R_lat={float(R_lat):.4f} bpp  R_pup={float(R_pup):.0f} b")
        if cfg.on_step is not None:
            cfg.on_step(step, {"loss": float(loss), "D": float(D),
                                "R_lat": float(R_lat), "R_pup": float(R_pup)})

    return history
