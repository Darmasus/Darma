"""Fp32 overfit + QAT for NeRV backbone.

Differences from hash-grid training:
- Forward is per-frame (returns entire H x W), not per-coord. Loss is
  computed over ALL pixels in a sampled frame subset (picking frames
  cheaper than resampling coords because CNN decoders process whole
  frames anyway).
- Reusable pieces: `attach_quantizers`, `encode_codec`, `total_rate_bits`,
  and the plain per-tensor Laplace rate term all work unchanged since
  they just iterate over `nn.Parameter`s.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .nerv import NeRVBackbone
from .quantize import attach_quantizers, total_rate_bits
from .prior import prior_params_for_quants, rate_bits_from_prior


@dataclass
class NeRVTrainConfig:
    steps: int = 2000
    lr: float = 5e-3
    log_every: int = 200
    frames_per_step: int = 0   # 0 = all frames every step (small clips)


@dataclass
class NeRVQATConfig:
    steps: int = 1500
    lr: float = 2e-3
    scale_lr: float = 5e-3
    lambda_rate: float = 0.0
    log_every: int = 200
    init_scale: float = 5e-3
    lambda_warmup_frac: float = 0.25
    frames_per_step: int = 0


def _pick_frames(n_frames: int, k: int, device) -> torch.Tensor:
    if k <= 0 or k >= n_frames:
        return torch.arange(n_frames, device=device)
    return torch.randperm(n_frames, device=device)[:k]


def overfit_nerv(video: torch.Tensor, model: NeRVBackbone,
                  tcfg: NeRVTrainConfig | None = None,
                  device: str | torch.device = "cpu",
                  verbose: bool = True):
    tcfg = tcfg or NeRVTrainConfig()
    video = video.to(device)
    model = model.to(device)
    opt = Adam(model.parameters(), lr=tcfg.lr)

    T = video.shape[0]
    t0 = time.time()
    for step in range(tcfg.steps):
        opt.zero_grad(set_to_none=True)
        idx = _pick_frames(T, tcfg.frames_per_step, device)
        pred = model(idx)
        loss = F.mse_loss(pred, video[idx])
        loss.backward()
        opt.step()

        if step % tcfg.log_every == 0 or step == tcfg.steps - 1:
            with torch.no_grad():
                full = model.reconstruct().clamp(0, 1)
                mse_full = F.mse_loss(full, video).item()
                psnr_full = 10 * math.log10(1.0 / max(mse_full, 1e-12))
            if verbose:
                print(f"  [nerv fp32 {step:5d}] loss={float(loss):.5f}  "
                      f"full PSNR={psnr_full:.2f}  ({time.time()-t0:.1f}s)",
                      flush=True)
    return model


def overfit_nerv_qat(video: torch.Tensor, model: NeRVBackbone,
                      qcfg: NeRVQATConfig | None = None,
                      device: str | torch.device = "cpu",
                      verbose: bool = True,
                      prior: nn.Module | None = None,
                      caption_emb: torch.Tensor | None = None):
    qcfg = qcfg or NeRVQATConfig()
    video = video.to(device)
    model = model.to(device)
    quants = attach_quantizers(model, init_scale=qcfg.init_scale).to(device)

    if prior is not None:
        prior = prior.to(device).eval()
        for p in prior.parameters():
            p.requires_grad_(False)

    weight_params = [q.param for q in quants.values()]
    scale_params  = [q.log_scale for q in quants.values()]
    opt = Adam([
        {"params": weight_params, "lr": qcfg.lr},
        {"params": scale_params,  "lr": qcfg.scale_lr},
    ])

    T = video.shape[0]
    N_pixels = video.numel() // video.shape[0]    # pixels*3 per frame

    t0 = time.time()
    for step in range(qcfg.steps):
        opt.zero_grad(set_to_none=True)
        idx = _pick_frames(T, qcfg.frames_per_step, device)
        pred = model.forward_q(idx, quants)
        D = F.mse_loss(pred, video[idx])

        if prior is not None:
            prior_params = prior_params_for_quants(prior, quants,
                                                      caption_emb=caption_emb)
            R = rate_bits_from_prior(quants, prior_params)
        else:
            R = total_rate_bits(quants)
        warmup_steps = max(int(qcfg.steps * qcfg.lambda_warmup_frac), 1)
        lam_eff = qcfg.lambda_rate * min(step / warmup_steps, 1.0)
        loss = D + lam_eff * (R / (len(idx) * N_pixels))
        loss.backward()
        opt.step()

        if step % qcfg.log_every == 0 or step == qcfg.steps - 1:
            with torch.no_grad():
                full = reconstruct_quantized_nerv(model, quants).clamp(0, 1)
                psnr_full = 10 * math.log10(
                    1.0 / max(F.mse_loss(full, video).item(), 1e-12))
                bits = float(R.detach())
            if verbose:
                print(f"  [nerv qat {step:5d}] D={float(D):.5f}  "
                      f"full PSNR={psnr_full:.2f}  R~{bits:.0f} bits  "
                      f"({time.time()-t0:.1f}s)", flush=True)
    return quants


@torch.no_grad()
def reconstruct_quantized_nerv(model: NeRVBackbone, quants: nn.ModuleDict
                                 ) -> torch.Tensor:
    device = next(model.parameters()).device
    idx = torch.arange(model.n_frames, device=device)
    return model.forward_q(idx, quants)
