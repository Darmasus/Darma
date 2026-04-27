"""Fp32 overfit + QAT for M-NeRV with optional rate-aware QAT."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .mnerv import MNeRVBackbone, MNeRVConfig
from .quantize import attach_quantizers, total_rate_bits
from .prior import prior_params_for_quants, rate_bits_from_prior


@dataclass
class MNeRVTrainConfig:
    steps: int = 2000
    lr: float = 5e-3
    log_every: int = 200
    # Scheduled sampling: probability of using the target frame (vs the
    # model's own previous reconstruction) as the warp source ramps from
    # 1.0 (always TF) at step 0 down to 0.0 (always free-run) at this
    # fraction of total steps. Smooth transition avoids the abrupt
    # exposure-bias collapse we saw with a hard switch.
    tf_decay_frac: float = 0.5


@dataclass
class MNeRVQATConfig:
    steps: int = 1500
    lr: float = 2e-3
    scale_lr: float = 5e-3
    lambda_rate: float = 0.0
    log_every: int = 200
    init_scale: float = 5e-3
    lambda_warmup_frac: float = 0.25
    tf_decay_frac: float = 0.4


def overfit_mnerv(video: torch.Tensor, model: MNeRVBackbone,
                    tcfg: MNeRVTrainConfig | None = None,
                    device: str | torch.device = "cpu",
                    verbose: bool = True):
    tcfg = tcfg or MNeRVTrainConfig()
    video = video.to(device)
    model = model.to(device)
    opt = Adam(model.parameters(), lr=tcfg.lr)

    T = video.shape[0]
    idx_full = torch.arange(T, device=device)
    t0 = time.time()
    decay_steps = max(int(tcfg.steps * tcfg.tf_decay_frac), 1)
    for step in range(tcfg.steps):
        opt.zero_grad(set_to_none=True)
        # Scheduled sampling: tf_prob ramps 1.0 -> 0.0 over decay_steps.
        tf_prob = max(1.0 - step / decay_steps, 0.0)
        teacher = video if torch.rand(1).item() < tf_prob else None
        pred = model(idx_full, teacher_frames=teacher)
        loss = F.mse_loss(pred, video)
        loss.backward(); opt.step()

        if step % tcfg.log_every == 0 or step == tcfg.steps - 1:
            with torch.no_grad():
                full = model(idx_full, teacher_frames=None).clamp(0, 1)
                psnr = 10 * math.log10(1.0 / max(F.mse_loss(full, video).item(), 1e-12))
            if verbose:
                print(f"  [mnerv fp32 {step:5d}] loss={float(loss):.5f}  "
                      f"freerun PSNR={psnr:.2f}  ({time.time()-t0:.1f}s)",
                      flush=True)
    return model


def overfit_mnerv_qat(video: torch.Tensor, model: MNeRVBackbone,
                       qcfg: MNeRVQATConfig | None = None,
                       device: str | torch.device = "cpu",
                       verbose: bool = True,
                       prior: nn.Module | None = None,
                       caption_emb: torch.Tensor | None = None):
    qcfg = qcfg or MNeRVQATConfig()
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
    idx_full = torch.arange(T, device=device)
    N_pixels_per_frame = video.numel() // T

    t0 = time.time()
    decay_steps = max(int(qcfg.steps * qcfg.tf_decay_frac), 1)
    for step in range(qcfg.steps):
        opt.zero_grad(set_to_none=True)
        tf_prob = max(1.0 - step / decay_steps, 0.0)
        teacher = video if torch.rand(1).item() < tf_prob else None
        pred = model.forward_q(idx_full, quants, teacher_frames=teacher)
        D = F.mse_loss(pred, video)

        if prior is not None:
            prior_params = prior_params_for_quants(prior, quants,
                                                      caption_emb=caption_emb)
            R = rate_bits_from_prior(quants, prior_params)
        else:
            R = total_rate_bits(quants)

        warmup_steps = max(int(qcfg.steps * qcfg.lambda_warmup_frac), 1)
        lam_eff = qcfg.lambda_rate * min(step / warmup_steps, 1.0)
        loss = D + lam_eff * (R / (T * N_pixels_per_frame))
        loss.backward(); opt.step()

        if step % qcfg.log_every == 0 or step == qcfg.steps - 1:
            with torch.no_grad():
                full = model.forward_q(idx_full, quants, teacher_frames=None).clamp(0, 1)
                psnr = 10 * math.log10(1.0 / max(F.mse_loss(full, video).item(), 1e-12))
                bits = float(R.detach())
            if verbose:
                print(f"  [mnerv qat {step:5d} tf={tf_prob:.2f}] D={float(D):.5f}  "
                      f"freerun PSNR={psnr:.2f}  R~{bits:.0f} bits  "
                      f"({time.time()-t0:.1f}s)", flush=True)
    return quants


@torch.no_grad()
def reconstruct_quantized_mnerv(model: MNeRVBackbone, quants: nn.ModuleDict
                                  ) -> torch.Tensor:
    device = next(model.parameters()).device
    idx = torch.arange(model.n_frames, device=device)
    return model.forward_q(idx, quants, teacher_frames=None)
