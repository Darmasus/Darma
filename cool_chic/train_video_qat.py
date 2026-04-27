"""Quantization-aware training for VideoINR.

Same idea as image QAT but stochastic-pixel. Three tri-plane tables get
their own quantizers, plus the MLP layers. The grid lookups during
training use the quantized table values.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .codec import VideoINR
from .quantize import attach_quantizers, total_rate_bits
from .hash_grid import _hash
from .prior import prior_params_for_quants, rate_bits_from_prior


@dataclass
class VideoQATConfig:
    steps: int = 3000
    lr: float = 2e-3
    scale_lr: float = 5e-3
    lambda_rate: float = 0.0
    pixels_per_step: int = 1 << 13
    log_every: int = 200
    init_scale: float = 5e-3
    # Fraction of QAT steps to linearly ramp lambda_rate from 0 -> target.
    # Lets reconstruction settle before the rate term dominates. 0 = no warmup.
    lambda_warmup_frac: float = 0.25
    # L1 sparsity penalty on the raw INR weights (pre-quantization). Pushes
    # weights toward exactly zero so they compress trivially under the
    # entropy coder. 0 = off.
    l1_lambda: float = 0.0


def _psnr(x, y):
    mse = F.mse_loss(x.clamp(0, 1), y.clamp(0, 1)).clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse))


def _hash_grid_2d_quantized(grid, qtables: list[torch.Tensor],
                              coords_2d: torch.Tensor) -> torch.Tensor:
    """Replays HashGrid2D.forward but reads from the supplied qtables."""
    feats = []
    for l in range(grid.L):
        N = grid.resolutions[l]
        scaled = coords_2d * N
        i0 = scaled.floor().long()
        t = scaled - i0.float()
        corners = []
        for dy in (0, 1):
            for dx in (0, 1):
                c = (i0 + torch.tensor([dx, dy], device=coords_2d.device)) % N
                h = _hash(c, grid.T)
                corners.append(qtables[l][h])
        c00, c10, c01, c11 = corners
        wx, wy = t[..., 0:1], t[..., 1:2]
        f0 = c00 * (1 - wx) + c10 * wx
        f1 = c01 * (1 - wx) + c11 * wx
        feats.append(f0 * (1 - wy) + f1 * wy)
    return torch.cat(feats, dim=-1)


def _forward_video_quantized(model: VideoINR, quants: nn.ModuleDict,
                               coords: torch.Tensor) -> torch.Tensor:
    """coords: (N, 3) (x,y,t) in [0,1]. Returns (N, 3) RGB."""
    # Tri-plane lookups via quantized tables.
    x, y, t = coords[..., 0:1], coords[..., 1:2], coords[..., 2:3]

    qxy = [quants[f"grid_xy_tables_{l}"].quantized() for l in range(model.grid.xy.L)]
    qxt = [quants[f"grid_xt_tables_{l}"].quantized() for l in range(model.grid.xt.L)]
    qyt = [quants[f"grid_yt_tables_{l}"].quantized() for l in range(model.grid.yt.L)]

    f_xy = _hash_grid_2d_quantized(model.grid.xy, qxy, torch.cat([x, y], dim=-1))
    f_xt = _hash_grid_2d_quantized(model.grid.xt, qxt, torch.cat([x, t], dim=-1))
    f_yt = _hash_grid_2d_quantized(model.grid.yt, qyt, torch.cat([y, t], dim=-1))
    h = f_xy + f_xt + f_yt

    layer_idx = 0
    for layer in model.mlp.net:
        if isinstance(layer, nn.Linear):
            qw = quants[f"mlp_net_{layer_idx}_weight"].quantized()
            qb = quants[f"mlp_net_{layer_idx}_bias"].quantized()
            h = F.linear(h, qw, qb)
        elif isinstance(layer, nn.ReLU):
            h = F.relu(h)
        layer_idx += 1
    return h


def overfit_video_qat(video: torch.Tensor, model: VideoINR,
                       qcfg: VideoQATConfig | None = None,
                       device: str | torch.device = "cpu",
                       verbose: bool = True,
                       prior: nn.Module | None = None,
                       caption_emb: torch.Tensor | None = None):
    """If `prior` is passed, the rate term uses the prior's per-weight
    Gaussian NLL. Otherwise falls back to the Week-1 per-tensor Laplace
    estimate. `lambda_rate` in the config controls the R-D tradeoff
    regardless."""
    qcfg = qcfg or VideoQATConfig()
    video = video.to(device)
    model = model.to(device)
    quants = attach_quantizers(model, init_scale=qcfg.init_scale).to(device)

    # Freeze the prior during INR overfit — it stays at its pretrained
    # state; only the INR weights adapt.
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

    T, _, H, W = video.shape
    target_flat = video.permute(0, 2, 3, 1).reshape(-1, 3)
    N = target_flat.shape[0]
    t_denom = max(T - 1, 1); h_denom = max(H - 1, 1); w_denom = max(W - 1, 1)

    history = {"step": [], "loss": [], "psnr": [], "bits": [], "wall_s": []}
    t0 = time.time()

    for step in range(qcfg.steps):
        opt.zero_grad(set_to_none=True)
        idx = torch.randint(0, N, (qcfg.pixels_per_step,), device=device)
        ti = idx // (H * W); rem = idx % (H * W); yi = rem // W; xi = rem % W
        coords = torch.stack([
            xi.float() / w_denom,
            yi.float() / h_denom,
            ti.float() / t_denom,
        ], dim=-1)

        pred = _forward_video_quantized(model, quants, coords)
        D = F.mse_loss(pred, target_flat[idx])
        if prior is not None:
            # Refresh prior predictions each step — they're cheap (the
            # prior is a small MLP) and keeping them fresh means the rate
            # term tracks any metadata drift. The prior outputs are
            # constants in the graph; gradient flows only through the
            # STE-rounded integer codes.
            prior_params = prior_params_for_quants(prior, quants,
                                                      caption_emb=caption_emb)
            R = rate_bits_from_prior(quants, prior_params)
        else:
            R = total_rate_bits(quants)
        warmup_steps = max(int(qcfg.steps * qcfg.lambda_warmup_frac), 1)
        lam_eff = qcfg.lambda_rate * min(step / warmup_steps, 1.0)
        loss = D + lam_eff * (R / N)
        if qcfg.l1_lambda > 0.0:
            l1 = sum(q.param.abs().sum() for q in quants.values())
            loss = loss + qcfg.l1_lambda * l1
        loss.backward()
        opt.step()

        if step % qcfg.log_every == 0 or step == qcfg.steps - 1:
            with torch.no_grad():
                ps = _psnr(pred, target_flat[idx])
                bits = float(R.detach())
            elapsed = time.time() - t0
            history["step"].append(step); history["loss"].append(float(loss.detach()))
            history["psnr"].append(ps);   history["bits"].append(bits)
            history["wall_s"].append(elapsed)
            if verbose:
                print(f"[vqat {step:5d} t={elapsed:6.1f}s] D={float(D.detach()):.5f}  "
                      f"PSNR={ps:5.2f}  R~{bits:.0f} bits", flush=True)
    return quants, history


@torch.no_grad()
def reconstruct_quantized_video(model: VideoINR, quants: nn.ModuleDict,
                                  chunk: int = 1 << 17) -> torch.Tensor:
    device = next(model.parameters()).device
    coords = model._coord_grid(device)
    outs = []
    for i in range(0, coords.shape[0], chunk):
        c = coords[i:i + chunk]
        outs.append(_forward_video_quantized(model, quants, c))
    rgb = torch.cat(outs, dim=0)
    return rgb.reshape(model.T_frames, model.H, model.W, 3).permute(0, 3, 1, 2)
