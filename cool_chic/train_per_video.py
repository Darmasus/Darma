"""Per-video overfit loop for the tri-plane INR codec.

Reconstructing a full T*H*W coordinate grid every step is expensive, so we
sample a random subset of pixels per step (stochastic gradient over space-
time). This is also more memory-efficient than dense reconstruction.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.optim import Adam

from .codec import VideoINR, VideoConfig


@dataclass
class VideoTrainConfig:
    steps: int = 4000
    lr: float = 5e-3
    pixels_per_step: int = 1 << 14   # 16k random pixels per step
    log_every: int = 100
    grid_lr: float | None = None


def _psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = F.mse_loss(x.clamp(0, 1), y.clamp(0, 1)).clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse))


def overfit_video(video: torch.Tensor,
                   cfg: VideoConfig | None = None,
                   tcfg: VideoTrainConfig | None = None,
                   device: str | torch.device = "cpu",
                   verbose: bool = True) -> tuple[VideoINR, dict]:
    """video: (T, 3, H, W) float in [0, 1]."""
    cfg = cfg or VideoConfig()
    tcfg = tcfg or VideoTrainConfig()

    assert video.dim() == 4 and video.shape[1] == 3, "expected (T, 3, H, W)"
    video = video.to(device)
    T, _, H, W = video.shape

    model = VideoINR(T_frames=T, H=H, W=W, cfg=cfg).to(device)
    opt = Adam(model.parameters(), lr=tcfg.lr)

    # Pre-flatten target for fast indexed lookup.
    target_flat = video.permute(0, 2, 3, 1).reshape(-1, 3)   # (T*H*W, 3)
    N = target_flat.shape[0]
    t_denom = max(T - 1, 1)
    h_denom = max(H - 1, 1)
    w_denom = max(W - 1, 1)

    history = {"step": [], "loss": [], "psnr": [], "wall_s": []}
    t0 = time.time()

    for step in range(tcfg.steps):
        opt.zero_grad(set_to_none=True)

        # Random pixel sample.
        idx = torch.randint(0, N, (tcfg.pixels_per_step,), device=device)
        ti = idx // (H * W)
        rem = idx % (H * W)
        yi = rem // W
        xi = rem % W
        coords = torch.stack([
            xi.float() / w_denom,
            yi.float() / h_denom,
            ti.float() / t_denom,
        ], dim=-1)
        feats = model.grid(coords)
        pred = model.mlp(feats)
        loss = F.mse_loss(pred, target_flat[idx])
        loss.backward()
        opt.step()

        if step % tcfg.log_every == 0 or step == tcfg.steps - 1:
            with torch.no_grad():
                # Cheap eval on the same sampled pixels.
                ps = _psnr(pred, target_flat[idx])
                loss_val = float(loss.detach())
            elapsed = time.time() - t0
            history["step"].append(step)
            history["loss"].append(loss_val)
            history["psnr"].append(ps)
            history["wall_s"].append(elapsed)
            if verbose:
                print(f"[step {step:5d} t={elapsed:6.1f}s] "
                      f"loss={loss_val:.5f}  PSNR={ps:5.2f} dB (sampled)",
                      flush=True)

    return model, history


@torch.no_grad()
def evaluate_video(model: VideoINR, target: torch.Tensor) -> dict:
    """Full reconstruction + per-frame PSNR. Returns mean/min PSNR."""
    device = next(model.parameters()).device
    target = target.to(device)
    recon = model.reconstruct(device=device).clamp(0, 1)
    per_frame = []
    for i in range(target.shape[0]):
        per_frame.append(_psnr(recon[i], target[i]))
    return {
        "mean_psnr": sum(per_frame) / len(per_frame),
        "min_psnr": min(per_frame),
        "max_psnr": max(per_frame),
        "per_frame": per_frame,
    }
