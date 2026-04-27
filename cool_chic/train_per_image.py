"""Per-image overfit loop. Encoder = optimizer.

Args:
  image      (3, H, W) float in [0, 1]
  steps      number of Adam iterations
  lr         learning rate
  device     cuda or cpu

Returns:
  trained ImageINR model + history dict (loss, psnr per logged step)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.optim import Adam

from .codec import ImageINR, ImageConfig


@dataclass
class TrainConfig:
    steps: int = 2000
    lr: float = 5e-3        # large because the model is tiny
    log_every: int = 50
    grid_lr: float | None = None   # if set, separate LR for hash tables


def _psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = F.mse_loss(x.clamp(0, 1), y.clamp(0, 1)).clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse))


def overfit_image(image: torch.Tensor,
                   cfg: ImageConfig | None = None,
                   tcfg: TrainConfig | None = None,
                   device: str | torch.device = "cpu",
                   verbose: bool = True) -> tuple[ImageINR, dict]:
    cfg = cfg or ImageConfig()
    tcfg = tcfg or TrainConfig()

    assert image.dim() == 3 and image.shape[0] == 3, "expected (3, H, W)"
    image = image.to(device)
    _, H, W = image.shape

    model = ImageINR(H=H, W=W, cfg=cfg).to(device)
    if tcfg.grid_lr is not None:
        opt = Adam([
            {"params": model.grid.parameters(), "lr": tcfg.grid_lr},
            {"params": model.mlp.parameters(),  "lr": tcfg.lr},
        ])
    else:
        opt = Adam(model.parameters(), lr=tcfg.lr)

    history = {"step": [], "loss": [], "psnr": [], "wall_s": []}
    target = image                               # (3, H, W)
    t0 = time.time()

    for step in range(tcfg.steps):
        opt.zero_grad(set_to_none=True)
        recon = model.reconstruct(device=device)
        loss = F.mse_loss(recon, target)
        loss.backward()
        opt.step()

        if step % tcfg.log_every == 0 or step == tcfg.steps - 1:
            with torch.no_grad():
                ps = _psnr(recon, target)
                loss_val = float(loss.detach())
            elapsed = time.time() - t0
            history["step"].append(step)
            history["loss"].append(loss_val)
            history["psnr"].append(ps)
            history["wall_s"].append(elapsed)
            if verbose:
                print(f"[step {step:5d} t={elapsed:6.1f}s] "
                      f"loss={loss_val:.5f}  PSNR={ps:5.2f} dB", flush=True)

    return model, history
