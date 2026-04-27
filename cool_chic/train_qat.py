"""Quantization-aware training for the INR codec.

Adds a learnable per-tensor scale to every parameter group, runs the model
through `quantizer.quantized()` instead of the raw weight, and minimizes
distortion + lambda * rate.

This is what produces a model whose weights compress well — without QAT,
naive rounding of the trained fp32 weights typically loses 3-6 dB of PSNR.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .codec import ImageINR
from .quantize import attach_quantizers, total_rate_bits


@dataclass
class QATConfig:
    steps: int = 2500
    lr: float = 5e-3
    scale_lr: float = 5e-3
    lambda_rate: float = 0.0   # 0 = pure distortion (good for testing QAT-only)
    log_every: int = 200
    init_scale: float = 1e-3


def _psnr(x, y):
    mse = F.mse_loss(x.clamp(0, 1), y.clamp(0, 1)).clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse))


def _swap_in_quantized(model: nn.Module, quants: nn.ModuleDict) -> None:
    """Monkey-patch each parameter to a property that returns the quantized
    view. We keep the original Parameter as the storage; what changes is
    the value the forward pass sees.

    The cleanest way: walk modules and rebind attribute reads. For our
    small INR (Linear + ParameterList) we can just override the underlying
    param tensors right before the forward by writing the quantized view
    back into a non-leaf cache used by the layer.

    To stay simple and correct, we instead let `quants` own the scales,
    and redirect each parameter via a small forward hook that writes the
    quantized tensor into the parameter's `.data` *only* during forward.
    Backward still flows via STE inside `quantized()`.
    """
    # We use a much simpler scheme: replace each param in-place with a
    # detached copy of the quantized view at every forward. Gradients on
    # the underlying param come from the QAT graph below.
    pass  # see overfit_image_qat: we run forward through a custom path.


def _forward_image_quantized(model: ImageINR, quants: nn.ModuleDict,
                              coords: torch.Tensor) -> torch.Tensor:
    """Run ImageINR forward but with each parameter replaced by its
    quantized counterpart. Returns RGB in (H*W, 3)."""
    # 1) Hash grid forward, but using quantized table values.
    feats = []
    grid = model.grid
    for l in range(grid.L):
        N = grid.resolutions[l]
        scaled = coords * N
        i0 = scaled.floor().long()
        t = scaled - i0.float()
        # Get quantized table.
        qtable = quants[f"grid_tables_{l}"].quantized()
        # Replicate HashGrid2D.forward inline (4 corners + bilinear).
        from .hash_grid import _hash
        corners = []
        for dy in (0, 1):
            for dx in (0, 1):
                c = i0 + torch.tensor([dx, dy], device=coords.device)
                c = c % N
                h = _hash(c, grid.T)
                corners.append(qtable[h])
        c00, c10, c01, c11 = corners
        wx, wy = t[..., 0:1], t[..., 1:2]
        f0 = c00 * (1 - wx) + c10 * wx
        f1 = c01 * (1 - wx) + c11 * wx
        feats.append(f0 * (1 - wy) + f1 * wy)
    h = torch.cat(feats, dim=-1)

    # 2) MLP forward, using quantized weights+biases.
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


def overfit_image_qat(image: torch.Tensor,
                       model: ImageINR,
                       qcfg: QATConfig | None = None,
                       device: str | torch.device = "cpu",
                       verbose: bool = True):
    """Continue training `model` with quantization-aware training.

    Typical workflow:
      model, _ = overfit_image(image, ...)            # fp32 warm start
      quants  = overfit_image_qat(image, model, ...)  # add QAT
    """
    qcfg = qcfg or QATConfig()
    image = image.to(device)
    model = model.to(device)
    quants = attach_quantizers(model, init_scale=qcfg.init_scale).to(device)

    # Two parameter groups: fp32 weights vs scale.
    weight_params = [q.param for q in quants.values()]
    scale_params  = [q.log_scale for q in quants.values()]
    opt = Adam([
        {"params": weight_params, "lr": qcfg.lr},
        {"params": scale_params,  "lr": qcfg.scale_lr},
    ])

    coords = model._coord_grid(device)
    target_flat = image.permute(1, 2, 0).reshape(-1, 3)
    H, W = image.shape[-2:]

    history = {"step": [], "loss": [], "psnr": [], "bits": [], "wall_s": []}
    t0 = time.time()

    for step in range(qcfg.steps):
        opt.zero_grad(set_to_none=True)
        rgb = _forward_image_quantized(model, quants, coords)
        D = F.mse_loss(rgb, target_flat)
        R = total_rate_bits(quants)
        loss = D + qcfg.lambda_rate * (R / (H * W))   # rate in bits/pixel
        loss.backward()
        opt.step()

        if step % qcfg.log_every == 0 or step == qcfg.steps - 1:
            with torch.no_grad():
                ps = _psnr(rgb, target_flat)
                bits = float(R.detach())
            elapsed = time.time() - t0
            history["step"].append(step)
            history["loss"].append(float(loss.detach()))
            history["psnr"].append(ps)
            history["bits"].append(bits)
            history["wall_s"].append(elapsed)
            if verbose:
                print(f"[qat {step:5d} t={elapsed:6.1f}s] "
                      f"D={float(D.detach()):.5f}  PSNR={ps:5.2f}  "
                      f"R={bits:.0f} bits ({bits/8/1024:.1f} KB)",
                      flush=True)
    return quants, history
