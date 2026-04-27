"""Proof of life: hash grid + MLP can overfit a single image.

Target: PSNR > 30 dB on a 128x128 synthetic image in <60 s on CPU.
If this fails, the rest of the architecture is moot.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from cool_chic.codec import ImageConfig
from cool_chic.train_per_image import overfit_image, TrainConfig


def _make_test_image(H: int = 128, W: int = 128) -> torch.Tensor:
    """Synthetic image with low + high frequency content."""
    ys = torch.linspace(-1, 1, H)
    xs = torch.linspace(-1, 1, W)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    r = (gx ** 2 + gy ** 2).sqrt()

    # Three channels: radial gradient, angular pattern, fine sinusoid.
    ch_r = (1 - r).clamp(0, 1)
    ch_g = 0.5 + 0.5 * torch.cos(8 * torch.atan2(gy, gx))
    ch_b = 0.5 + 0.5 * torch.sin(20 * gx) * torch.cos(20 * gy)
    img = torch.stack([ch_r, ch_g, ch_b], dim=0).clamp(0, 1)
    return img


def test_image_overfit_psnr():
    img = _make_test_image(128, 128)
    cfg = ImageConfig(L=6, T=1 << 12, F=2, N_min=8, N_max=128,
                      mlp_hidden=64, mlp_depth=4)
    tcfg = TrainConfig(steps=1500, lr=5e-3, log_every=200)

    model, hist = overfit_image(img, cfg=cfg, tcfg=tcfg, device="cpu",
                                  verbose=True)
    final_psnr = hist["psnr"][-1]
    print(f"\nfinal: PSNR={final_psnr:.2f} dB  "
          f"params={model.total_params}  ({model.total_params * 4 / 1024:.1f} KB at fp32)")

    assert final_psnr > 25.0, f"INR failed to overfit: PSNR={final_psnr:.2f}"


if __name__ == "__main__":
    test_image_overfit_psnr()
    print("ok")
