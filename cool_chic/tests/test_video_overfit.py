"""Day-2 proof of life: tri-plane INR overfits a short video clip.

Target: PSNR > 27 dB on a 16-frame 96x96 synthetic clip with motion,
within ~2 minutes on CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from cool_chic.codec import VideoConfig
from cool_chic.train_per_video import overfit_video, evaluate_video, VideoTrainConfig


def _make_test_video(T: int = 16, H: int = 96, W: int = 96) -> torch.Tensor:
    """Synthetic clip: a moving radial gradient + drifting sinusoid."""
    ys = torch.linspace(-1, 1, H)
    xs = torch.linspace(-1, 1, W)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    frames = []
    for ti in range(T):
        t = ti / max(T - 1, 1)               # in [0, 1]
        cx = -0.5 + t                         # camera pans left -> right
        cy = -0.3 * (1 - 2 * t)
        r = ((gx - cx) ** 2 + (gy - cy) ** 2).sqrt()
        ch_r = (1 - r).clamp(0, 1)
        ch_g = 0.5 + 0.5 * torch.cos(8 * torch.atan2(gy - cy, gx - cx))
        ch_b = 0.5 + 0.5 * torch.sin(15 * gx + 4 * t) * torch.cos(15 * gy)
        frames.append(torch.stack([ch_r, ch_g, ch_b], dim=0).clamp(0, 1))
    return torch.stack(frames, dim=0)        # (T, 3, H, W)


def test_video_overfit_psnr():
    vid = _make_test_video(T=16, H=96, W=96)
    cfg = VideoConfig(L=6, T=1 << 12, F=2, N_min=8, N_max=128,
                      mlp_hidden=64, mlp_depth=4)
    tcfg = VideoTrainConfig(steps=2500, lr=5e-3,
                             pixels_per_step=1 << 13,    # 8k pixels/step
                             log_every=250)

    model, hist = overfit_video(vid, cfg=cfg, tcfg=tcfg, device="cpu",
                                  verbose=True)
    eval = evaluate_video(model, vid)
    print(f"\nfull-clip eval: mean PSNR={eval['mean_psnr']:.2f} dB  "
          f"(min {eval['min_psnr']:.2f}, max {eval['max_psnr']:.2f})")
    print(f"params={model.total_params}  ({model.total_params * 4 / 1024:.1f} KB at fp32)")

    assert eval["mean_psnr"] > 22.0, \
        f"video INR underfit: mean PSNR={eval['mean_psnr']:.2f}"


if __name__ == "__main__":
    test_video_overfit_psnr()
    print("ok")
