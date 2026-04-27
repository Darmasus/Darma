"""End-to-end video: train -> QAT -> encode -> decode -> reconstruct."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from torch.optim import Adam

from cool_chic.codec import VideoINR, VideoConfig
from cool_chic.train_video_qat import (
    overfit_video_qat, VideoQATConfig, reconstruct_quantized_video,
)
from cool_chic.bitstream import encode_codec, decode_codec, KIND_VIDEO
from cool_chic.tests.test_video_overfit import _make_test_video


def _psnr(x, y):
    mse = F.mse_loss(x.clamp(0, 1), y.clamp(0, 1)).clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse))


def test_video_roundtrip():
    vid = _make_test_video(T=8, H=64, W=64)
    cfg = VideoConfig(L=4, T=1 << 12, F=2, N_min=8, N_max=64,
                      mlp_hidden=48, mlp_depth=3)
    device = "cpu"

    model = VideoINR(T_frames=8, H=64, W=64, cfg=cfg).to(device)
    print(f"params: {model.total_params}  ({model.total_params*4/1024:.1f} KB fp32)")

    # fp32 warm-up.
    print("\n=== fp32 warm-up ===")
    opt = Adam(model.parameters(), lr=5e-3)
    target_flat = vid.to(device).permute(0, 2, 3, 1).reshape(-1, 3)
    N = target_flat.shape[0]
    t_d, h_d, w_d = 7, 63, 63
    for step in range(1500):
        opt.zero_grad(set_to_none=True)
        idx = torch.randint(0, N, (8192,), device=device)
        ti = idx // (64*64); rem = idx % (64*64); yi = rem // 64; xi = rem % 64
        coords = torch.stack([xi.float()/w_d, yi.float()/h_d, ti.float()/t_d], -1)
        pred = model.mlp(model.grid(coords))
        loss = F.mse_loss(pred, target_flat[idx])
        loss.backward(); opt.step()
        if step % 300 == 0:
            print(f"  step {step}  loss={float(loss.detach()):.5f}", flush=True)

    # QAT fine-tune.
    print("\n=== QAT ===")
    quants, _ = overfit_video_qat(vid, model,
                                    qcfg=VideoQATConfig(steps=600, lr=2e-3,
                                                        scale_lr=5e-3,
                                                        lambda_rate=0.0,
                                                        pixels_per_step=8192,
                                                        log_every=150,
                                                        init_scale=5e-3),
                                    device=device, verbose=True)

    full_qat = reconstruct_quantized_video(model, quants).clamp(0, 1)
    psnr_qat = _psnr(full_qat, vid)
    print(f"QAT full-clip PSNR: {psnr_qat:.2f} dB")

    # Serialize + reload.
    blob = encode_codec(quants, kind=KIND_VIDEO, H=64, W=64, T_frames=8, cfg=cfg)
    print(f"bitstream: {len(blob)} B  ({len(blob)/1024:.2f} KB)  "
          f"({len(blob)*8/(8*64*64):.4f} bpp)")

    decoded = decode_codec(blob)
    fresh = VideoINR(T_frames=8, H=64, W=64, cfg=cfg)
    sd = {n.replace(".", "_"): p for n, p in fresh.named_parameters()}
    for safe, t in decoded["weights"].items():
        if safe in sd and sd[safe].shape == t.shape:
            with torch.no_grad():
                sd[safe].copy_(t)
    full_dec = fresh.reconstruct().clamp(0, 1)
    psnr_dec = _psnr(full_dec, vid)
    print(f"decoded full-clip PSNR: {psnr_dec:.2f} dB  "
          f"(delta vs QAT: {psnr_dec - psnr_qat:+.2f} dB)")

    assert abs(psnr_dec - psnr_qat) < 1.0
    assert psnr_dec > 17.0


if __name__ == "__main__":
    test_video_roundtrip()
    print("\nok")
