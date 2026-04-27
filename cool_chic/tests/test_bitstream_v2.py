"""Roundtrip a small INR through the v2 bitstream (prior-conditioned rANS)."""
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
from cool_chic.bitstream_v2 import (
    encode_codec_v2, decode_codec_v2, load_prior, MAGIC_V2,
)
from cool_chic.bitstream import KIND_VIDEO
from cool_chic.tests.test_video_overfit import _make_test_video


def _psnr(x, y):
    mse = F.mse_loss(x.clamp(0, 1), y.clamp(0, 1)).clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse))


def test_v2_roundtrip():
    vid = _make_test_video(T=8, H=64, W=64)
    cfg = VideoConfig(L=4, T=1 << 12, F=2, N_min=8, N_max=64,
                      mlp_hidden=48, mlp_depth=3)
    device = "cpu"

    model = VideoINR(T_frames=8, H=64, W=64, cfg=cfg).to(device)

    # fp32 warm-up.
    print("=== fp32 warm-up ===")
    opt = Adam(model.parameters(), lr=5e-3)
    target_flat = vid.to(device).permute(0, 2, 3, 1).reshape(-1, 3)
    N = target_flat.shape[0]
    t_d, h_d, w_d = 7, 63, 63
    for step in range(800):
        opt.zero_grad(set_to_none=True)
        idx = torch.randint(0, N, (8192,), device=device)
        ti = idx // (64*64); rem = idx % (64*64); yi = rem // 64; xi = rem % 64
        coords = torch.stack([xi.float()/w_d, yi.float()/h_d, ti.float()/t_d], -1)
        pred = model.mlp(model.grid(coords))
        loss = F.mse_loss(pred, target_flat[idx])
        loss.backward(); opt.step()

    # QAT.
    print("\n=== QAT ===")
    quants, _ = overfit_video_qat(vid, model,
                                    qcfg=VideoQATConfig(steps=400, lr=2e-3,
                                                        scale_lr=5e-3,
                                                        lambda_rate=0.0,
                                                        pixels_per_step=8192,
                                                        log_every=100,
                                                        init_scale=5e-3),
                                    device=device, verbose=True)
    full_qat = reconstruct_quantized_video(model, quants).clamp(0, 1)
    psnr_qat = _psnr(full_qat, vid)
    print(f"\nQAT full-clip PSNR: {psnr_qat:.2f} dB")

    # Encode/decode via v2.
    # Use the latest hash-grid prior. test_bitstream_v2 is a *roundtrip*
    # test — what matters is encode/decode consistency, not PSNR vs target.
    prior = load_prior("cool_chic/data/prior_v4_nocap.pt", device=device)
    blob = encode_codec_v2(quants, kind=KIND_VIDEO, H=64, W=64, T_frames=8,
                            cfg=cfg, prior=prior)
    print(f"\nv2 bitstream: {len(blob)} B  ({len(blob)/1024:.2f} KB)  "
          f"({len(blob)*8/(8*64*64):.4f} bpp)")

    decoded = decode_codec_v2(blob, prior=prior)
    assert decoded["H"] == 64 and decoded["W"] == 64

    fresh = VideoINR(T_frames=8, H=64, W=64, cfg=cfg)
    sd = {n.replace(".", "_"): p for n, p in fresh.named_parameters()}
    n_loaded = 0
    for safe, t in decoded["weights"].items():
        if safe in sd and sd[safe].shape == t.shape:
            with torch.no_grad():
                sd[safe].copy_(t)
            n_loaded += 1
    print(f"loaded {n_loaded}/{len(sd)} tensors")
    assert n_loaded == len(sd)

    full_dec = fresh.reconstruct().clamp(0, 1)
    psnr_dec = _psnr(full_dec, vid)
    print(f"decoded full-clip PSNR: {psnr_dec:.2f} dB  "
          f"(delta vs QAT: {psnr_dec - psnr_qat:+.2f} dB)")

    assert abs(psnr_dec - psnr_qat) < 1.0, \
        f"v2 roundtrip drifted: {psnr_dec:.2f} vs {psnr_qat:.2f}"


if __name__ == "__main__":
    test_v2_roundtrip()
    print("\nok")
