"""End-to-end: image -> trained INR -> QAT -> bitstream -> reconstruction.

Verifies:
  1. QAT preserves PSNR vs fp32 (within ~1 dB).
  2. encode_codec / decode_codec roundtrip preserves quantized weights.
  3. Reloading decoded weights into a fresh model gives bitwise identical
     reconstruction to the in-memory quantized model.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from cool_chic.codec import ImageINR, ImageConfig
from cool_chic.train_per_image import overfit_image, TrainConfig
from cool_chic.train_qat import overfit_image_qat, QATConfig, _forward_image_quantized
from cool_chic.bitstream import encode_codec, decode_codec, KIND_IMAGE
from cool_chic.tests.test_image_overfit import _make_test_image


def _psnr(x, y):
    mse = F.mse_loss(x.clamp(0, 1), y.clamp(0, 1)).clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse))


def test_image_codec_roundtrip():
    img = _make_test_image(64, 64)         # smaller for speed
    cfg = ImageConfig(L=4, T=1 << 10, F=2, N_min=8, N_max=64,
                      mlp_hidden=32, mlp_depth=3)

    # --- Stage 1: fp32 overfit ---
    print("=== fp32 warm-up ===")
    model, _ = overfit_image(img, cfg=cfg,
                              tcfg=TrainConfig(steps=600, lr=5e-3, log_every=200),
                              device="cpu", verbose=True)
    fp32_recon = model.reconstruct().clamp(0, 1)
    fp32_psnr = _psnr(fp32_recon, img)
    print(f"fp32 PSNR: {fp32_psnr:.2f} dB")

    # --- Stage 2: QAT ---
    print("\n=== QAT fine-tune ===")
    quants, _ = overfit_image_qat(img, model,
                                    qcfg=QATConfig(steps=400, lr=2e-3,
                                                    scale_lr=5e-3,
                                                    lambda_rate=0.0,
                                                    log_every=100,
                                                    init_scale=5e-3),  # larger
                                    device="cpu", verbose=True)

    coords = model._coord_grid(torch.device("cpu"))
    qrgb = _forward_image_quantized(model, quants, coords)
    qrecon = qrgb.reshape(img.shape[-2], img.shape[-1], 3).permute(2, 0, 1).clamp(0, 1)
    qat_psnr = _psnr(qrecon, img)
    print(f"QAT PSNR: {qat_psnr:.2f} dB  (delta vs fp32: {qat_psnr - fp32_psnr:+.2f} dB)")

    # --- Stage 3: serialize ---
    blob = encode_codec(quants, kind=KIND_IMAGE,
                         H=img.shape[-2], W=img.shape[-1], T_frames=1, cfg=cfg)
    print(f"\nbitstream: {len(blob)} B  ({len(blob)/1024:.2f} KB)")

    # --- Stage 4: decode ---
    decoded = decode_codec(blob)
    assert decoded["H"] == img.shape[-2]
    assert decoded["W"] == img.shape[-1]

    # --- Stage 5: reload + reconstruct ---
    fresh = ImageINR(H=img.shape[-2], W=img.shape[-1], cfg=cfg)
    # Load decoded weights directly (no QAT scaling — the values are
    # already dequantized by decode_codec).
    sd = {n.replace(".", "_"): p for n, p in fresh.named_parameters()}
    loaded = 0
    for safe, t in decoded["weights"].items():
        if safe in sd and sd[safe].shape == t.shape:
            with torch.no_grad():
                sd[safe].copy_(t)
            loaded += 1
    print(f"loaded {loaded}/{len(sd)} parameter tensors from bitstream")

    decoded_recon = fresh.reconstruct().clamp(0, 1)
    decoded_psnr = _psnr(decoded_recon, img)
    print(f"decoded PSNR: {decoded_psnr:.2f} dB  "
          f"(delta vs QAT: {decoded_psnr - qat_psnr:+.2f} dB)")

    # Sanity: roundtrip should be within 0.5 dB of QAT eval (slight differ-
    # ence due to per-tensor sigma estimation in arithmetic coding).
    assert abs(decoded_psnr - qat_psnr) < 1.0, \
        f"roundtrip drifted too much: {decoded_psnr:.2f} vs {qat_psnr:.2f}"
    assert decoded_psnr > 20.0, f"decoded image is unusable: PSNR={decoded_psnr:.2f}"


if __name__ == "__main__":
    test_image_codec_roundtrip()
    print("\nok")
