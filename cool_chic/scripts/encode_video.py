"""Encode a video file to a Cool-Chic .cc bitstream.

Pipeline:
  1. Read video -> (T, 3, H, W) tensor.
  2. fp32 overfit (warm start).
  3. Quantization-aware fine-tune.
  4. Serialize to .cc bytes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from cool_chic.codec import VideoINR, VideoConfig
from cool_chic.train_per_video import overfit_video, VideoTrainConfig
from cool_chic.train_video_qat import overfit_video_qat, VideoQATConfig, reconstruct_quantized_video
from cool_chic.bitstream import encode_codec, KIND_VIDEO
from utils.video_io import read_video


def _psnr_video(orig: torch.Tensor, recon: torch.Tensor) -> float:
    import torch.nn.functional as F
    mse = F.mse_loss(orig.clamp(0, 1), recon.clamp(0, 1)).clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=str)
    ap.add_argument("output", type=str)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fp32-steps", type=int, default=2000)
    ap.add_argument("--qat-steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--init-scale", type=float, default=5e-3)
    ap.add_argument("--pixels", type=int, default=1 << 14)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    # Capacity knob: bigger T -> more params -> bigger file -> better PSNR.
    ap.add_argument("--T", type=int, default=1 << 13,
                    help="hash table size per level")
    ap.add_argument("--L", type=int, default=6, help="hash levels")
    args = ap.parse_args()

    print(f"reading {args.input}")
    video = read_video(args.input, max_frames=args.max_frames)
    T, _, H, W = video.shape
    print(f"clip: {T} frames {W}x{H}")

    cfg = VideoConfig(L=args.L, T=args.T, F=2, N_min=16, N_max=256,
                       mlp_hidden=64, mlp_depth=4)
    model = VideoINR(T_frames=T, H=H, W=W, cfg=cfg).to(args.device)
    print(f"INR params: {model.total_params}  ({model.total_params * 4 / 1024:.1f} KB fp32)")

    print("\n=== fp32 warm-up ===")
    t0 = time.time()
    overfit_video(video, cfg=cfg,
                   tcfg=VideoTrainConfig(steps=args.fp32_steps, lr=args.lr,
                                           pixels_per_step=args.pixels,
                                           log_every=max(args.fp32_steps // 10, 1)),
                   device=args.device, verbose=True)
    # overfit_video creates its own model; but we already have ours. Redo:
    # (Refactor needed — for now repeat with model in-hand.)

    # Repeat using the *external* model so we can carry state into QAT.
    import torch.nn.functional as F
    from torch.optim import Adam
    opt = Adam(model.parameters(), lr=args.lr)
    target_flat = video.to(args.device).permute(0, 2, 3, 1).reshape(-1, 3)
    N = target_flat.shape[0]
    t_d, h_d, w_d = max(T - 1, 1), max(H - 1, 1), max(W - 1, 1)
    for step in range(args.fp32_steps):
        opt.zero_grad(set_to_none=True)
        idx = torch.randint(0, N, (args.pixels,), device=args.device)
        ti = idx // (H * W); rem = idx % (H * W); yi = rem // W; xi = rem % W
        coords = torch.stack([xi.float()/w_d, yi.float()/h_d, ti.float()/t_d], -1)
        pred = model.mlp(model.grid(coords))
        loss = F.mse_loss(pred, target_flat[idx])
        loss.backward(); opt.step()
        if step % max(args.fp32_steps // 10, 1) == 0:
            ps = -10 * torch.log10(loss.detach().clamp_min(1e-12))
            print(f"  fp32 step {step:5d}  loss={float(loss):.5f}  PSNR≈{float(ps):.2f}",
                  flush=True)

    print(f"\nfp32 elapsed: {time.time()-t0:.1f}s")

    print("\n=== QAT fine-tune ===")
    quants, _ = overfit_video_qat(video, model,
                                    qcfg=VideoQATConfig(steps=args.qat_steps,
                                                        lr=args.lr * 0.4,
                                                        scale_lr=5e-3,
                                                        lambda_rate=0.0,
                                                        pixels_per_step=args.pixels,
                                                        log_every=max(args.qat_steps // 10, 1),
                                                        init_scale=args.init_scale),
                                    device=args.device, verbose=True)

    # Full-clip eval on the QAT model.
    full = reconstruct_quantized_video(model, quants).clamp(0, 1)
    psnr_qat = _psnr_video(video.to(args.device), full)
    print(f"\nQAT full-clip PSNR: {psnr_qat:.2f} dB")

    print("\n=== serialize ===")
    blob = encode_codec(quants, kind=KIND_VIDEO, H=H, W=W, T_frames=T, cfg=cfg)
    Path(args.output).write_bytes(blob)
    bpp = len(blob) * 8 / (T * H * W)
    print(f"wrote {args.output}: {len(blob)/1024:.2f} KB  ({bpp:.4f} bpp)")
    print(f"final PSNR: {psnr_qat:.2f} dB @ {bpp:.4f} bpp")


if __name__ == "__main__":
    main()
