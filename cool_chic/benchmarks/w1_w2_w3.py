"""Head-to-head: Week-1 baseline vs Week-2 prior-aware QAT vs Week-3
caption-conditioned prior-aware QAT.

Same clip, same fp32 warm-up, same capacity. Only the QAT rate term
changes: no prior / non-caption prior / caption prior. All three legs
use the Week-1 per-tensor Laplace encoder (we established in W2 that
per-tensor empirical sigma fits any single INR better than a universal
prior, so the prior's value is purely as a training regularizer).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from torch.optim import Adam

from cool_chic.codec import VideoINR, VideoConfig
from cool_chic.train_video_qat import (
    overfit_video_qat, VideoQATConfig, reconstruct_quantized_video,
)
from cool_chic.bitstream import encode_codec, KIND_VIDEO
from cool_chic.bitstream_v2 import load_prior
from cool_chic.caption import encode_caption
from utils.video_io import read_video


def _psnr(x, y):
    mse = F.mse_loss(x.clamp(0, 1), y.clamp(0, 1)).clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse))


def _fp32_warm(video, model, *, steps, lr, pixels, device):
    target_flat = video.to(device).permute(0, 2, 3, 1).reshape(-1, 3)
    N = target_flat.shape[0]
    T, _, H, W = video.shape
    t_d, h_d, w_d = max(T-1, 1), max(H-1, 1), max(W-1, 1)
    opt = Adam(model.parameters(), lr=lr)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        idx = torch.randint(0, N, (pixels,), device=device)
        ti = idx // (H*W); rem = idx % (H*W); yi = rem // W; xi = rem % W
        coords = torch.stack([xi.float()/w_d, yi.float()/h_d, ti.float()/t_d], -1)
        pred = model.mlp(model.grid(coords))
        loss = F.mse_loss(pred, target_flat[idx])
        loss.backward(); opt.step()
    return model


def _run_qat(warm_state, video, cfg, *, qat_steps, lr, pixels, device,
              lam, prior=None, caption_emb=None):
    model = VideoINR(T_frames=video.shape[0], H=video.shape[-2], W=video.shape[-1],
                      cfg=cfg).to(device)
    model.load_state_dict(warm_state)
    quants, _ = overfit_video_qat(video, model,
                                    qcfg=VideoQATConfig(steps=qat_steps,
                                                        lr=lr * 0.4,
                                                        scale_lr=5e-3,
                                                        lambda_rate=lam,
                                                        pixels_per_step=pixels,
                                                        log_every=max(qat_steps // 4, 1),
                                                        init_scale=5e-3),
                                    device=device, verbose=True,
                                    prior=prior, caption_emb=caption_emb)
    recon = reconstruct_quantized_video(model, quants).clamp(0, 1)
    psnr = _psnr(recon.cpu(), video.cpu())
    Tf, _, H, W = video.shape
    blob = encode_codec(quants, kind=KIND_VIDEO, H=H, W=W, T_frames=Tf, cfg=cfg)
    bpp = len(blob) * 8 / (Tf * H * W)
    del model, quants, recon
    if device == "cuda": torch.cuda.empty_cache()
    return psnr, len(blob), bpp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=str)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--caption", type=str,
                    default="color bars with timecode, rainbow diagonal, green checker pattern, ffmpeg test signal")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32-steps", type=int, default=2000)
    ap.add_argument("--qat-steps",  type=int, default=1500)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--pixels", type=int, default=16384)
    ap.add_argument("--lam", type=float, default=1e-3,
                    help="single lambda for all three legs (pick from W2's sweet spot)")
    ap.add_argument("--prior-nocap", type=str,
                    default="cool_chic/data/prior_v2_nocap.pt")
    ap.add_argument("--prior-cap", type=str,
                    default="cool_chic/data/prior_v2_cap.pt")
    args = ap.parse_args()

    video = read_video(args.source, max_frames=args.frames)
    Tf, _, H, W = video.shape
    total_px = Tf * H * W
    print(f"source: {Tf}x{W}x{H}  ({total_px} px)")
    print(f"caption: {args.caption!r}")

    cfg = VideoConfig(L=5, T=1 << 12, F=2, N_min=16, N_max=256,
                       mlp_hidden=48, mlp_depth=3)

    # Warm up once, reuse for all legs.
    base = VideoINR(T_frames=Tf, H=H, W=W, cfg=cfg).to(args.device)
    print(f"params: {base.total_params}")
    t0 = time.time()
    _fp32_warm(video, base, steps=args.fp32_steps, lr=args.lr,
                pixels=args.pixels, device=args.device)
    warm_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    del base
    if args.device == "cuda": torch.cuda.empty_cache()
    print(f"fp32 warm-up: {time.time()-t0:.1f}s")

    prior_nocap = load_prior(args.prior_nocap, device=args.device)
    prior_cap   = load_prior(args.prior_cap,   device=args.device)
    caption_emb = encode_caption(args.caption, device=args.device)

    results = []

    print(f"\n=== W1 (no prior, lambda=0) ===")
    psnr, bytes_, bpp = _run_qat(warm_state, video, cfg,
                                    qat_steps=args.qat_steps, lr=args.lr,
                                    pixels=args.pixels, device=args.device,
                                    lam=0.0, prior=None)
    results.append(("W1", psnr, bytes_, bpp))
    print(f"-> {bytes_} B  bpp={bpp:.4f}  PSNR={psnr:.2f}")

    print(f"\n=== W2 (non-caption prior, lambda={args.lam}) ===")
    psnr, bytes_, bpp = _run_qat(warm_state, video, cfg,
                                    qat_steps=args.qat_steps, lr=args.lr,
                                    pixels=args.pixels, device=args.device,
                                    lam=args.lam, prior=prior_nocap)
    results.append(("W2", psnr, bytes_, bpp))
    print(f"-> {bytes_} B  bpp={bpp:.4f}  PSNR={psnr:.2f}")

    print(f"\n=== W3 (caption prior, lambda={args.lam}) ===")
    psnr, bytes_, bpp = _run_qat(warm_state, video, cfg,
                                    qat_steps=args.qat_steps, lr=args.lr,
                                    pixels=args.pixels, device=args.device,
                                    lam=args.lam, prior=prior_cap,
                                    caption_emb=caption_emb)
    results.append(("W3", psnr, bytes_, bpp))
    print(f"-> {bytes_} B  bpp={bpp:.4f}  PSNR={psnr:.2f}")

    print("\n=== SUMMARY ===")
    w1_bytes = results[0][2]
    print(f"{'leg':6s} {'bytes':>8s} {'bpp':>7s} {'PSNR':>6s}  {'vs W1':>8s}")
    for leg, psnr, bytes_, bpp in results:
        delta = (bytes_ - w1_bytes) / w1_bytes * 100.0
        print(f"{leg:6s} {bytes_:>8d} {bpp:>7.4f} {psnr:>6.2f}  {delta:>+7.1f}%")

    out = Path("cool_chic/benchmarks/out/w1_w2_w3.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        [{"leg": r[0], "psnr": r[1], "bytes": r[2], "bpp": r[3]} for r in results],
        indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
