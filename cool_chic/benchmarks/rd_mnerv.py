"""RD benchmark: M-NeRV (motion-compensated) vs vanilla NeRV vs AV1/x265.

Tests on a clip that should benefit from motion comp (real continuous
content). M-NeRV is decoded sequentially (each frame depends on
previous), so we can't randomly subsample frames at training time.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from cool_chic.mnerv import MNeRVBackbone, MNeRVConfig
from cool_chic.train_mnerv import (
    overfit_mnerv, overfit_mnerv_qat, reconstruct_quantized_mnerv,
    MNeRVTrainConfig, MNeRVQATConfig,
)
from cool_chic.nerv import NeRVBackbone, NeRVConfig
from cool_chic.train_nerv import overfit_nerv, overfit_nerv_qat, reconstruct_quantized_nerv, NeRVTrainConfig, NeRVQATConfig
from cool_chic.bitstream import encode_codec, KIND_VIDEO
from cool_chic.bitstream_v2 import load_prior
from cool_chic.caption import encode_caption
from utils.video_io import read_video


@dataclass
class RDPoint:
    codec: str
    label: str
    bytes: int
    bpp: float
    psnr: float
    seconds: float


def _psnr(x, y):
    mse = F.mse_loss(x.clamp(0, 1), y.clamp(0, 1)).clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse))


def _ffmpeg_point(codec: str, src: str, kbps: int, video, out_dir: Path) -> RDPoint:
    out = out_dir / f"{codec}_{kbps}.mp4"
    common = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
              "-pix_fmt", "yuv420p", "-an"]
    if codec == "x265":
        cmd = common + ["-c:v", "libx265", "-preset", "medium",
                          "-b:v", f"{kbps}k", str(out)]
    elif codec == "av1":
        cmd = common + ["-c:v", "libaom-av1", "-cpu-used", "6",
                          "-b:v", f"{kbps}k", str(out)]
    else:
        raise ValueError(codec)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    bytes_ = out.stat().st_size
    Tf, _, H, W = video.shape
    bpp = bytes_ * 8 / (Tf * H * W)
    recon = read_video(str(out), max_frames=Tf)
    n = min(Tf, recon.shape[0])
    psnr = _psnr(recon[:n].cpu(), video[:n].cpu())
    return RDPoint(codec, f"{kbps}k", bytes_, bpp, psnr, time.time() - t0)


def _mnerv_point(video, *, embed_dim, base_ch, fp32_steps, qat_steps, lr,
                   lam, device, label, prior=None, caption_emb=None) -> RDPoint:
    Tf, _, H, W = video.shape
    t0 = time.time()
    model = MNeRVBackbone(n_frames=Tf, H=H, W=W,
                            cfg=MNeRVConfig(embed_dim=embed_dim, base_ch=base_ch)).to(device)
    overfit_mnerv(video, model,
                    tcfg=MNeRVTrainConfig(steps=fp32_steps, lr=lr,
                                              log_every=max(fp32_steps // 4, 1),
                                              tf_decay_frac=0.5),
                    device=device, verbose=True)
    quants = overfit_mnerv_qat(video, model,
                                  qcfg=MNeRVQATConfig(steps=qat_steps,
                                                        lr=lr * 0.4, scale_lr=5e-3,
                                                        lambda_rate=lam,
                                                        log_every=max(qat_steps // 4, 1),
                                                        init_scale=5e-3,
                                                        lambda_warmup_frac=0.25,
                                                        tf_decay_frac=0.4),
                                  device=device, verbose=True,
                                  prior=prior, caption_emb=caption_emb)
    recon = reconstruct_quantized_mnerv(model, quants).clamp(0, 1)
    psnr = _psnr(recon.cpu(), video.cpu())
    blob = encode_codec(quants, kind=KIND_VIDEO, H=H, W=W, T_frames=Tf, cfg=None)
    bpp = len(blob) * 8 / (Tf * H * W)
    del model, quants, recon
    if device == "cuda": torch.cuda.empty_cache()
    return RDPoint("mnerv", label, len(blob), bpp, psnr, time.time() - t0)


def _nerv_point(video, *, embed_dim, base_ch, fp32_steps, qat_steps, lr, lam,
                  device, label, frames_per_step=16, prior=None, caption_emb=None) -> RDPoint:
    Tf, _, H, W = video.shape
    t0 = time.time()
    model = NeRVBackbone(n_frames=Tf, H=H, W=W,
                          cfg=NeRVConfig(embed_dim=embed_dim, base_ch=base_ch)).to(device)
    overfit_nerv(video, model,
                  tcfg=NeRVTrainConfig(steps=fp32_steps, lr=lr,
                                          log_every=max(fp32_steps // 4, 1),
                                          frames_per_step=frames_per_step),
                  device=device, verbose=True)
    quants = overfit_nerv_qat(video, model,
                                qcfg=NeRVQATConfig(steps=qat_steps,
                                                    lr=lr * 0.4, scale_lr=5e-3,
                                                    lambda_rate=lam,
                                                    log_every=max(qat_steps // 4, 1),
                                                    init_scale=5e-3,
                                                    lambda_warmup_frac=0.25,
                                                    frames_per_step=frames_per_step),
                                device=device, verbose=True,
                                prior=prior, caption_emb=caption_emb)
    recon = reconstruct_quantized_nerv(model, quants).clamp(0, 1)
    psnr = _psnr(recon.cpu(), video.cpu())
    blob = encode_codec(quants, kind=KIND_VIDEO, H=H, W=W, T_frames=Tf, cfg=None)
    bpp = len(blob) * 8 / (Tf * H * W)
    del model, quants, recon
    if device == "cuda": torch.cuda.empty_cache()
    return RDPoint("nerv", label, len(blob), bpp, psnr, time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=str)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32-steps", type=int, default=3000)
    ap.add_argument("--qat-steps",  type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--mnerv-sizes", default="tiny:16:32;small:24:48")
    ap.add_argument("--nerv-sizes",  default="tiny:16:32;small:32:64")
    ap.add_argument("--lambdas",     default="0.0,1e-2,1e-1")
    ap.add_argument("--ffmpeg-kbps", default="20,50,100,250,500")
    ap.add_argument("--prior", default=None,
                    help="optional NeRV-trained prior path (used for both nerv and mnerv)")
    ap.add_argument("--caption", default="")
    ap.add_argument("--out-dir", default="cool_chic/benchmarks/out")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    video = read_video(args.source)
    Tf, _, H, W = video.shape
    total_px = Tf * H * W
    print(f"source: {args.source}  {Tf}x{W}x{H}  ({total_px} px)")

    prior = None; caption_emb = None
    if args.prior:
        prior = load_prior(args.prior, device=args.device)
        if args.caption:
            caption_emb = encode_caption(args.caption, device=args.device)
        print(f"prior loaded ({prior.caption_dim=})  caption_used={caption_emb is not None}")

    points: list[RDPoint] = []

    # --- M-NeRV sweep ---
    for size_spec in args.mnerv_sizes.split(";"):
        ls, eds, bcs = size_spec.split(":")
        ed = int(eds); bc = int(bcs)
        for lam_str in args.lambdas.split(","):
            lam = float(lam_str)
            label = f"{ls}_lam{lam:.0e}"
            print(f"\n=== mnerv {label} ===")
            p = _mnerv_point(video, embed_dim=ed, base_ch=bc,
                              fp32_steps=args.fp32_steps,
                              qat_steps=args.qat_steps, lr=args.lr,
                              lam=lam, device=args.device, label=label,
                              prior=prior, caption_emb=caption_emb)
            points.append(p)
            print(f"-> {p.bytes} B  bpp={p.bpp:.4f}  PSNR={p.psnr:.2f}")

    # --- vanilla NeRV sweep ---
    for size_spec in args.nerv_sizes.split(";"):
        ls, eds, bcs = size_spec.split(":")
        ed = int(eds); bc = int(bcs)
        for lam_str in args.lambdas.split(","):
            lam = float(lam_str)
            label = f"{ls}_lam{lam:.0e}"
            print(f"\n=== nerv {label} ===")
            p = _nerv_point(video, embed_dim=ed, base_ch=bc,
                              fp32_steps=args.fp32_steps,
                              qat_steps=args.qat_steps, lr=args.lr,
                              lam=lam, device=args.device, label=label,
                              prior=prior, caption_emb=caption_emb)
            points.append(p)
            print(f"-> {p.bytes} B  bpp={p.bpp:.4f}  PSNR={p.psnr:.2f}")

    # --- ffmpeg baselines ---
    for codec in ("x265", "av1"):
        for kbps in [int(k) for k in args.ffmpeg_kbps.split(",")]:
            p = _ffmpeg_point(codec, args.source, kbps, video, out_dir)
            points.append(p)
            print(f"  [{codec:4s} {kbps:>5d}k] {p.bytes:>6d} B  bpp={p.bpp:.4f}  "
                  f"PSNR={p.psnr:.2f}")

    print("\n=== RD TABLE ===")
    print(f"{'codec':6s} {'label':>16s} {'bytes':>7s} {'bpp':>7s} {'PSNR':>6s}")
    for p in sorted(points, key=lambda p: p.bpp):
        print(f"{p.codec:6s} {p.label:>16s} {p.bytes:>7d} {p.bpp:>7.4f} {p.psnr:>6.2f}")

    name = Path(args.source).stem
    out = out_dir / f"rd_mnerv_{name}.json"
    out.write_text(json.dumps([asdict(p) for p in points], indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
