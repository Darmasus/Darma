"""RD benchmark: NeRV backbone vs hash-grid INR vs AV1 / x265.

For each lambda_rate, fp32-overfit the NeRV on the held-out clip, then
prior-free QAT (per-tensor Laplace rate term), then encode via existing
`encode_codec`. Report bytes / bpp / PSNR. Compare to ffmpeg baselines
at sweep of bitrates.
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

from cool_chic.nerv import NeRVBackbone, NeRVConfig
from cool_chic.train_nerv import (
    overfit_nerv, overfit_nerv_qat, reconstruct_quantized_nerv,
    NeRVTrainConfig, NeRVQATConfig,
)
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


def _ffmpeg_encode(codec: str, src: str, kbps: int, out_dir: Path) -> Path:
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
    subprocess.run(cmd, check=True)
    return out


def _ffmpeg_point(codec: str, src: str, kbps: int, video: torch.Tensor,
                    out_dir: Path) -> RDPoint:
    t0 = time.time()
    mp4 = _ffmpeg_encode(codec, src, kbps, out_dir)
    bytes_ = mp4.stat().st_size
    Tf, _, H, W = video.shape
    bpp = bytes_ * 8 / (Tf * H * W)
    recon = read_video(str(mp4), max_frames=Tf)
    n = min(Tf, recon.shape[0])
    psnr = _psnr(recon[:n].cpu(), video[:n].cpu())
    return RDPoint(codec, f"{kbps}k", bytes_, bpp, psnr, time.time() - t0)


def _nerv_point(video, *, embed_dim, base_ch, fp32_steps, qat_steps,
                  lr, lam, device, label, frames_per_step=0,
                  prior=None, caption_emb=None) -> RDPoint:
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
    ap.add_argument("--fp32-steps", type=int, default=2500)
    ap.add_argument("--qat-steps",  type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-3)
    # NeRV size ladder: (label, embed_dim, base_ch)
    ap.add_argument("--sizes", default="tiny:16:32;small:32:64;medium:48:96")
    ap.add_argument("--lambdas", default="0.0,1e-3,1e-2")
    ap.add_argument("--ffmpeg-kbps", default="50,100,250,500,1000,2000")
    ap.add_argument("--frames-per-step", type=int, default=0,
                    help="0 = full clip every step; >0 samples this many frames")
    ap.add_argument("--prior", default=None,
                    help="path to NeRV-trained prior .pt (caption-conditioned). "
                          "If unset, uses per-tensor Laplace rate term.")
    ap.add_argument("--caption", default="",
                    help="caption to feed the prior (FiLM conditioning)")
    ap.add_argument("--out-dir", default="cool_chic/benchmarks/out")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    video = read_video(args.source)
    Tf, _, H, W = video.shape
    total_px = Tf * H * W
    print(f"source: {args.source}  {Tf}x{W}x{H}  ({total_px} px)")

    prior = None
    caption_emb = None
    if args.prior:
        prior = load_prior(args.prior, device=args.device)
        if args.caption:
            caption_emb = encode_caption(args.caption, device=args.device)
        print(f"loaded prior from {args.prior}  caption_dim={prior.caption_dim}  "
              f"using_caption={'yes' if caption_emb is not None else 'no'}")

    points: list[RDPoint] = []
    for size_spec in args.sizes.split(";"):
        label_size, ed_s, bc_s = size_spec.split(":")
        ed = int(ed_s); bc = int(bc_s)
        for lam_str in args.lambdas.split(","):
            lam = float(lam_str)
            label = f"{label_size}_lam{lam:.0e}"
            print(f"\n=== nerv {label}  embed={ed} base_ch={bc} ===")
            p = _nerv_point(video, embed_dim=ed, base_ch=bc,
                              fp32_steps=args.fp32_steps,
                              qat_steps=args.qat_steps, lr=args.lr,
                              lam=lam, device=args.device, label=label,
                              frames_per_step=args.frames_per_step,
                              prior=prior, caption_emb=caption_emb)
            points.append(p)
            print(f"-> {p.bytes} B  bpp={p.bpp:.4f}  PSNR={p.psnr:.2f}")

    for codec in ("x265", "av1"):
        for kbps in [int(k) for k in args.ffmpeg_kbps.split(",")]:
            p = _ffmpeg_point(codec, args.source, kbps, video, out_dir)
            points.append(p)
            print(f"  [{codec:4s} {kbps:>5d}k] {p.bytes:>6d} B  bpp={p.bpp:.4f}  "
                  f"PSNR={p.psnr:.2f}")

    print("\n=== RD TABLE (sorted by bpp) ===")
    print(f"{'codec':6s} {'label':>20s} {'bytes':>7s} {'bpp':>8s} {'PSNR':>6s}")
    for p in sorted(points, key=lambda p: p.bpp):
        print(f"{p.codec:6s} {p.label:>20s} {p.bytes:>7d} {p.bpp:>8.4f} {p.psnr:>6.2f}")

    name = Path(args.source).stem
    out = out_dir / f"rd_nerv_{name}.json"
    out.write_text(json.dumps([asdict(p) for p in points], indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
