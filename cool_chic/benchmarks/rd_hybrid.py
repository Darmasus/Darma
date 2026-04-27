"""RD benchmark: hybrid (x265 baseline + INR residual + L1 sparsity) vs
pure libaom-av1 / pure libx265, on a held-out Vimeo clip.

Each hybrid point is a (x265 CRF, INR lambda, INR L1) tuple. We sweep a
small grid so the RD curve comes out clearly.
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

from cool_chic.codec import VideoConfig
from cool_chic.residual_codec import encode_hybrid, HybridConfig
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
    mp4 = _ffmpeg_encode(codec, src, kbps, out_dir)
    bytes_ = mp4.stat().st_size
    Tf, _, H, W = video.shape
    bpp = bytes_ * 8 / (Tf * H * W)
    recon = read_video(str(mp4), max_frames=Tf)
    n = min(Tf, recon.shape[0])
    psnr = _psnr(recon[:n].cpu(), video[:n].cpu())
    return RDPoint(codec, f"{kbps}k", bytes_, bpp, psnr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=str)
    ap.add_argument("--caption", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prior-cap", default="cool_chic/data/prior_v4_cap.pt")
    ap.add_argument("--fp32-steps", type=int, default=1500)
    ap.add_argument("--qat-steps",  type=int, default=2000)
    ap.add_argument("--pixels", type=int, default=16384)
    # Hybrid sweep: (x265 CRF, INR lambda, INR L1)
    ap.add_argument("--sweep", type=str,
                    default="40,1e-2,0;40,1e-1,0;36,1e-2,0;36,1e-1,0;32,1e-1,0;32,1e-1,1e-4;28,1e-1,0")
    ap.add_argument("--ffmpeg-kbps", default="50,100,250,500,1000,2000")
    ap.add_argument("--inr-config", default="small")
    ap.add_argument("--out-dir", default="cool_chic/benchmarks/out")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    video = read_video(args.source)
    Tf, _, H, W = video.shape
    total_px = Tf * H * W
    print(f"source: {args.source}  {Tf}x{W}x{H}  ({total_px} px)")
    print(f"caption: {args.caption}")

    if args.inr_config == "small":
        cfg = VideoConfig(L=5, T=1 << 12, F=2, N_min=16, N_max=256,
                           mlp_hidden=48, mlp_depth=3)
    else:
        cfg = VideoConfig(L=6, T=1 << 13, F=2, N_min=16, N_max=256,
                           mlp_hidden=64, mlp_depth=4)

    prior = load_prior(args.prior_cap, device=args.device)
    caption_emb = encode_caption(args.caption, device=args.device)

    points: list[RDPoint] = []

    # --- hybrid sweep ---
    for spec in args.sweep.split(";"):
        crf_s, lam_s, l1_s = spec.strip().split(",")
        crf = int(crf_s); lam = float(lam_s); l1 = float(l1_s)
        label = f"crf{crf}_lam{lam:.0e}_l1{l1:.0e}"
        print(f"\n=== HYBRID {label} ===")
        t0 = time.time()
        hcfg = HybridConfig(
            x265_crf=crf, fp32_steps=args.fp32_steps, qat_steps=args.qat_steps,
            lr=5e-3, pixels_per_step=args.pixels,
            lambda_rate=lam, l1_lambda=l1,
            lambda_warmup_frac=0.25, init_scale=2e-3,
        )
        result = encode_hybrid(video, cfg=cfg, hcfg=hcfg, prior=prior,
                                 caption_emb=caption_emb, device=args.device,
                                 verbose=True)
        points.append(RDPoint("hybrid", label, result["bytes"],
                                result["bpp"], result["psnr"]))
        print(f"  took {time.time()-t0:.1f}s")

    # --- pure codec baselines ---
    kbps_list = [int(k) for k in args.ffmpeg_kbps.split(",")]
    for codec in ("x265", "av1"):
        for kbps in kbps_list:
            p = _ffmpeg_point(codec, args.source, kbps, video, out_dir)
            points.append(p)
            print(f"  [{codec:4s} {kbps:5d}k] {p.bytes:>6d} B  bpp={p.bpp:.4f}  "
                  f"PSNR={p.psnr:.2f}")

    print("\n=== RD TABLE (sorted by bpp) ===")
    print(f"{'codec':7s} {'label':>26s} {'bytes':>7s} {'bpp':>8s} {'PSNR':>6s}")
    for p in sorted(points, key=lambda p: p.bpp):
        print(f"{p.codec:7s} {p.label:>26s} {p.bytes:>7d} {p.bpp:>8.4f} {p.psnr:>6.2f}")

    name = Path(args.source).stem
    out = out_dir / f"rd_hybrid_{name}_{args.inr_config}.json"
    out.write_text(json.dumps([asdict(p) for p in points], indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
