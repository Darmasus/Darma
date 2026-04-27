"""RD curve: our caption-conditioned INR codec vs libaom-av1 vs libx265
on a held-out video clip (small, 7-frame Vimeo septuplet by default).

For each codec we collect (bpp, PSNR) points across its operating range:
  - Ours: sweep lambda at fixed capacity
  - AV1/x265: sweep target bitrate

Produces a summary table + JSON dump. An ASCII RD plot is printed for
quick comparison.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
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


def _fp32_warm(video, model, *, steps, lr, pixels, device):
    target_flat = video.to(device).permute(0, 2, 3, 1).reshape(-1, 3)
    N = target_flat.shape[0]
    T, _, H, W = video.shape
    t_d, h_d, w_d = max(T-1, 1), max(H-1, 1), max(W-1, 1)
    opt = Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        idx = torch.randint(0, N, (pixels,), device=device)
        ti = idx // (H*W); rem = idx % (H*W); yi = rem // W; xi = rem % W
        coords = torch.stack([xi.float()/w_d, yi.float()/h_d, ti.float()/t_d], -1)
        pred = model.mlp(model.grid(coords))
        loss = F.mse_loss(pred, target_flat[idx])
        loss.backward(); opt.step()


def _inr_point(video, cfg, *, warm_state, prior, caption_emb, lam, qat_steps,
                 lr, pixels, device, label) -> RDPoint:
    t0 = time.time()
    model = VideoINR(T_frames=video.shape[0], H=video.shape[-2], W=video.shape[-1],
                      cfg=cfg).to(device)
    model.load_state_dict(warm_state)
    quants, _ = overfit_video_qat(video, model,
                                    qcfg=VideoQATConfig(steps=qat_steps,
                                                        lr=lr * 0.4,
                                                        scale_lr=5e-3,
                                                        lambda_rate=lam,
                                                        pixels_per_step=pixels,
                                                        log_every=qat_steps,
                                                        init_scale=5e-3,
                                                        lambda_warmup_frac=0.25),
                                    device=device, verbose=False,
                                    prior=prior, caption_emb=caption_emb)
    recon = reconstruct_quantized_video(model, quants).clamp(0, 1)
    psnr = _psnr(recon.cpu(), video.cpu())
    Tf, _, H, W = video.shape
    blob = encode_codec(quants, kind=KIND_VIDEO, H=H, W=W, T_frames=Tf, cfg=cfg)
    bpp = len(blob) * 8 / (Tf * H * W)
    del model, quants, recon
    if device == "cuda": torch.cuda.empty_cache()
    return RDPoint("inr", label, len(blob), bpp, psnr, time.time() - t0)


def _ffmpeg_encode(codec: str, src: str, kbps: int, out_dir: Path) -> tuple[Path, float]:
    t0 = time.time()
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
    return out, time.time() - t0


def _ffmpeg_point(codec: str, src: str, kbps: int, video: torch.Tensor,
                    out_dir: Path) -> RDPoint:
    mp4, wall = _ffmpeg_encode(codec, src, kbps, out_dir)
    bytes_ = mp4.stat().st_size
    Tf, _, H, W = video.shape
    bpp = bytes_ * 8 / (Tf * H * W)
    recon = read_video(str(mp4), max_frames=Tf)
    n = min(Tf, recon.shape[0])
    psnr = _psnr(recon[:n].cpu(), video[:n].cpu())
    return RDPoint(codec, f"{kbps}k", bytes_, bpp, psnr, wall)


def _ascii_plot(points: list[RDPoint], title: str,
                  cols: int = 60, rows: int = 14) -> str:
    """Tiny ASCII scatter plot for quick inspection."""
    if not points: return ""
    bpps = [p.bpp for p in points]
    psnrs = [p.psnr for p in points]
    x_min, x_max = min(bpps), max(bpps)
    y_min, y_max = min(psnrs), max(psnrs)
    dx = (x_max - x_min) or 1; dy = (y_max - y_min) or 1
    grid = [[" "] * cols for _ in range(rows)]
    codec_char = {"inr": "I", "av1": "A", "x265": "X"}
    for p in points:
        x = int((p.bpp - x_min) / dx * (cols - 1))
        y = rows - 1 - int((p.psnr - y_min) / dy * (rows - 1))
        grid[y][x] = codec_char.get(p.codec, "?")
    lines = [title, f"  PSNR {y_max:.1f} ^"]
    for r in grid:
        lines.append("       |" + "".join(r))
    lines.append(f"  PSNR {y_min:.1f} +" + "-" * cols + "-> bpp")
    lines.append(f"            {x_min:.3f} {' '*(cols - 8)} {x_max:.3f}")
    lines.append("  legend: I=INR  A=av1  X=x265")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=str, help="mp4 of the held-out clip (small)")
    ap.add_argument("--caption", type=str, required=True,
                    help="BLIP-style description for W3 conditioning")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prior-cap", default="cool_chic/data/prior_v3_cap.pt")
    ap.add_argument("--fp32-steps", type=int, default=2500)
    ap.add_argument("--qat-steps",  type=int, default=3000)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--pixels", type=int, default=16384)
    ap.add_argument("--lambdas", default="1e-3,1e-2,1e-1,1e0,3e0")
    ap.add_argument("--ffmpeg-kbps", default="50,100,250,500,1000,2000")
    ap.add_argument("--inr-config", default="small",
                    choices=["small", "medium"],
                    help="small=L5 T4096 mlp48x3, medium=L6 T8192 mlp64x4")
    ap.add_argument("--out-dir", default="cool_chic/benchmarks/out")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    video = read_video(args.source)
    Tf, _, H, W = video.shape
    total_px = Tf * H * W
    print(f"source: {args.source}  {Tf}x{W}x{H}  ({total_px} px)")
    print(f"caption: {args.caption}")
    print(f"capacity: {args.inr_config}")

    if args.inr_config == "small":
        cfg = VideoConfig(L=5, T=1 << 12, F=2, N_min=16, N_max=256,
                           mlp_hidden=48, mlp_depth=3)
    else:
        cfg = VideoConfig(L=6, T=1 << 13, F=2, N_min=16, N_max=256,
                           mlp_hidden=64, mlp_depth=4)

    points: list[RDPoint] = []

    # ---------- INR (W3 caption prior) ----------
    base = VideoINR(T_frames=Tf, H=H, W=W, cfg=cfg).to(args.device)
    print(f"INR params: {base.total_params}")
    t0 = time.time()
    _fp32_warm(video, base, steps=args.fp32_steps, lr=args.lr,
                pixels=args.pixels, device=args.device)
    warm_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    del base
    if args.device == "cuda": torch.cuda.empty_cache()
    print(f"fp32 warm-up: {time.time()-t0:.1f}s")

    prior_cap = load_prior(args.prior_cap, device=args.device)
    caption_emb = encode_caption(args.caption, device=args.device)

    for lam_str in args.lambdas.split(","):
        lam = float(lam_str)
        p = _inr_point(video, cfg, warm_state=warm_state, prior=prior_cap,
                        caption_emb=caption_emb, lam=lam,
                        qat_steps=args.qat_steps, lr=args.lr,
                        pixels=args.pixels, device=args.device,
                        label=f"lam={lam:.0e}")
        points.append(p)
        print(f"  [inr {p.label:>10s}] {p.bytes:>6d} B  bpp={p.bpp:.4f}  "
              f"PSNR={p.psnr:.2f}  ({p.seconds:.0f}s)")

    # ---------- AV1 and x265 ----------
    kbps_list = [int(k) for k in args.ffmpeg_kbps.split(",")]
    for codec in ("x265", "av1"):
        for kbps in kbps_list:
            p = _ffmpeg_point(codec, args.source, kbps, video, out_dir)
            points.append(p)
            print(f"  [{codec} {p.label:>6s}] {p.bytes:>6d} B  bpp={p.bpp:.4f}  "
                  f"PSNR={p.psnr:.2f}  ({p.seconds:.1f}s)")

    # ---------- table & ascii plot ----------
    print("\n=== RD TABLE (sorted by bpp) ===")
    print(f"{'codec':6s} {'label':>10s} {'bytes':>7s} {'bpp':>8s} {'PSNR':>6s}")
    for p in sorted(points, key=lambda p: p.bpp):
        print(f"{p.codec:6s} {p.label:>10s} {p.bytes:>7d} {p.bpp:>8.4f} {p.psnr:>6.2f}")

    print("\n" + _ascii_plot(points, f"RD: {Path(args.source).name}"))

    name = Path(args.source).stem
    out = out_dir / f"rd_vs_ffmpeg_{name}_{args.inr_config}.json"
    out.write_text(json.dumps([asdict(p) for p in points], indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
