"""End-to-end benchmark: WANVC vs libaom-av1 vs libx265.

For each target bitrate:
  * Encode with each codec
  * Decode back to mp4
  * Measure PSNR / MS-SSIM / VMAF against the source
  * Record bpp = (file_bytes * 8) / (T * H * W)

Writes an RD plot per metric and prints BD-rate savings.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from utils.video_io import read_video, write_video
from utils.metrics import psnr, ms_ssim_metric, vmaf_score
from benchmarks.ffmpeg_codecs import encode_av1, encode_x265
from benchmarks.rd_plot import RDPoint, bd_rate, plot_rd


# Default kbps rungs. The λ values are *auto-calibrated* per source video via
# benchmarks/calibrate_lambdas.py. If --lambdas-json is not supplied, we fall
# back to a coarse default ladder; rerun calibration for accurate RD curves.
KBPS_LADDER = [250, 500, 1000, 2000, 4000, 8000]
DEFAULT_LAMBDAS = [0.05, 0.025, 0.013, 0.0067, 0.0035, 0.0018]  # high→low bitrate


def _decode_to_mp4(path_bitstream: str, path_out: str, ckpt: str):
    subprocess.run(
        ["python", "scripts/decode.py", path_bitstream, path_out, "--ckpt", ckpt],
        check=True,
    )


def _wanvc_point(src: str, lam: float, ckpt: str, tmp: Path) -> tuple[str, int]:
    bs = tmp / f"wanvc_{lam:.4f}.wanvc"
    subprocess.run(
        ["python", "scripts/encode.py", src, str(bs), "--ckpt", ckpt,
         "--lambda-rd", str(lam)],
        check=True,
    )
    rebuilt = tmp / f"wanvc_{lam:.4f}.mp4"
    _decode_to_mp4(str(bs), str(rebuilt), ckpt)
    return str(rebuilt), bs.stat().st_size


def _measure(src_mp4: str, dist_mp4: str, T: int, H: int, W: int, bytes_: int) -> RDPoint:
    orig = read_video(src_mp4)
    recon = read_video(dist_mp4, max_frames=orig.shape[0])
    n = min(orig.shape[0], recon.shape[0])
    orig, recon = orig[:n], recon[:n]
    bpp = bytes_ * 8 / (n * H * W)
    p = psnr(orig, recon)
    m = ms_ssim_metric(orig, recon)
    try:
        v = vmaf_score(src_mp4, dist_mp4)
    except Exception as e:
        print(f"  vmaf unavailable: {e}")
        v = float("nan")
    return RDPoint(bpp=bpp, psnr=p, ms_ssim=m, vmaf=v)


def _resolve_wanvc_lambdas(args) -> list[float]:
    """Return the λ ladder to evaluate, preferring calibration JSON if given."""
    if args.lambdas_json:
        data = json.loads(Path(args.lambdas_json).read_text())
        # Sort by kbps so high-rate point is first (matches plotting order).
        entries = sorted(data["lambdas_for_kbps"], key=lambda e: -e["kbps"])
        return [float(e["lambda"]) for e in entries]
    return DEFAULT_LAMBDAS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=str)
    ap.add_argument("--ckpt", type=str, default="checkpoints/base.pt")
    ap.add_argument("--out",  type=str, default="bench_out")
    ap.add_argument("--skip-wanvc", action="store_true")
    ap.add_argument("--lambdas-json", type=str, default=None,
                    help="output of benchmarks/calibrate_lambdas.py")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    video = read_video(args.source)
    T, _, H, W = video.shape
    print(f"source: {T}x{H}x{W}")

    curves: dict[str, list[RDPoint]] = {"av1": [], "x265": [], "wanvc": []}

    for kbps in KBPS_LADDER:
        print(f"--- {kbps} kbps ---")
        av1  = encode_av1 (args.source, kbps, str(out))
        x265 = encode_x265(args.source, kbps, str(out))
        curves["av1"] .append(_measure(args.source, av1.bitstream_path,  T, H, W, av1.bytes))
        curves["x265"].append(_measure(args.source, x265.bitstream_path, T, H, W, x265.bytes))

    if not args.skip_wanvc:
        lambdas = _resolve_wanvc_lambdas(args)
        print(f"wanvc λ ladder: {lambdas}")
        with tempfile.TemporaryDirectory() as td:
            for lam in lambdas:
                print(f"--- wanvc λ={lam} ---")
                mp4, size = _wanvc_point(args.source, lam, args.ckpt, Path(td))
                curves["wanvc"].append(_measure(args.source, mp4, T, H, W, size))

    for metric in ("psnr", "ms_ssim", "vmaf"):
        plot_rd(curves, metric, str(out / f"rd_{metric}.png"))

    # BD-rate vs AV1 anchor.
    if curves["wanvc"]:
        for metric in ("psnr", "ms_ssim", "vmaf"):
            try:
                bd = bd_rate(curves["av1"], curves["wanvc"], metric)
                print(f"BD-rate (wanvc vs av1, {metric}):  {bd:+.2f}%")
            except Exception as e:
                print(f"BD-rate {metric} failed: {e}")

    (out / "curves.json").write_text(json.dumps(
        {k: [p.__dict__ for p in v] for k, v in curves.items()}, indent=2
    ))
    print(f"wrote {out}/rd_*.png and curves.json")


if __name__ == "__main__":
    main()
