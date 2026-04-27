"""Map target bitrates (kbps) to the λ values that produce them.

Workflow:
  1. Run WANVC encode at a set of probe λ values on a representative clip.
  2. Measure true bpp = file_bytes * 8 / (T * H * W).
  3. Fit bpp = exp(a * log(λ) + b).  (log-log-linear works well in practice.)
  4. Invert to pick a λ for each kbps rung.

The fit is per-video; rerun on each benchmark source. Output is a JSON that
`run_benchmark.py` can consume.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from utils.video_io import read_video


def _encode_at(source: str, lam: float, ckpt: str, tmp: Path, adapt_steps: int) -> int:
    bs = tmp / f"probe_{lam:.5f}.wanvc"
    subprocess.run(
        ["python", "scripts/encode.py", source, str(bs),
         "--ckpt", ckpt, "--lambda-rd", str(lam),
         "--adapt-steps", str(adapt_steps)],
        check=True,
    )
    return bs.stat().st_size


def _target_bpps(kbps_ladder: Iterable[int], T: int, fps: float, H: int, W: int) -> list[float]:
    # bpp = (kbps * 1000) * (T / fps) / (T * H * W) = kbps * 1000 / (fps * H * W)
    return [kbps * 1000.0 / (fps * H * W) for kbps in kbps_ladder]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=str, help="representative calibration clip")
    ap.add_argument("--ckpt", type=str, default="checkpoints/base.pt")
    ap.add_argument("--probes", type=float, nargs="+",
                    default=[0.001, 0.003, 0.008, 0.02, 0.05, 0.12])
    ap.add_argument("--kbps", type=int, nargs="+",
                    default=[250, 500, 1000, 2000, 4000, 8000])
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--adapt-steps", type=int, default=40,
                    help="low value speeds up calibration without skewing the λ→bpp fit much")
    ap.add_argument("--out", type=str, default="bench_out/lambdas.json")
    args = ap.parse_args()

    video = read_video(args.source)
    T, _, H, W = video.shape
    print(f"calibration source: {T} frames @ {W}x{H}")

    # 1) Probe runs.
    records = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for lam in args.probes:
            print(f"  probing λ={lam}")
            n_bytes = _encode_at(args.source, lam, args.ckpt, tmp, args.adapt_steps)
            bpp = n_bytes * 8 / (T * H * W)
            records.append({"lambda": lam, "bytes": n_bytes, "bpp": bpp})
            print(f"    -> {n_bytes} B  ({bpp:.5f} bpp)")

    # 2) Fit log(bpp) = a * log(λ) + b
    xs = np.log(np.array([r["lambda"] for r in records]))
    ys = np.log(np.array([r["bpp"] for r in records]))
    a, b = np.polyfit(xs, ys, 1)
    print(f"fit: log(bpp) = {a:.4f} * log(λ) + {b:.4f}")

    # 3) Invert for target kbps rungs.
    targets = _target_bpps(args.kbps, T, args.fps, H, W)
    inverted = []
    for kbps, target_bpp in zip(args.kbps, targets):
        log_bpp = math.log(target_bpp)
        lam = math.exp((log_bpp - b) / a)
        inverted.append({"kbps": kbps, "target_bpp": target_bpp, "lambda": lam})
        print(f"  {kbps:>5d} kbps -> bpp={target_bpp:.5f} -> λ≈{lam:.5f}")

    out = {
        "source": args.source,
        "probes": records,
        "fit": {"a": float(a), "b": float(b)},
        "lambdas_for_kbps": inverted,
        "video": {"T": T, "H": H, "W": W, "fps": args.fps},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
