"""Render the RD plot from one or more rd_vs_ffmpeg JSON dumps as a PNG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+",
                    help="one or more rd_vs_ffmpeg_*.json files")
    ap.add_argument("--out", default="cool_chic/benchmarks/out/rd_plot.png")
    ap.add_argument("--title", default="Cool-Chic INR vs AV1 / x265")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Aggregate points per codec/capacity label across files.
    series: dict[str, list[tuple[float, float]]] = {}
    styles = {
        "av1":            dict(color="tab:red",    marker="o", ls="-"),
        "x265":           dict(color="tab:blue",   marker="s", ls="-"),
        "inr small":      dict(color="tab:green",  marker="^", ls="--"),
        "inr medium":     dict(color="tab:purple", marker="v", ls="--"),
    }

    for path in args.jsons:
        p = Path(path)
        points = json.loads(p.read_text())
        # capacity from filename: rd_vs_ffmpeg_<clip>_<cap>.json
        cap = p.stem.rsplit("_", 1)[-1]  # "small" / "medium"
        for pt in points:
            if pt["codec"] == "inr":
                key = f"inr {cap}"
            else:
                key = pt["codec"]
            series.setdefault(key, []).append((pt["bpp"], pt["psnr"]))

    for key, pts in sorted(series.items()):
        pts.sort()
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        style = styles.get(key, dict(marker="x", ls=":"))
        ax.plot(xs, ys, label=key, **style, linewidth=1.5, markersize=7)

    ax.set_xlabel("bpp (bits per pixel)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
