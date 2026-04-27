"""Rate–distortion curves and Bjøntegaard-Delta (BD-rate) computation."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RDPoint:
    bpp: float
    psnr: float = float("nan")
    ms_ssim: float = float("nan")
    vmaf: float = float("nan")


def bd_rate(anchor: list[RDPoint], test: list[RDPoint], metric: str = "psnr") -> float:
    """Bjøntegaard-Delta rate savings (%) of `test` vs `anchor`.

    Negative => test uses fewer bits at equal quality (good).

    Implementation: piecewise-cubic interpolation of log(rate) vs metric,
    integrated over the overlapping metric range.
    """
    a_m = np.array([getattr(p, metric) for p in anchor])
    a_r = np.log(np.array([p.bpp for p in anchor]))
    t_m = np.array([getattr(p, metric) for p in test])
    t_r = np.log(np.array([p.bpp for p in test]))

    if len(a_m) < 4 or len(t_m) < 4:
        raise ValueError("BD-rate needs at least 4 RD points per curve")

    # Sort by metric
    ai = np.argsort(a_m); a_m, a_r = a_m[ai], a_r[ai]
    ti = np.argsort(t_m); t_m, t_r = t_m[ti], t_r[ti]

    lo = max(a_m.min(), t_m.min())
    hi = min(a_m.max(), t_m.max())
    if hi <= lo:
        return float("nan")

    pa = np.polyfit(a_m, a_r, 3)
    pt = np.polyfit(t_m, t_r, 3)
    Pa = np.poly1d(pa); Pt = np.poly1d(pt)
    Ia = np.polyint(Pa); It = np.polyint(Pt)
    avg_a = (Ia(hi) - Ia(lo)) / (hi - lo)
    avg_t = (It(hi) - It(lo)) / (hi - lo)
    return (math.exp(avg_t - avg_a) - 1) * 100


def plot_rd(curves: dict[str, list[RDPoint]], metric: str, out_png: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, pts in curves.items():
        pts = sorted(pts, key=lambda p: p.bpp)
        ax.plot([p.bpp for p in pts], [getattr(p, metric) for p in pts], marker="o", label=name)
    ax.set_xscale("log")
    ax.set_xlabel("bits per pixel (bpp)")
    ax.set_ylabel(metric.upper())
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
