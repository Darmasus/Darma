"""GOP segmentation with simple shot-boundary detection.

We insert an I-frame at any position where color-histogram L1 distance between
consecutive frames exceeds a threshold, and at least every `max_gop_len` frames.
The goal: every GOP ≈ a single shot, so per-GOP weight adaptation sees
statistically coherent content.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GOP:
    start: int
    end: int          # exclusive
    length: int       # end - start
    is_keyshot: bool  # True if started at a shot boundary


def shot_boundary_score(prev: torch.Tensor, curr: torch.Tensor, bins: int = 32) -> float:
    """L1 distance of per-channel normalized histograms."""
    def hist(x):
        x = (x.clamp(0, 1) * (bins - 1)).round().long()
        out = []
        for c in range(x.shape[0]):
            h = torch.bincount(x[c].view(-1), minlength=bins).float()
            out.append(h / h.sum().clamp(min=1))
        return torch.stack(out)
    return float((hist(prev) - hist(curr)).abs().sum())


def segment_gops(video: torch.Tensor, max_gop_len: int = 16,
                 min_gop_len: int = 4, shot_thresh: float = 0.35) -> list[GOP]:
    """video: (T, 3, H, W) in [0, 1]. Returns GOPs that respect shot cuts."""
    T = video.shape[0]
    if T == 0:
        return []
    cuts = [0]
    last = 0
    for t in range(1, T):
        score = shot_boundary_score(video[t - 1], video[t])
        at_max = (t - last) >= max_gop_len
        at_cut = score > shot_thresh and (t - last) >= min_gop_len
        if at_max or at_cut:
            cuts.append(t)
            last = t
    cuts.append(T)

    gops: list[GOP] = []
    for i, start in enumerate(cuts[:-1]):
        end = cuts[i + 1]
        gops.append(GOP(start=start, end=end, length=end - start,
                        is_keyshot=(i == 0 or end - start < max_gop_len)))
    return gops
