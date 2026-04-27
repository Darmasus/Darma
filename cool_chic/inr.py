"""Tiny MLP that maps hash-grid features to RGB.

Deliberately small: 4 hidden layers of width 64, ReLU. ~10K params.
Together with a ~50K-param hash grid the entire codec for one video
fits in roughly 50-200 KB depending on quantization.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class INRDecoder(nn.Module):
    """`features -> RGB` mapping. No batchnorm, no GDN — those introduce
    numerical-stability traps we already learned to avoid the hard way."""

    def __init__(self, in_dim: int, hidden: int = 64, depth: int = 4,
                 out_dim: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU(inplace=True))
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # No sigmoid: train against raw output, clamp at inference.
        return self.net(feats)
