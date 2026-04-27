"""Multi-resolution hash encoding (Instant-NGP, Müller et al. 2022).

Maps a continuous coordinate `(x, y[, t]) ∈ [0, 1]^d` to an `L*F`-dim
feature vector by lookup in `L` levels of hash tables at geometrically-
increasing resolutions. The MLP that follows maps these features to RGB.

Why a hash table instead of a dense grid?
  Dense grids cost O(N^d) parameters; hash tables let you set the table
  size T independently of the resolution N. At fine levels, collisions
  are common but the MLP learns to disambiguate from coarser levels.

For video we offer two encoders:
  * HashGrid3D   true 3D hash over (x, y, t). 8-corner trilinear interp.
                 Heavier but captures full spatiotemporal correlation.
  * TriPlaneGrid three 2D planes (XY, XT, YT). O(N²) instead of O(N³)
                 parameters. Much smaller, slight loss on diagonal motion.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


# Large primes from the Instant-NGP paper. Used to mix dimensions.
_PRIMES = (1, 2_654_435_761, 805_459_861, 3_674_653_429)


def _hash(coords: torch.Tensor, table_size: int) -> torch.Tensor:
    """Spatial hash: XOR of (coord_i * prime_i) mod T."""
    h = torch.zeros_like(coords[..., 0])
    for i in range(coords.shape[-1]):
        h = h ^ (coords[..., i] * _PRIMES[i])
    return h % table_size


def _resolutions(L: int, N_min: int, N_max: int) -> list[int]:
    """Geometric progression of per-level resolutions."""
    if L == 1:
        return [N_min]
    b = (N_max / N_min) ** (1 / (L - 1))
    return [int(round(N_min * (b ** l))) for l in range(L)]


# --------------------------------------------------------------------------- #
# 2D hash grid (single image)
# --------------------------------------------------------------------------- #
class HashGrid2D(nn.Module):
    """For each query (x, y) ∈ [0, 1]², return an L*F-dim feature."""

    def __init__(
        self,
        L: int = 6,
        T: int = 1 << 12,    # 4096 entries per level
        F: int = 2,
        N_min: int = 16,
        N_max: int = 256,
    ):
        super().__init__()
        self.L = L
        self.T = T
        self.F = F
        self.resolutions = _resolutions(L, N_min, N_max)
        # One table per level; init small to avoid early divergence.
        self.tables = nn.ParameterList(
            [nn.Parameter(torch.empty(T, F).uniform_(-1e-4, 1e-4)) for _ in range(L)]
        )

    @property
    def out_dim(self) -> int:
        return self.L * self.F

    @property
    def n_params(self) -> int:
        return self.L * self.T * self.F

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """xy: (..., 2) in [0, 1]. Returns: (..., L*F)."""
        feats = []
        for l in range(self.L):
            N = self.resolutions[l]
            scaled = xy * N                       # (..., 2)
            i0 = scaled.floor().long()            # corner indices
            t = scaled - i0.float()               # fractional in [0,1)

            # 4 corners: (0,0), (1,0), (0,1), (1,1)
            corners = []
            for dy in (0, 1):
                for dx in (0, 1):
                    c = i0 + torch.tensor([dx, dy], device=xy.device)
                    c = c % N                     # wrap to keep in-range
                    h = _hash(c, self.T)
                    corners.append(self.tables[l][h])  # (..., F)
            c00, c10, c01, c11 = corners

            wx, wy = t[..., 0:1], t[..., 1:2]
            # Bilinear interp.
            f0 = c00 * (1 - wx) + c10 * wx
            f1 = c01 * (1 - wx) + c11 * wx
            f = f0 * (1 - wy) + f1 * wy
            feats.append(f)
        return torch.cat(feats, dim=-1)


# --------------------------------------------------------------------------- #
# 3D hash grid (video, full xyz hashing)
# --------------------------------------------------------------------------- #
class HashGrid3D(nn.Module):
    """For each query (x, y, t) ∈ [0, 1]³, return an L*F-dim feature."""

    def __init__(
        self,
        L: int = 8,
        T: int = 1 << 14,
        F: int = 2,
        N_min: int = 8,
        N_max: int = 128,
    ):
        super().__init__()
        self.L = L
        self.T = T
        self.F = F
        self.resolutions = _resolutions(L, N_min, N_max)
        self.tables = nn.ParameterList(
            [nn.Parameter(torch.empty(T, F).uniform_(-1e-4, 1e-4)) for _ in range(L)]
        )

    @property
    def out_dim(self) -> int: return self.L * self.F

    @property
    def n_params(self) -> int: return self.L * self.T * self.F

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        """xyt: (..., 3) in [0, 1]. Returns: (..., L*F)."""
        feats = []
        for l in range(self.L):
            N = self.resolutions[l]
            s = xyt * N
            i0 = s.floor().long()
            t = s - i0.float()
            wx, wy, wz = t[..., 0:1], t[..., 1:2], t[..., 2:3]

            # 8 corners of the trilinear cell.
            corners = {}
            for dz in (0, 1):
                for dy in (0, 1):
                    for dx in (0, 1):
                        c = i0 + torch.tensor([dx, dy, dz], device=xyt.device)
                        c = c % N
                        h = _hash(c, self.T)
                        corners[(dx, dy, dz)] = self.tables[l][h]

            # Trilinear interp.
            c00 = corners[(0,0,0)] * (1-wx) + corners[(1,0,0)] * wx
            c10 = corners[(0,1,0)] * (1-wx) + corners[(1,1,0)] * wx
            c01 = corners[(0,0,1)] * (1-wx) + corners[(1,0,1)] * wx
            c11 = corners[(0,1,1)] * (1-wx) + corners[(1,1,1)] * wx
            c0 = c00 * (1-wy) + c10 * wy
            c1 = c01 * (1-wy) + c11 * wy
            f = c0 * (1-wz) + c1 * wz
            feats.append(f)
        return torch.cat(feats, dim=-1)


# --------------------------------------------------------------------------- #
# Tri-plane decomposition (much smaller for video)
# --------------------------------------------------------------------------- #
class TriPlaneGrid(nn.Module):
    """Three 2D hash grids over (XY), (XT), (YT). Sum the per-plane lookups.

    For a length-T video at HxW resolution, this gives O(H*W + H*T + W*T)
    parameters instead of O(H*W*T) for a true 3D grid. For most videos
    that's a 10-100x parameter saving.
    """

    def __init__(
        self,
        L: int = 6,
        T: int = 1 << 12,
        F: int = 2,
        N_min: int = 16,
        N_max: int = 256,
    ):
        super().__init__()
        self.xy = HashGrid2D(L=L, T=T, F=F, N_min=N_min, N_max=N_max)
        self.xt = HashGrid2D(L=L, T=T, F=F, N_min=N_min, N_max=N_max)
        self.yt = HashGrid2D(L=L, T=T, F=F, N_min=N_min, N_max=N_max)

    @property
    def out_dim(self) -> int: return self.xy.out_dim

    @property
    def n_params(self) -> int: return 3 * self.xy.n_params

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        x, y, t = xyt[..., 0:1], xyt[..., 1:2], xyt[..., 2:3]
        f_xy = self.xy(torch.cat([x, y], dim=-1))
        f_xt = self.xt(torch.cat([x, t], dim=-1))
        f_yt = self.yt(torch.cat([y, t], dim=-1))
        return f_xy + f_xt + f_yt
