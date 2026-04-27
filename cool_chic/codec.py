"""Per-instance codecs.

ImageINR: hash grid + MLP for one image. The "bitstream" is the union of
          hash table entries and MLP weights.

VideoINR: tri-plane hash grid + MLP for one video clip. Same idea, three
          spatial planes for parameter efficiency.

Both expose:
  reconstruct()      -> tensor of pixel values
  parameters()       -> all trainable weights (the bitstream-to-be)
  total_params       -> bytes-equivalent before quantization
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .hash_grid import HashGrid2D, TriPlaneGrid
from .inr import INRDecoder


@dataclass
class ImageConfig:
    L: int = 6
    T: int = 1 << 12
    F: int = 2
    N_min: int = 16
    N_max: int = 256
    mlp_hidden: int = 64
    mlp_depth: int = 4


class ImageINR(nn.Module):
    """Per-image codec: each image gets its own (hash_grid, mlp)."""

    def __init__(self, H: int, W: int, cfg: ImageConfig | None = None):
        super().__init__()
        cfg = cfg or ImageConfig()
        self.H, self.W = H, W
        self.grid = HashGrid2D(L=cfg.L, T=cfg.T, F=cfg.F,
                                N_min=cfg.N_min, N_max=cfg.N_max)
        self.mlp = INRDecoder(in_dim=self.grid.out_dim,
                               hidden=cfg.mlp_hidden, depth=cfg.mlp_depth)

    @property
    def total_params(self) -> int:
        return self.grid.n_params + self.mlp.n_params

    def _coord_grid(self, device: torch.device) -> torch.Tensor:
        """Returns (H*W, 2) coords in [0, 1]."""
        ys = torch.linspace(0, 1, self.H, device=device)
        xs = torch.linspace(0, 1, self.W, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy], dim=-1).reshape(-1, 2)

    def reconstruct(self, device: torch.device | None = None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        coords = self._coord_grid(device)
        feats = self.grid(coords)
        rgb = self.mlp(feats)             # (H*W, 3)
        return rgb.reshape(self.H, self.W, 3).permute(2, 0, 1)  # (3, H, W)


@dataclass
class VideoConfig:
    L: int = 6
    T: int = 1 << 13          # bigger table for video
    F: int = 2
    N_min: int = 16
    N_max: int = 256
    mlp_hidden: int = 64
    mlp_depth: int = 4


class VideoINR(nn.Module):
    """Per-clip codec: tri-plane hash grid + MLP. Coordinates are (x, y, t)
    in [0, 1]^3 with t = frame_index / (T_frames - 1)."""

    def __init__(self, T_frames: int, H: int, W: int,
                 cfg: VideoConfig | None = None):
        super().__init__()
        cfg = cfg or VideoConfig()
        self.T_frames, self.H, self.W = T_frames, H, W
        self.grid = TriPlaneGrid(L=cfg.L, T=cfg.T, F=cfg.F,
                                  N_min=cfg.N_min, N_max=cfg.N_max)
        self.mlp = INRDecoder(in_dim=self.grid.out_dim,
                               hidden=cfg.mlp_hidden, depth=cfg.mlp_depth)

    @property
    def total_params(self) -> int:
        return self.grid.n_params + self.mlp.n_params

    def _coord_grid(self, device: torch.device) -> torch.Tensor:
        """Returns (T*H*W, 3) coords in [0, 1]."""
        T = max(self.T_frames - 1, 1)
        ts = torch.linspace(0, 1, self.T_frames, device=device)
        ys = torch.linspace(0, 1, self.H, device=device)
        xs = torch.linspace(0, 1, self.W, device=device)
        gt, gy, gx = torch.meshgrid(ts, ys, xs, indexing="ij")
        return torch.stack([gx, gy, gt], dim=-1).reshape(-1, 3)

    def reconstruct(self, device: torch.device | None = None,
                    chunk: int = 1 << 18) -> torch.Tensor:
        """Returns (T_frames, 3, H, W). Chunked to keep VRAM bounded."""
        device = device or next(self.parameters()).device
        coords = self._coord_grid(device)
        outs = []
        for i in range(0, coords.shape[0], chunk):
            c = coords[i:i + chunk]
            outs.append(self.mlp(self.grid(c)))
        rgb = torch.cat(outs, dim=0)
        return rgb.reshape(self.T_frames, self.H, self.W, 3).permute(0, 3, 1, 2)
