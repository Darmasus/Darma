"""NeRV-style video backbone: per-frame embedding + upsampling CNN decoder.

Motivation: our hash-grid + tri-plane INR hits a ~28 dB PSNR wall on
natural content. NeRV (Chen et al., 2021) is known to reach 30-45 dB on
natural video at similar parameter budgets — the CNN decoder is just a
much better inductive bias for natural spatial content than hash tables.

Architecture:
    frame_idx (N,)
        -> frame_embed: (N, D)
        -> stem: Linear -> reshape to (N, C, h0, w0) tiny spatial
        -> K upsample blocks: Upsample(x2) + Conv3x3 + GELU + Conv3x3 + GELU
        -> out_conv: Conv3x3 -> (N, 3, H, W)

Chosen small enough that even a 7-frame 96x160 clip can afford the
parameters. Scales up cleanly by increasing `base_ch` and `embed_dim`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NeRVConfig:
    embed_dim:  int = 32
    base_ch:    int = 64
    # We'll auto-derive (start_h, start_w, n_ups) from target (H, W)
    # by halving until both dims are <=8.


def _factor_pyramid(H: int, W: int, min_dim: int = 4) -> tuple[int, int, int]:
    """Pick (start_h, start_w, n_ups) so that start*2^n_ups == (H, W).

    Requires H and W both divisible by a common power of 2. Raises if not.
    We pick the largest n_ups such that H/2^n >= min_dim and W/2^n >= min_dim
    and both H/2^n and W/2^n are integers.
    """
    h, w, n = H, W, 0
    while h % 2 == 0 and w % 2 == 0 and h // 2 >= min_dim and w // 2 >= min_dim:
        h //= 2; w //= 2; n += 1
    return h, w, n


class NeRVBackbone(nn.Module):
    """Per-frame embedding + CNN decoder. Shape-agnostic for H/W that have
    a common power-of-2 pyramid to a small starting resolution."""

    def __init__(self, n_frames: int, H: int, W: int,
                  cfg: NeRVConfig | None = None):
        super().__init__()
        cfg = cfg or NeRVConfig()
        self.n_frames = n_frames
        self.H, self.W = H, W
        self.cfg = cfg
        sh, sw, n_ups = _factor_pyramid(H, W)
        if n_ups == 0:
            raise ValueError(f"({H}, {W}) must have a common pow-of-2 pyramid; "
                             f"got start=({sh},{sw}), 0 upsamples.")
        self.start_h, self.start_w, self.n_ups = sh, sw, n_ups

        # Learnable per-frame embedding
        self.frame_embed = nn.Parameter(torch.randn(n_frames, cfg.embed_dim) * 0.02)

        # Stem: map embed -> (C, sh, sw)
        self.stem = nn.Linear(cfg.embed_dim, cfg.base_ch * sh * sw)

        # Upsample blocks. Channels halve each level down to >= 8.
        blocks = []
        ch = cfg.base_ch
        for _ in range(n_ups):
            next_ch = max(ch // 2, 8)
            blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(ch, next_ch, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(next_ch, next_ch, kernel_size=3, padding=1),
                nn.GELU(),
            ))
            ch = next_ch
        self.blocks = nn.ModuleList(blocks)

        self.out_conv = nn.Conv2d(ch, 3, kernel_size=3, padding=1)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, frame_idx: torch.Tensor) -> torch.Tensor:
        """frame_idx: (N,) long. Returns (N, 3, H, W) in raw (unclamped) RGB."""
        e = self.frame_embed[frame_idx]                            # (N, D)
        x = self.stem(e).view(-1, self.cfg.base_ch, self.start_h, self.start_w)
        for b in self.blocks:
            x = b(x)
        return self.out_conv(x)

    def reconstruct(self) -> torch.Tensor:
        """Full video (T, 3, H, W)."""
        device = next(self.parameters()).device
        idx = torch.arange(self.n_frames, device=device)
        return self(idx)

    def forward_q(self, frame_idx: torch.Tensor, quants) -> torch.Tensor:
        """Same as forward, but reads every weight from `quants` (an
        `nn.ModuleDict[name -> TensorQuantizer]`). The quantizer returns
        the STE-rounded tensor, so gradients still flow into the raw
        params during QAT.

        Name mapping matches `attach_quantizers`: dots in parameter names
        are replaced with underscores.
        """
        fe  = quants["frame_embed"].quantized()
        e   = fe[frame_idx]

        sw  = quants["stem_weight"].quantized()
        sb  = quants["stem_bias"].quantized()
        x   = F.linear(e, sw, sb).view(-1, self.cfg.base_ch,
                                         self.start_h, self.start_w)
        for i in range(len(self.blocks)):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            w1 = quants[f"blocks_{i}_1_weight"].quantized()
            b1 = quants[f"blocks_{i}_1_bias"].quantized()
            x = F.conv2d(x, w1, b1, padding=1)
            x = F.gelu(x)
            w2 = quants[f"blocks_{i}_3_weight"].quantized()
            b2 = quants[f"blocks_{i}_3_bias"].quantized()
            x = F.conv2d(x, w2, b2, padding=1)
            x = F.gelu(x)

        ow = quants["out_conv_weight"].quantized()
        ob = quants["out_conv_bias"].quantized()
        return F.conv2d(x, ow, ob, padding=1)
