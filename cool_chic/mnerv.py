"""Motion-compensated NeRV (M-NeRV).

Frame 0 is a keyframe, decoded directly from its embedding.
Frame t > 0 is reconstructed as:

    frame_t = warp(frame_{t-1}, flow_t) + residual_t

where flow_t and residual_t are both decoded from a per-frame
embedding e_t through a shared CNN backbone with two heads.

This lets the model spend its bits on motion (flow) and small
appearance corrections (residual) instead of re-encoding the full
frame from scratch — exactly the trick AV1 uses to leave us 4-8 dB
behind on natural content.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nerv import _factor_pyramid


@dataclass
class MNeRVConfig:
    embed_dim: int = 24
    base_ch:   int = 48
    flow_scale: float = 4.0   # output flow is in pixels; scale tanh by this
    keyframe_interval: int = 8   # every Kth frame is a full-decode keyframe;
                                  # bounds autoregressive error propagation
                                  # to <= K-1 frames


def warp_with_flow(prev_frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """prev_frame: (B,3,H,W). flow: (B,2,H,W) in pixel units (dx, dy)."""
    B, _, H, W = prev_frame.shape
    device = prev_frame.device
    yy, xx = torch.meshgrid(torch.arange(H, device=device),
                              torch.arange(W, device=device), indexing="ij")
    grid = torch.stack([xx.float(), yy.float()], dim=-1)        # (H, W, 2)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)               # (B, H, W, 2)
    # add flow (which is in pixel units, dx then dy)
    grid = grid + flow.permute(0, 2, 3, 1)                       # (B, H, W, 2)
    # normalize to [-1, 1]
    grid_x = 2.0 * grid[..., 0] / max(W - 1, 1) - 1.0
    grid_y = 2.0 * grid[..., 1] / max(H - 1, 1) - 1.0
    grid_n = torch.stack([grid_x, grid_y], dim=-1)
    return F.grid_sample(prev_frame, grid_n, mode="bilinear",
                          padding_mode="border", align_corners=True)


class MNeRVBackbone(nn.Module):
    def __init__(self, n_frames: int, H: int, W: int,
                  cfg: MNeRVConfig | None = None):
        super().__init__()
        cfg = cfg or MNeRVConfig()
        self.cfg = cfg
        self.n_frames = n_frames; self.H = H; self.W = W

        sh, sw, n_ups = _factor_pyramid(H, W)
        if n_ups == 0:
            raise ValueError(f"({H},{W}) has no factor-2 pyramid; can't build decoder")
        self.start_h, self.start_w, self.n_ups = sh, sw, n_ups

        self.frame_embed = nn.Parameter(torch.randn(n_frames, cfg.embed_dim) * 0.02)
        self.stem = nn.Linear(cfg.embed_dim, cfg.base_ch * sh * sw)

        blocks = []; ch = cfg.base_ch
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
        self.last_ch = ch

        # Output heads. Residual is unbounded RGB (we add to warped frame).
        # Flow is tanh-bounded then scaled — discourages crazy flow vectors.
        self.residual_head = nn.Conv2d(ch, 3, 3, padding=1)
        self.flow_head     = nn.Conv2d(ch, 2, 3, padding=1)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _decode_features(self, frame_idx: torch.Tensor) -> torch.Tensor:
        e = self.frame_embed[frame_idx]
        x = self.stem(e).view(-1, self.cfg.base_ch, self.start_h, self.start_w)
        for b in self.blocks:
            x = b(x)
        return x

    def forward(self, frame_idx: torch.Tensor,
                  teacher_frames: torch.Tensor | None = None) -> torch.Tensor:
        """`frame_idx` MUST be a contiguous arange(0, T) (or any contiguous
        range starting at 0) — M-NeRV is causal. If `teacher_frames` is
        given, frame t-1's previous reference is the *target* frame (teacher
        forcing during training); otherwise it's the model's own previous
        reconstruction."""
        feats = self._decode_features(frame_idx)
        residual = self.residual_head(feats)               # (T,3,H,W)
        flow_raw = self.flow_head(feats)                   # (T,2,H,W)
        flow = torch.tanh(flow_raw) * self.cfg.flow_scale  # bounded

        K = self.cfg.keyframe_interval
        out = []
        for i, t in enumerate(frame_idx.tolist()):
            if i == 0 or t == 0 or (K > 0 and t % K == 0):
                # keyframe: residual IS the frame
                out.append(residual[i])
            else:
                prev = (teacher_frames[i - 1] if teacher_frames is not None
                        else out[-1])
                warped = warp_with_flow(prev.unsqueeze(0), flow[i:i+1])[0]
                out.append(warped + residual[i])
        return torch.stack(out, dim=0)

    @torch.no_grad()
    def reconstruct(self) -> torch.Tensor:
        device = next(self.parameters()).device
        idx = torch.arange(self.n_frames, device=device)
        return self.forward(idx)

    def is_keyframe(self, t: int) -> bool:
        K = self.cfg.keyframe_interval
        return t == 0 or (K > 0 and t % K == 0)

    def forward_q(self, frame_idx: torch.Tensor, quants,
                    teacher_frames: torch.Tensor | None = None) -> torch.Tensor:
        """Same as forward but reads weights from `quants` (per-tensor
        quantizers from `attach_quantizers`). Used during QAT."""
        fe = quants["frame_embed"].quantized()
        e = fe[frame_idx]

        sw = quants["stem_weight"].quantized()
        sb = quants["stem_bias"].quantized()
        x = F.linear(e, sw, sb).view(-1, self.cfg.base_ch, self.start_h, self.start_w)
        for i in range(len(self.blocks)):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            w1 = quants[f"blocks_{i}_1_weight"].quantized()
            b1 = quants[f"blocks_{i}_1_bias"].quantized()
            x = F.conv2d(x, w1, b1, padding=1); x = F.gelu(x)
            w2 = quants[f"blocks_{i}_3_weight"].quantized()
            b2 = quants[f"blocks_{i}_3_bias"].quantized()
            x = F.conv2d(x, w2, b2, padding=1); x = F.gelu(x)

        rw = quants["residual_head_weight"].quantized()
        rb = quants["residual_head_bias"].quantized()
        residual = F.conv2d(x, rw, rb, padding=1)
        fw = quants["flow_head_weight"].quantized()
        fb = quants["flow_head_bias"].quantized()
        flow_raw = F.conv2d(x, fw, fb, padding=1)
        flow = torch.tanh(flow_raw) * self.cfg.flow_scale

        K = self.cfg.keyframe_interval
        out = []
        for i, t in enumerate(frame_idx.tolist()):
            if i == 0 or t == 0 or (K > 0 and t % K == 0):
                out.append(residual[i])
            else:
                prev = (teacher_frames[i - 1] if teacher_frames is not None
                        else out[-1])
                warped = warp_with_flow(prev.unsqueeze(0), flow[i:i+1])[0]
                out.append(warped + residual[i])
        return torch.stack(out, dim=0)
