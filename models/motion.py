"""Motion compensation sub-network for P-frames.

Architecture (compact DVC-Pro variant):
  flow_net(x_prev, x_curr) -> f            # 2-channel optical flow
  warp(x_prev, f)          -> x_pred       # differentiable bilinear warp
  refine(concat(x_pred, x_prev, f)) -> x_mc # residual refinement

Both `flow_net` and `refine`'s last layers are AdaptableConv2d so the encoder
can overfit them to the GOP's motion statistics (panning, handheld shake, etc.)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptation import AdaptableConv2d, AdaptationConfig
from .numerics import safe_flow, safe_act


def _bilinear_warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp tensor x (B, C, H, W) by optical flow (B, 2, H, W) in pixels."""
    B, _, H, W = x.shape
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, H, device=x.device),
        torch.linspace(-1, 1, W, device=x.device),
        indexing="ij",
    )
    base = torch.stack((xs, ys), dim=-1).expand(B, -1, -1, -1)
    # Normalize flow from pixel units -> grid_sample's [-1, 1] range.
    flow_n = flow.clone()
    flow_n[:, 0] = flow_n[:, 0] * (2.0 / max(W - 1, 1))
    flow_n[:, 1] = flow_n[:, 1] * (2.0 / max(H - 1, 1))
    grid = base + flow_n.permute(0, 2, 3, 1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


class _EncBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=2):
        super().__init__()
        self.c = nn.Conv2d(in_c, out_c, k, stride=s, padding=k // 2)
        self.a = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.a(self.c(x))


class FlowNet(nn.Module):
    """Lightweight hourglass that predicts optical flow from (x_prev, x_curr)."""

    def __init__(self, base: int = 32):
        super().__init__()
        self.enc1 = _EncBlock(6, base, s=2)
        self.enc2 = _EncBlock(base, base * 2, s=2)
        self.enc3 = _EncBlock(base * 2, base * 4, s=2)
        self.mid  = nn.Conv2d(base * 4, base * 4, 3, padding=1)
        self.up3  = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.up2  = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.up1  = nn.ConvTranspose2d(base, base, 4, stride=2, padding=1)
        self.flow = AdaptableConv2d(base, 2, 3, padding=1, cfg=AdaptationConfig(rank=4))

    def forward(self, x_prev: torch.Tensor, x_curr: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_prev, x_curr], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        m  = F.leaky_relu(self.mid(e3), 0.1, inplace=True)
        u3 = F.leaky_relu(self.up3(m) + e2, 0.1, inplace=True)
        u2 = F.leaky_relu(self.up2(u3) + e1, 0.1, inplace=True)
        u1 = F.leaky_relu(self.up1(u2), 0.1, inplace=True)
        # Clamp flow magnitude — values >>image extent extrapolate to inf in
        # bilinear warp, contaminating g_a(x_mc) and the temporal prior.
        return safe_flow(self.flow(u1))


class MotionCompensationNet(nn.Module):
    """flow -> warp -> refine."""

    def __init__(self, base: int = 32):
        super().__init__()
        self.flow_net = FlowNet(base=base)
        self.refine = nn.Sequential(
            nn.Conv2d(3 + 3 + 2, base, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            AdaptableConv2d(base, 3, 3, padding=1, cfg=AdaptationConfig(rank=4)),
        )

    def forward(self, x_prev: torch.Tensor, x_curr: torch.Tensor):
        flow = self.flow_net(x_prev, x_curr)
        x_warp = safe_act(_bilinear_warp(x_prev, flow), bound=10.0)
        x_mc = safe_act(self.refine(torch.cat([x_warp, x_prev, flow], dim=1)), bound=10.0)
        return {"flow": flow, "warp": x_warp, "x_mc": x_mc}
