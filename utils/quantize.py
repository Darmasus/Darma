"""Quantization primitives used across the codec."""
from __future__ import annotations

import torch


class _STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return _STERound.apply(x)


def uniform_quantize(x: torch.Tensor, step: torch.Tensor | float) -> torch.Tensor:
    if isinstance(step, float):
        step = torch.tensor(step, device=x.device, dtype=x.dtype)
    return ste_round(x / step) * step


def add_uniform_noise(x: torch.Tensor, step: torch.Tensor | float) -> torch.Tensor:
    """Training-time surrogate for rounding: additive uniform noise."""
    if isinstance(step, float):
        step = torch.tensor(step, device=x.device, dtype=x.dtype)
    return x + (torch.rand_like(x) - 0.5) * step
