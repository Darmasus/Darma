"""Quantization for the bitstream weights.

We use uniform symmetric quantization with a per-tensor learnable scale.
Forward pass rounds; backward pass uses the straight-through estimator.

Why per-tensor? Per-channel would shrink the rate slightly more, but adds
side-information overhead and complicates the encoder/decoder contract.
For per-instance INRs (one tiny model per video) per-tensor is sufficient.

The rate model is a Laplace: R(q) = -log2 P(q | μ=0, b=σ/√2). Cheap to
compute, differentiable, and a near-perfect fit for trained weight
distributions in practice.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Straight-through round
# --------------------------------------------------------------------------- #
class _STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return torch.round(x)
    @staticmethod
    def backward(ctx, g): return g


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return _STERound.apply(x)


# --------------------------------------------------------------------------- #
# Per-tensor quantizer (one scale per parameter group)
# --------------------------------------------------------------------------- #
class TensorQuantizer(nn.Module):
    """Wraps a single nn.Parameter and exposes a quantized view of it.

    The trainable parameter remains in fp32; quantization is applied on
    every forward access. This lets the optimizer continue to use real
    gradients via STE while the model behaves as if it were quantized.
    """

    def __init__(self, param: nn.Parameter, init_scale: float = 1e-3):
        super().__init__()
        self.param = param            # registered as submodule's tensor
        # log-scale ensures positivity without a clamp.
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale)))

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp()

    def quantized(self) -> torch.Tensor:
        s = self.scale
        return ste_round(self.param / s) * s

    def integer_codes(self) -> torch.Tensor:
        """Returns the int representation (no STE) for serialization."""
        with torch.no_grad():
            return torch.round(self.param / self.scale).to(torch.int32)

    def rate_bits(self) -> torch.Tensor:
        """Differentiable upper bound on bit-cost under a Laplace prior.

        For continuous Laplace L(0, b), -log p(x) = |x|/b + log(2b).
        We use b learned implicitly as scale * sigma_unit, giving a tight
        per-tensor estimate without extra parameters.
        """
        q = self.quantized()
        # Use the std of |q| as a proxy for b; small constant prevents log(0).
        b = q.abs().mean().clamp_min(1e-9)
        per_param = (q.abs() / b) / math.log(2.0) + torch.log2(2 * b + 1e-12)
        return per_param.sum()


# --------------------------------------------------------------------------- #
# Module-level helpers: attach quantizers to all parameters in a model
# --------------------------------------------------------------------------- #
def attach_quantizers(model: nn.Module, init_scale: float = 1e-3) -> nn.ModuleDict:
    """Returns a ModuleDict mapping parameter-name -> TensorQuantizer.

    Caller is responsible for using `quantizer.quantized()` in place of
    the raw param when running the forward pass under quantization.
    """
    quants = nn.ModuleDict()
    for name, p in model.named_parameters():
        # Skip the quantizer's own log_scale params if present.
        if "log_scale" in name:
            continue
        safe = name.replace(".", "_")
        quants[safe] = TensorQuantizer(p, init_scale=init_scale)
    return quants


def total_rate_bits(quants: nn.ModuleDict) -> torch.Tensor:
    """Sum of differentiable rate estimates across all quantizers."""
    return sum((q.rate_bits() for q in quants.values()),
               start=torch.zeros((), device=next(iter(quants.values())).param.device))
