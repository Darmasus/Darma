"""Numerical-stability primitives shared across the codec.

Neural video codecs are uniquely vulnerable to NaN explosions because:
  * GDN normalises by sqrt(beta + sum(gamma * x^2)) — a small denominator
    means a huge output, which then gets squared in the next layer.
  * GaussianConditional computes -log p(y|mu, sigma) which has a
    (y - mu)^2 / sigma^2 term — a small sigma blows up.
  * Optical-flow warps interpolate outside the image; flow values much
    larger than the spatial extent produce extrapolated nonsense.

These helpers don't change the math when inputs are well-behaved, they
just prevent any single batch from producing inf/NaN that poisons the
whole model.
"""
from __future__ import annotations

import torch


# --- magnitude bounds ---------------------------------------------------- #
# Generous defaults; only kick in for clearly-pathological values.
ACT_BOUND       = 1.0e3      # any conv activation outside ±1000 is bogus
FLOW_BOUND      = 64.0       # max |flow| in pixels (1080p panning fits easily)
LOGIT_BOUND     = 30.0       # softmax inputs
SCALE_FLOOR     = 0.05       # min std for likelihood Gaussian
MEAN_BOUND      = 50.0       # max |mu| for likelihood
LIKELIHOOD_FLOOR = 1.0e-9    # min likelihood (avoids log(0))


def safe_act(x: torch.Tensor, bound: float = ACT_BOUND) -> torch.Tensor:
    """Replace inf/NaN with 0, clamp to ±bound. Cheap (one nan_to_num + clamp)."""
    x = torch.nan_to_num(x, nan=0.0, posinf=bound, neginf=-bound)
    return x.clamp_(-bound, bound)


def safe_flow(flow: torch.Tensor, bound: float = FLOW_BOUND) -> torch.Tensor:
    return safe_act(flow, bound=bound)


def safe_scales(scales: torch.Tensor, floor: float = SCALE_FLOOR) -> torch.Tensor:
    """Floor + nan-replace for Gaussian-conditional sigma."""
    scales = torch.nan_to_num(scales, nan=floor, posinf=ACT_BOUND, neginf=floor)
    return scales.clamp_(min=floor)


def safe_means(mu: torch.Tensor, bound: float = MEAN_BOUND) -> torch.Tensor:
    return safe_act(mu, bound=bound)


def state_dict_finite(state_dict) -> bool:
    """True if every tensor in a state dict is finite."""
    return all(torch.isfinite(v).all().item() for v in state_dict.values()
               if torch.is_tensor(v))
