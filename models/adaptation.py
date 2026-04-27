"""LoRA-style Weight-Adaptation layer.

W' = W + (alpha / r) * B @ A,   A: (r, c_in*k*k), B: (c_out, r)

Only A and B are transmitted as Parameter Update Packets (PUPs). Base W stays
fixed on both encoder and decoder. At decode time, we compose W' once per GOP
and run a standard conv — no autograd needed on the decode path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AdaptationConfig:
    rank: int = 4
    alpha: float = 8.0
    # Per-layer quantization step. Encoder can re-tune this via RD search.
    init_scale: float = 2.0e-3
    # Gaussian prior std for the PUP entropy model. Also re-tuned per GOP.
    init_sigma: float = 5.0e-3


class AdaptableConv2d(nn.Module):
    """Conv2d whose weights can be adapted by a transmitted low-rank delta.

    The base weight `W` is frozen at deploy time. `A` and `B` are populated
    per-GOP by `apply_pup` from a decoded PUP. Encoder-side training fits them.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        cfg: AdaptationConfig | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.cfg = cfg or AdaptationConfig()

        self.base = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias,
        )

        r = self.cfg.rank
        fan_in = in_channels * kernel_size * kernel_size
        # A, B start at zero -> W' == W at initialization.
        self.A = nn.Parameter(torch.zeros(r, fan_in))
        self.B = nn.Parameter(torch.zeros(out_channels, r))

        # Encoder-tunable quantization parameters, also transmitted in PUP hdr.
        self.log_scale = nn.Parameter(torch.tensor(self.cfg.init_scale).log())
        self.log_sigma = nn.Parameter(torch.tensor(self.cfg.init_sigma).log())

        # Flag: when True, compose W' = W + (alpha/r) * B @ A on the fly.
        self._adapted = False

    # ------------------------------------------------------------------ #
    # encoder-side helpers
    # ------------------------------------------------------------------ #
    def reset_delta(self) -> None:
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        nn.init.zeros_(self.B)
        self._adapted = True

    def freeze_base(self) -> None:
        for p in self.base.parameters():
            p.requires_grad_(False)

    def adaptable_parameters(self) -> Iterator[nn.Parameter]:
        yield self.A
        yield self.B
        yield self.log_scale
        yield self.log_sigma

    # ------------------------------------------------------------------ #
    # quantization with straight-through estimator
    # ------------------------------------------------------------------ #
    def quantized_AB(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.log_scale.exp()
        # STE: round in forward, identity gradient in backward.
        A_q = self.A + (torch.round(self.A / s) * s - self.A).detach()
        B_q = self.B + (torch.round(self.B / s) * s - self.B).detach()
        return A_q, B_q, s

    def rate_bits(self) -> torch.Tensor:
        """Differentiable upper bound on PUP payload bits (Gaussian prior)."""
        A_q, B_q, _ = self.quantized_AB()
        sigma = self.log_sigma.exp().clamp(min=1e-6)
        # -log2 p(x) under N(0, sigma^2), integrated over a quantization bin.
        nll = 0.5 * ((A_q / sigma) ** 2) + torch.log(sigma * (2 * torch.pi) ** 0.5)
        bits = nll.sum() / torch.log(torch.tensor(2.0))
        nll_b = 0.5 * ((B_q / sigma) ** 2) + torch.log(sigma * (2 * torch.pi) ** 0.5)
        bits = bits + nll_b.sum() / torch.log(torch.tensor(2.0))
        return bits

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.base.weight
        b = self.base.bias
        if self._adapted:
            A_q, B_q, _ = self.quantized_AB()
            delta = (self.cfg.alpha / self.cfg.rank) * (B_q @ A_q)
            delta = delta.view_as(W)
            W = W + delta
        return F.conv2d(x, W, b, stride=self.stride, padding=self.padding)


def collect_adaptable_layers(module: nn.Module) -> list[tuple[str, AdaptableConv2d]]:
    return [(n, m) for n, m in module.named_modules() if isinstance(m, AdaptableConv2d)]


def apply_pup(module: nn.Module, pup: dict[str, dict[str, torch.Tensor]]) -> None:
    """Load decoded PUP tensors into the model's AdaptableConv2d layers.

    `pup[layer_name]` must contain {"A", "B", "log_scale", "log_sigma"}.
    """
    layers = dict(collect_adaptable_layers(module))
    for name, entry in pup.items():
        if name not in layers:
            continue
        layer = layers[name]
        with torch.no_grad():
            layer.A.copy_(entry["A"])
            layer.B.copy_(entry["B"])
            layer.log_scale.copy_(entry["log_scale"])
            layer.log_sigma.copy_(entry["log_sigma"])
            layer._adapted = True
