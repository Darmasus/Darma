"""Learned weight prior — predicts (mu, log_sigma) per integer code.

Week-2 v1 is *non-autoregressive*: each weight's distribution is conditioned
only on its position metadata (tensor type, level, offset). This loses the
inter-weight correlations a true AR transformer would catch, but gives:
  - O(1) parallel forward at decode time
  - A clean foundation Week 3 can extend (caption-conditioning) and
    Week 2.5 can upgrade to AR/cross-tensor context if the bpp gain
    isn't enough.

The expected win vs Week 1's per-tensor empirical sigma comes from the
prior knowing globally that "level-0 xy hash entries are ~3x noisier than
level-5", learned from many overfit INRs rather than estimated per-tensor.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .serialize import N_TYPES, MAX_LEVEL


class WeightPrior(nn.Module):
    """Maps (type_id, level, offset [, caption_emb]) -> (mu, log_sigma).

    If `caption_dim > 0`, the prior is conditioned on a per-INR caption
    embedding via FiLM modulation (scale + shift applied at each hidden
    layer). The encoder re-runs the text model on the caption string
    shipped in the bitstream, so the decoder recovers the same
    conditioning.
    """

    def __init__(self, hidden: int = 64, depth: int = 3,
                  log_sigma_min: float = -4.0, log_sigma_max: float = 10.0,
                  log_sigma_init: float = 4.0,
                  caption_dim: int = 0):
        super().__init__()
        self.type_emb  = nn.Embedding(N_TYPES, hidden)
        self.level_emb = nn.Embedding(MAX_LEVEL, hidden)
        self.offset_in = nn.Linear(1, hidden)

        self.caption_dim = caption_dim
        self.depth = depth
        if caption_dim > 0:
            # Map caption embedding to per-layer (gamma, beta) FiLM params.
            # We emit 2*hidden per layer so every hidden layer gets its
            # own (gamma, beta). Gamma is applied as (1 + gamma) so zero
            # init reduces to identity.
            self.caption_proj = nn.Linear(caption_dim, 2 * hidden * depth)
            nn.init.zeros_(self.caption_proj.weight)
            nn.init.zeros_(self.caption_proj.bias)

        # Explicit per-layer Linears (instead of nn.Sequential) so FiLM
        # can be applied between them.
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(depth)]
        )
        self.head_last = nn.Linear(hidden, 2)
        # Standard init on weight so hidden features affect the output;
        # bias starts at (mu=0, log_sigma=log_sigma_init) so initial
        # sigma is in the ~55 ballpark instead of exp(random).
        nn.init.xavier_uniform_(self.head_last.weight, gain=0.01)
        with torch.no_grad():
            self.head_last.bias.copy_(torch.tensor([0.0, log_sigma_init]))

        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        self.hidden = hidden

    def _saturate(self, x: torch.Tensor) -> torch.Tensor:
        center = 0.5 * (self.log_sigma_min + self.log_sigma_max)
        half   = 0.5 * (self.log_sigma_max - self.log_sigma_min)
        return center + half * torch.tanh((x - center) / half)

    def forward(self, type_id: torch.Tensor, level: torch.Tensor,
                  offset: torch.Tensor,
                  caption_emb: torch.Tensor | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        h = (self.type_emb(type_id)
             + self.level_emb(level.clamp(max=MAX_LEVEL - 1))
             + self.offset_in(offset.unsqueeze(-1)))

        film = None
        if self.caption_dim > 0:
            if caption_emb is None:
                # Allow caption-less usage by pretending caption=zeros;
                # because caption_proj is zero-initialized this degrades
                # to the non-conditioned prior at init, and training
                # decides how much to lean on the caption.
                film = torch.zeros(2 * self.hidden * self.depth,
                                     device=h.device, dtype=h.dtype)
            else:
                film = self.caption_proj(caption_emb)  # (2*H*D,)
            # split into per-layer (gamma, beta)
            film = film.view(self.depth, 2, self.hidden)

        for i, layer in enumerate(self.hidden_layers):
            h = layer(h)
            if film is not None:
                gamma = 1.0 + film[i, 0]   # (H,) -- broadcasts over tokens
                beta  = film[i, 1]
                h = h * gamma + beta
            h = F.gelu(h)

        out = self.head_last(h)
        mu = out[..., 0]
        log_sigma = self._saturate(out[..., 1])
        return mu, log_sigma


@torch.no_grad()
def prior_params_for_quants(prior: "WeightPrior",
                              quants,
                              caption_emb: torch.Tensor | None = None
                              ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Run the prior for each tensor in `quants` and return
    {name: (mu[n], sigma[n])} where mu/sigma match the flattened shape.

    Used by rate-aware QAT so the rate term is differentiable w.r.t. the
    INR weights (via STE-round) while the prior's (mu, sigma) outputs are
    treated as constants (the prior is frozen during INR overfit)."""
    from .serialize import canonical_order, parse_tensor_name
    device = next(prior.parameters()).device
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name in canonical_order(quants):
        q = quants[name]
        t, l = parse_tensor_name(name)
        n = q.param.numel()
        type_id = torch.full((n,), t, dtype=torch.int64, device=device)
        level   = torch.full((n,), l, dtype=torch.int64, device=device)
        offset  = torch.arange(n, dtype=torch.float32, device=device) / max(n, 1)
        mu, log_sigma = prior(type_id, level, offset, caption_emb=caption_emb)
        out[name] = (mu, log_sigma.exp().clamp_min(0.5))
    return out


def rate_bits_from_prior(quants, prior_params: dict[str, tuple[torch.Tensor, torch.Tensor]]
                          ) -> torch.Tensor:
    """Differentiable rate estimate using prior's (mu, sigma).

    The prior outputs are constants; the STE-rounded integer codes carry
    the gradient back into the INR weights. Bits per weight is:
        0.5 * z^2 + log(sigma) + 0.5 log(2 pi)    (in nats)
    summed across all weights and converted to bits.
    """
    import math
    total = None
    for name, (mu, sigma) in prior_params.items():
        q = quants[name]
        # STE-rounded integer codes, flat:
        s = q.scale
        q_int = (q.param / s).round().detach() + (q.param / s) - (q.param / s).detach()
        q_int = q_int.reshape(-1)
        z = (q_int - mu) / sigma
        nats = 0.5 * z * z + sigma.log() + 0.5 * math.log(2 * math.pi)
        bits = nats.sum() / math.log(2.0)
        total = bits if total is None else total + bits
    return total


def gaussian_nll_bits(ints: torch.Tensor, mu: torch.Tensor,
                       log_sigma: torch.Tensor) -> torch.Tensor:
    """Per-weight bit cost under the prior (continuous Gaussian approx).

    For a discretized-Gaussian PMF the bit cost converges to the continuous
    NLL / log(2) when sigma >> 1. For our integer-code distributions sigma
    is typically 5..200, so the approximation is tight (<1% off).
    """
    sigma = log_sigma.exp()
    z = (ints.float() - mu) / sigma
    # nats -> bits
    return (0.5 * z * z + log_sigma + 0.5 * math.log(2 * math.pi)) / math.log(2.0)
