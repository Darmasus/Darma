"""Caption-conditioned hypernet that emits INR starting weights.

Given a per-weight position (tensor_type, level, offset) and an
optional caption embedding, the hypernet predicts the dequantized
weight value. At encode time, we query the hypernet for every weight
position of the target INR and use the output as initialization,
skipping most of the fp32 warm-up.

The hypernet does NOT ship in the bitstream — it's part of the codec,
like the prior. Encoder and decoder both keep a pinned copy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .serialize import N_TYPES, MAX_LEVEL


class WeightHypernet(nn.Module):
    """Maps (type_id, level, offset [, caption_emb]) -> weight value.

    Same architecture as WeightPrior but with a single scalar output per
    weight (the value) instead of (mu, log_sigma). FiLM conditioning on
    caption is optional.
    """

    def __init__(self, hidden: int = 64, depth: int = 3,
                  caption_dim: int = 0, value_scale: float = 0.05):
        super().__init__()
        self.type_emb  = nn.Embedding(N_TYPES, hidden)
        self.level_emb = nn.Embedding(MAX_LEVEL, hidden)
        self.offset_in = nn.Linear(1, hidden)

        self.caption_dim = caption_dim
        self.depth = depth
        self.hidden = hidden
        if caption_dim > 0:
            self.caption_proj = nn.Linear(caption_dim, 2 * hidden * depth)
            nn.init.zeros_(self.caption_proj.weight)
            nn.init.zeros_(self.caption_proj.bias)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(depth)]
        )
        self.head = nn.Linear(hidden, 1)
        # Small init so the hypernet starts near zero — a mean-zero init
        # is a safe starting point (matches typical INR weight distributions).
        nn.init.xavier_uniform_(self.head.weight, gain=0.1)
        nn.init.zeros_(self.head.bias)

        # Output is tanh-bounded then scaled: predicted weights stay in
        # [-value_scale, value_scale]. Trained INR weights we've seen are
        # ~|w| < 0.05, so 0.05 is a good default.
        self.value_scale = value_scale

    def forward(self, type_id: torch.Tensor, level: torch.Tensor,
                  offset: torch.Tensor,
                  caption_emb: torch.Tensor | None = None) -> torch.Tensor:
        h = (self.type_emb(type_id)
             + self.level_emb(level.clamp(max=MAX_LEVEL - 1))
             + self.offset_in(offset.unsqueeze(-1)))

        film = None
        if self.caption_dim > 0:
            if caption_emb is None:
                film = torch.zeros(2 * self.hidden * self.depth,
                                     device=h.device, dtype=h.dtype)
            else:
                film = self.caption_proj(caption_emb)
            film = film.view(self.depth, 2, self.hidden)

        for i, layer in enumerate(self.hidden_layers):
            h = layer(h)
            if film is not None:
                gamma = 1.0 + film[i, 0]
                beta  = film[i, 1]
                h = h * gamma + beta
            h = F.gelu(h)

        out = self.head(h).squeeze(-1)
        return self.value_scale * torch.tanh(out)


@torch.no_grad()
def apply_hypernet_init(model, hypernet: WeightHypernet,
                          caption_emb: torch.Tensor | None = None) -> int:
    """Initialize every parameter of `model` from the hypernet's output.

    Returns the number of parameters initialized. Parameters whose names
    don't match the INR convention are skipped (silently — nn.Linear
    biases / anything else handled by PyTorch init is fine).
    """
    from .serialize import parse_tensor_name
    device = next(hypernet.parameters()).device
    n_inited = 0
    for name, p in model.named_parameters():
        safe = name.replace(".", "_")
        try:
            t, l = parse_tensor_name(safe)
        except ValueError:
            continue
        n = p.numel()
        type_id = torch.full((n,), t, dtype=torch.int64, device=device)
        level   = torch.full((n,), l, dtype=torch.int64, device=device)
        offset  = torch.arange(n, dtype=torch.float32, device=device) / max(n, 1)
        values  = hypernet(type_id, level, offset, caption_emb=caption_emb)
        p.data.copy_(values.view(p.shape).to(p.device))
        n_inited += n
    return n_inited
