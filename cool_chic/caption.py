"""Text -> embedding via MiniLM (sentence-transformers/all-MiniLM-L6-v2).

Used by Week-3 to condition the WeightPrior on a short caption describing
the clip. The text encoder is deterministic given the caption string, so
encoder and decoder produce identical embeddings from the same caption.

Cheap: 22M params, 384-dim output, runs in a few ms on CPU.
"""
from __future__ import annotations

import functools

import torch
import torch.nn.functional as F


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CAPTION_DIM  = 384


@functools.lru_cache(maxsize=2)
def _load(model_name: str, device: str):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


@torch.no_grad()
def encode_captions(texts: list[str], device: str | torch.device = "cpu",
                     model_name: str = DEFAULT_MODEL) -> torch.Tensor:
    """Returns (N, 384) L2-normalized embeddings.

    Mean-pool over tokens with the attention mask (standard MiniLM
    sentence-embedding recipe); then L2-normalize so cosine similarity
    between captions is a plain dot product.
    """
    if isinstance(device, torch.device):
        device = str(device)
    tok, model = _load(model_name, device)
    enc = tok(texts, padding=True, truncation=True, max_length=64,
               return_tensors="pt").to(device)
    hidden = model(**enc).last_hidden_state            # (N, L, D)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
    return F.normalize(pooled, dim=-1)


def encode_caption(text: str, device: str | torch.device = "cpu",
                    model_name: str = DEFAULT_MODEL) -> torch.Tensor:
    """Returns a (384,) embedding."""
    return encode_captions([text], device=device, model_name=model_name).squeeze(0)
