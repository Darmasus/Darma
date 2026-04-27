"""Masked diffusion residual coder.

Rather than run a full DDPM at decode time (too slow), we use a masked
discrete-token diffusion scheme over VQ residual tokens (MaskGIT-style),
which converges in ~8 steps. This file provides:

  * a small VQ tokenizer for the (x_curr - x_mc) residual
  * a masked-token transformer that predicts tokens from a partially-masked
    sequence, conditioned on the motion-compensated frame
  * parallel iterative decoding: 8-step mask schedule

At the bitstream level we transmit only the VQ indices *that the encoder
chose to keep* (the rest are predicted by the decoder from context). A
confidence threshold decides which tokens are skipped — this is the
quality/bitrate knob.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VQTokenizer(nn.Module):
    def __init__(self, codebook_size: int = 1024, dim: int = 64, patch: int = 8):
        super().__init__()
        self.patch = patch
        self.dim = dim
        self.codebook_size = codebook_size
        self.enc = nn.Conv2d(3, dim, patch, stride=patch)
        self.dec = nn.ConvTranspose2d(dim, 3, patch, stride=patch)
        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # z: (B, D, H, W)
        B, D, H, W = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, D)
        d = (flat.pow(2).sum(1, keepdim=True)
             - 2 * flat @ self.codebook.weight.t()
             + self.codebook.weight.pow(2).sum(1))
        idx = d.argmin(1)
        zq = self.codebook(idx).view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        # STE
        zq_ste = z + (zq - z).detach()
        return zq_ste, idx.view(B, H, W)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def decode_from_indices(self, idx: torch.Tensor) -> torch.Tensor:
        B, H, W = idx.shape
        z = self.codebook(idx.view(-1)).view(B, H, W, self.dim).permute(0, 3, 1, 2).contiguous()
        return self.dec(z)


class MaskedTransformer(nn.Module):
    def __init__(self, codebook_size: int, dim: int = 256, depth: int = 6, heads: int = 8):
        super().__init__()
        self.mask_id = codebook_size            # extra token id
        self.tok_emb = nn.Embedding(codebook_size + 1, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, dim))
        self.ctx_proj = nn.Conv2d(3, dim, 1)     # from x_mc
        enc_layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True, activation="gelu")
        self.net = nn.TransformerEncoder(enc_layer, depth)
        self.head = nn.Linear(dim, codebook_size)

    def forward(self, tok_ids: torch.Tensor, x_mc: torch.Tensor) -> torch.Tensor:
        B, H, W = tok_ids.shape
        tok = self.tok_emb(tok_ids).view(B, H * W, -1)
        ctx = F.adaptive_avg_pool2d(self.ctx_proj(x_mc), (H, W))
        ctx = ctx.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        h = tok + ctx + self.pos_emb[:, : H * W]
        h = self.net(h)
        return self.head(h).view(B, H, W, -1)


class MaskedDiffusionResidual(nn.Module):
    """Residual coder combining VQ tokenization with masked-diffusion priors."""

    def __init__(self, codebook_size: int = 1024, dim: int = 64, patch: int = 8):
        super().__init__()
        self.tok = VQTokenizer(codebook_size, dim, patch)
        self.prior = MaskedTransformer(codebook_size, dim=256, depth=4, heads=8)
        self.codebook_size = codebook_size

    def encode_residual(self, residual: torch.Tensor, x_mc: torch.Tensor):
        z = self.tok.encode(residual)
        _, idx = self.tok.quantize(z)
        # Encoder decides which tokens to transmit based on prior confidence.
        with torch.no_grad():
            masked = torch.full_like(idx, self.prior.mask_id)
            logits = self.prior(masked, x_mc)
            probs = logits.softmax(-1)
            keep_prob = probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        # Lower keep_prob -> prior can't predict it -> must transmit.
        return idx, keep_prob

    @torch.no_grad()
    def decode_residual(self, transmitted_idx: torch.Tensor, keep_mask: torch.Tensor,
                        x_mc: torch.Tensor, steps: int = 8) -> torch.Tensor:
        """Iterative parallel decoding (MaskGIT)."""
        idx = torch.where(keep_mask, transmitted_idx,
                          torch.full_like(transmitted_idx, self.prior.mask_id))
        B, H, W = idx.shape
        total = H * W
        for t in range(steps):
            logits = self.prior(idx, x_mc)
            probs = logits.softmax(-1)
            sampled = probs.argmax(-1)
            conf = probs.max(-1).values
            mask = idx.eq(self.prior.mask_id)
            # schedule: unmask cos(pi/2 * (1 - t/steps)) fraction each step
            frac = math.cos(math.pi / 2 * (1 - (t + 1) / steps))
            k = int(total * frac)
            conf_masked = conf.masked_fill(~mask, -1)
            thresh = conf_masked.view(B, -1).topk(min(k, total), dim=-1).values[:, -1:]
            thresh = thresh.view(B, 1, 1)
            to_fill = mask & (conf_masked >= thresh)
            idx = torch.where(to_fill, sampled, idx)
        return self.tok.decode_from_indices(idx)
