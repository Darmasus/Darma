"""W3 D1 smoke test: caption encoder reproducibility + FiLM prior shapes."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from cool_chic.caption import encode_caption, encode_captions, CAPTION_DIM
from cool_chic.prior import WeightPrior, gaussian_nll_bits


def test_encode_reproducible():
    a = encode_caption("a green grass field waving in the wind")
    b = encode_caption("a green grass field waving in the wind")
    c = encode_caption("a cat on a couch")
    assert a.shape == (CAPTION_DIM,), a.shape
    assert torch.allclose(a, b, atol=1e-6), "same text should give same embedding"
    cos_same = float(a @ b)
    cos_diff = float(a @ c)
    print(f"cosine same-text : {cos_same:.4f}")
    print(f"cosine diff-text : {cos_diff:.4f}")
    assert cos_same > cos_diff + 0.05


def test_film_prior():
    prior = WeightPrior(hidden=64, depth=3, caption_dim=CAPTION_DIM)
    n = 1024
    type_id = torch.randint(0, 5, (n,))
    level   = torch.randint(0, 6, (n,))
    offset  = torch.rand(n)

    # caption=None -> identity FiLM (weight init is zero) -> works
    mu0, log_sigma0 = prior(type_id, level, offset, caption_emb=None)
    print(f"no-caption mu stats:  mean={mu0.mean():.3f}  std={mu0.std():.3f}")
    print(f"no-caption log_sigma: mean={log_sigma0.mean():.3f}  std={log_sigma0.std():.3f}")

    # with a real caption embedding
    emb = encode_caption("grass moving in the breeze, green")
    mu1, log_sigma1 = prior(type_id, level, offset, caption_emb=emb)
    assert mu1.shape == mu0.shape

    # At init (caption_proj zero-init) the captioned and un-captioned
    # outputs should be identical.
    assert torch.allclose(mu0, mu1, atol=1e-6), "zero-init FiLM should be identity"
    print("zero-init FiLM is identity: OK")

    # Perturb caption_proj so FiLM is active, verify outputs now differ.
    with torch.no_grad():
        prior.caption_proj.weight.normal_(std=0.5)
        prior.caption_proj.bias.normal_(std=0.1)
    mu2, _ = prior(type_id, level, offset, caption_emb=emb)
    mu3, _ = prior(type_id, level, offset,
                    caption_emb=encode_caption("a cat on a couch"))
    delta = (mu2 - mu3).abs().mean().item()
    print(f"different-caption delta mu: {delta:.4f}")
    assert delta > 1e-4, \
        "different captions should give different outputs once FiLM is active"


if __name__ == "__main__":
    test_encode_reproducible()
    test_film_prior()
    print("\nok")
