"""W2 D1 smoke test: tokenize -> prior -> nll. Just shape sanity."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from cool_chic.codec import VideoINR, VideoConfig
from cool_chic.quantize import attach_quantizers
from cool_chic.serialize import (
    tokenize, detokenize, canonical_order, parse_tensor_name,
    TYPE_GRID_XY, TYPE_GRID_XT, TYPE_GRID_YT, TYPE_MLP_W, TYPE_MLP_B,
)
from cool_chic.prior import WeightPrior, gaussian_nll_bits


def test_smoke():
    cfg = VideoConfig(L=3, T=1 << 8, F=2, N_min=8, N_max=32,
                       mlp_hidden=16, mlp_depth=2)
    model = VideoINR(T_frames=4, H=16, W=16, cfg=cfg)
    quants = attach_quantizers(model, init_scale=1e-2)

    # Move some weights so integer codes are non-zero.
    with torch.no_grad():
        for q in quants.values():
            q.param.add_(torch.randn_like(q.param) * 0.05)

    order = canonical_order(quants)
    print("canonical order:")
    for n in order:
        t, l = parse_tensor_name(n)
        print(f"  {n:35s} type={t} level={l}")

    tok = tokenize(quants)
    print(f"\ntotal tokens: {tok['ints'].numel()}")
    print(f"  type_id unique: {sorted(set(tok['type_id'].tolist()))}")
    print(f"  level   unique: {sorted(set(tok['level'].tolist()))}")
    print(f"  offset  range : [{tok['offset'].min():.3f}, {tok['offset'].max():.3f}]")
    print(f"  ints    stats : min={tok['ints'].min().item()} "
          f"max={tok['ints'].max().item()} std={tok['ints'].float().std().item():.2f}")

    # Detokenize roundtrip
    rebuilt = detokenize(tok["ints"], tok["entries"])
    for name in quants.keys():
        a = quants[name].integer_codes()
        b = rebuilt[name]
        assert a.shape == b.shape, (name, a.shape, b.shape)
        assert torch.equal(a, b), name
    print("detokenize roundtrip: OK")

    # Prior forward
    prior = WeightPrior(hidden=32, depth=2)
    mu, log_sigma = prior(tok["type_id"], tok["level"], tok["offset"])
    assert mu.shape == log_sigma.shape == tok["ints"].shape
    bits = gaussian_nll_bits(tok["ints"], mu, log_sigma)
    print(f"untrained prior bits: total={bits.sum().item():.1f}  "
          f"mean/weight={bits.mean().item():.3f}  "
          f"params={sum(p.numel() for p in prior.parameters())}")


if __name__ == "__main__":
    test_smoke()
    print("\nok")
