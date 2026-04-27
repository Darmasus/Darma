"""Train the WeightPrior on a dataset of tokenized INRs.

Loss: continuous Gaussian NLL of the integer codes under the prior's
predicted (mu, sigma).

If `--caption` is set, the prior is FiLM-conditioned on each sample's
caption embedding (MiniLM, 384-dim). Per-sample loop: all tokens from
one INR get that INR's caption embedding.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.optim import AdamW

from cool_chic.prior import WeightPrior, gaussian_nll_bits
from cool_chic.caption import encode_captions, CAPTION_DIM


def _baseline_per_tensor_bits(sample: dict) -> float:
    """Bits under an *idealized* per-tensor Gaussian — anchor that matches
    what Week-1's bitstream costs."""
    total = 0.0
    cur = 0
    for (_name, _t, _l, _shape, n) in sample["entries"]:
        ints = sample["ints"][cur:cur + n].float()
        sigma = float(ints.std().clamp_min(0.5))
        z = ints / sigma
        bits_t = (0.5 * z * z + math.log(sigma) + 0.5 * math.log(2 * math.pi)) / math.log(2.0)
        total += float(bits_t.sum())
        cur += n
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cool_chic/data/inr_dataset.pt")
    ap.add_argument("--out",  default="cool_chic/data/prior.pt")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth",  type=int, default=3)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--caption", action="store_true",
                    help="enable caption-conditioned FiLM prior")
    ap.add_argument("--batch-tokens", type=int, default=0,
                    help="If >0 and caption disabled, sample this many tokens per step "
                          "(minibatch training). 0 = full-batch per epoch.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    samples: list[dict] = torch.load(args.data, weights_only=False)
    print(f"loaded {len(samples)} INR samples")

    # Precompute per-sample caption embeddings if requested.
    caption_embs: list[torch.Tensor | None] = [None] * len(samples)
    if args.caption:
        texts = [s.get("caption", "") for s in samples]
        missing = sum(1 for t in texts if not t)
        if missing:
            raise SystemExit(f"{missing} samples missing captions")
        print("encoding captions...")
        embs = encode_captions(texts, device=args.device)    # (N, 384)
        for i in range(len(samples)):
            caption_embs[i] = embs[i]
        # pairwise cosine to confirm embeddings are diverse
        sim = embs @ embs.T
        print(f"caption cosine min/mean/max: "
              f"{sim.min().item():.3f} / "
              f"{(sim.sum() - sim.trace()).item() / (sim.numel() - sim.shape[0]):.3f} / "
              f"{(sim - torch.eye(sim.shape[0], device=sim.device)).max().item():.3f}")

    # Move per-sample tensors to device
    for s in samples:
        for k in ("ints", "type_id", "level", "offset"):
            s[k] = s[k].to(args.device)

    total_tokens = sum(s["ints"].numel() for s in samples)
    print(f"total tokens: {total_tokens}")

    baseline_bits = sum(_baseline_per_tensor_bits({
        "ints": s["ints"].cpu(), "entries": s["entries"]}) for s in samples)
    print(f"per-tensor Gaussian baseline: {baseline_bits:.0f} bits  "
          f"({baseline_bits / total_tokens:.3f} bits/weight)")

    caption_dim = CAPTION_DIM if args.caption else 0
    prior = WeightPrior(hidden=args.hidden, depth=args.depth,
                          caption_dim=caption_dim).to(args.device)
    n_params = sum(p.numel() for p in prior.parameters())
    print(f"prior params: {n_params}  ({n_params * 4 / 1024:.1f} KB fp32)  "
          f"caption={'yes' if args.caption else 'no'}")

    opt = AdamW(prior.parameters(), lr=args.lr, weight_decay=1e-4)

    # Fast-path for the non-caption prior: concat ALL tokens into a single
    # tensor so each epoch is a single forward + backward over the whole
    # dataset. With the caption path we still have to loop because FiLM
    # modulation is per-sample.
    if not args.caption:
        all_ints   = torch.cat([s["ints"]    for s in samples])
        all_types  = torch.cat([s["type_id"] for s in samples])
        all_levels = torch.cat([s["level"]   for s in samples])
        all_offs   = torch.cat([s["offset"]  for s in samples])

    t0 = time.time()
    best_loss = float("inf")
    best_state = None
    N_all = all_ints.numel() if not args.caption else total_tokens
    for epoch in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        if not args.caption:
            if args.batch_tokens > 0 and args.batch_tokens < N_all:
                idx = torch.randint(0, N_all, (args.batch_tokens,), device=args.device)
                ti, le, of, it = all_types[idx], all_levels[idx], all_offs[idx], all_ints[idx]
            else:
                ti, le, of, it = all_types, all_levels, all_offs, all_ints
            mu, log_sigma = prior(ti, le, of)
            bits = gaussian_nll_bits(it, mu, log_sigma)
            total_loss = bits.mean()
            # For logging extrapolate to full-dataset bits:
            total_bits_tensor = float(bits.mean().detach()) * N_all
        else:
            # For caption conditioning we still loop over samples, but we
            # can minibatch *within* each sample too if batch_tokens>0.
            total_loss = 0.0
            total_bits_tensor = 0.0
            # Sample a random subset of samples each step to keep it fast.
            sample_count = len(samples)
            if args.batch_tokens > 0:
                # Pick enough samples to roughly hit batch_tokens target.
                avg_n = total_tokens / sample_count
                k = max(1, min(sample_count, int(args.batch_tokens / avg_n)))
                picks = torch.randperm(sample_count)[:k].tolist()
            else:
                picks = list(range(sample_count))
            picked_total = 0
            for i in picks:
                s, emb = samples[i], caption_embs[i]
                mu, log_sigma = prior(s["type_id"], s["level"], s["offset"],
                                        caption_emb=emb)
                bits = gaussian_nll_bits(s["ints"], mu, log_sigma)
                total_loss = total_loss + bits.sum()
                total_bits_tensor = total_bits_tensor + float(bits.sum().detach())
                picked_total += bits.numel()
            total_loss = total_loss / max(picked_total, 1)
            total_bits_tensor = total_bits_tensor / max(picked_total, 1) * total_tokens
        loss = total_loss if not args.caption else total_loss
        loss.backward()
        # FiLM conditioning can go unstable — cap gradient norm.
        torch.nn.utils.clip_grad_norm_(prior.parameters(), max_norm=1.0)
        opt.step()

        # Keep the best checkpoint seen during training (simplest early-stop).
        cur = float(loss.detach())
        if cur < best_loss:
            best_loss = cur
            best_state = {k: v.detach().clone() for k, v in prior.state_dict().items()}

        if epoch % max(args.epochs // 20, 1) == 0 or epoch == args.epochs - 1:
            ratio = baseline_bits / max(total_bits_tensor, 1e-9)
            print(f"  epoch {epoch:4d}  loss(bits/w)={float(loss.detach()):.3f}  "
                  f"total={total_bits_tensor:.0f}  vs baseline={baseline_bits:.0f}  "
                  f"-> {ratio:.3f}x  ({time.time()-t0:.1f}s)")

    # Prefer the best-seen state to insulate from late-training blow-ups.
    final_state = best_state if best_state is not None else prior.state_dict()
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": final_state,
        "hidden": args.hidden,
        "depth":  args.depth,
        "caption_dim": caption_dim,
    }, out_path)
    print(f"\nsaved prior to {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)  "
          f"best_loss={best_loss:.3f} bits/weight")


if __name__ == "__main__":
    main()
