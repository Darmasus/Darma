"""Train the caption-conditioned WeightHypernet on a dataset of trained
INRs. Loss: MSE between hypernet output and the dequantized weights
(= integer codes * per-tensor scale).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from cool_chic.hypernet import WeightHypernet
from cool_chic.caption import encode_captions, CAPTION_DIM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cool_chic/data/inr_dataset_v2.pt")
    ap.add_argument("--out",  default="cool_chic/data/hypernet.pt")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth",  type=int, default=3)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--caption", action="store_true", default=True)
    ap.add_argument("--no-caption", dest="caption", action="store_false")
    ap.add_argument("--value-scale", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    samples: list[dict] = torch.load(args.data, weights_only=False)
    print(f"loaded {len(samples)} INR samples")

    # Precompute per-sample dequantized weights = ints * scale, matched
    # with per-weight metadata. We flatten across ALL tensors so each
    # sample is a single long vector with matching (type_id, level, offset).
    prepared = []
    for s in samples:
        ints = s["ints"].to(args.device).float()
        type_id = s["type_id"].to(args.device)
        level   = s["level"].to(args.device)
        offset  = s["offset"].to(args.device)
        # Build per-weight scale vector from (entries, scales)
        per_weight_scale = torch.empty_like(ints)
        cur = 0
        for (_name, _t, _l, _shape, n), sc in zip(s["entries"], s["scales"]):
            per_weight_scale[cur:cur + n] = sc
            cur += n
        values = ints * per_weight_scale       # dequantized weights
        prepared.append({
            "type_id": type_id, "level": level, "offset": offset,
            "values":  values,
            "caption": s.get("caption", ""),
        })

    caption_embs: list[torch.Tensor | None] = [None] * len(prepared)
    if args.caption:
        texts = [p["caption"] for p in prepared]
        if any(not t for t in texts):
            raise SystemExit("caption conditioning requested but some samples missing captions")
        print("encoding captions...")
        embs = encode_captions(texts, device=args.device)
        for i in range(len(prepared)):
            caption_embs[i] = embs[i]

    total_weights = sum(p["values"].numel() for p in prepared)
    mean_target = float(torch.cat([p["values"] for p in prepared]).mean())
    std_target  = float(torch.cat([p["values"] for p in prepared]).std())
    print(f"total weights: {total_weights}  target mean={mean_target:.4f}  std={std_target:.4f}")

    caption_dim = CAPTION_DIM if args.caption else 0
    hypernet = WeightHypernet(hidden=args.hidden, depth=args.depth,
                                caption_dim=caption_dim,
                                value_scale=args.value_scale).to(args.device)
    n_params = sum(p.numel() for p in hypernet.parameters())
    print(f"hypernet params: {n_params}  ({n_params * 4 / 1024:.1f} KB fp32)  "
          f"caption={'yes' if args.caption else 'no'}")

    opt = AdamW(hypernet.parameters(), lr=args.lr, weight_decay=1e-4)

    t0 = time.time()
    best_mse = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_mse_sum = 0.0
        for p, emb in zip(prepared, caption_embs):
            pred = hypernet(p["type_id"], p["level"], p["offset"],
                              caption_emb=emb)
            mse = F.mse_loss(pred, p["values"], reduction="sum")
            total_loss = total_loss + mse
            total_mse_sum = total_mse_sum + float(mse.detach())
        loss = total_loss / total_weights
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=1.0)
        opt.step()

        mse_avg = total_mse_sum / total_weights
        if mse_avg < best_mse:
            best_mse = mse_avg
            best_state = {k: v.detach().clone() for k, v in hypernet.state_dict().items()}

        if epoch % max(args.epochs // 20, 1) == 0 or epoch == args.epochs - 1:
            # "Naive init" MSE = just predict the mean: variance of the data.
            naive_mse = std_target * std_target
            print(f"  epoch {epoch:4d}  mse/w={mse_avg:.6f}  "
                  f"(naive={naive_mse:.6f}, ratio={naive_mse/max(mse_avg,1e-12):.2f}x)  "
                  f"({time.time()-t0:.1f}s)")

    final_state = best_state if best_state is not None else hypernet.state_dict()
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": final_state,
        "hidden": args.hidden,
        "depth":  args.depth,
        "caption_dim": caption_dim,
        "value_scale": args.value_scale,
    }, out_path)
    print(f"\nsaved hypernet to {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)  "
          f"best mse/w={best_mse:.6f}")


if __name__ == "__main__":
    main()
