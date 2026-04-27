"""Build a dataset of overfit NeRV INRs for prior training.

Same idea as `make_inr_dataset_v2.py` but uses the NeRV CNN-decoder
backbone instead of hash-grid + MLP. Each entry is a tokenized weight
sequence (ints + per-position metadata) the prior trainer can consume.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from torch.optim import Adam

from cool_chic.nerv import NeRVBackbone, NeRVConfig
from cool_chic.train_nerv import overfit_nerv, overfit_nerv_qat, NeRVTrainConfig, NeRVQATConfig
from cool_chic.serialize import tokenize
from utils.video_io import read_video


def _train_one(video, ncfg, *, fp32_steps, qat_steps, lr, frames_per_step, device):
    Tf, _, H, W = video.shape
    model = NeRVBackbone(n_frames=Tf, H=H, W=W, cfg=ncfg).to(device)
    overfit_nerv(video, model,
                  tcfg=NeRVTrainConfig(steps=fp32_steps, lr=lr,
                                          log_every=fp32_steps,
                                          frames_per_step=frames_per_step),
                  device=device, verbose=False)
    quants = overfit_nerv_qat(video, model,
                                qcfg=NeRVQATConfig(steps=qat_steps,
                                                    lr=lr * 0.4, scale_lr=5e-3,
                                                    lambda_rate=0.0,
                                                    log_every=qat_steps,
                                                    init_scale=5e-3,
                                                    lambda_warmup_frac=0.0,
                                                    frames_per_step=frames_per_step),
                                device=device, verbose=False)
    tok = tokenize(quants)
    return {
        "ints":    tok["ints"].cpu(),
        "type_id": tok["type_id"].cpu(),
        "level":   tok["level"].cpu(),
        "offset":  tok["offset"].cpu(),
        "scales":  tok["scales"],
        "entries": [(e.name, e.type_id, e.level, e.shape, e.n)
                    for e in tok["entries"]],
        "ncfg": ncfg, "T": Tf, "H": H, "W": W,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="cool_chic/data/combined_manifest_v2.json")
    ap.add_argument("--out", default="cool_chic/data/nerv_dataset.pt")
    ap.add_argument("--exclude", default="testsrc,testsrc2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32-steps", type=int, default=1500)
    ap.add_argument("--qat-steps",  type=int, default=500)
    ap.add_argument("--lr", type=float, default=5e-3)
    # Tiny NeRV is enough for prior diversity; we want many samples not
    # high per-clip PSNR.
    ap.add_argument("--embed-dim", type=int, default=16)
    ap.add_argument("--base-ch",   type=int, default=32)
    ap.add_argument("--frames-per-step", type=int, default=0)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}
    todo = [c for c in manifest if c["name"] not in exclude]
    print(f"{len(todo)} clips to train (excluded: {sorted(exclude)})")

    ncfg = NeRVConfig(embed_dim=args.embed_dim, base_ch=args.base_ch)
    samples = []
    t0 = time.time()
    for i, entry in enumerate(todo):
        tt = time.time()
        video = read_video(entry["path"])
        try:
            sample = _train_one(video, ncfg,
                                  fp32_steps=args.fp32_steps,
                                  qat_steps=args.qat_steps, lr=args.lr,
                                  frames_per_step=args.frames_per_step,
                                  device=args.device)
        except Exception as e:
            print(f"  [{i+1:3d}/{len(todo)}] {entry['name']:24s}  SKIP ({e})")
            continue
        sample["source"]  = entry["name"]
        sample["caption"] = entry["caption"]
        ints_std = sample["ints"].float().std().item()
        samples.append(sample)
        if i % 10 == 0 or i == len(todo) - 1:
            print(f"  [{i+1:3d}/{len(todo)}] {entry['name']:24s}  "
                  f"std={ints_std:6.2f}  ({time.time()-tt:.1f}s)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, args.out)
    total_tokens = sum(s["ints"].numel() for s in samples)
    print(f"\ndone: {len(samples)} samples, {total_tokens} tokens, "
          f"{time.time()-t0:.1f}s total -> {args.out}")


if __name__ == "__main__":
    main()
