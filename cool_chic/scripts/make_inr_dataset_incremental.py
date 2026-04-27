"""Extend an existing INR dataset with new clips. Skips any clip whose
`name` already appears in the base dataset, trains INRs for the rest,
and writes the union to `--out`.
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

from cool_chic.codec import VideoINR, VideoConfig
from cool_chic.train_video_qat import overfit_video_qat, VideoQATConfig
from cool_chic.serialize import tokenize
from utils.video_io import read_video


def _fp32_overfit(video, model, *, steps, lr, pixels, device):
    target_flat = video.to(device).permute(0, 2, 3, 1).reshape(-1, 3)
    N = target_flat.shape[0]
    T, _, H, W = video.shape
    t_d, h_d, w_d = max(T - 1, 1), max(H - 1, 1), max(W - 1, 1)
    opt = Adam(model.parameters(), lr=lr)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        idx = torch.randint(0, N, (pixels,), device=device)
        ti = idx // (H * W); rem = idx % (H * W); yi = rem // W; xi = rem % W
        coords = torch.stack([xi.float()/w_d, yi.float()/h_d, ti.float()/t_d], -1)
        pred = model.mlp(model.grid(coords))
        loss = F.mse_loss(pred, target_flat[idx])
        loss.backward(); opt.step()


def _train_one(video, cfg, *, fp32_steps, qat_steps, lr, pixels, device):
    Tf, _, H, W = video.shape
    model = VideoINR(T_frames=Tf, H=H, W=W, cfg=cfg).to(device)
    _fp32_overfit(video, model, steps=fp32_steps, lr=lr, pixels=pixels,
                   device=device)
    quants, _ = overfit_video_qat(video, model,
                                    qcfg=VideoQATConfig(steps=qat_steps,
                                                        lr=lr * 0.4,
                                                        scale_lr=5e-3,
                                                        lambda_rate=0.0,
                                                        pixels_per_step=pixels,
                                                        log_every=max(qat_steps, 1),
                                                        init_scale=5e-3),
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
        "cfg": cfg, "T": Tf, "H": H, "W": W,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",  required=True, help="full combined manifest")
    ap.add_argument("--base",      default="cool_chic/data/inr_dataset_v3.pt",
                    help="existing dataset to extend (name-keyed skip)")
    ap.add_argument("--out",       required=True)
    ap.add_argument("--exclude",   default="testsrc,testsrc2")
    ap.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32-steps", type=int, default=700)
    ap.add_argument("--qat-steps",  type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--pixels", type=int, default=8192)
    args = ap.parse_args()

    base = []
    if Path(args.base).exists():
        base = torch.load(args.base, weights_only=False)
    existing_names = {s.get("source") for s in base}
    print(f"base dataset: {len(base)} samples already present")

    manifest = json.loads(Path(args.manifest).read_text())
    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}
    todo = [c for c in manifest
            if c["name"] not in existing_names and c["name"] not in exclude]
    print(f"{len(todo)} new clips to train (skipping {len(manifest) - len(todo)})")

    cfg = VideoConfig(L=4, T=1 << 11, F=2, N_min=8, N_max=96,
                       mlp_hidden=32, mlp_depth=3)

    new_samples = []
    t0 = time.time()
    for i, entry in enumerate(todo):
        tt = time.time()
        video = read_video(entry["path"])
        sample = _train_one(video, cfg, fp32_steps=args.fp32_steps,
                              qat_steps=args.qat_steps, lr=args.lr,
                              pixels=args.pixels, device=args.device)
        sample["source"] = entry["name"]
        sample["caption"] = entry["caption"]
        new_samples.append(sample)
        ints_std = sample["ints"].float().std().item()
        print(f"  [{i+1:3d}/{len(todo)}] {entry['name']:24s}  ints std={ints_std:6.2f}  "
              f"({time.time()-tt:.1f}s)")

    all_samples = list(base) + new_samples
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_samples, args.out)
    total_tokens = sum(s["ints"].numel() for s in all_samples)
    print(f"\ndone: base {len(base)} + new {len(new_samples)} = {len(all_samples)}, "
          f"{total_tokens} tokens, {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
