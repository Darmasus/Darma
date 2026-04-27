"""Build an INR dataset from a manifest of (video, caption) pairs.

Trains a small INR per clip. Smaller capacity + fewer steps than the
v1 dataset generator because we want *diversity* over per-clip PSNR —
the prior cares about the spread of weight distributions, not that any
single INR hits a PSNR target.
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
    ap.add_argument("--manifest", default="cool_chic/data/clips/manifest.json")
    ap.add_argument("--out", default="cool_chic/data/inr_dataset_v2.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32-steps", type=int, default=700)
    ap.add_argument("--qat-steps",  type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--pixels", type=int, default=8192)
    ap.add_argument("--exclude", type=str, default="",
                    help="comma-separated clip names to hold out of training")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    exclude = set(s.strip() for s in args.exclude.split(",") if s.strip())
    manifest = [c for c in manifest if c["name"] not in exclude]
    print(f"{len(manifest)} clips to process (excluded: {sorted(exclude) or 'none'})")

    # One compact capacity for every clip keeps the prior's input space
    # uniform (same set of tensor shapes across samples). Small enough
    # that 24 INRs finish in ~10 min on the GPU.
    cfg = VideoConfig(L=4, T=1 << 11, F=2, N_min=8, N_max=96,
                       mlp_hidden=32, mlp_depth=3)

    samples = []
    t0 = time.time()
    for i, entry in enumerate(manifest):
        tt = time.time()
        video = read_video(entry["path"])
        print(f"  [{i+1:2d}/{len(manifest)}] {entry['name']:20s} "
              f"{tuple(video.shape)}  caption={entry['caption'][:40]}...",
              end="", flush=True)
        sample = _train_one(video, cfg, fp32_steps=args.fp32_steps,
                              qat_steps=args.qat_steps, lr=args.lr,
                              pixels=args.pixels, device=args.device)
        sample["source"]  = entry["name"]
        sample["caption"] = entry["caption"]
        # Quick per-sample stats
        ints_std = sample["ints"].float().std().item()
        ints_range = (int(sample["ints"].min()), int(sample["ints"].max()))
        samples.append(sample)
        print(f" ints std={ints_std:6.2f} range={ints_range}  ({time.time()-tt:.1f}s)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, out_path)
    elapsed = time.time() - t0
    total_tokens = sum(s["ints"].numel() for s in samples)
    print(f"\ndone: {len(samples)} samples, {total_tokens} tokens, "
          f"{elapsed:.1f}s, {out_path.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
