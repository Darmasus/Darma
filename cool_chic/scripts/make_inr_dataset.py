"""Generate a dataset of trained Cool-Chic INRs.

Each sample is one tokenized INR (flat int codes + per-position metadata).
The prior network in `cool_chic/prior.py` is trained over the union to
learn per-(tensor-type, level) weight statistics.

Diversity matters more than per-clip PSNR here. We sweep:
  - real clip windows (sliding over sample.mp4)
  - one synthetic clip from test_video_overfit
  - small / medium grid configs
so the prior sees different scales / distributions, not 30 copies of the
same INR's tensor stats.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from torch.optim import Adam

from cool_chic.codec import VideoINR, VideoConfig
from cool_chic.quantize import attach_quantizers
from cool_chic.train_video_qat import overfit_video_qat, VideoQATConfig
from cool_chic.serialize import tokenize, parse_tensor_name
from cool_chic.tests.test_video_overfit import _make_test_video
from utils.video_io import read_video


def _fp32_overfit(video, model, *, steps, lr, pixels, device, log_every=400):
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
        if step % log_every == 0 or step == steps - 1:
            print(f"      fp32 {step:5d}  loss={float(loss.detach()):.5f}",
                  flush=True)


def _train_one(video, cfg: VideoConfig, *, fp32_steps, qat_steps, lr, pixels,
                 device) -> dict:
    Tf, _, H, W = video.shape
    model = VideoINR(T_frames=Tf, H=H, W=W, cfg=cfg).to(device)
    _fp32_overfit(video, model, steps=fp32_steps, lr=lr, pixels=pixels,
                   device=device, log_every=max(fp32_steps // 4, 1))
    quants, _ = overfit_video_qat(video, model,
                                    qcfg=VideoQATConfig(steps=qat_steps,
                                                        lr=lr * 0.4,
                                                        scale_lr=5e-3,
                                                        lambda_rate=0.0,
                                                        pixels_per_step=pixels,
                                                        log_every=max(qat_steps // 3, 1),
                                                        init_scale=5e-3),
                                    device=device, verbose=True)
    tok = tokenize(quants)
    return {
        "ints":    tok["ints"].cpu(),
        "type_id": tok["type_id"].cpu(),
        "level":   tok["level"].cpu(),
        "offset":  tok["offset"].cpu(),
        "scales":  tok["scales"],
        "entries": [(e.name, e.type_id, e.level, e.shape, e.n)
                    for e in tok["entries"]],
        "cfg": cfg,
        "T": Tf, "H": H, "W": W,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sample.mp4",
                    help="real clip to slide over")
    ap.add_argument("--out", default="cool_chic/data/inr_dataset.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32-steps", type=int, default=1500)
    ap.add_argument("--qat-steps",  type=int, default=800)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--pixels", type=int, default=8192)
    ap.add_argument("--frames-per-window", type=int, default=8)
    ap.add_argument("--n-windows", type=int, default=4,
                    help="how many overlapping windows to slice from --source")
    ap.add_argument("--include-synthetic", action="store_true", default=True)
    args = ap.parse_args()

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    t0 = time.time()

    # --- real clip windows ---
    print(f"=== reading {args.source} ===")
    full = read_video(args.source)
    Tf_total = full.shape[0]
    fpw = args.frames_per_window
    if Tf_total < fpw:
        raise SystemExit(f"clip too short: {Tf_total} < {fpw} frames")
    stride = max((Tf_total - fpw) // max(args.n_windows - 1, 1), 1)
    starts = [min(i * stride, Tf_total - fpw) for i in range(args.n_windows)]
    cfg_a = VideoConfig(L=4, T=1 << 11, F=2, N_min=8,  N_max=128,
                          mlp_hidden=32, mlp_depth=3)
    cfg_b = VideoConfig(L=5, T=1 << 12, F=2, N_min=16, N_max=128,
                          mlp_hidden=48, mlp_depth=3)
    for i, start in enumerate(starts):
        clip = full[start:start + fpw]
        cfg = cfg_a if i % 2 == 0 else cfg_b
        print(f"\n--- real window {i+1}/{len(starts)} (frames {start}..{start+fpw-1}) "
              f"L={cfg.L} T={cfg.T} ---")
        s = _train_one(clip, cfg, fp32_steps=args.fp32_steps,
                         qat_steps=args.qat_steps, lr=args.lr,
                         pixels=args.pixels, device=args.device)
        s["source"] = f"real_w{i}"
        samples.append(s)

    # --- synthetic clips (different statistical character) ---
    if args.include_synthetic:
        for i, (T, H, W) in enumerate([(8, 64, 64), (8, 96, 96)]):
            clip = _make_test_video(T=T, H=H, W=W)
            print(f"\n--- synthetic {i+1} ({T}x{H}x{W}) ---")
            s = _train_one(clip, cfg_a, fp32_steps=args.fp32_steps,
                             qat_steps=args.qat_steps, lr=args.lr,
                             pixels=args.pixels, device=args.device)
            s["source"] = f"synth_{i}"
            samples.append(s)

    elapsed = time.time() - t0
    total_tokens = sum(s["ints"].numel() for s in samples)
    print(f"\n=== done: {len(samples)} INRs, {total_tokens} total tokens, "
          f"{elapsed:.1f}s ===")
    torch.save(samples, out_path)
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
