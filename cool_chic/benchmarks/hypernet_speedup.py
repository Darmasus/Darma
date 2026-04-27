"""Benchmark encode-time speedup from hypernet initialization.

For each init strategy (random, hypernet), measure PSNR on the held-out
clip at a ladder of fp32 step counts. If hypernet gives a higher PSNR
at step=0 or converges faster, it's useful for fast encoding. If not,
we've confirmed the per-position values aren't predictable and the
"hypernet" concept reduces to per-tensor sigma (which WeightPrior
already captures).
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
from cool_chic.hypernet import WeightHypernet, apply_hypernet_init
from cool_chic.caption import encode_caption
from utils.video_io import read_video


def _psnr_batch(pred, target):
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1)).clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse))


def load_hypernet(path: str, device: str) -> WeightHypernet:
    blob = torch.load(path, weights_only=False, map_location=device)
    h = WeightHypernet(hidden=blob["hidden"], depth=blob["depth"],
                         caption_dim=blob.get("caption_dim", 0),
                         value_scale=blob.get("value_scale", 0.05)).to(device)
    h.load_state_dict(blob["state_dict"])
    h.eval()
    return h


def _init_random(video, cfg, device):
    Tf, _, H, W = video.shape
    return VideoINR(T_frames=Tf, H=H, W=W, cfg=cfg).to(device)


def _init_hypernet(video, cfg, device, hypernet, caption):
    model = _init_random(video, cfg, device)
    caption_emb = encode_caption(caption, device=device) if caption else None
    n = apply_hypernet_init(model, hypernet, caption_emb=caption_emb)
    print(f"    hypernet initialized {n} weights")
    return model


def _overfit(model, video, *, steps, lr, pixels, device):
    target_flat = video.to(device).permute(0, 2, 3, 1).reshape(-1, 3)
    N = target_flat.shape[0]
    T, _, H, W = video.shape
    t_d, h_d, w_d = max(T-1, 1), max(H-1, 1), max(W-1, 1)
    opt = Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        idx = torch.randint(0, N, (pixels,), device=device)
        ti = idx // (H*W); rem = idx % (H*W); yi = rem // W; xi = rem % W
        coords = torch.stack([xi.float()/w_d, yi.float()/h_d, ti.float()/t_d], -1)
        pred = model.mlp(model.grid(coords))
        loss = F.mse_loss(pred, target_flat[idx])
        loss.backward(); opt.step()


def _reconstruct_psnr(model, video):
    recon = model.reconstruct().clamp(0, 1)
    return _psnr_batch(recon.cpu(), video.cpu())


def _run(video, cfg, *, init_fn, step_ladder, lr, pixels, device, label):
    t0 = time.time()
    model = init_fn()
    psnr0 = _reconstruct_psnr(model, video)
    print(f"  [{label}] step=0  PSNR={psnr0:.2f}  ({time.time()-t0:.1f}s)")
    results = [(0, psnr0, time.time() - t0)]
    prev = 0
    for s in step_ladder:
        delta = s - prev
        _overfit(model, video, steps=delta, lr=lr, pixels=pixels, device=device)
        prev = s
        psnr = _reconstruct_psnr(model, video)
        el = time.time() - t0
        print(f"  [{label}] step={s:5d}  PSNR={psnr:.2f}  ({el:.1f}s)")
        results.append((s, psnr, el))
    del model
    if device == "cuda": torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=str)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--caption", type=str,
                    default="color bars with timecode, rainbow diagonal, green checker pattern, ffmpeg test signal")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--pixels", type=int, default=16384)
    ap.add_argument("--hypernet", default="cool_chic/data/hypernet.pt")
    ap.add_argument("--steps", type=str, default="100,300,700,1500,3000")
    args = ap.parse_args()

    video = read_video(args.source, max_frames=args.frames)
    Tf, _, H, W = video.shape
    print(f"source: {Tf}x{W}x{H}")

    # Match the capacity the hypernet was trained on (v2 dataset).
    cfg = VideoConfig(L=4, T=1 << 11, F=2, N_min=8, N_max=96,
                       mlp_hidden=32, mlp_depth=3)
    hyp = load_hypernet(args.hypernet, device=args.device)
    step_ladder = [int(s) for s in args.steps.split(",")]

    print("\n=== random init ===")
    torch.manual_seed(0)
    rnd_results = _run(video, cfg,
                        init_fn=lambda: _init_random(video, cfg, args.device),
                        step_ladder=step_ladder, lr=args.lr, pixels=args.pixels,
                        device=args.device, label="rnd")

    print("\n=== hypernet init (captioned) ===")
    torch.manual_seed(0)
    hyp_results = _run(video, cfg,
                        init_fn=lambda: _init_hypernet(video, cfg, args.device,
                                                          hyp, args.caption),
                        step_ladder=step_ladder, lr=args.lr, pixels=args.pixels,
                        device=args.device, label="hyp")

    print("\n=== SUMMARY ===")
    print(f"{'steps':>5s}  {'PSNR_rnd':>8s}  {'time_rnd':>8s}  "
          f"{'PSNR_hyp':>8s}  {'time_hyp':>8s}  {'dPSNR':>6s}")
    for (s_r, p_r, t_r), (s_h, p_h, t_h) in zip(rnd_results, hyp_results):
        assert s_r == s_h
        dp = p_h - p_r
        print(f"{s_r:>5d}  {p_r:>8.2f}  {t_r:>7.1f}s  "
              f"{p_h:>8.2f}  {t_h:>7.1f}s  {dp:>+6.2f}")


if __name__ == "__main__":
    main()
