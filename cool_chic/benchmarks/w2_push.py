"""Week-2 push: longer QAT + lambda warmup + high-end lambda sweep.

Held-out sample.mp4 evaluation using the v2 priors trained on the
22-clip diverse corpus.
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
from cool_chic.train_video_qat import (
    overfit_video_qat, VideoQATConfig, reconstruct_quantized_video,
)
from cool_chic.bitstream import encode_codec, KIND_VIDEO
from cool_chic.bitstream_v2 import load_prior
from utils.video_io import read_video


def _psnr(x, y):
    mse = F.mse_loss(x.clamp(0, 1), y.clamp(0, 1)).clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse))


def _fp32_warm(video, model, *, steps, lr, pixels, device):
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


def _run(warm_state, video, cfg, *, qat_steps, lr, pixels, device,
          lam, prior, lambda_warmup_frac):
    model = VideoINR(T_frames=video.shape[0], H=video.shape[-2], W=video.shape[-1],
                      cfg=cfg).to(device)
    model.load_state_dict(warm_state)
    quants, _ = overfit_video_qat(video, model,
                                    qcfg=VideoQATConfig(steps=qat_steps,
                                                        lr=lr * 0.4,
                                                        scale_lr=5e-3,
                                                        lambda_rate=lam,
                                                        pixels_per_step=pixels,
                                                        log_every=max(qat_steps // 4, 1),
                                                        init_scale=5e-3,
                                                        lambda_warmup_frac=lambda_warmup_frac),
                                    device=device, verbose=True,
                                    prior=prior)
    recon = reconstruct_quantized_video(model, quants).clamp(0, 1)
    psnr = _psnr(recon.cpu(), video.cpu())
    Tf, _, H, W = video.shape
    blob = encode_codec(quants, kind=KIND_VIDEO, H=H, W=W, T_frames=Tf, cfg=cfg)
    bpp = len(blob) * 8 / (Tf * H * W)
    del model, quants, recon
    if device == "cuda": torch.cuda.empty_cache()
    return psnr, len(blob), bpp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=str)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32-steps", type=int, default=2500)
    ap.add_argument("--qat-steps",  type=int, default=3000)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--pixels", type=int, default=16384)
    ap.add_argument("--warmup-frac", type=float, default=0.25)
    ap.add_argument("--prior", type=str, default="cool_chic/data/prior_v2_nocap.pt")
    ap.add_argument("--lambdas", type=str,
                    default="0.0,1e-3,3e-3,1e-2,3e-2,1e-1")
    args = ap.parse_args()

    video = read_video(args.source, max_frames=args.frames)
    Tf, _, H, W = video.shape
    total_px = Tf * H * W
    print(f"source: {Tf}x{W}x{H}  ({total_px} px)")

    cfg = VideoConfig(L=5, T=1 << 12, F=2, N_min=16, N_max=256,
                       mlp_hidden=48, mlp_depth=3)

    base = VideoINR(T_frames=Tf, H=H, W=W, cfg=cfg).to(args.device)
    print(f"INR params: {base.total_params}")
    t0 = time.time()
    _fp32_warm(video, base, steps=args.fp32_steps, lr=args.lr,
                pixels=args.pixels, device=args.device)
    warm_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    del base
    if args.device == "cuda": torch.cuda.empty_cache()
    print(f"fp32 warm-up: {time.time()-t0:.1f}s")

    prior = load_prior(args.prior, device=args.device)

    lambdas = [float(x.strip()) for x in args.lambdas.split(",")]
    results = []
    for lam in lambdas:
        label = f"lam={lam:.0e}"
        print(f"\n=== {label}  (warmup_frac={args.warmup_frac}) ===")
        psnr, bytes_, bpp = _run(warm_state, video, cfg,
                                    qat_steps=args.qat_steps, lr=args.lr,
                                    pixels=args.pixels, device=args.device,
                                    lam=lam, prior=prior,
                                    lambda_warmup_frac=args.warmup_frac)
        results.append((label, psnr, bytes_, bpp))
        print(f"-> {bytes_} B  bpp={bpp:.4f}  PSNR={psnr:.2f}")

    print("\n=== SUMMARY ===")
    # use the lowest-lambda leg as reference baseline (equivalent to W1 at lam=0)
    ref_bytes = results[0][2]; ref_psnr = results[0][1]
    print(f"{'leg':12s}  {'bytes':>8s}  {'bpp':>7s}  {'PSNR':>6s}  {'dBytes':>8s}  {'dPSNR':>6s}")
    for leg, psnr, bytes_, bpp in results:
        db = (bytes_ - ref_bytes) / ref_bytes * 100.0
        dp = psnr - ref_psnr
        print(f"{leg:12s}  {bytes_:>8d}  {bpp:>7.4f}  {psnr:>6.2f}  "
              f"{db:>+7.1f}%  {dp:>+6.2f}")

    out = Path("cool_chic/benchmarks/out/w2_push.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        [{"leg": r[0], "psnr": r[1], "bytes": r[2], "bpp": r[3]} for r in results],
        indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
