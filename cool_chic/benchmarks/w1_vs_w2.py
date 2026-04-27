"""Head-to-head: Week-1 (per-tensor Laplace, no rate term) vs
Week-2 v2 (prior-aware QAT + v2 bitstream).

Same clip, same fp32 warm-up, same capacity. Only the QAT rate term
and the bitstream encoder differ.
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
from cool_chic.train_video_qat import (
    overfit_video_qat, VideoQATConfig, reconstruct_quantized_video,
)
from cool_chic.bitstream   import encode_codec, KIND_VIDEO
from cool_chic.bitstream_v2 import encode_codec_v2, load_prior
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
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        idx = torch.randint(0, N, (pixels,), device=device)
        ti = idx // (H*W); rem = idx % (H*W); yi = rem // W; xi = rem % W
        coords = torch.stack([xi.float()/w_d, yi.float()/h_d, ti.float()/t_d], -1)
        pred = model.mlp(model.grid(coords))
        loss = F.mse_loss(pred, target_flat[idx])
        loss.backward(); opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=str)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32-steps", type=int, default=2000)
    ap.add_argument("--qat-steps",  type=int, default=1500)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--pixels", type=int, default=16384)
    # R-D lambdas. For the prior-aware path we sweep multiple.
    ap.add_argument("--lambdas", type=str, default="0.0,1e-5,5e-5,2e-4",
                    help="comma-separated lambda_rate values for W2")
    args = ap.parse_args()

    video = read_video(args.source, max_frames=args.frames)
    Tf, _, H, W = video.shape
    total_px = Tf * H * W
    print(f"source: {Tf}x{W}x{H}  ({total_px} px)")

    cfg = VideoConfig(L=5, T=1 << 12, F=2, N_min=16, N_max=256,
                       mlp_hidden=48, mlp_depth=3)

    # ----- fp32 warm-up once, cloned for both legs -----
    base_model = VideoINR(T_frames=Tf, H=H, W=W, cfg=cfg).to(args.device)
    print(f"params: {base_model.total_params}")
    t0 = time.time()
    _fp32_warm(video, base_model, steps=args.fp32_steps, lr=args.lr,
                pixels=args.pixels, device=args.device)
    warm_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
    print(f"fp32 warm-up done in {time.time()-t0:.1f}s")

    results = []

    # ============ Week-1 leg (no prior, no rate) ============
    print("\n=== Week-1 (per-tensor Laplace, lambda=0) ===")
    model_w1 = VideoINR(T_frames=Tf, H=H, W=W, cfg=cfg).to(args.device)
    model_w1.load_state_dict(warm_state)
    quants_w1, _ = overfit_video_qat(video, model_w1,
                                        qcfg=VideoQATConfig(steps=args.qat_steps,
                                                            lr=args.lr * 0.4,
                                                            scale_lr=5e-3,
                                                            lambda_rate=0.0,
                                                            pixels_per_step=args.pixels,
                                                            log_every=max(args.qat_steps // 4, 1),
                                                            init_scale=5e-3),
                                        device=args.device, verbose=True)
    rec_w1 = reconstruct_quantized_video(model_w1, quants_w1).clamp(0, 1)
    psnr_w1 = _psnr(rec_w1.cpu(), video.cpu())
    blob_w1 = encode_codec(quants_w1, kind=KIND_VIDEO, H=H, W=W, T_frames=Tf, cfg=cfg)
    bpp_w1 = len(blob_w1) * 8 / total_px
    print(f"  -> {len(blob_w1)} B  bpp={bpp_w1:.4f}  PSNR={psnr_w1:.2f}")
    results.append(("W1", "none", psnr_w1, bpp_w1, len(blob_w1)))

    # ============ Week-2 legs (prior-aware QAT, sweep lambda) ============
    prior = load_prior("cool_chic/data/prior.pt", device=args.device)
    # Free the Week-1 model before spinning up parallel legs.
    del model_w1, quants_w1, rec_w1
    if args.device == "cuda":
        torch.cuda.empty_cache()
    for lam_str in args.lambdas.split(","):
        lam = float(lam_str.strip())
        print(f"\n=== Week-2 prior-aware QAT  lambda={lam:.1e} ===")
        model_w2 = VideoINR(T_frames=Tf, H=H, W=W, cfg=cfg).to(args.device)
        model_w2.load_state_dict(warm_state)
        quants_w2, _ = overfit_video_qat(video, model_w2,
                                            qcfg=VideoQATConfig(steps=args.qat_steps,
                                                                lr=args.lr * 0.4,
                                                                scale_lr=5e-3,
                                                                lambda_rate=lam,
                                                                pixels_per_step=args.pixels,
                                                                log_every=max(args.qat_steps // 4, 1),
                                                                init_scale=5e-3),
                                            device=args.device, verbose=True,
                                            prior=prior)
        rec_w2 = reconstruct_quantized_video(model_w2, quants_w2).clamp(0, 1)
        psnr_w2 = _psnr(rec_w2.cpu(), video.cpu())
        # Both bitstream variants so we can compare apples-to-apples
        blob_v1 = encode_codec   (quants_w2, kind=KIND_VIDEO, H=H, W=W, T_frames=Tf, cfg=cfg)
        blob_v2 = encode_codec_v2(quants_w2, kind=KIND_VIDEO, H=H, W=W, T_frames=Tf,
                                    cfg=cfg, prior=prior)
        print(f"  v1 bitstream: {len(blob_v1)} B  bpp={len(blob_v1)*8/total_px:.4f}")
        print(f"  v2 bitstream: {len(blob_v2)} B  bpp={len(blob_v2)*8/total_px:.4f}  "
              f"PSNR={psnr_w2:.2f}")
        results.append((f"W2 v2 lam={lam:.0e}", "prior", psnr_w2,
                         len(blob_v2) * 8 / total_px, len(blob_v2)))
        results.append((f"W2 v1 lam={lam:.0e}", "prior_train+v1_code",
                         psnr_w2, len(blob_v1) * 8 / total_px, len(blob_v1)))
        # Free model/quants before next leg so the GPU doesn't OOM.
        del model_w2, quants_w2, rec_w2
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print("\n=== SUMMARY ===")
    print(f"{'leg':32s} {'bytes':>8s} {'bpp':>7s} {'PSNR':>6s}  {'vs W1':>8s}")
    w1_bytes = results[0][4]
    for label, _, psnr, bpp, bytes_ in results:
        delta = (bytes_ - w1_bytes) / w1_bytes * 100.0
        print(f"{label:32s} {bytes_:>8d} {bpp:>7.4f} {psnr:>6.2f}  "
              f"{delta:>+7.1f}%")

    # persist numbers for later reference
    out = Path("cool_chic/benchmarks/out/w1_vs_w2.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    out.write_text(json.dumps(
        [{"leg": r[0], "mode": r[1], "psnr": r[2], "bpp": r[3],
          "bytes": r[4]} for r in results], indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
