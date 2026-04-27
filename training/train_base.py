"""Base-model pretraining on Vimeo-90k (septuplet).

Adaptation disabled (A, B frozen at zero). Produces θ₀ for per-GOP overfitting.

Usage
-----
  # Real training against Vimeo-90k septuplet:
  python training/train_base.py \
      --dataset vimeo --data-root /data/vimeo_septuplet \
      --crop 256 --num-frames 5 --batch 8 --epochs 30 \
      --lambda-rd 0.013 --out checkpoints/base_lam013.pt

  # Fallback to any folder of .mp4 clips:
  python training/train_base.py --dataset folder --data-root /clips/ --batch 2

  # Quick smoke run on synthetic data:
  python training/train_base.py --dataset synthetic --epochs 1 --batch 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python training/train_base.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from models import WANVCAutoencoder, collect_adaptable_layers
from utils.datasets import Vimeo90kSeptuplet, VideoFolderDataset


class _SyntheticDataset(Dataset):
    """Procedurally-generated clips with real learnable structure.

    Each sample is a sum of randomly-coloured "blob" layers + a sinusoidal
    texture, all translated at independent velocities (parallax). This gives:
      * a non-trivial spatial distribution (forces g_a/g_s to learn a real
        texture transform, not an identity),
      * frame-to-frame motion that the optical-flow subnet must estimate,
      * temporal correlation that the temporal prior can exploit.

    Loss/PSNR will improve smoothly over thousands of steps, unlike the
    moving-gradient version which the model memorizes in ~50 steps.
    """

    def __init__(self, n: int = 1024, T: int = 5, crop: int = 256,
                 seed_base: int = 0):
        self.n, self.T, self.crop = n, T, crop
        self.seed_base = seed_base

    def __len__(self): return self.n

    @staticmethod
    def _gauss_blob(crop: int, cx: float, cy: float, sigma: float,
                    color: torch.Tensor) -> torch.Tensor:
        ys, xs = torch.meshgrid(
            torch.arange(crop).float(), torch.arange(crop).float(), indexing="ij"
        )
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        m = torch.exp(-d2 / (2 * sigma * sigma))         # (H, W)
        return color.view(3, 1, 1) * m.unsqueeze(0)

    def __getitem__(self, i):
        # Deterministic per-index so different epochs see the same samples
        # (useful for tracking PSNR; epochs differ only via DataLoader shuffle).
        g = torch.Generator().manual_seed(self.seed_base + i)
        crop = self.crop

        # Layer set: 3-6 colored blobs, each with its own velocity.
        n_blobs = int(torch.randint(3, 7, (1,), generator=g).item())
        blobs = []
        for _ in range(n_blobs):
            cx = float(torch.rand(1, generator=g) * crop)
            cy = float(torch.rand(1, generator=g) * crop)
            sigma = float(torch.rand(1, generator=g) * crop * 0.25 + crop * 0.05)
            color = torch.rand(3, generator=g)
            vx = float((torch.rand(1, generator=g) - 0.5) * 6)   # px/frame
            vy = float((torch.rand(1, generator=g) - 0.5) * 6)
            blobs.append((cx, cy, sigma, color, vx, vy))

        # Sinusoidal texture w/ random orientation for high-frequency content.
        theta = float(torch.rand(1, generator=g) * 2 * 3.14159)
        freq = float(torch.rand(1, generator=g) * 0.10 + 0.02)
        phase_v = float((torch.rand(1, generator=g) - 0.5) * 0.4)

        # Background tint.
        bg = torch.rand(3, generator=g) * 0.3

        ys, xs = torch.meshgrid(
            torch.arange(crop).float(), torch.arange(crop).float(), indexing="ij"
        )

        frames = []
        for t in range(self.T):
            img = bg.view(3, 1, 1).expand(3, crop, crop).clone()
            for cx, cy, sigma, color, vx, vy in blobs:
                img = img + self._gauss_blob(crop, cx + vx * t, cy + vy * t, sigma, color)
            tex = 0.15 * torch.sin(2 * 3.14159 * freq *
                                   (xs * torch.cos(torch.tensor(theta))
                                    + ys * torch.sin(torch.tensor(theta)))
                                   + phase_v * t)
            img = img + tex.unsqueeze(0)
            frames.append(img.clamp(0, 1))
        return torch.stack(frames, dim=0)


def _build_dataset(args) -> Dataset:
    if args.dataset == "vimeo":
        return Vimeo90kSeptuplet(
            root=args.data_root, split="train",
            crop=args.crop, num_frames=args.num_frames,
            augment=True, temporal_stride=args.temporal_stride,
        )
    if args.dataset == "folder":
        return VideoFolderDataset(
            root=args.data_root, crop=args.crop, num_frames=args.num_frames,
        )
    if args.dataset == "synthetic":
        return _SyntheticDataset(n=args.synthetic_size, T=args.num_frames, crop=args.crop)
    raise ValueError(f"unknown dataset {args.dataset}")


def _rate_bpp(likelihoods: dict[str, torch.Tensor], num_pixels: int) -> torch.Tensor:
    total = sum((-torch.log2(l.clamp(min=1e-9)).sum() for l in likelihoods.values()))
    return total / num_pixels


def _aux_loss(model: WANVCAutoencoder) -> torch.Tensor:
    """CompressAI's EntropyBottleneck needs a secondary aux_loss on its
    quantile parameters to stay in a sensible range."""
    total = torch.zeros((), device=next(model.parameters()).device)
    try:
        total = total + model.hyper.entropy_bottleneck.loss()
    except Exception:
        pass
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["vimeo", "folder", "synthetic"], default="synthetic")
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--num-frames", type=int, default=5)
    ap.add_argument("--temporal-stride", type=int, default=1)
    ap.add_argument("--synthetic-size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--aux-lr", type=float, default=1e-3)
    ap.add_argument("--lambda-rd", type=float, default=0.013)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="checkpoints/base.pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=200,
                    help="step interval between intermediate checkpoint writes")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="hard cap on global steps; overrides --epochs when set")
    ap.add_argument("--max-seconds", type=int, default=None,
                    help="hard cap on wall time in seconds; honoured between batches")
    ap.add_argument("--log-jsonl", type=str, default=None,
                    help="if set, append one JSON line per logged step")
    ap.add_argument("--resume", type=str, default=None,
                    help="checkpoint path to resume from (loads model weights only)")
    args = ap.parse_args()

    if args.dataset in {"vimeo", "folder"} and not args.data_root:
        ap.error(f"--data-root required for --dataset {args.dataset}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    model = WANVCAutoencoder().to(args.device)
    # Resume from a prior checkpoint if requested. We load only the model
    # weights (not optimizer state) — Adam re-warming over a few hundred
    # steps is much cheaper than handling state-dict version skew.
    resume_path = args.resume
    if resume_path is None and Path(args.out).exists():
        # Auto-resume from the output path if it already exists.
        resume_path = args.out
    if resume_path and Path(resume_path).exists():
        state = torch.load(resume_path, map_location=args.device)
        # Tolerate both missing/unexpected keys AND shape mismatches. The
        # latter happens if the checkpoint predates an architecture change
        # (e.g. VQ codebook size, activation swap). Keys that don't match
        # current tensor shapes are silently skipped and trained from scratch.
        current = model.state_dict()
        compatible = {k: v for k, v in state["model"].items()
                      if k in current and current[k].shape == v.shape}
        skipped = [k for k in state["model"] if k not in compatible]
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        prior_step = state.get("step", 0)
        print(f"resumed from {resume_path} (prior_step={prior_step}, "
              f"loaded={len(compatible)} skipped_shape_mismatch={len(skipped)} "
              f"missing={len(missing)} unexpected={len(unexpected)})",
              flush=True)
        if skipped:
            print("  skipped keys:", ", ".join(skipped[:6]),
                  "..." if len(skipped) > 6 else "", flush=True)
    else:
        prior_step = 0

    for _, layer in collect_adaptable_layers(model):
        for p in layer.adaptable_parameters():
            p.requires_grad_(False)

    ds = _build_dataset(args)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
                    drop_last=True)

    # CompressAI convention: separate optimizer for aux params.
    aux_params = []
    try:
        aux_params = list(model.hyper.entropy_bottleneck.parameters())
    except Exception:
        aux_params = []
    aux_ids = {id(p) for p in aux_params}
    main_params = [p for p in model.parameters() if p.requires_grad and id(p) not in aux_ids]

    # Larger eps in Adam's denominator keeps the per-param effective LR
    # bounded when v_t collapses; otherwise a single small-gradient param
    # can get a huge update and destabilise the model.
    opt = Adam(main_params, lr=args.lr, eps=1e-6)
    aux_opt = Adam(aux_params, lr=args.aux_lr, eps=1e-6) if aux_params else None

    import copy
    import json
    import math
    import time
    log_fp = open(args.log_jsonl, "a", buffering=1) if args.log_jsonl else None
    t0 = time.time()
    peak_mem_mb = 0
    global_step = prior_step    # continue numbering across resumes
    stop = False

    # Auto-rollback snapshot: kept in-memory, refreshed every SNAP_EVERY
    # steps when the model is healthy. Triggers on either persistent NaN or
    # a loss explosion (D > LOSS_EXPLOSION_D).
    SNAP_EVERY = 100
    NAN_TOLERANCE = 5
    LOSS_EXPLOSION_D = 1.0     # MSE > 1.0 on [0,1] targets is pathological
    MAX_WEIGHT_MAG = 100.0     # covers CompressAI quantile params (~20)
    nan_skip_count = 0
    explosion_count = 0
    last_snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
    last_snap_step = global_step
    rollback_events = 0

    # Track the best (lowest-D) checkpoint separately so we always have a
    # recoverable good model even if the "latest" save gets polluted.
    best_out = str(Path(args.out).with_name(Path(args.out).stem + "_best.pt"))
    best_D = float("inf")
    for epoch in range(args.epochs):
        if stop:
            break
        for clip in dl:
            clip = clip.to(args.device, non_blocking=True)   # (B, T, 3, H, W)
            B, T, C, H, W = clip.shape
            num_pixels_per_frame = H * W

            opt.zero_grad(set_to_none=True)
            if aux_opt is not None:
                aux_opt.zero_grad(set_to_none=True)

            total_D = 0.0
            total_R = 0.0
            for b in range(B):
                x0 = clip[b, 0:1]
                out_i = model.encode_iframe(x0)
                # Compute MSE on raw output — clamping kills gradients on
                # out-of-range pixels and prevents the codec from learning
                # to stay in [0, 1] naturally.
                D = F.mse_loss(out_i["x_hat"], x0)
                R = _rate_bpp(out_i["likelihoods"], num_pixels_per_frame)

                # Motion estimation needs valid pixels, so clamp ONLY when
                # feeding x_prev into the next P-frame.
                x_prev = out_i["x_hat"].clamp(0, 1).detach()
                for t in range(1, T):
                    xt = clip[b, t:t + 1]
                    out_p = model.encode_pframe(x_prev, xt)
                    zq, _ = model.residual.tok.quantize(
                        model.residual.tok.encode(xt - out_p["x_from_latent"])
                    )
                    r_hat = model.residual.tok.dec(zq)
                    x_hat = out_p["x_from_latent"] + r_hat       # raw, no clamp
                    D = D + F.mse_loss(x_hat, xt)
                    R = R + _rate_bpp(out_p["likelihoods"], num_pixels_per_frame)
                    x_prev = x_hat.clamp(0, 1).detach()

                total_D = total_D + D / T
                total_R = total_R + R / T

            main_loss = (total_D + args.lambda_rd * total_R) / B

            # Two unhealthy conditions both trigger rollback:
            #  (a) non-finite loss (NaN/Inf — the original failure mode)
            #  (b) huge-but-finite distortion (model drifted to out-of-range
            #      outputs and is stuck producing them). D > 1.0 is
            #      impossible for healthy training on [0,1] targets.
            _D_value = float(total_D.detach() / B) if hasattr(total_D, "detach") else float(total_D / B)
            unhealthy = (not torch.isfinite(main_loss)) or (_D_value > LOSS_EXPLOSION_D)

            if unhealthy:
                reason = "non-finite" if not torch.isfinite(main_loss) else f"D={_D_value:.2f}>{LOSS_EXPLOSION_D}"
                if not torch.isfinite(main_loss):
                    nan_skip_count += 1
                else:
                    explosion_count += 1
                total_bad = nan_skip_count + explosion_count
                if total_bad <= 3 or total_bad % 50 == 0:
                    print(f"  [warn] unhealthy step {global_step} ({reason}), "
                          f"skipping (bad={total_bad})", flush=True)
                if total_bad >= NAN_TOLERANCE:
                    rollback_events += 1
                    new_lr = opt.param_groups[0]["lr"] * 0.5
                    print(f"  [ROLLBACK #{rollback_events}] restoring snapshot "
                          f"from step {last_snap_step}, lr "
                          f"{opt.param_groups[0]['lr']:.2e} -> {new_lr:.2e}", flush=True)
                    model.load_state_dict(last_snapshot)
                    for g in opt.param_groups: g["lr"] = new_lr
                    for p in main_params:
                        if p in opt.state: opt.state[p] = {}
                    nan_skip_count = 0
                    explosion_count = 0
                global_step += 1
                if args.max_steps and global_step >= args.max_steps: stop = True; break
                if args.max_seconds and (time.time() - t0) >= args.max_seconds:
                    stop = True; break
                continue
            else:
                # Healthy step — periodically refresh the in-memory snapshot
                # but only if weight magnitudes are reasonable.
                nan_skip_count = 0
                explosion_count = 0
                if (global_step - last_snap_step) >= SNAP_EVERY:
                    sd = model.state_dict()
                    max_w = max((float(v.abs().max()) for v in sd.values()
                                 if torch.is_tensor(v) and v.dtype.is_floating_point
                                 and v.numel() > 0), default=0.0)
                    if (all(torch.isfinite(v).all() for v in sd.values() if torch.is_tensor(v))
                            and max_w < MAX_WEIGHT_MAG):
                        last_snapshot = {k: v.detach().clone() for k, v in sd.items()}
                        last_snap_step = global_step

            main_loss.backward()
            torch.nn.utils.clip_grad_norm_(main_params, args.grad_clip)
            opt.step()

            if aux_opt is not None:
                # main_loss.backward() flowed gradients through likelihoods,
                # which depend on the EntropyBottleneck quantile params.
                # Those gradients are *not* what aux_loss wants to optimise;
                # zero them before computing the actual aux_loss gradient.
                aux_opt.zero_grad(set_to_none=True)
                aux_l = _aux_loss(model)
                aux_l.backward()
                torch.nn.utils.clip_grad_norm_(aux_params, args.grad_clip)
                aux_opt.step()

            if args.device == "cuda":
                cur_mem = torch.cuda.max_memory_allocated() // (1024 * 1024)
                if cur_mem > peak_mem_mb:
                    peak_mem_mb = cur_mem

            if global_step % args.log_every == 0:
                with torch.no_grad():
                    D_val = float((total_D / B).detach()) if hasattr(total_D, "detach") else float(total_D / B)
                    R_val = float((total_R / B).detach()) if hasattr(total_R, "detach") else float(total_R / B)
                    L_val = float(main_loss.detach())
                # PSNR over the *training* sample — strictly an indicator, not a true val score.
                psnr = float(-10.0 * math.log10(max(D_val, 1e-12)))
                elapsed = time.time() - t0
                print(f"[ep {epoch} step {global_step:5d} t={elapsed:7.0f}s "
                      f"vram={peak_mem_mb:5d}MB] "
                      f"L={L_val:.4f}  D={D_val:.4f}  "
                      f"PSNR={psnr:5.2f}  R={R_val:.4f} bpp", flush=True)
                if log_fp:
                    log_fp.write(json.dumps({
                        "step": global_step, "epoch": epoch, "wall_s": elapsed,
                        "loss": L_val, "D": D_val, "R_bpp": R_val,
                        "psnr": psnr, "vram_mb": peak_mem_mb,
                    }) + "\n")

            if args.ckpt_every and global_step > 0 and global_step % args.ckpt_every == 0:
                # Save the *latest* checkpoint only if weights are sane:
                # finite AND reasonable magnitude. A huge-but-finite weight
                # is a sign the model drifted into a bad equilibrium.
                sd = model.state_dict()
                max_w = max((float(v.abs().max()) for v in sd.values()
                             if torch.is_tensor(v) and v.dtype.is_floating_point
                             and v.numel() > 0), default=0.0)
                all_finite = all(torch.isfinite(v).all() for v in sd.values()
                                 if torch.is_tensor(v))
                if all_finite and max_w < MAX_WEIGHT_MAG:
                    torch.save({"model": sd, "args": vars(args), "step": global_step}, args.out)
                    # Best checkpoint: only updated when this step's D set a new min.
                    if _D_value < best_D:
                        best_D = _D_value
                        torch.save({"model": sd, "args": vars(args),
                                    "step": global_step, "best_D": best_D}, best_out)
                else:
                    print(f"  [warn] skipping checkpoint at step {global_step} "
                          f"(max_w={max_w:.1f}, all_finite={all_finite})", flush=True)

            global_step += 1
            if args.max_steps and global_step >= args.max_steps:
                stop = True; break
            if args.max_seconds and (time.time() - t0) >= args.max_seconds:
                print(f"reached --max-seconds {args.max_seconds}, stopping")
                stop = True; break

        sd = model.state_dict()
        max_w = max((float(v.abs().max()) for v in sd.values()
                     if torch.is_tensor(v) and v.dtype.is_floating_point
                     and v.numel() > 0), default=0.0)
        all_finite = all(torch.isfinite(v).all() for v in sd.values() if torch.is_tensor(v))
        if all_finite and max_w < MAX_WEIGHT_MAG:
            torch.save({"model": sd, "args": vars(args), "epoch": epoch,
                        "step": global_step}, args.out)
            print(f"saved {args.out} (epoch {epoch}, step {global_step}, "
                  f"best_D={best_D:.5f})")
        else:
            print(f"  [warn] skipping epoch-save (max_w={max_w:.1f}, "
                  f"all_finite={all_finite})", flush=True)
    if log_fp:
        log_fp.close()


if __name__ == "__main__":
    main()
