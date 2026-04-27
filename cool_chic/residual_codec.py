"""Hybrid codec: x265 baseline + INR residual.

Workflow:
    encoder(video, caption)
        1. Encode video with libx265 at given CRF -> baseline.mp4 (few KB)
        2. Decode baseline to x265_rec tensor
        3. Compute residual = video - x265_rec (small magnitude, mean~0)
        4. Train an INR on (coords -> residual), with rate-aware QAT
           against the WeightPrior + optional L1 sparsity
        5. Serialize: [header | x265.mp4 bytes | INR bytes]

    decoder(blob)
        1. Split blob into x265 and INR parts
        2. Decode x265 -> x265_rec tensor
        3. Decode INR -> residual_hat tensor
        4. output = clip(x265_rec + residual_hat, 0, 1)

The INR doesn't have to represent the whole frame content — only the
mistakes x265 makes (block boundaries, fine textures, quantization
noise). This is what AV1/HEVC already exploit via motion comp; we're
recreating that advantage architecturally.
"""
from __future__ import annotations

import io
import pickle
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam

from .codec import VideoINR, VideoConfig
from .train_video_qat import overfit_video_qat, VideoQATConfig, reconstruct_quantized_video
from .bitstream import encode_codec, decode_codec, KIND_VIDEO, load_weights_into_model
from .serialize import tokenize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.video_io import read_video, write_video


MAGIC_HYBRID = 0x48594231   # 'HYB1'


def _x265_encode(video: torch.Tensor, crf: int, tmp_dir: Path,
                   fps: float = 8.0) -> bytes:
    """Write `video` (T,3,H,W float [0,1]) to a temp mp4 via x265, return bytes."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    raw = tmp_dir / "src.mp4"
    out = tmp_dir / "x265.mp4"
    # First write raw as lossless mp4 via libx264 -qp 0 (lossless x264)
    write_video(str(raw), video, fps=fps, crf=0)
    # Then x265 at target CRF
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(raw),
            "-c:v", "libx265", "-preset", "medium",
            "-crf", str(crf), "-pix_fmt", "yuv420p", "-an", str(out)]
    subprocess.run(cmd, check=True)
    blob = out.read_bytes()
    return blob


def _x265_decode(blob: bytes, tmp_dir: Path, T: int) -> torch.Tensor:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / "x265_dec.mp4"
    out.write_bytes(blob)
    rec = read_video(str(out), max_frames=T)
    return rec


@dataclass
class HybridConfig:
    x265_crf: int = 32
    fp32_steps: int = 1500
    qat_steps:  int = 2000
    lr: float = 5e-3
    pixels_per_step: int = 16384
    lambda_rate: float = 1e-2
    l1_lambda: float = 0.0
    lambda_warmup_frac: float = 0.25
    init_scale: float = 2e-3     # residuals are small, tighter init


def _fp32_warm_residual(residual, model, *, steps, lr, pixels, device):
    """Overfit the INR to the pre-computed residual (not the raw frames)."""
    target_flat = residual.to(device).permute(0, 2, 3, 1).reshape(-1, 3)
    N = target_flat.shape[0]
    T, _, H, W = residual.shape
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


def encode_hybrid(video: torch.Tensor, *, cfg: VideoConfig, hcfg: HybridConfig,
                    prior=None, caption_emb=None, device: str = "cpu",
                    verbose: bool = True) -> dict:
    """Returns dict with bytes, psnr, bpp stats so callers can benchmark.

    Shipping format produced by `serialize_hybrid` below.
    """
    video = video.to(device)
    Tf, _, H, W = video.shape

    with tempfile.TemporaryDirectory() as tdir:
        tmp = Path(tdir)
        t_x265 = time.time()
        x265_bytes = _x265_encode(video.cpu(), hcfg.x265_crf, tmp / "enc")
        x265_rec = _x265_decode(x265_bytes, tmp / "dec", Tf).to(device)
    x265_bpp = len(x265_bytes) * 8 / (Tf * H * W)
    x265_psnr = float(10.0 * torch.log10(1.0 / F.mse_loss(
        x265_rec.clamp(0, 1), video.clamp(0, 1)).clamp_min(1e-12)))
    if verbose:
        print(f"  x265 baseline @ CRF {hcfg.x265_crf}: {len(x265_bytes)} B  "
              f"bpp={x265_bpp:.4f}  PSNR={x265_psnr:.2f}  ({time.time()-t_x265:.1f}s)")

    residual = (video - x265_rec).clamp(-1, 1)
    res_std = float(residual.std())
    if verbose:
        print(f"  residual: std={res_std:.4f}  range=[{residual.min():.3f}, {residual.max():.3f}]")

    model = VideoINR(T_frames=Tf, H=H, W=W, cfg=cfg).to(device)
    t_fp32 = time.time()
    _fp32_warm_residual(residual, model, steps=hcfg.fp32_steps,
                         lr=hcfg.lr, pixels=hcfg.pixels_per_step, device=device)
    if verbose:
        print(f"  INR fp32 warm-up: {time.time()-t_fp32:.1f}s")

    t_qat = time.time()
    quants, _ = overfit_video_qat(residual, model,
                                    qcfg=VideoQATConfig(steps=hcfg.qat_steps,
                                                        lr=hcfg.lr * 0.4,
                                                        scale_lr=5e-3,
                                                        lambda_rate=hcfg.lambda_rate,
                                                        pixels_per_step=hcfg.pixels_per_step,
                                                        log_every=max(hcfg.qat_steps, 1),
                                                        init_scale=hcfg.init_scale,
                                                        lambda_warmup_frac=hcfg.lambda_warmup_frac,
                                                        l1_lambda=hcfg.l1_lambda),
                                    device=device, verbose=False,
                                    prior=prior, caption_emb=caption_emb)
    if verbose:
        print(f"  INR QAT: {time.time()-t_qat:.1f}s")

    # Serialize INR part as a regular cool_chic codec (per-tensor Laplace).
    inr_bytes = encode_codec(quants, kind=KIND_VIDEO,
                              H=H, W=W, T_frames=Tf, cfg=cfg)

    # Evaluate combined reconstruction
    residual_hat = reconstruct_quantized_video(model, quants)
    output = (x265_rec + residual_hat).clamp(0, 1)
    total_bytes = len(x265_bytes) + len(inr_bytes)
    psnr = float(10.0 * torch.log10(1.0 / F.mse_loss(
        output, video.clamp(0, 1)).clamp_min(1e-12)))
    bpp = total_bytes * 8 / (Tf * H * W)

    # Sparsity stats for introspection
    total_w = sum(q.param.numel() for q in quants.values())
    n_zero = sum((q.integer_codes() == 0).sum().item() for q in quants.values())
    if verbose:
        print(f"  hybrid: x265 {len(x265_bytes)} B + INR {len(inr_bytes)} B = "
              f"{total_bytes} B  bpp={bpp:.4f}  PSNR={psnr:.2f}  "
              f"(residual INR PSNR alone={float(-10.0 * torch.log10(F.mse_loss(residual_hat, residual).clamp_min(1e-12))):.2f})")
        print(f"  INR weight zero fraction: {n_zero / total_w:.2%}")

    return {
        "x265_bytes": x265_bytes, "inr_bytes": inr_bytes,
        "bytes": total_bytes, "bpp": bpp, "psnr": psnr,
        "x265_bpp": x265_bpp, "x265_psnr": x265_psnr,
        "residual_std": res_std,
        "zero_fraction": n_zero / total_w,
    }


def serialize_hybrid(x265_bytes: bytes, inr_bytes: bytes, *, H: int, W: int,
                       T: int) -> bytes:
    out = io.BytesIO()
    out.write(struct.pack("<IHHHII", MAGIC_HYBRID, H, W, T,
                            len(x265_bytes), len(inr_bytes)))
    out.write(x265_bytes)
    out.write(inr_bytes)
    return out.getvalue()


def deserialize_hybrid(data: bytes) -> dict:
    buf = io.BytesIO(data)
    (magic, H, W, T, nx, ni) = struct.unpack("<IHHHII", buf.read(18))
    assert magic == MAGIC_HYBRID, f"bad magic 0x{magic:08x}"
    return {"H": H, "W": W, "T": T,
             "x265_bytes": buf.read(nx), "inr_bytes": buf.read(ni)}
