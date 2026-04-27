"""Compile the decoder hot path for ≥30 fps @ 1080p.

Strategy:
  1. Fold A, B into base conv weights once per GOP (no runtime delta math).
  2. torch.compile(mode='reduce-overhead') the forward pass.
  3. Optionally export to ONNX -> TensorRT for true C++ inference.

The fold step turns AdaptableConv2d into a plain nn.Conv2d, eliminating all
per-forward overhead of the low-rank multiply. After the PUP has been applied
and folded, the model is a standard convolutional graph that TensorRT can
fully fuse.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from models import WANVCAutoencoder
from models.adaptation import AdaptableConv2d


def fold_adaptation_into_base(model: nn.Module) -> None:
    """Rewrite every AdaptableConv2d so that base.weight <- W + (alpha/r) B A.

    Safe to call any time after `apply_pup`. After folding, A and B are set
    to zero so a second fold would be a no-op.
    """
    for m in model.modules():
        if not isinstance(m, AdaptableConv2d):
            continue
        if not m._adapted:
            continue
        A_q, B_q, _ = m.quantized_AB()
        delta = (m.cfg.alpha / m.cfg.rank) * (B_q @ A_q)
        delta = delta.view_as(m.base.weight)
        with torch.no_grad():
            m.base.weight.add_(delta)
            m.A.zero_()
            m.B.zero_()
        m._adapted = False


def trace_synthesis(model: WANVCAutoencoder, H: int = 1080, W: int = 1920,
                    device: str = "cuda") -> torch.jit.ScriptModule:
    """Trace only g_s (the hot path on the decoder side)."""
    model = model.to(device).eval()
    y = torch.zeros(1, model.M, H // 16, W // 16, device=device)
    with torch.inference_mode():
        traced = torch.jit.trace(model.g_s, y, check_trace=False)
    traced = torch.jit.freeze(traced)
    return traced


def export_onnx(model: WANVCAutoencoder, out_path: str,
                H: int = 1080, W: int = 1920, device: str = "cuda") -> None:
    """Export g_s to ONNX. Consume with `trtexec --onnx=out_path --fp16`."""
    y = torch.zeros(1, model.M, H // 16, W // 16, device=device)
    torch.onnx.export(
        model.g_s.to(device).eval(), y, out_path,
        input_names=["y"], output_names=["x_hat"],
        dynamic_axes={"y": {2: "H16", 3: "W16"}, "x_hat": {2: "H", 3: "W"}},
        opset_version=17,
    )


def benchmark_fps(traced, H: int = 1080, W: int = 1920, device: str = "cuda", iters: int = 50):
    """Rough fps measurement — warm-up + CUDA event timing."""
    y = torch.zeros(1, 320, H // 16, W // 16, device=device)
    for _ in range(10):
        traced(y)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        traced(y)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    print(f"synthesis @ {H}x{W}: {ms:.2f} ms -> {1000/ms:.1f} fps")
    return 1000.0 / ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/base.pt")
    ap.add_argument("--out", type=str, default="checkpoints/g_s.ts")
    ap.add_argument("--onnx", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = WANVCAutoencoder().to(args.device).eval()
    import pathlib
    if pathlib.Path(args.ckpt).exists():
        state = torch.load(args.ckpt, map_location=args.device)
        model.load_state_dict(state["model"], strict=False)

    fold_adaptation_into_base(model)
    traced = trace_synthesis(model, device=args.device)
    traced.save(args.out)
    print(f"saved {args.out}")

    if args.device == "cuda":
        benchmark_fps(traced)

    if args.onnx:
        export_onnx(model, args.onnx, device=args.device)
        print(f"saved {args.onnx}")


if __name__ == "__main__":
    main()
