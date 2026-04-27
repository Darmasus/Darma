"""Decode a Cool-Chic .cc video bitstream back to an (T, 3, H, W) tensor
and, optionally, a near-lossless mp4 proxy.

The .cc blob carries H/W/T_frames/cfg and the quantized weights. We
instantiate a fresh VideoINR, load weights, and run reconstruct().
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from cool_chic.codec import VideoINR
from cool_chic.bitstream import decode_codec, load_weights_into_model
from utils.video_io import write_video


def decode_to_tensor(cc_path: str, device: str = "cpu") -> tuple[torch.Tensor, dict]:
    blob = Path(cc_path).read_bytes()
    decoded = decode_codec(blob)
    cfg = decoded["cfg"]
    model = VideoINR(T_frames=decoded["T_frames"], H=decoded["H"], W=decoded["W"],
                     cfg=cfg).to(device)
    load_weights_into_model(model, decoded["weights"])
    recon = model.reconstruct().clamp(0, 1).cpu()
    return recon, decoded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=str, help=".cc bitstream")
    ap.add_argument("output", type=str, help="output mp4")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    recon, meta = decode_to_tensor(args.input, device=args.device)
    T, _, H, W = recon.shape
    print(f"decoded: {T} frames {W}x{H}")
    write_video(args.output, recon, fps=args.fps)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
