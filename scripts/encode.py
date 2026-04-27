"""Encode a video file to a WANVC bitstream.

Pipeline:
  1. Load video -> (T, 3, H, W)
  2. Shot-boundary-aware GOP segmentation
  3. For each GOP:
       a. adapt_to_gop() -> populate A, B in AdaptableConv2d layers
       b. encode_pup(adaptable_layers) -> PUP bytes
       c. I-frame: hyperprior.compress(y) -> real rANS bytes
       d. P-frames: flow rANS + dy hyperprior rANS + packed VQ indices/keep mask

Container layout (little-endian):

  magic                u32     'WAN1'
  width, height        u16, u16
  fps_num, fps_den     u16, u16
  n_gops               u32
  for each GOP:
    gop_start, gop_end u32, u32
    pup_len            u32
    pup_bytes          bytes
    gop_payload_len    u32
    gop_payload        pickled dict (I-frame streams + per-P-frame packet)
"""
from __future__ import annotations

import argparse
import io
import pickle
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models import WANVCAutoencoder, collect_adaptable_layers
from training.adapt_gop import adapt_to_gop, AdaptConfig
from entropy_coder import encode_pup
from utils.video_io import read_video
from utils.gop import segment_gops


CONTAINER_MAGIC = 0x57414E31  # 'WAN1'


def _encode_gop(model: WANVCAutoencoder, gop_frames: torch.Tensor) -> bytes:
    """Run the model on a GOP and serialize every stream into one pickle blob.

    The pickle contains:
        iframe: dict from hyperprior.compress
        pframes: list of dicts from autoencoder.compress_pframe
        hw: (H, W) so decoder can size y_hat correctly
    """
    device = gop_frames.device
    T = gop_frames.shape[0]
    H, W = gop_frames.shape[-2:]

    x0 = gop_frames[0:1]
    i_streams = model.compress_iframe(x0)
    x_hat_prev = model.decompress_iframe(i_streams, (H, W))    # decoder-side recon

    p_packets = []
    for t in range(1, T):
        xt = gop_frames[t:t + 1]
        pkt = model.compress_pframe(x_hat_prev, xt)
        p_packets.append(pkt)
        x_hat_prev = model.decompress_pframe(x_hat_prev, pkt)

    blob = {"iframe": i_streams, "pframes": p_packets, "hw": (H, W)}
    return pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=str, help="input video path")
    ap.add_argument("output", type=str, help="output .wanvc bitstream")
    ap.add_argument("--ckpt", type=str, default="checkpoints/base.pt")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--adapt-steps", type=int, default=80)
    ap.add_argument("--lambda-rd", type=float, default=0.013)
    ap.add_argument("--no-adapt", action="store_true",
                    help="skip per-GOP adaptation (baseline mode)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    video = read_video(args.input, max_frames=args.max_frames)
    T, _, H, W = video.shape
    print(f"loaded {T} frames @ {W}x{H}")

    model = WANVCAutoencoder().to(args.device)
    if Path(args.ckpt).exists():
        state = torch.load(args.ckpt, map_location=args.device)
        model.load_state_dict(state["model"], strict=False)
    model.eval()
    model.update_entropy_tables(force=True)

    gops = segment_gops(video)
    print(f"segmented into {len(gops)} GOPs")

    out_buf = io.BytesIO()
    out_buf.write(struct.pack("<IHHHHI", CONTAINER_MAGIC, W, H, 30, 1, len(gops)))

    total_pup = 0
    total_payload = 0
    for gi, gop in enumerate(gops):
        frames = video[gop.start:gop.end].to(args.device)
        if args.no_adapt:
            # Just clear any stale A/B and carry on.
            from models import collect_adaptable_layers as _cal
            for _, layer in _cal(model):
                layer._adapted = False
            pup_bytes = b""
        else:
            adapt_to_gop(model, frames,
                         AdaptConfig(steps=args.adapt_steps,
                                     lambda_rd=args.lambda_rd, device=args.device))
            pup_bytes = encode_pup(collect_adaptable_layers(model))

        payload = _encode_gop(model, frames)

        out_buf.write(struct.pack("<II", gop.start, gop.end))
        out_buf.write(struct.pack("<I", len(pup_bytes))); out_buf.write(pup_bytes)
        out_buf.write(struct.pack("<I", len(payload))); out_buf.write(payload)
        total_pup += len(pup_bytes); total_payload += len(payload)
        print(f"[gop {gi:3d}] {gop.start:4d}:{gop.end:4d}  PUP={len(pup_bytes)}B  payload={len(payload)}B")

    total = len(out_buf.getvalue())
    bpp = total * 8 / (T * H * W)
    print(f"wrote {args.output}: {total/1024:.1f} KiB  ({bpp:.4f} bpp, "
          f"PUP share={100*total_pup/max(total,1):.1f}%)")
    Path(args.output).write_bytes(out_buf.getvalue())


if __name__ == "__main__":
    main()
