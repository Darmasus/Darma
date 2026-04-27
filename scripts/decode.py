"""Decode a .wanvc bitstream back to an MP4."""
from __future__ import annotations

import argparse
import io
import pickle
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models import WANVCAutoencoder, apply_pup
from entropy_coder import decode_pup
from utils.video_io import write_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input",  type=str)
    ap.add_argument("output", type=str)
    ap.add_argument("--ckpt", type=str, default="checkpoints/base.pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data = open(args.input, "rb").read()
    buf = io.BytesIO(data)
    magic, W, H, fps_n, fps_d, n_gops = struct.unpack("<IHHHHI", buf.read(16))
    assert magic == 0x57414E31, "not a WANVC bitstream"
    fps = fps_n / fps_d

    model = WANVCAutoencoder().to(args.device)
    import pathlib
    if pathlib.Path(args.ckpt).exists():
        state = torch.load(args.ckpt, map_location=args.device)
        model.load_state_dict(state["model"], strict=False)
    model.eval()
    model.update_entropy_tables(force=True)

    # Snapshot base weights so every GOP can reset before applying its PUP.
    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    frames_out: list[torch.Tensor] = []
    with torch.inference_mode():
        for gi in range(n_gops):
            (gop_start, gop_end) = struct.unpack("<II", buf.read(8))
            (pup_len,) = struct.unpack("<I", buf.read(4))
            pup_bytes = buf.read(pup_len)
            (payload_len,) = struct.unpack("<I", buf.read(4))
            payload = buf.read(payload_len)

            # Reset and adapt.
            model.load_state_dict(base_state, strict=True)
            if pup_len > 0:
                apply_pup(model, decode_pup(pup_bytes, device=args.device))

            blob = pickle.loads(payload)
            Hp, Wp = blob["hw"]
            x_prev = model.decompress_iframe(blob["iframe"], (Hp, Wp))
            frames_out.append(x_prev.squeeze(0).cpu())

            for pkt in blob["pframes"]:
                x_curr = model.decompress_pframe(x_prev, pkt)
                frames_out.append(x_curr.squeeze(0).cpu())
                x_prev = x_curr

    video = torch.stack(frames_out, dim=0)
    write_video(args.output, video, fps=fps)
    print(f"wrote {args.output}  {video.shape[0]} frames")


if __name__ == "__main__":
    main()
