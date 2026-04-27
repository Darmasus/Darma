"""Sample N real video clips from Vimeo-90K septuplets, caption each with
BLIP, write as (160x96, 7-frame) mp4s, and emit a manifest compatible with
make_inr_dataset_v2.py.

Vimeo-90K sequences live at:
  D:\\datasets\\vimeo_septuplet\\vimeo_septuplet\\sequences\\AAAAA\\BBBB\\im{1..7}.png

Each septuplet is 7 frames at 448x256. We downsample via ffmpeg and
caption the first frame so the caption-conditioned prior has real
per-clip semantic signal.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image


DEFAULT_VIMEO_ROOT = Path(r"D:\datasets\vimeo_septuplet\vimeo_septuplet")


def _load_blip(device: str):
    from transformers import BlipProcessor, BlipForConditionalGeneration
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return proc, model


@torch.no_grad()
def _caption(img_path: Path, proc, model, device: str) -> str:
    img = Image.open(img_path).convert("RGB")
    inputs = proc(img, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_length=40, num_beams=3)
    return proc.decode(out[0], skip_special_tokens=True)


def _write_mp4(png_paths: list[Path], out_path: Path, w: int, h: int) -> None:
    # Feed PNGs to ffmpeg via the "concat" demuxer with per-frame durations.
    # Simpler: use `-pattern_type glob` -- but glob ordering is risky on Win.
    # Easiest: build a temporary file list.
    listfile = out_path.with_suffix(".list.txt")
    listfile.write_text("\n".join([f"file '{p.as_posix()}'" for p in png_paths]))
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-r", "8", "-i", str(listfile),
        "-vf", f"scale={w}:{h}:flags=area",
        "-c:v", "libx264", "-crf", "10", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", str(out_path),
    ], check=True)
    listfile.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_VIMEO_ROOT))
    ap.add_argument("--n", type=int, default=50,
                    help="how many septuplets to sample")
    ap.add_argument("--out-dir", default="cool_chic/data/vimeo_clips")
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    root = Path(args.root)
    train_list = (root / "sep_trainlist.txt").read_text().strip().splitlines()
    rng = random.Random(args.seed)
    picks = rng.sample(train_list, args.n)
    print(f"picked {len(picks)} septuplets from {len(train_list)} total")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    proc, model = _load_blip(args.device)

    manifest = []
    for i, rel in enumerate(picks):
        seq_dir = root / "sequences" / rel.replace("/", "\\") / ""  # rel like 00001/0001
        seq_dir = (root / "sequences" / rel)
        pngs = sorted(seq_dir.glob("im*.png"))
        if len(pngs) != 7:
            print(f"  skip {rel}: {len(pngs)} pngs")
            continue
        name = "vimeo_" + rel.replace("/", "_").replace("\\", "_")
        mp4_path = out_dir / f"{name}.mp4"

        try:
            _write_mp4(pngs, mp4_path, args.width, args.height)
        except subprocess.CalledProcessError as e:
            print(f"  ffmpeg failed on {rel}: {e}")
            continue

        caption = _caption(pngs[0], proc, model, args.device)
        print(f"  [{i+1:2d}/{len(picks)}] {name}  "
              f"caption=\"{caption}\"")
        manifest.append({"path": str(mp4_path), "name": name,
                          "caption": caption})

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {manifest_path} ({len(manifest)} clips)")


if __name__ == "__main__":
    main()
