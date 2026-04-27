"""Sample MORE Vimeo clips, avoiding ones already in an existing manifest
and avoiding held-out IDs. Writes a separate manifest for the new batch
so the old batch isn't re-captioned.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--existing-manifest", default="cool_chic/data/vimeo_clips/manifest.json")
    ap.add_argument("--heldout-file", default="cool_chic/data/heldout_ids.txt")
    ap.add_argument("--out-dir", default="cool_chic/data/vimeo_clips_b")
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=96)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    root = Path(args.root)
    train_list = (root / "sep_trainlist.txt").read_text().strip().splitlines()
    rng = random.Random(args.seed)
    rng.shuffle(train_list)

    existing = set()
    if Path(args.existing_manifest).exists():
        for c in json.loads(Path(args.existing_manifest).read_text()):
            # c["name"] is e.g. 'vimeo_00062_0025' -> original path '00062/0025'
            existing.add(c["name"].replace("vimeo_", "").replace("_", "/", 1))
    heldout = set()
    if Path(args.heldout_file).exists():
        heldout = set(l.strip() for l in Path(args.heldout_file).read_text().splitlines() if l.strip())
    print(f"already-trained: {len(existing)}, held-out: {sorted(heldout)}")

    skip = existing | heldout
    picks = []
    for rel in train_list:
        if rel in skip: continue
        picks.append(rel)
        if len(picks) >= args.n: break
    print(f"picked {len(picks)} new clips")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    proc, model = _load_blip(args.device)

    manifest = []
    for i, rel in enumerate(picks):
        seq_dir = root / "sequences" / rel
        pngs = sorted(seq_dir.glob("im*.png"))
        if len(pngs) != 7:
            continue
        name = "vimeo_" + rel.replace("/", "_")
        mp4_path = out_dir / f"{name}.mp4"
        try:
            _write_mp4(pngs, mp4_path, args.width, args.height)
        except subprocess.CalledProcessError:
            continue
        caption = _caption(pngs[0], proc, model, args.device)
        # Drop "merry merry" degenerate loops and any caption that has a slur/swear.
        bad_tokens = {"merry merry", "fucked", "nigger"}
        if any(t in caption.lower() for t in bad_tokens):
            print(f"  DROP {name}: {caption[:50]}")
            mp4_path.unlink(missing_ok=True)
            continue
        manifest.append({"path": str(mp4_path), "name": name,
                          "caption": caption})
        if i % 10 == 0:
            print(f"  [{i+1:3d}/{len(picks)}] {name}: {caption[:55]}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {out_dir}/manifest.json ({len(manifest)} clips)")


if __name__ == "__main__":
    main()
