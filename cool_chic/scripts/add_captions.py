"""Patch the existing INR dataset with per-sample captions.

Adds a string `caption` field to each sample dict. Captions for
sample.mp4 windows describe the test-pattern content; captions for
synthetic clips describe the procedural pattern. The idea isn't perfect
descriptions — just captions diverse enough that the text encoder
produces embeddings with meaningful inter-sample cosine distance.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch


CAPTIONS = {
    "real_w0":  "color bars with timecode, rainbow diagonal, green checker pattern, ffmpeg test signal",
    "real_w1":  "color bars with timecode, rainbow diagonal, green checker pattern, ffmpeg test signal",
    "real_w2":  "color bars with timecode, rainbow diagonal, green checker pattern, ffmpeg test signal",
    "real_w3":  "color bars with timecode, rainbow diagonal, green checker pattern, ffmpeg test signal",
    "synth_0":  "synthetic radial gradient with drifting sinusoid, low frequency, smooth colors",
    "synth_1":  "synthetic radial gradient with drifting sinusoid, medium frequency, smooth colors",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cool_chic/data/inr_dataset.pt")
    args = ap.parse_args()

    samples = torch.load(args.data, weights_only=False)
    for s in samples:
        src = s["source"]
        s["caption"] = CAPTIONS.get(src, "generic short video clip")
        print(f"  {src:10s} -> {s['caption'][:60]}...")
    torch.save(samples, args.data)
    print(f"\nupdated {args.data}  ({len(samples)} samples captioned)")


if __name__ == "__main__":
    main()
