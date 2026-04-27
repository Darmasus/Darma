"""Generate a diverse set of short clips with ffmpeg lavfi sources.

Each clip gets an associated caption describing its content. The goal
isn't photorealism — it's *content diversity* in the weight statistics
of the resulting INRs (flat vs high-freq, deterministic vs chaotic,
color-banded vs smooth, etc.), so a caption-conditioned prior has
something to learn from.

Writes (H, W=160, 96) RGB mp4 clips, 16 frames each, to `cool_chic/data/clips/`.
Small resolution keeps per-INR overfit fast on the GPU.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CLIPS: list[dict] = [
    dict(name="smpte",      caption="smpte color bars, static vertical stripes, no motion",
         lavfi="smptebars=size=160x96:rate=8:duration=2"),
    dict(name="smptehd",    caption="hd color bars, static pattern, broadcast reference",
         lavfi="smptehdbars=size=160x96:rate=8:duration=2"),
    dict(name="pal75bars",  caption="pal 75 percent color bars, broadcast test pattern",
         lavfi="pal75bars=size=160x96:rate=8:duration=2"),
    dict(name="pal100bars", caption="pal 100 percent color bars, saturated broadcast pattern",
         lavfi="pal100bars=size=160x96:rate=8:duration=2"),
    dict(name="testsrc",    caption="ffmpeg test pattern with timecode counter, rainbow bars",
         lavfi="testsrc=size=160x96:rate=8:duration=2"),
    dict(name="testsrc2",   caption="ffmpeg test source two, animated timecode and color bars",
         lavfi="testsrc2=size=160x96:rate=8:duration=2"),
    dict(name="rgbtest",    caption="rgb test pattern with primary colors",
         lavfi="rgbtestsrc=size=160x96:rate=8:duration=2"),
    dict(name="yuvtest",    caption="yuv test pattern with chroma fields",
         lavfi="yuvtestsrc=size=160x96:rate=8:duration=2"),
    dict(name="mandel",     caption="animated mandelbrot fractal zoom, fine detail, chaotic edges",
         lavfi="mandelbrot=size=160x96:rate=8:end_scale=0.3:start_scale=0.6"),
    dict(name="life",       caption="conway game of life cellular automaton, black and white cells evolving",
         lavfi="life=size=160x96:rate=8:mold=10:life_color=#ffffff:ratio=0.5:death_color=#000000"),
    dict(name="cellauto",   caption="elementary cellular automaton, binary pattern rule 30",
         lavfi="cellauto=rule=30:size=160x96:rate=8"),
    dict(name="gradients1", caption="animated smooth linear gradient, soft color transitions",
         lavfi="gradients=size=160x96:rate=8:duration=2:type=linear:c0=red:c1=blue:speed=0.05"),
    dict(name="gradients2", caption="animated radial gradient, center outward color change, smooth",
         lavfi="gradients=size=160x96:rate=8:duration=2:type=radial:c0=yellow:c1=purple:speed=0.02"),
    dict(name="solidred",   caption="solid red color frame, flat uniform image, no motion",
         lavfi="color=color=red:size=160x96:rate=8:duration=2"),
    dict(name="solidgreen", caption="solid green color frame, flat uniform image, no motion",
         lavfi="color=color=green:size=160x96:rate=8:duration=2"),
    dict(name="solidblue",  caption="solid blue color frame, flat uniform image, no motion",
         lavfi="color=color=blue:size=160x96:rate=8:duration=2"),
    dict(name="blackframe", caption="solid black frame, zero intensity, silent image",
         lavfi="color=color=black:size=160x96:rate=8:duration=2"),
    dict(name="whiteframe", caption="solid white frame, full intensity uniform",
         lavfi="color=color=white:size=160x96:rate=8:duration=2"),
    dict(name="noise1",     caption="random pixel noise, high frequency static, chaotic uncorrelated",
         lavfi="color=color=gray:size=160x96:rate=8:duration=2,noise=alls=80:allf=t"),
    dict(name="noise2",     caption="low intensity grain noise, mostly gray with subtle texture",
         lavfi="color=color=gray:size=160x96:rate=8:duration=2,noise=alls=20:allf=t"),
    dict(name="spectrum",   caption="color spectrum gradient, rainbow hue sweep across frame",
         lavfi="colorspectrum=size=160x96:rate=8:duration=2"),
    # A couple of "complex" synthetic clips derived by compositing.
    dict(name="blur_bars",  caption="color bars softened by blur, low frequency pastel stripes",
         lavfi="smptebars=size=160x96:rate=8:duration=2,boxblur=10:2"),
    dict(name="hued_test",  caption="test pattern with inverted hue, shifted color wheel",
         lavfi="testsrc=size=160x96:rate=8:duration=2,hue=h=180"),
    dict(name="mirror_mandel",
         caption="mandelbrot fractal mirrored horizontally, symmetric fractal edges",
         lavfi="mandelbrot=size=160x96:rate=8:end_scale=0.3,hflip"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="cool_chic/data/clips")
    ap.add_argument("--crf", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for c in CLIPS:
        out = out_dir / f"{c['name']}.mp4"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", c["lavfi"],
            "-frames:v", "16",
            "-c:v", "libx264", "-crf", str(args.crf), "-preset", "veryfast",
            "-pix_fmt", "yuv420p", str(out),
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"  SKIP {c['name']} (ffmpeg failed)")
            continue
        size_kb = out.stat().st_size / 1024
        print(f"  [{size_kb:6.1f} KB] {out.name:20s} {c['caption'][:55]}")
        manifest.append({"path": str(out), "name": c["name"],
                          "caption": c["caption"]})

    import json
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)} clips, manifest at {out_dir}/manifest.json")


if __name__ == "__main__":
    main()
