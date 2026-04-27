"""FFmpeg wrappers for libaom-av1 and libx265 at target bitrates.

Each encoder is invoked with constant-bitrate (CBR, 2-pass for AV1) to give a
fair RD point. For each bitrate we record the actual file size on disk (true
bits), then measure PSNR/MS-SSIM/VMAF against the original.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


def _check_ffmpeg():
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")


@dataclass
class EncodeResult:
    codec: str
    target_kbps: int
    bitstream_path: str
    bytes: int


def encode_av1(src: str, kbps: int, out_dir: str, preset: int = 8) -> EncodeResult:
    _check_ffmpeg()
    out = Path(out_dir) / f"av1_{kbps}.mp4"
    log = Path(tempfile.mkdtemp()) / "av1log"
    # 2-pass for rate control accuracy at low bitrates.
    base = ["ffmpeg", "-hide_banner", "-y", "-i", src,
            "-c:v", "libaom-av1", "-cpu-used", str(preset),
            "-b:v", f"{kbps}k", "-pix_fmt", "yuv420p"]
    subprocess.run(base + ["-pass", "1", "-passlogfile", str(log),
                           "-an", "-f", "null", "-"], check=True)
    subprocess.run(base + ["-pass", "2", "-passlogfile", str(log),
                           "-an", str(out)], check=True)
    return EncodeResult("libaom-av1", kbps, str(out), out.stat().st_size)


def encode_x265(src: str, kbps: int, out_dir: str, preset: str = "medium") -> EncodeResult:
    _check_ffmpeg()
    out = Path(out_dir) / f"x265_{kbps}.mp4"
    log = Path(tempfile.mkdtemp()) / "x265log"
    base = ["ffmpeg", "-hide_banner", "-y", "-i", src,
            "-c:v", "libx265", "-preset", preset,
            "-b:v", f"{kbps}k", "-pix_fmt", "yuv420p"]
    subprocess.run(base + ["-x265-params", f"pass=1:stats={log}",
                           "-an", "-f", "null", "-"], check=True)
    subprocess.run(base + ["-x265-params", f"pass=2:stats={log}",
                           "-an", str(out)], check=True)
    return EncodeResult("libx265", kbps, str(out), out.stat().st_size)
