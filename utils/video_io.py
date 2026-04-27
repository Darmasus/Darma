"""Thin video I/O wrapper. Prefers decord, falls back to ffmpeg piping."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


def _decord_reader(path: str):
    try:
        import decord
    except ImportError:
        return None
    decord.bridge.set_bridge("native")
    return decord.VideoReader(path)


def iter_frames(path: str, start: int = 0, count: int | None = None) -> Iterator[torch.Tensor]:
    """Yields HWC uint8 tensors."""
    vr = _decord_reader(path)
    if vr is not None:
        n = len(vr) if count is None else min(start + count, len(vr))
        for i in range(start, n):
            frame = vr[i].asnumpy()            # H, W, 3, uint8
            yield torch.from_numpy(frame)
        return

    # ffmpeg fallback
    import ffmpeg
    probe = ffmpeg.probe(path)
    stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    W, H = int(stream["width"]), int(stream["height"])
    out_kwargs = {"format": "rawvideo", "pix_fmt": "rgb24"}
    if count is not None:
        out_kwargs["vframes"] = start + count
    cmd = ffmpeg.input(path).output("pipe:", **out_kwargs).global_args("-loglevel", "error").compile()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = W * H * 3
    try:
        i = 0
        while True:
            buf = proc.stdout.read(frame_size)
            if len(buf) < frame_size:
                break
            if i >= start and (count is None or i < start + count):
                arr = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
                yield torch.from_numpy(arr)
            i += 1
            if count is not None and i >= start + count:
                break
    finally:
        proc.stdout.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()


def read_video(path: str, max_frames: int | None = None) -> torch.Tensor:
    """Returns a (T, 3, H, W) float tensor in [0, 1]."""
    frames = list(iter_frames(path, count=max_frames))
    t = torch.stack(frames, dim=0).float().div_(255.0)
    return t.permute(0, 3, 1, 2).contiguous()


def write_video(path: str, frames: torch.Tensor, fps: float = 30.0, crf: int = 10) -> None:
    """frames: (T, 3, H, W) float [0,1]. Writes a high-quality x264 proxy."""
    import ffmpeg
    f = (frames.clamp(0, 1) * 255).round().to(torch.uint8)
    T, _, H, W = f.shape
    proc = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{W}x{H}", r=fps)
        .output(path, vcodec="libx264", crf=crf, pix_fmt="yuv420p")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for t in range(T):
        proc.stdin.write(f[t].permute(1, 2, 0).cpu().numpy().tobytes())
    proc.stdin.close()
    proc.wait()
