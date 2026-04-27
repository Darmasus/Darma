"""Quality metrics: PSNR, MS-SSIM, VMAF (via libvmaf through ffmpeg)."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch


def psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((x - y) ** 2).clamp_min(1e-12)
    return float(10.0 * torch.log10((max_val ** 2) / mse))


def ms_ssim_metric(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> float:
    from pytorch_msssim import ms_ssim
    if x.dim() == 3:
        x, y = x.unsqueeze(0), y.unsqueeze(0)
    return float(ms_ssim(x.clamp(0, 1), y.clamp(0, 1), data_range=data_range))


def vmaf_score(reference_mp4: str, distorted_mp4: str, model: str = "vmaf_v0.6.1") -> float:
    """Requires ffmpeg built with libvmaf. Returns mean VMAF over the clip.

    On Windows, libvmaf's filter argument parser uses `:` as a separator, so
    any literal `:` in the log path (e.g. `C:\\Users\\...`) must be escaped.
    We cd into the temp dir and pass a relative filename to avoid this.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")
    with tempfile.TemporaryDirectory() as tmp:
        log_name = "vmaf.json"
        ref = Path(reference_mp4).resolve()
        dist = Path(distorted_mp4).resolve()
        cmd = [
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(dist), "-i", str(ref),
            "-lavfi", f"libvmaf=log_path={log_name}:log_fmt=json:model=version={model}",
            "-f", "null", "-",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg/libvmaf failed:\n{res.stderr[-2000:]}")
        data = json.loads((Path(tmp) / log_name).read_text())
        return float(data["pooled_metrics"]["vmaf"]["mean"])
