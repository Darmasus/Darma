"""Track B orchestrator — download Vimeo-90k, extract, then train.

Designed for unattended multi-day operation on a workstation that's also
being used for other things (browsing, video, etc):

  * Spawns the training subprocess at BELOW_NORMAL priority on Windows so
    the desktop compositor stays responsive.
  * Auto-resumes from `checkpoints/base.pt` if it exists (so a Windows
    update reboot during training doesn't lose work — just rerun this).
  * Single 30-hour wall-time cap on the training step. Adjust via
    --max-seconds.
  * Streams the training subprocess's stdout to both the console and a
    log file so you can `tail -f` from another terminal.

Usage:
  python scripts/track_b_run.py --data-root D:/datasets/vimeo_septuplet
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))

# Official mirror of the septuplet dataset (~82 GB zip). Download is HTTP, no
# auth, supports range requests so partial downloads resume.
VIMEO_URL = "https://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip"
VIMEO_ZIP_NAME = "vimeo_septuplet.zip"
VIMEO_ZIP_BYTES = 87_930_374_183   # measured via HEAD on 2026-04-22


# ------------------------------------------------------------------ #
# download with resume
# ------------------------------------------------------------------ #
def _resume_download(url: str, out_path: Path) -> None:
    """HTTP GET with Range header for partial resume."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    have = out_path.stat().st_size if out_path.exists() else 0
    print(f"download: {url}\n  -> {out_path}  (resuming from {have/1e9:.2f} GB)")

    req = urllib.request.Request(url)
    if have > 0:
        req.add_header("Range", f"bytes={have}-")

    t0 = time.time()
    last_print = t0
    with urllib.request.urlopen(req) as resp, open(out_path, "ab") as f:
        total_remote = have + int(resp.headers.get("Content-Length", 0))
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            have += len(chunk)
            now = time.time()
            if now - last_print > 5:
                pct = 100.0 * have / max(total_remote, 1)
                rate_mbps = have / (now - t0) / 1e6
                print(f"  {have/1e9:6.2f} GB / {total_remote/1e9:6.2f} GB  "
                      f"({pct:5.1f}%)  {rate_mbps:5.1f} MB/s",
                      flush=True)
                last_print = now
    print(f"download complete in {(time.time()-t0)/60:.1f} min")


# ------------------------------------------------------------------ #
# extraction (parallel-ish via stdlib zipfile)
# ------------------------------------------------------------------ #
def _extract_if_needed(zip_path: Path, target_dir: Path) -> None:
    sentinel = target_dir / "sequences"
    if sentinel.exists() and any(sentinel.iterdir()):
        print(f"already extracted (found {sentinel}), skipping")
        return
    print(f"extracting {zip_path} -> {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        t0 = time.time()
        for i, n in enumerate(names):
            z.extract(n, path=target_dir)
            if i % 5000 == 0 and i > 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(names) - i) / rate
                print(f"  {i:>7d} / {len(names)}  ({rate:.0f} files/s, ETA {eta/60:.1f} min)",
                      flush=True)
    print(f"extracted in {(time.time()-t0)/60:.1f} min")


# ------------------------------------------------------------------ #
# launch training as subprocess at BELOW_NORMAL priority
# ------------------------------------------------------------------ #
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000      # Windows constant


def _launch_training(args) -> int:
    py = sys.executable
    cmd = [
        py, str(THIS.parents[1] / "training" / "train_base.py"),
        "--dataset", "vimeo",
        "--data-root", str(args.data_root),
        "--crop", "256",
        "--num-frames", "5",
        "--batch", str(args.batch),
        "--num-workers", str(args.num_workers),
        "--epochs", "100",
        "--lr", "1e-5",                 # lowered again: 3e-5 still exploded at step 646
        "--aux-lr", "1e-3",
        "--lambda-rd", str(args.lambda_rd),
        "--grad-clip", "0.25",          # tightened further
        "--max-seconds", str(args.max_seconds),
        "--log-every", "50",
        "--ckpt-every", "500",
        "--log-jsonl", str(args.log_jsonl),
        "--out", str(args.out),
        "--device", "cuda",
        "--resume", str(args.out),     # auto-resume if file exists
    ]
    print("launch:", " ".join(cmd), flush=True)

    creationflags = 0
    if os.name == "nt":
        creationflags = BELOW_NORMAL_PRIORITY_CLASS

    log_path = Path(args.log_text)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as logf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 creationflags=creationflags, bufsize=0)
        try:
            for line in proc.stdout:
                sys.stdout.buffer.write(line); sys.stdout.buffer.flush()
                logf.write(line); logf.flush()
        except KeyboardInterrupt:
            print("\nctrl-c received, terminating training subprocess...")
            proc.terminate()
        rc = proc.wait()
    print(f"training subprocess exit: {rc}")
    return rc


# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="D:/datasets/vimeo_septuplet",
                    help="extracted dataset root")
    ap.add_argument("--zip-path", type=str, default="D:/datasets/vimeo_septuplet.zip",
                    help="where to download the zip")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--lambda-rd", type=float, default=0.013)
    ap.add_argument("--max-seconds", type=int, default=108_000)   # 30 h
    ap.add_argument("--out", type=str, default="checkpoints/base.pt")
    ap.add_argument("--log-jsonl", type=str, default="logs/track_b.jsonl")
    ap.add_argument("--log-text", type=str, default="logs/track_b.out")
    args = ap.parse_args()

    zip_path = Path(args.zip_path)
    data_root = Path(args.data_root)

    # 1. Disk-space sanity check.
    free = shutil.disk_usage(zip_path.parent if zip_path.parent.exists()
                              else Path(zip_path.anchor)).free
    print(f"free disk on {zip_path.parent}: {free/1e9:.1f} GB")
    if free < 200 * 1024 ** 3:
        print("WARNING: less than 200 GB free; extraction may fail.")

    # 2. Download.
    if not args.skip_download and not (zip_path.exists() and zip_path.stat().st_size > VIMEO_ZIP_BYTES * 0.95):
        _resume_download(VIMEO_URL, zip_path)
    else:
        print(f"skip download (zip exists at {zip_path}, size {zip_path.stat().st_size/1e9:.1f} GB)")

    # 3. Extract.
    if not args.skip_extract:
        _extract_if_needed(zip_path, data_root)

    # 4. Train.
    if not args.skip_train:
        rc = _launch_training(args)
        sys.exit(rc)


if __name__ == "__main__":
    main()
