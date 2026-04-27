"""Robust install helper for WANVC.

CompressAI ships a C++ range-coder extension. On Windows and some Linux
CIs the default `pip install compressai` fails because:
  * it builds from sdist,
  * the build isolation env doesn't pin `torch`, so the CMake step can't
    find Torch,
  * MSVC has to be present and on PATH.

This script installs in the order that works most consistently:
  1. torch + torchvision (pinned to the user's CUDA if requested)
  2. numpy, pillow, etc.
  3. constriction (pure wheel on every platform — our primary rANS)
  4. compressai with --no-build-isolation so it picks up the just-installed
     torch from the active env. If it still fails, we fall back to installing
     *without* compressai; the codebase will raise an informative error if
     a code path that needs it is actually exercised.

Run as:  python scripts/install.py [--cuda 12.1] [--skip-compressai]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], allow_fail: bool = False) -> int:
    print(">>", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0 and not allow_fail:
        raise SystemExit(f"failed: {' '.join(cmd)}  (exit {rc})")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", type=str, default=None,
                    help="CUDA version, e.g. 12.1 or 11.8. Omit for CPU-only.")
    ap.add_argument("--skip-compressai", action="store_true",
                    help="Skip compressai install (hyperprior won't be available).")
    ap.add_argument("--extra-index-url", type=str, default=None)
    args = ap.parse_args()

    python = sys.executable
    pip = [python, "-m", "pip", "install", "--upgrade"]

    # 1) torch
    if args.cuda:
        tag = {"11.8": "cu118", "12.1": "cu121", "12.4": "cu124"}.get(args.cuda)
        if tag is None:
            raise SystemExit(f"unknown CUDA {args.cuda}")
        url = f"https://download.pytorch.org/whl/{tag}"
        _run(pip + ["torch", "torchvision", "--index-url", url])
    else:
        _run(pip + ["torch", "torchvision"])

    # 2) small deps
    _run(pip + ["numpy>=1.26", "pillow>=10", "tqdm>=4.66", "pyyaml>=6",
                "matplotlib>=3.8", "scipy>=1.12", "einops>=0.8",
                "ffmpeg-python>=0.2", "pytorch-msssim>=1.0",
                "lpips>=0.1.4"])

    # 3) rANS (pure wheel, no build needed)
    _run(pip + ["constriction>=0.3.5"])

    # decord for fast frame access; optional
    _run(pip + ["decord>=0.6.0"], allow_fail=True)

    # 4) compressai — the one that likes to break
    if args.skip_compressai:
        print("skipping compressai (hyperprior code paths will be unavailable)")
        return

    # First try the normal wheel — works on Linux/macOS recent releases.
    rc = _run(pip + ["compressai>=1.2.6"], allow_fail=True)
    if rc == 0:
        return

    print("wheel install failed; retrying with --no-build-isolation against the installed torch")
    rc = _run(pip + ["--no-build-isolation", "compressai>=1.2.6"], allow_fail=True)
    if rc == 0:
        return

    print()
    print("============================================================")
    print("compressai install failed. WANVC can run without it *only* if")
    print("you avoid the hyperprior (dev/testing). On Windows, install")
    print("Visual Studio Build Tools with the 'Desktop development with")
    print("C++' workload, then rerun this script.")
    print("============================================================")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
