# Installing WANVC on Windows

The tricky dependency is **CompressAI** — it ships a C++ range-coder and
needs MSVC to build from source when no matching wheel exists. Follow the
order below for a reliable install.

## 1. Prerequisites

- **Python 3.10 or 3.11** (3.13 wheels are spotty on Windows right now).
- **Visual Studio 2022 Build Tools**, "Desktop development with C++" workload.
  Direct installer: <https://aka.ms/vs/17/release/vs_BuildTools.exe>
- **FFmpeg** with `libvmaf` and `libaom`:
  - Easiest: `winget install Gyan.FFmpeg.Essentials` then verify `ffmpeg -buildconf`
    reports `--enable-libvmaf --enable-libaom`.
  - If not, grab a "full" build from <https://www.gyan.dev/ffmpeg/builds/>.

## 2. Create a venv and install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

# CPU build:
python scripts/install.py

# CUDA 12.1 build:
python scripts/install.py --cuda 12.1
```

`scripts/install.py` installs torch first, then retries `compressai` with
`--no-build-isolation` if the initial attempt fails — that flag lets the
CompressAI build find the torch you just installed.

## 3. If compressai still won't build

- Confirm MSVC is on PATH: open a fresh "x64 Native Tools Command Prompt
  for VS 2022", then `cl.exe` should print a banner.
- Delete `%LOCALAPPDATA%\pip\Cache` and retry (corrupt downloads are the
  #1 cause on Windows).
- As a last resort, run with `--skip-compressai`. The hyperprior-based
  code path (`models.hyperprior.ScaleHyperprior`) will raise a clear error
  if actually exercised; every other part of the codebase (PUP serializer,
  rANS via `constriction`, benchmarks against FFmpeg) still works.

## 4. Sanity check

```powershell
python -m pytest tests/test_smoke_forward.py tests/test_pup_roundtrip.py -v
```

Both should pass without a CUDA device. With CUDA:

```powershell
python scripts/jit_trace.py --ckpt checkpoints/base.pt --out checkpoints/g_s.ts
# Reports synthesis FPS at 1080p.
```

## 5. Known issues

- **`decord`** sometimes ships broken Windows wheels. The code falls back to
  `ffmpeg-python` automatically, so `pip install decord` being skipped is fine.
- **Paths with spaces**: the FFmpeg wrappers quote arguments, but
  `--data-root` for Vimeo does not. Keep the dataset on a simple path like
  `C:\datasets\vimeo_septuplet`.
