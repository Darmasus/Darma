"""Vimeo-90k septuplet dataset.

Expected on-disk layout (matches the official release at
http://toflow.csail.mit.edu/index.html#septuplet):

    root/
      sequences/
        00001/0266/im1.png ... im7.png
        00001/0268/im1.png ... im7.png
        ...
      sep_trainlist.txt      # lines "00001/0266"
      sep_testlist.txt

Each sample is a (T, 3, H, W) float tensor in [0, 1]. Typical native
resolution is 448x256; we random-crop to a configurable size. The loader
supports:

  * temporal subsampling (e.g. take every 2nd of 7 frames => T=4)
  * random reverse (treat clip backwards with 50% prob — cheap augmentation)
  * random horizontal flip
  * deterministic mode for validation (center crop, no flip)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class Vimeo90kSeptuplet(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",           # "train" | "test"
        crop: int = 256,
        num_frames: int = 5,
        augment: bool = True,
        temporal_stride: int = 1,
        seed: int | None = None,
    ):
        root = Path(root)
        self.root = root
        self.seq_dir = root / "sequences"
        list_path = root / f"sep_{'trainlist' if split == 'train' else 'testlist'}.txt"
        if not list_path.exists():
            raise FileNotFoundError(
                f"Vimeo split file missing: {list_path}. "
                "Expected the septuplet release layout."
            )
        self.samples = [ln.strip() for ln in list_path.read_text().splitlines() if ln.strip()]
        self.crop = crop
        self.num_frames = num_frames
        self.augment = augment
        self.temporal_stride = temporal_stride
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_seven(self, clip_dir: Path) -> list[Image.Image]:
        imgs = []
        for i in range(1, 8):
            p = clip_dir / f"im{i}.png"
            imgs.append(Image.open(p).convert("RGB"))
        return imgs

    def _pick_window(self) -> Sequence[int]:
        # 1..7 inclusive; pick a contiguous sub-window of length num_frames at temporal_stride
        span = (self.num_frames - 1) * self.temporal_stride + 1
        if span > 7:
            raise ValueError(f"num_frames * stride exceeds septuplet length (got {span})")
        start = self._rng.randint(0, 7 - span) if self.augment else 0
        return list(range(start, start + span, self.temporal_stride))

    def __getitem__(self, idx: int) -> torch.Tensor:
        clip = self.seq_dir / self.samples[idx]
        imgs = self._load_seven(clip)
        indices = self._pick_window()
        imgs = [imgs[i] for i in indices]

        W0, H0 = imgs[0].size
        crop = self.crop
        if self.augment:
            x = self._rng.randint(0, max(W0 - crop, 0))
            y = self._rng.randint(0, max(H0 - crop, 0))
            hflip = self._rng.random() < 0.5
            reverse = self._rng.random() < 0.5
        else:
            x = (W0 - crop) // 2
            y = (H0 - crop) // 2
            hflip = False
            reverse = False

        frames: list[torch.Tensor] = []
        for im in imgs:
            t = TF.to_tensor(im)                 # (3, H, W), float [0,1]
            t = TF.crop(t, y, x, crop, crop)
            if hflip:
                t = TF.hflip(t)
            frames.append(t)
        if reverse:
            frames = frames[::-1]
        return torch.stack(frames, dim=0)        # (T, 3, H, W)


class VideoFolderDataset(Dataset):
    """Fallback dataset: recursively globs .mp4/.mov and reads short clips.

    Useful when Vimeo-90k is not available. Each __getitem__ returns a
    (num_frames, 3, crop, crop) tensor.
    """

    def __init__(self, root: str | Path, crop: int = 256, num_frames: int = 5,
                 max_clips: int = 10_000):
        from utils.video_io import iter_frames
        root = Path(root)
        self._paths = sorted(
            [p for p in root.rglob("*") if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}]
        )[:max_clips]
        if not self._paths:
            raise FileNotFoundError(f"No videos found under {root}")
        self.crop = crop
        self.num_frames = num_frames
        self._iter_frames = iter_frames

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        import random as _r
        path = self._paths[idx]
        frames = list(self._iter_frames(str(path), count=self.num_frames + 4))
        if len(frames) < self.num_frames:
            raise RuntimeError(f"Short video {path}")
        start = _r.randint(0, len(frames) - self.num_frames)
        out = torch.stack([frames[i] for i in range(start, start + self.num_frames)], dim=0)
        # HWC uint8 -> CHW float
        out = out.permute(0, 3, 1, 2).float().div_(255.0)
        T, C, H, W = out.shape
        y = _r.randint(0, max(H - self.crop, 0))
        x = _r.randint(0, max(W - self.crop, 0))
        return out[:, :, y:y + self.crop, x:x + self.crop]
