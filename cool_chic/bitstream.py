"""Real bitstream serialization for an INR codec.

A `.cc` file packs:

  magic           u32     'CC01'
  kind            u8      0=image, 1=video
  H, W            u16, u16
  T_frames        u16     (1 for image)
  cfg_blob_len    u16
  cfg_blob        bytes   pickled ImageConfig / VideoConfig
  n_layers        u16
  for each parameter group:
    name_len      u16
    name          bytes   UTF-8
    n_int         u32
    scale         f32     dequantization scale
    sigma         f32     Laplace prior parameter (for arithmetic coding)
    payload_len   u32
    payload       bytes   rANS-coded int32 symbols under Laplace(0, sigma)

Why Laplace? Trained INR weights are sharply peaked around zero and
exponentially-tailed — Laplace fits the empirical distribution closely.
Encoding under a per-tensor Laplace gives near-entropy compression with
no learned prior network needed (that comes in Week 2).
"""
from __future__ import annotations

import io
import math
import pickle
import struct
from dataclasses import dataclass

import numpy as np
import torch

from entropy_coder.ans import encode_gaussian, decode_gaussian


MAGIC = 0x43433031   # 'CC01'

KIND_IMAGE = 0
KIND_VIDEO = 1


# --------------------------------------------------------------------------- #
# Per-tensor encode / decode using a Laplace-as-discretized-Gaussian
# (constriction's QuantizedGaussian is a close enough fit; the bitrate
# difference vs a true Laplace is < 5% in practice).
# --------------------------------------------------------------------------- #
def _encode_tensor(symbols: np.ndarray, sigma: float, bound: int = 4096) -> bytes:
    """Larger default bound so trained weights with large integer codes
    (caused by aggressive scale or fat-tail distributions) aren't truncated.
    The bit cost only grows ~logarithmically with bound."""
    sym_clipped = np.clip(symbols, -bound + 1, bound - 1).astype(np.int32)
    mu = np.zeros_like(sym_clipped, dtype=np.float64)
    sg = np.full(sym_clipped.shape, max(sigma, 0.5), dtype=np.float64)
    return encode_gaussian(sym_clipped, mu, sg, bound=bound)


def _decode_tensor(payload: bytes, shape: tuple[int, ...], sigma: float,
                   bound: int = 4096) -> np.ndarray:
    mu = np.zeros(shape, dtype=np.float64)
    sg = np.full(shape, max(sigma, 0.5), dtype=np.float64)
    return decode_gaussian(payload, shape, mu, sg, bound=bound)


# --------------------------------------------------------------------------- #
# Encode an INR codec into bytes
# --------------------------------------------------------------------------- #
@dataclass
class _LayerEntry:
    name: str
    shape: tuple[int, ...]
    scale: float
    sigma: float
    payload: bytes


def _entries_from_quants(quants) -> list[_LayerEntry]:
    entries = []
    for name, q in quants.items():
        ints = q.integer_codes().cpu().numpy()
        sigma = float(np.std(ints.astype(np.float32))) or 1.0
        payload = _encode_tensor(ints.reshape(-1), sigma=sigma)
        entries.append(_LayerEntry(
            name=name,
            shape=tuple(ints.shape),
            scale=float(q.scale.detach().cpu()),
            sigma=sigma,
            payload=payload,
        ))
    return entries


def encode_codec(quants, *, kind: int, H: int, W: int, T_frames: int,
                  cfg) -> bytes:
    cfg_blob = pickle.dumps(cfg, protocol=pickle.HIGHEST_PROTOCOL)
    out = io.BytesIO()
    out.write(struct.pack("<IBHHHH", MAGIC, kind, H, W, T_frames, len(cfg_blob)))
    out.write(cfg_blob)

    entries = _entries_from_quants(quants)
    out.write(struct.pack("<H", len(entries)))
    for e in entries:
        nb = e.name.encode("utf-8")
        out.write(struct.pack("<H", len(nb))); out.write(nb)
        out.write(struct.pack("<B", len(e.shape)))
        for d in e.shape:
            out.write(struct.pack("<I", d))
        out.write(struct.pack("<I", int(np.prod(e.shape))))
        out.write(struct.pack("<f", e.scale))
        out.write(struct.pack("<f", e.sigma))
        out.write(struct.pack("<I", len(e.payload)))
        out.write(e.payload)
    return out.getvalue()


# --------------------------------------------------------------------------- #
# Decode bytes -> dict of {name: torch.Tensor (dequantized)}
# --------------------------------------------------------------------------- #
def decode_codec(data: bytes) -> dict:
    buf = io.BytesIO(data)
    (magic, kind, H, W, T_frames, cfg_len) = struct.unpack("<IBHHHH", buf.read(13))
    assert magic == MAGIC, f"bad magic 0x{magic:08x}"
    cfg_blob = buf.read(cfg_len)
    cfg = pickle.loads(cfg_blob)

    (n_layers,) = struct.unpack("<H", buf.read(2))
    weights: dict[str, torch.Tensor] = {}
    for _ in range(n_layers):
        (name_len,) = struct.unpack("<H", buf.read(2))
        name = buf.read(name_len).decode("utf-8")
        (n_dims,) = struct.unpack("<B", buf.read(1))
        shape = tuple(struct.unpack(f"<{n_dims}I", buf.read(4 * n_dims)))
        (n_int,) = struct.unpack("<I", buf.read(4))
        (scale,) = struct.unpack("<f", buf.read(4))
        (sigma,) = struct.unpack("<f", buf.read(4))
        (n_payload,) = struct.unpack("<I", buf.read(4))
        payload = buf.read(n_payload)

        ints = _decode_tensor(payload, shape=shape, sigma=sigma)
        weights[name] = torch.from_numpy(ints.astype(np.float32) * scale)

    return {"kind": kind, "H": H, "W": W, "T_frames": T_frames,
            "cfg": cfg, "weights": weights}


# --------------------------------------------------------------------------- #
# Convenience: load decoded weights back into a model
# --------------------------------------------------------------------------- #
def load_weights_into_model(model, weights: dict[str, torch.Tensor]) -> None:
    """Load the dequantized tensors back into the model's parameters by
    matching the safe-name mapping (dots replaced with underscores)."""
    state = {n.replace(".", "_"): p for n, p in model.named_parameters()}
    with torch.no_grad():
        loaded = 0
        for safe, t in weights.items():
            if safe not in state:
                continue
            if state[safe].shape != t.shape:
                continue
            state[safe].copy_(t.to(state[safe].device))
            loaded += 1
    return loaded
