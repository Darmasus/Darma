"""Week-2 bitstream: per-weight (mu, sigma) come from a learned prior, not
from per-tensor sigma carried in the file.

Format (`.cc2`):
  magic           u32     'CC02'
  kind            u8
  H, W            u16, u16
  T_frames        u16
  cfg_blob_len    u16
  cfg_blob        bytes   pickled VideoConfig / ImageConfig
  prior_id_len    u16
  prior_id        bytes   UTF-8 (e.g. "v1-mlp-h64-d3")
  n_tensors       u16
  for each tensor (canonical_order):
    name_len      u16
    name          bytes
    n_dims        u8
    shape         u32 * n_dims
    scale         f32
  payload_len     u32
  payload         bytes   single rANS stream covering ALL tensors

Encoder and decoder must agree on:
  - canonical_order (sorted by (type_id, level, name))
  - prior weights (loaded by prior_id)
  - quantization bound (constant)

The prior is *not* in the bitstream; it ships with the decoder. So even
a 60 KB prior is amortized across every encoded video.
"""
from __future__ import annotations

import io
import pickle
import struct
from pathlib import Path

import numpy as np
import torch

from entropy_coder.ans import encode_gaussian, decode_gaussian
from .bitstream import KIND_IMAGE, KIND_VIDEO  # reuse
from .prior import WeightPrior
from .serialize import (
    canonical_order, tokenize, detokenize, parse_tensor_name,
    TensorEntry,
)


MAGIC_V2 = 0x43433032   # 'CC02' -- non-captioned
MAGIC_V3 = 0x43433033   # 'CC03' -- caption-conditioned
DEFAULT_PRIOR_ID = "v1-mlp"
DEFAULT_BOUND = 4096


# --------------------------------------------------------------------------- #
# Prior loading
# --------------------------------------------------------------------------- #
def load_prior(path: str = "cool_chic/data/prior.pt",
                device: str | torch.device = "cpu") -> WeightPrior:
    """Loads a `WeightPrior` checkpoint. Tolerates two legacy variants:

    - Older priors were saved when `serialize.N_TYPES` was 5 (hash-grid
      only). After NeRV was added it bumped to 12. We zero-pad the
      `type_emb` in that case so old hash-grid-trained priors still load
      and behave correctly on hash-grid weights.
    - Even older priors used a `Sequential` head (`head.0/2/4/...`)
      instead of `hidden_layers + head_last`. Those cannot be loaded
      transparently — retrain.
    """
    blob = torch.load(path, weights_only=False, map_location=device)
    caption_dim = blob.get("caption_dim", 0)
    p = WeightPrior(hidden=blob["hidden"], depth=blob["depth"],
                     caption_dim=caption_dim).to(device)
    state = dict(blob["state_dict"])
    # Pad/truncate type_emb if the saved checkpoint used a different N_TYPES.
    cur_n = p.type_emb.weight.shape[0]
    saved_te = state.get("type_emb.weight")
    if saved_te is not None and saved_te.shape[0] != cur_n:
        if saved_te.shape[0] < cur_n:
            pad = torch.zeros(cur_n - saved_te.shape[0], saved_te.shape[1],
                                device=saved_te.device, dtype=saved_te.dtype)
            state["type_emb.weight"] = torch.cat([saved_te, pad], dim=0)
        else:
            state["type_emb.weight"] = saved_te[:cur_n]
    p.load_state_dict(state)
    p.eval()
    return p


# --------------------------------------------------------------------------- #
# Encode
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_codec_v2(quants, *, kind: int, H: int, W: int, T_frames: int,
                     cfg, prior: WeightPrior,
                     prior_id: str = DEFAULT_PRIOR_ID,
                     bound: int = DEFAULT_BOUND) -> bytes:
    tok = tokenize(quants)
    device = next(prior.parameters()).device

    type_id = tok["type_id"].to(device)
    level   = tok["level"].to(device)
    offset  = tok["offset"].to(device)
    mu, log_sigma = prior(type_id, level, offset)
    sigma = log_sigma.exp().clamp_min(0.5)   # guard tiny sigma

    ints = tok["ints"].cpu().numpy().astype(np.int32)
    means  = mu.cpu().numpy().astype(np.float64)
    scales = sigma.cpu().numpy().astype(np.float64)

    # One single rANS stream over the full flattened sequence.
    sym_clipped = np.clip(ints, -bound + 1, bound - 1)
    payload = encode_gaussian(sym_clipped, means, scales, bound=bound)

    cfg_blob   = pickle.dumps(cfg, protocol=pickle.HIGHEST_PROTOCOL)
    prior_blob = prior_id.encode("utf-8")
    out = io.BytesIO()
    out.write(struct.pack("<IBHHHH", MAGIC_V2, kind, H, W, T_frames, len(cfg_blob)))
    out.write(cfg_blob)
    out.write(struct.pack("<H", len(prior_blob))); out.write(prior_blob)
    out.write(struct.pack("<H", len(tok["entries"])))
    for entry, scale in zip(tok["entries"], tok["scales"]):
        nb = entry.name.encode("utf-8")
        out.write(struct.pack("<H", len(nb))); out.write(nb)
        out.write(struct.pack("<B", len(entry.shape)))
        for d in entry.shape:
            out.write(struct.pack("<I", d))
        out.write(struct.pack("<f", scale))
    out.write(struct.pack("<I", len(payload)))
    out.write(payload)
    return out.getvalue()


# --------------------------------------------------------------------------- #
# Decode
# --------------------------------------------------------------------------- #
@torch.no_grad()
def decode_codec_v2(data: bytes, prior: WeightPrior,
                     bound: int = DEFAULT_BOUND) -> dict:
    buf = io.BytesIO(data)
    (magic, kind, H, W, T_frames, cfg_len) = struct.unpack("<IBHHHH", buf.read(13))
    assert magic == MAGIC_V2, f"bad magic 0x{magic:08x}"
    cfg = pickle.loads(buf.read(cfg_len))
    (prior_id_len,) = struct.unpack("<H", buf.read(2))
    prior_id = buf.read(prior_id_len).decode("utf-8")

    (n_tensors,) = struct.unpack("<H", buf.read(2))
    entries: list[TensorEntry] = []
    scales: list[float] = []
    for _ in range(n_tensors):
        (name_len,) = struct.unpack("<H", buf.read(2))
        name = buf.read(name_len).decode("utf-8")
        (n_dims,) = struct.unpack("<B", buf.read(1))
        shape = tuple(struct.unpack(f"<{n_dims}I", buf.read(4 * n_dims)))
        (scale,) = struct.unpack("<f", buf.read(4))
        t, l = parse_tensor_name(name)
        n = int(np.prod(shape))
        entries.append(TensorEntry(name=name, type_id=t, level=l,
                                     shape=shape, n=n))
        scales.append(scale)

    (payload_len,) = struct.unpack("<I", buf.read(4))
    payload = buf.read(payload_len)

    # Reconstruct the metadata sequence (must match encoder's tokenize order).
    type_ids, levels, offsets = [], [], []
    for e in entries:
        type_ids.append(torch.full((e.n,), e.type_id, dtype=torch.int64))
        levels  .append(torch.full((e.n,), e.level,   dtype=torch.int64))
        offsets .append(torch.arange(e.n, dtype=torch.float32) / max(e.n, 1))
    type_id = torch.cat(type_ids); level = torch.cat(levels); offset = torch.cat(offsets)

    device = next(prior.parameters()).device
    mu, log_sigma = prior(type_id.to(device), level.to(device), offset.to(device))
    sigma = log_sigma.exp().clamp_min(0.5)

    means_np  = mu.cpu().numpy().astype(np.float64)
    scales_np = sigma.cpu().numpy().astype(np.float64)
    n_total = sum(e.n for e in entries)
    ints_flat_np = decode_gaussian(payload, (n_total,), means_np, scales_np,
                                     bound=bound)
    ints_flat = torch.from_numpy(ints_flat_np)

    per_tensor_ints = detokenize(ints_flat, entries)

    weights: dict[str, torch.Tensor] = {}
    for e, sc in zip(entries, scales):
        ints = per_tensor_ints[e.name]
        weights[e.name] = (ints.float() * sc)

    return {
        "kind": kind, "H": H, "W": W, "T_frames": T_frames,
        "cfg": cfg, "prior_id": prior_id, "weights": weights,
    }


# --------------------------------------------------------------------------- #
# V3: caption-conditioned stream.
# Same layout as V2 but with a caption string inserted after prior_id.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_codec_v3(quants, *, kind: int, H: int, W: int, T_frames: int,
                     cfg, prior: WeightPrior, caption: str,
                     prior_id: str = DEFAULT_PRIOR_ID,
                     bound: int = DEFAULT_BOUND) -> bytes:
    from .caption import encode_caption
    device = next(prior.parameters()).device
    caption_emb = encode_caption(caption, device=device)

    tok = tokenize(quants)
    type_id = tok["type_id"].to(device)
    level   = tok["level"].to(device)
    offset  = tok["offset"].to(device)
    mu, log_sigma = prior(type_id, level, offset, caption_emb=caption_emb)
    sigma = log_sigma.exp().clamp_min(0.5)

    ints = tok["ints"].cpu().numpy().astype(np.int32)
    means  = mu.cpu().numpy().astype(np.float64)
    scales = sigma.cpu().numpy().astype(np.float64)
    sym_clipped = np.clip(ints, -bound + 1, bound - 1)
    payload = encode_gaussian(sym_clipped, means, scales, bound=bound)

    cfg_blob   = pickle.dumps(cfg, protocol=pickle.HIGHEST_PROTOCOL)
    prior_blob = prior_id.encode("utf-8")
    caption_blob = caption.encode("utf-8")
    out = io.BytesIO()
    out.write(struct.pack("<IBHHHH", MAGIC_V3, kind, H, W, T_frames, len(cfg_blob)))
    out.write(cfg_blob)
    out.write(struct.pack("<H", len(prior_blob)));   out.write(prior_blob)
    out.write(struct.pack("<H", len(caption_blob))); out.write(caption_blob)
    out.write(struct.pack("<H", len(tok["entries"])))
    for entry, scale in zip(tok["entries"], tok["scales"]):
        nb = entry.name.encode("utf-8")
        out.write(struct.pack("<H", len(nb))); out.write(nb)
        out.write(struct.pack("<B", len(entry.shape)))
        for d in entry.shape:
            out.write(struct.pack("<I", d))
        out.write(struct.pack("<f", scale))
    out.write(struct.pack("<I", len(payload)))
    out.write(payload)
    return out.getvalue()


@torch.no_grad()
def decode_codec_v3(data: bytes, prior: WeightPrior,
                     bound: int = DEFAULT_BOUND) -> dict:
    from .caption import encode_caption

    buf = io.BytesIO(data)
    (magic, kind, H, W, T_frames, cfg_len) = struct.unpack("<IBHHHH", buf.read(13))
    assert magic == MAGIC_V3, f"bad magic 0x{magic:08x}"
    cfg = pickle.loads(buf.read(cfg_len))
    (prior_id_len,) = struct.unpack("<H", buf.read(2))
    prior_id = buf.read(prior_id_len).decode("utf-8")
    (caption_len,) = struct.unpack("<H", buf.read(2))
    caption = buf.read(caption_len).decode("utf-8")

    (n_tensors,) = struct.unpack("<H", buf.read(2))
    entries: list[TensorEntry] = []; scales: list[float] = []
    for _ in range(n_tensors):
        (name_len,) = struct.unpack("<H", buf.read(2))
        name = buf.read(name_len).decode("utf-8")
        (n_dims,) = struct.unpack("<B", buf.read(1))
        shape = tuple(struct.unpack(f"<{n_dims}I", buf.read(4 * n_dims)))
        (scale,) = struct.unpack("<f", buf.read(4))
        t, l = parse_tensor_name(name)
        n = int(np.prod(shape))
        entries.append(TensorEntry(name=name, type_id=t, level=l, shape=shape, n=n))
        scales.append(scale)

    (payload_len,) = struct.unpack("<I", buf.read(4))
    payload = buf.read(payload_len)

    device = next(prior.parameters()).device
    caption_emb = encode_caption(caption, device=device)
    type_ids, levels, offsets = [], [], []
    for e in entries:
        type_ids.append(torch.full((e.n,), e.type_id, dtype=torch.int64))
        levels  .append(torch.full((e.n,), e.level,   dtype=torch.int64))
        offsets .append(torch.arange(e.n, dtype=torch.float32) / max(e.n, 1))
    type_id = torch.cat(type_ids); level = torch.cat(levels); offset = torch.cat(offsets)

    mu, log_sigma = prior(type_id.to(device), level.to(device), offset.to(device),
                            caption_emb=caption_emb)
    sigma = log_sigma.exp().clamp_min(0.5)
    means_np  = mu.cpu().numpy().astype(np.float64)
    scales_np = sigma.cpu().numpy().astype(np.float64)
    n_total = sum(e.n for e in entries)
    ints_flat_np = decode_gaussian(payload, (n_total,), means_np, scales_np,
                                     bound=bound)
    ints_flat = torch.from_numpy(ints_flat_np)

    per_tensor_ints = detokenize(ints_flat, entries)
    weights = {e.name: per_tensor_ints[e.name].float() * sc
                for e, sc in zip(entries, scales)}
    return {
        "kind": kind, "H": H, "W": W, "T_frames": T_frames,
        "cfg": cfg, "prior_id": prior_id, "caption": caption, "weights": weights,
    }
