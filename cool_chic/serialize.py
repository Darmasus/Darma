"""Canonical token-order for INR weight tensors.

Every quantizer (`TensorQuantizer`) holds an int32 tensor (`integer_codes`)
with arbitrary shape. To feed those to a learned prior we need:
  - A canonical *flat* order across an entire model
  - Per-position metadata (which tensor type, which level, where in the
    tensor) the prior can condition on

The metadata vocabulary is intentionally tiny (5 tensor types, integer
level + normalized offset) — small enough to embed into a few hundred
parameters and learn a useful prior from ~30 trained INRs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Tensor type IDs
# --------------------------------------------------------------------------- #
TYPE_GRID_XY = 0
TYPE_GRID_XT = 1
TYPE_GRID_YT = 2
TYPE_MLP_W   = 3
TYPE_MLP_B   = 4

# NeRV backbone tensor types (added in W4 backbone-swap).
TYPE_NERV_FRAME_EMBED = 5
TYPE_NERV_STEM_W      = 6
TYPE_NERV_STEM_B      = 7
TYPE_NERV_BLOCK_W     = 8   # level = 2*block_idx + conv_idx (0/1 within block)
TYPE_NERV_BLOCK_B     = 9
TYPE_NERV_OUT_W       = 10
TYPE_NERV_OUT_B       = 11

N_TYPES   = 12
MAX_LEVEL = 16


def parse_tensor_name(name: str) -> tuple[int, int]:
    """quants-key -> (type_id, level).

    Examples (names come from `attach_quantizers` which replaces '.' with '_'):
      'grid_xy_tables_0'  -> (TYPE_GRID_XY, 0)
      'grid_xt_tables_3'  -> (TYPE_GRID_XT, 3)
      'grid_yt_tables_2'  -> (TYPE_GRID_YT, 2)
      'mlp_net_0_weight'  -> (TYPE_MLP_W, 0)   # 0 = first Linear layer in mlp.net
      'mlp_net_2_weight'  -> (TYPE_MLP_W, 2)
      'mlp_net_2_bias'    -> (TYPE_MLP_B, 2)
    """
    if name.startswith("grid_xy_tables_"):
        return TYPE_GRID_XY, int(name.split("_")[-1])
    if name.startswith("grid_xt_tables_"):
        return TYPE_GRID_XT, int(name.split("_")[-1])
    if name.startswith("grid_yt_tables_"):
        return TYPE_GRID_YT, int(name.split("_")[-1])
    if name.startswith("mlp_net_") and name.endswith("_weight"):
        return TYPE_MLP_W, int(name.split("_")[-2])
    if name.startswith("mlp_net_") and name.endswith("_bias"):
        return TYPE_MLP_B, int(name.split("_")[-2])

    # --- NeRV backbone names (see cool_chic/nerv.py) ---
    if name == "frame_embed":
        return TYPE_NERV_FRAME_EMBED, 0
    if name == "stem_weight":
        return TYPE_NERV_STEM_W, 0
    if name == "stem_bias":
        return TYPE_NERV_STEM_B, 0
    if name == "out_conv_weight":
        return TYPE_NERV_OUT_W, 0
    if name == "out_conv_bias":
        return TYPE_NERV_OUT_B, 0
    # M-NeRV adds two output heads (residual_head, flow_head) but no
    # out_conv. Map them onto the existing OUT_W/OUT_B types and use
    # the level field to distinguish: 0 = residual, 1 = flow.
    if name == "residual_head_weight":  return TYPE_NERV_OUT_W, 0
    if name == "residual_head_bias":    return TYPE_NERV_OUT_B, 0
    if name == "flow_head_weight":      return TYPE_NERV_OUT_W, 1
    if name == "flow_head_bias":        return TYPE_NERV_OUT_B, 1
    if name.startswith("blocks_") and (name.endswith("_weight") or name.endswith("_bias")):
        # blocks_{B}_{IDX_WITHIN_SEQ}_(weight|bias) -- IDX is 1 or 3
        # (sequence is Up, Conv, GELU, Conv, GELU; index 1 and 3 are Convs)
        parts = name.split("_")
        block_idx = int(parts[1])
        seq_idx   = int(parts[2])    # 1 or 3
        conv_idx  = 0 if seq_idx == 1 else 1
        level = 2 * block_idx + conv_idx
        if name.endswith("_weight"):
            return TYPE_NERV_BLOCK_W, level
        return TYPE_NERV_BLOCK_B, level

    raise ValueError(f"can't parse tensor name: {name!r}")


@dataclass
class TensorEntry:
    name: str
    type_id: int
    level: int
    shape: tuple[int, ...]
    n: int     # numel


def canonical_order(quants: nn.ModuleDict) -> list[str]:
    """Stable iteration order shared by encoder and decoder.

    Sort by (type_id, level, name) so the encoder and decoder always agree
    even if Python dict-iteration order changes.
    """
    keyed = []
    for name in quants.keys():
        t, l = parse_tensor_name(name)
        keyed.append(((t, l, name), name))
    keyed.sort()
    return [k[1] for k in keyed]


def tokenize(quants: nn.ModuleDict) -> dict:
    """Flatten a quants dict into one long sequence of int codes plus
    per-position metadata.

    Returns a dict with:
      ints       : int32 tensor (N,)        -- the integer codes
      type_id    : int64 tensor (N,)        -- tensor type
      level      : int64 tensor (N,)        -- level (or layer idx)
      offset     : float32 tensor (N,)      -- in [0, 1)
      entries    : list[TensorEntry]        -- per-tensor descriptors (in order)
      scales     : list[float]              -- per-tensor dequant scale
    """
    order = canonical_order(quants)
    flat_ints, flat_type, flat_lvl, flat_off = [], [], [], []
    entries, scales = [], []
    for name in order:
        q = quants[name]
        ints = q.integer_codes().detach().cpu().reshape(-1)
        n = ints.numel()
        t, l = parse_tensor_name(name)
        entries.append(TensorEntry(name=name, type_id=t, level=l,
                                     shape=tuple(q.param.shape), n=n))
        scales.append(float(q.scale.detach().cpu()))
        flat_ints.append(ints.to(torch.int32))
        flat_type.append(torch.full((n,), t, dtype=torch.int64))
        flat_lvl .append(torch.full((n,), l, dtype=torch.int64))
        # offset normalized so the prior sees position-within-tensor on a
        # consistent scale regardless of tensor size.
        flat_off .append(torch.arange(n, dtype=torch.float32) / max(n, 1))
    return dict(
        ints   = torch.cat(flat_ints),
        type_id= torch.cat(flat_type),
        level  = torch.cat(flat_lvl),
        offset = torch.cat(flat_off),
        entries= entries,
        scales = scales,
    )


def detokenize(ints_flat: torch.Tensor,
                entries: list[TensorEntry]) -> dict[str, torch.Tensor]:
    """Inverse of `tokenize`'s flat ints layout.

    Returns {tensor_name: int32 tensor reshaped to original shape}.
    """
    out = {}
    cur = 0
    for e in entries:
        chunk = ints_flat[cur:cur + e.n].reshape(e.shape).to(torch.int32)
        out[e.name] = chunk
        cur += e.n
    if cur != ints_flat.numel():
        raise ValueError(f"flat length mismatch: {cur} vs {ints_flat.numel()}")
    return out
