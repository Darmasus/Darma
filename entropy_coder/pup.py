"""Parameter Update Packet (PUP) serializer.

Binary layout (little-endian):

   magic            u32     0x50555031  ('PUP1')
   n_layers         u16
   for each layer:
     name_len       u16
     name           bytes   UTF-8, not null-terminated
     layer_id       u16     ordinal in model.named_modules()
     c_out          u16
     r              u16
     fan_in         u32     c_in * k * k
     log_scale      f32
     log_sigma      f32
     alpha          f32
     n_bytes_A      u32
     payload_A      bytes   rANS-coded int32 symbols
     n_bytes_B      u32
     payload_B      bytes

Decoder: reads this header, reconstructs (A, B) from rANS+quantization, and
hands them to `models.adaptation.apply_pup`.

Total overhead per layer: 24 bytes of header + two rANS streams. For rank-4
adapters on a 192-channel ConvTranspose2d, A has 192·9 entries and B has 768,
so payloads are typically 200–600 bytes each at 5e-3 scale → ~1–2 KB / layer.
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from .ans import encode_gaussian, decode_gaussian


MAGIC = 0x50555031        # 'PUP1'
HEADER_FMT_LAYER = "<HHHIfff II"   # see comment in encode_pup; kept for ref


@dataclass
class LayerPUP:
    name: str
    layer_id: int
    c_out: int
    r: int
    fan_in: int
    log_scale: float
    log_sigma: float
    alpha: float
    A_payload: bytes
    B_payload: bytes


@dataclass
class ParameterUpdatePacket:
    layers: list[LayerPUP]


# ------------------------------------------------------------------ #
# encoder-side
# ------------------------------------------------------------------ #
def _encode_one(layer_name: str, layer_id: int, layer) -> LayerPUP:
    """Extract the compressed A/B streams from a single AdaptableConv2d."""
    A_q, B_q, s = layer.quantized_AB()
    sigma = layer.log_sigma.exp().clamp(min=1e-6)
    scale = s

    # Integer symbols = round(x / scale). Their distribution is Gaussian-ish
    # with std sigma/scale.
    A_int = torch.round(A_q / scale).to(torch.int32).cpu().numpy()
    B_int = torch.round(B_q / scale).to(torch.int32).cpu().numpy()
    sig_int = float((sigma / scale).item())

    A_payload = encode_gaussian(A_int, np.zeros_like(A_int, dtype=np.float64),
                                np.full(A_int.shape, sig_int, dtype=np.float64))
    B_payload = encode_gaussian(B_int, np.zeros_like(B_int, dtype=np.float64),
                                np.full(B_int.shape, sig_int, dtype=np.float64))

    return LayerPUP(
        name=layer_name,
        layer_id=layer_id,
        c_out=layer.out_channels,
        r=layer.cfg.rank,
        fan_in=layer.in_channels * layer.kernel_size * layer.kernel_size,
        log_scale=float(layer.log_scale.item()),
        log_sigma=float(layer.log_sigma.item()),
        alpha=float(layer.cfg.alpha),
        A_payload=A_payload,
        B_payload=B_payload,
    )


def encode_pup(named_layers: Iterable[tuple[str, object]]) -> bytes:
    """Serialize a full Parameter Update Packet to bytes.

    `named_layers` is the output of `collect_adaptable_layers(model)` (or a
    filtered subset thereof — e.g. only layers that actually moved).
    """
    buf = io.BytesIO()
    buf.write(struct.pack("<I", MAGIC))

    entries = [_encode_one(n, i, l) for i, (n, l) in enumerate(named_layers)]
    buf.write(struct.pack("<H", len(entries)))
    for e in entries:
        name_bytes = e.name.encode("utf-8")
        buf.write(struct.pack("<H", len(name_bytes)))
        buf.write(name_bytes)
        buf.write(struct.pack("<H", e.layer_id))
        buf.write(struct.pack("<H", e.c_out))
        buf.write(struct.pack("<H", e.r))
        buf.write(struct.pack("<I", e.fan_in))
        buf.write(struct.pack("<f", e.log_scale))
        buf.write(struct.pack("<f", e.log_sigma))
        buf.write(struct.pack("<f", e.alpha))
        buf.write(struct.pack("<I", len(e.A_payload)))
        buf.write(e.A_payload)
        buf.write(struct.pack("<I", len(e.B_payload)))
        buf.write(e.B_payload)
    return buf.getvalue()


# ------------------------------------------------------------------ #
# decoder-side
# ------------------------------------------------------------------ #
def decode_pup(data: bytes, device: str | torch.device = "cpu") -> dict[str, dict[str, torch.Tensor]]:
    """Parse PUP bytes -> dict consumable by `models.adaptation.apply_pup`."""
    buf = io.BytesIO(data)
    (magic,) = struct.unpack("<I", buf.read(4))
    if magic != MAGIC:
        raise ValueError(f"bad PUP magic: 0x{magic:08x}")
    (n_layers,) = struct.unpack("<H", buf.read(2))
    out: dict[str, dict[str, torch.Tensor]] = {}
    for _ in range(n_layers):
        (name_len,)  = struct.unpack("<H", buf.read(2))
        name         = buf.read(name_len).decode("utf-8")
        (layer_id,)  = struct.unpack("<H", buf.read(2))
        (c_out,)     = struct.unpack("<H", buf.read(2))
        (r,)         = struct.unpack("<H", buf.read(2))
        (fan_in,)    = struct.unpack("<I", buf.read(4))
        (log_scale,) = struct.unpack("<f", buf.read(4))
        (log_sigma,) = struct.unpack("<f", buf.read(4))
        (alpha,)     = struct.unpack("<f", buf.read(4))

        (n_a,)     = struct.unpack("<I", buf.read(4))
        A_payload  = buf.read(n_a)
        (n_b,)     = struct.unpack("<I", buf.read(4))
        B_payload  = buf.read(n_b)

        scale = np.exp(log_scale)
        sigma = np.exp(log_sigma)
        sig_int = float(sigma / scale)

        A_int = decode_gaussian(
            A_payload, (r, fan_in),
            np.zeros((r, fan_in), dtype=np.float64),
            np.full((r, fan_in), sig_int, dtype=np.float64),
        )
        B_int = decode_gaussian(
            B_payload, (c_out, r),
            np.zeros((c_out, r), dtype=np.float64),
            np.full((c_out, r), sig_int, dtype=np.float64),
        )

        out[name] = {
            "A": torch.from_numpy(A_int.astype(np.float32) * scale).to(device),
            "B": torch.from_numpy(B_int.astype(np.float32) * scale).to(device),
            "log_scale": torch.tensor(log_scale, device=device),
            "log_sigma": torch.tensor(log_sigma, device=device),
        }
    return out
