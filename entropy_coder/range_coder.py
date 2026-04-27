"""Range coder fallback for platforms without constriction.

Thin wrapper on CompressAI's `RangeEncoder` / `RangeDecoder` (which ships a
Rust implementation under the hood). Only used if rANS is unavailable.
"""
from __future__ import annotations

import numpy as np


class RangeCoder:
    def __init__(self):
        try:
            from compressai.ans import RangeEncoder, RangeDecoder
        except ImportError as e:           # pragma: no cover
            raise ImportError(
                "CompressAI's RangeEncoder is required for the range-coder "
                f"fallback. ({e})"
            )
        self._encoder_cls = RangeEncoder
        self._decoder_cls = RangeDecoder

    def encode(self, symbols: np.ndarray, cdfs, cdf_lengths, offsets) -> bytes:
        enc = self._encoder_cls()
        enc.encode_with_indexes(
            symbols.astype(np.int32).tolist(),
            list(range(len(symbols))),
            cdfs, cdf_lengths, offsets,
        )
        return enc.flush()

    def decode(self, buf: bytes, n: int, cdfs, cdf_lengths, offsets) -> np.ndarray:
        dec = self._decoder_cls()
        dec.set_stream(buf)
        return np.asarray(
            dec.decode_with_indexes(list(range(n)), cdfs, cdf_lengths, offsets),
            dtype=np.int32,
        )
