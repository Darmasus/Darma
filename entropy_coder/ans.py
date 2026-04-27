"""Asymmetric Numeral Systems (rANS) coder.

Implemented via the `constriction` library, which provides a fast Rust-backed
rANS with a Python API. Two wrappers are exposed:

  * `encode_gaussian(symbols, means, scales)` / `decode_gaussian(...)`
     for the hyperprior latent y_hat.
  * `ANSCoder` for the Parameter Update Packets (fixed zero-mean Gaussian
     with per-layer scale; symbols are the quantized A, B entries).

constriction's model is in floats and exact-entropy-coded, so we avoid
building huge CDF tables ourselves. Falls back gracefully if unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import constriction
    _HAS_CONSTRICTION = True
except ImportError:                    # pragma: no cover
    _HAS_CONSTRICTION = False


def _require():
    if not _HAS_CONSTRICTION:
        raise ImportError("Install `constriction` (pip install constriction) for rANS.")


@dataclass
class ANSCoder:
    """Stateful rANS encoder/decoder for a stream of integer symbols.

    The underlying stack is LIFO (rANS property): encode appends a symbol,
    decode pops one from the opposite side. We wrap that here so callers can
    work in FIFO order (encode the file in natural order; decoder reverses).
    """
    precision: int = 12

    def encode_bytes(self, symbols: np.ndarray, min_val: int, max_val: int,
                     probabilities: np.ndarray) -> bytes:
        """Encode `symbols` under a categorical with given `probabilities`
        (length == max_val - min_val + 1). Returns raw compressed bytes."""
        _require()
        encoder = constriction.stream.stack.AnsCoder()
        model = constriction.stream.model.Categorical(
            probabilities.astype(np.float64),
        )
        # Encode in REVERSE so that decoding (which pops from the top of the
        # stack) yields symbols in forward order.
        encoder.encode_reverse(symbols.astype(np.int32) - min_val, model)
        return encoder.get_compressed().tobytes()

    def decode_bytes(self, buf: bytes, n: int, min_val: int, max_val: int,
                     probabilities: np.ndarray) -> np.ndarray:
        _require()
        compressed = np.frombuffer(buf, dtype=np.uint32).copy()
        decoder = constriction.stream.stack.AnsCoder(compressed)
        model = constriction.stream.model.Categorical(
            probabilities.astype(np.float64),
        )
        return decoder.decode(model, n) + min_val


def _gaussian_pmf(mean: float, scale: float, lo: int, hi: int) -> np.ndarray:
    """PMF over integers [lo, hi] discretizing a N(mean, scale^2)."""
    import math
    xs = np.arange(lo, hi + 1, dtype=np.float64)
    cdf = 0.5 * (1 + np.vectorize(math.erf)((xs + 0.5 - mean) / (scale * 2 ** 0.5)))
    cdf_lo = 0.5 * (1 + np.vectorize(math.erf)((xs - 0.5 - mean) / (scale * 2 ** 0.5)))
    pmf = np.maximum(cdf - cdf_lo, 1e-12)
    pmf /= pmf.sum()
    return pmf


def encode_gaussian(symbols: np.ndarray, means: np.ndarray, scales: np.ndarray,
                    bound: int = 256) -> bytes:
    """Encode integer `symbols` (same shape as means/scales) under per-element
    discretized Gaussians.  All arrays must be broadcastable to the same shape."""
    _require()
    sym = symbols.astype(np.int32).reshape(-1)
    mu  = np.broadcast_to(means,  symbols.shape).reshape(-1).astype(np.float64)
    sig = np.broadcast_to(scales, symbols.shape).reshape(-1).astype(np.float64)
    encoder = constriction.stream.stack.AnsCoder()

    # constriction ships a `QuantizedGaussian` family; use it for speed.
    model = constriction.stream.model.QuantizedGaussian(-bound, bound)
    encoder.encode_reverse(sym, model, mu, sig)
    return encoder.get_compressed().tobytes()


def decode_gaussian(buf: bytes, shape: tuple[int, ...],
                    means: np.ndarray, scales: np.ndarray,
                    bound: int = 256) -> np.ndarray:
    _require()
    compressed = np.frombuffer(buf, dtype=np.uint32).copy()
    decoder = constriction.stream.stack.AnsCoder(compressed)
    n = int(np.prod(shape))
    mu  = np.broadcast_to(means,  shape).reshape(-1).astype(np.float64)
    sig = np.broadcast_to(scales, shape).reshape(-1).astype(np.float64)
    model = constriction.stream.model.QuantizedGaussian(-bound, bound)
    out = decoder.decode(model, mu, sig)
    return out.reshape(shape).astype(np.int32)
