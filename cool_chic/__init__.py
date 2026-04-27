"""cool_chic — INR-based neural video codec exploration.

Two backbone families coexist in this package:

  Hash-grid INR (Cool-Chic style)
      hash_grid.HashGrid2D / HashGrid3D / TriPlaneGrid
      inr.INRDecoder
      codec.ImageINR / VideoINR

  NeRV (and motion-compensated variant)
      nerv.NeRVBackbone, NeRVConfig
      mnerv.MNeRVBackbone, MNeRVConfig (negative result, kept for reference)

Shared infrastructure:

  quantize.attach_quantizers          per-tensor STE-round quantizers
  serialize.tokenize / parse_tensor_name  canonical weight tokenization
  prior.WeightPrior                   learned (mu,log_sigma) predictor with optional
                                      MiniLM caption FiLM modulation
  caption.encode_caption              MiniLM (384-d) wrapper
  bitstream.encode_codec              per-tensor Laplace + rANS bitstream (.cc)
  bitstream_v2.encode_codec_v{2,3}    prior-conditioned + caption-conditioned variants

The bitstream IS the network weights. Each video gets its own tiny network.
Encoder = optimization. Decoder = forward pass.

See README.md for the full architectural journey, RD curves, and what we learned.
"""
from .hash_grid import HashGrid2D, HashGrid3D, TriPlaneGrid
from .inr import INRDecoder
from .codec import ImageINR, VideoINR
from .nerv import NeRVBackbone, NeRVConfig
from .mnerv import MNeRVBackbone, MNeRVConfig

__all__ = [
    "HashGrid2D", "HashGrid3D", "TriPlaneGrid",
    "INRDecoder",
    "ImageINR", "VideoINR",
    "NeRVBackbone", "NeRVConfig",
    "MNeRVBackbone", "MNeRVConfig",
]
