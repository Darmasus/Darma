"""PUP encode->decode roundtrip.

Designed to run without CompressAI, so it remains useful on Windows CI when
the compressai build fails. Instead of instantiating WANVCAutoencoder, we
build a tiny nn.Module that holds a few AdaptableConv2d layers directly.
"""
import torch
import torch.nn as nn

from models import AdaptableConv2d, collect_adaptable_layers, apply_pup
from entropy_coder import encode_pup, decode_pup


class _TinyAdaptModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = AdaptableConv2d(16, 16, 3, padding=1)
        self.conv2 = AdaptableConv2d(16, 32, 3, padding=1)
        self.conv3 = AdaptableConv2d(32, 8, 3, padding=1)


def test_pup_roundtrip():
    model = _TinyAdaptModel()
    for _, layer in collect_adaptable_layers(model):
        layer.reset_delta()
        with torch.no_grad():
            layer.A.normal_(0, 1e-2)
            layer.B.normal_(0, 1e-2)

    saved = {n: (l.A.detach().clone(), l.B.detach().clone())
             for n, l in collect_adaptable_layers(model)}

    pup_bytes = encode_pup(collect_adaptable_layers(model))

    # Reset deltas to prove decode actually reloads them.
    for _, layer in collect_adaptable_layers(model):
        with torch.no_grad():
            layer.A.zero_()
            layer.B.zero_()
        layer._adapted = False

    pup = decode_pup(pup_bytes, device="cpu")
    apply_pup(model, pup)

    for name, layer in collect_adaptable_layers(model):
        A0, B0 = saved[name]
        step = layer.log_scale.exp().item()
        assert (layer.A - A0).abs().max().item() <= step + 1e-6
        assert (layer.B - B0).abs().max().item() <= step + 1e-6


if __name__ == "__main__":
    test_pup_roundtrip()
    print("ok")
