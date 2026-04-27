"""Smoke test: model forward passes on small synthetic input."""
import torch

from models import WANVCAutoencoder


def test_iframe_pframe_shapes():
    model = WANVCAutoencoder(N=32, M=48)
    model.update_entropy_tables(force=True)

    x0 = torch.rand(1, 3, 64, 64)
    x1 = torch.rand(1, 3, 64, 64)

    out_i = model.encode_iframe(x0)
    assert out_i["x_hat"].shape == x0.shape
    assert "y" in out_i["likelihoods"]

    out_p = model.encode_pframe(x0, x1)
    assert out_p["x_from_latent"].shape == x1.shape
    assert out_p["residual_idx"].shape[0] == 1


if __name__ == "__main__":
    test_iframe_pframe_shapes()
    print("ok")
