"""GLM-5.3-Flash vision tower: shapes and host streaming on a tiny random tower."""

from __future__ import annotations

import pytest
import torch

from freetoken.models.glm5_next import Glm5NextVisionModel, VisionConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _tiny_vc():
    return VisionConfig(
        hidden_size=64, depth=2, num_heads=4, intermediate_size=128, projection_intermediate_size=96, out_hidden_size=48,
        in_channels=3, patch_size=4, temporal_patch_size=2, spatial_merge_size=2, rms_norm_eps=1e-5, swiglu_limit=10.0,
        attention_bias=True,
    )


def _build(vc):
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            tower = Glm5NextVisionModel(vc)
    finally:
        torch.set_default_dtype(torch.float32)
    for p in tower.state_dict().values():
        p.normal_(0, 0.02)
    return tower


def test_tiny_tower_merges_four_patches_per_token():
    tower = _build(_tiny_vc())
    feature = torch.randn(16 + 64, 3 * 2 * 4 * 4, dtype=torch.bfloat16, device="cuda")
    out = tower.forward(feature, [[1, 4, 4], [1, 8, 8]])
    assert out.shape == (4 + 16, 48) and out.dtype == torch.bfloat16
    keys = tower.state_dict()
    assert keys["blocks.0.attn.qkv.bias"].shape == (192,) and keys["merger.post_projection_norm.bias"].shape == (48,)
    assert "merger.gate_proj.bias" not in keys and keys["downsample.weight"].shape == (48, 64, 2, 2)


def test_host_streamed_weights_compute_the_same_output():
    tower = _build(_tiny_vc())
    feature = torch.randn(64, 3 * 2 * 4 * 4, dtype=torch.bfloat16, device="cuda")
    resident = tower.forward(feature, [[1, 8, 8]])
    keys = tower.state_dict()
    tower.place_weights("host")
    assert tower.state_dict().keys() == keys.keys()
    assert tower.state_dict()["blocks.0.attn.qkv.weight"].device.type == "cpu"
    assert tower.state_dict()["merger.proj.weight"].device.type == "cuda"
    for _ in range(2):  # a second forward reuses the staging buffers behind the release events
        assert torch.equal(tower.forward(feature, [[1, 8, 8]]), resident)
    tower.place_weights("gpu")
    assert tower.state_dict()["blocks.0.attn.qkv.weight"].device.type == "cuda"
    assert torch.equal(tower.forward(feature, [[1, 8, 8]]), resident)
