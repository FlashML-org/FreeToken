"""MiniMax-M3 vision tower: shapes and host streaming on a tiny random tower."""

from __future__ import annotations

import pytest
import torch

from freetoken.models.minimax_m3 import MiniMaxM3VisionModel, VisionConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _single_rank():
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _tiny_vc():
    return VisionConfig(
        hidden_size=64, num_layers=2, num_heads=4, intermediate_size=128, num_channels=3, patch_size=4, temporal_patch_size=2,
        spatial_merge_size=2, layer_norm_eps=1e-5, rope_theta=10000.0, projector_hidden_size=96, text_hidden_size=48,
    )


def _build(vc):
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            tower = MiniMaxM3VisionModel(vc)
    finally:
        torch.set_default_dtype(torch.float32)
    for p in tower.state_dict().values():
        p.normal_(0, 0.02)
    return tower


def test_tiny_tower_merges_four_patches_per_token():
    tower = _build(_tiny_vc())
    feature = torch.randn(64, 3 * 2 * 4 * 4, dtype=torch.bfloat16, device="cuda")
    out = tower.forward(feature, [1, 8, 8])
    assert out.shape == (16, 48) and out.dtype == torch.bfloat16
    keys = tower.state_dict()
    assert keys["vision_model.embeddings.patch_embedding.weight"].shape == (64, 3, 2, 4, 4)
    assert keys["vision_model.encoder.layers.0.self_attn.qkv.bias"].shape == (192,)
    assert keys["patch_merge_mlp.linear_1.weight"].shape == (96, 4 * 48)
    assert not any(k.startswith("vision_model.post_layernorm") for k in keys)


def test_host_streamed_weights_compute_the_same_output():
    tower = _build(_tiny_vc())
    feature = torch.randn(64, 3 * 2 * 4 * 4, dtype=torch.bfloat16, device="cuda")
    resident = tower.forward(feature, [1, 8, 8])
    keys = tower.state_dict()
    tower.place_weights("host")
    assert tower.state_dict().keys() == keys.keys()
    assert tower.state_dict()["vision_model.encoder.layers.0.self_attn.qkv.weight"].device.type == "cpu"
    assert tower.state_dict()["vision_model.pre_layrnorm.weight"].device.type == "cuda"
    for _ in range(2):  # a second forward reuses the staging buffers behind the release events
        assert torch.equal(tower.forward(feature, [1, 8, 8]), resident)
    tower.place_weights("gpu")
    assert tower.state_dict()["vision_model.encoder.layers.0.self_attn.qkv.weight"].device.type == "cuda"
    assert torch.equal(tower.forward(feature, [1, 8, 8]), resident)
