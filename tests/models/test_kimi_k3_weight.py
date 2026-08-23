from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from freetoken.models.kimi_k3 import weight
from safetensors.torch import save_file


def _tp1():
    return SimpleNamespace(size=1, rank=0, is_primary=lambda: True)


def _config(
    *,
    experts: int = 1,
    hidden: int = 64,
    intermediate: int = 32,
    num_layers: int = 93,
):
    return SimpleNamespace(
        moe_weight_format="mxfp4",
        kimi_k3_args=SimpleNamespace(routed_expert_hidden_size=hidden),
        moe_intermediate_size=intermediate,
        first_k_dense_replace=1,
        num_layers=num_layers,
        num_experts=experts,
    )


def test_resident_names_strip_wrapper_and_skip_vision_mtp_and_experts():
    assert weight._resident_name("language_model.model.embed_tokens.weight", 93) == (
        "model.embed_tokens.weight"
    )
    assert weight._resident_name(
        "language_model.model.layers.92.self_attn.q_proj.weight", 93
    ) == ("model.layers.92.self_attn.q_proj.weight")
    assert (
        weight._resident_name(
            "language_model.model.layers.93.self_attn.q_proj.weight", 93
        )
        is None
    )
    assert weight._resident_name("vision_tower.blocks.0.weight", 93) is None
    assert weight._resident_name("mm_projector.post_norm.weight", 93) is None
    assert weight._resident_name("language_model.mtp.norm.weight", 93) is None
    assert (
        weight._resident_name(
            "language_model.model.layers.7.block_sparse_moe.gate.e_score_correction_bias",
            93,
        )
        == "model.layers.7.block_sparse_moe.e_score_correction_bias"
    )
    assert (
        weight._resident_name(
            "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed",
            93,
        )
        is None
    )


def test_resident_loader_fuses_dense_and_shared_mlp_across_shards(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(weight, "get_tp_info", _tp1)
    monkeypatch.setattr(weight, "cached_load_hf_config", lambda _: object())
    monkeypatch.setattr(
        weight, "parse_config", lambda _: SimpleNamespace(num_layers=93)
    )
    gate = torch.full((2, 3), 1.0)
    up = torch.full((2, 3), 2.0)
    shared_gate = torch.full((4, 3), 3.0)
    shared_up = torch.full((4, 3), 4.0)
    save_file(
        {
            "language_model.model.layers.0.mlp.gate_proj.weight": gate,
            "language_model.model.layers.1.block_sparse_moe.shared_experts.up_proj.weight": shared_up,
        },
        tmp_path / "model-00001-of-00002.safetensors",
    )
    save_file(
        {
            "language_model.model.layers.0.mlp.up_proj.weight": up,
            "language_model.model.layers.1.block_sparse_moe.shared_experts.gate_proj.weight": shared_gate,
            "language_model.model.layers.1.block_sparse_moe.gate.e_score_correction_bias": torch.ones(
                5
            ),
        },
        tmp_path / "model-00002-of-00002.safetensors",
    )
    loaded = dict(
        weight.iter_weights(
            str(tmp_path),
            torch.device("cpu"),
            include_moe_experts=False,
            include_non_moe=True,
        )
    )
    torch.testing.assert_close(
        loaded["model.layers.0.mlp.gate_up_proj.weight"], torch.cat((gate, up))
    )
    torch.testing.assert_close(
        loaded["model.layers.1.block_sparse_moe.shared_experts.gate_up_proj.weight"],
        torch.cat((shared_gate, shared_up)),
    )
    assert "model.layers.1.block_sparse_moe.e_score_correction_bias" in loaded


def test_resident_loader_rejects_incomplete_mlp_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(weight, "get_tp_info", _tp1)
    monkeypatch.setattr(weight, "cached_load_hf_config", lambda _: object())
    monkeypatch.setattr(
        weight, "parse_config", lambda _: SimpleNamespace(num_layers=93)
    )
    save_file(
        {"language_model.model.layers.0.mlp.gate_proj.weight": torch.ones(2, 3)},
        tmp_path / "model.safetensors",
    )
    with pytest.raises(ValueError, match="incomplete Kimi-K3 resident MLP fusions"):
        list(
            weight.iter_weights(
                str(tmp_path),
                torch.device("cpu"),
                include_moe_experts=False,
                include_non_moe=True,
            )
        )


def test_resident_loader_dequantizes_mxfp4_pairs_across_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(weight, "get_tp_info", _tp1)
    monkeypatch.setattr(weight, "cached_load_hf_config", lambda _: object())
    monkeypatch.setattr(weight, "parse_config", lambda _: SimpleNamespace(num_layers=8))
    prefix = "language_model.model.layers.1.block_sparse_moe.shared_experts"
    packed = torch.full((2, 16), 0x11, dtype=torch.uint8)
    scales = torch.full((2, 1), 127, dtype=torch.uint8)
    save_file(
        {
            f"{prefix}.gate_proj.weight_packed": packed,
            f"{prefix}.up_proj.weight_scale": scales,
        },
        tmp_path / "model-00001-of-00002.safetensors",
    )
    save_file(
        {
            f"{prefix}.gate_proj.weight_scale": scales,
            f"{prefix}.up_proj.weight_packed": packed,
        },
        tmp_path / "model-00002-of-00002.safetensors",
    )

    loaded = dict(
        weight.iter_weights(
            str(tmp_path),
            torch.device("cpu"),
            include_moe_experts=False,
            include_non_moe=True,
        )
    )
    fused = loaded["model.layers.1.block_sparse_moe.shared_experts.gate_up_proj.weight"]
    assert fused.dtype == torch.bfloat16
    torch.testing.assert_close(fused, torch.full_like(fused, 0.5))


def test_resident_fusion_rejects_incompatible_shapes():
    buffer = {}
    assert (
        weight._fuse_resident_mlp(
            "model.layers.0.mlp.gate_proj.weight", torch.ones(2, 3), buffer, 93
        )
        is None
    )
    with pytest.raises(ValueError, match="incompatible Kimi-K3 gate/up fusion"):
        weight._fuse_resident_mlp(
            "model.layers.0.mlp.up_proj.weight", torch.ones(3, 3), buffer, 93
        )


def test_copy_transposes_and_interleaves_gate_up():
    h, i = 64, 32
    banks = {
        "gate_up_blocks": [torch.zeros(1, h // 2, 2 * i, dtype=torch.uint8)],
        "gate_up_scales": [torch.zeros(1, h // 32, 2 * i, dtype=torch.uint8)],
        "down_blocks": [torch.zeros(1, i // 2, h, dtype=torch.uint8)],
        "down_scales": [torch.zeros(1, i // 32, h, dtype=torch.uint8)],
    }
    w1 = torch.randint(0, 256, (i, h // 2), dtype=torch.uint8)
    w3 = (w1 + 19).to(torch.uint8)
    w2 = torch.randint(0, 256, (h, i // 2), dtype=torch.uint8)
    for proj, value in (("w1", w1), ("w3", w3), ("w2", w2)):
        weight._copy_expert_tensor(
            banks,
            layer=0,
            expert=0,
            proj=proj,
            kind="weight_packed",
            value=value,
            hidden=h,
            intermediate=i,
        )
    torch.testing.assert_close(banks["gate_up_blocks"][0][0, :, ::2], w1.t())
    torch.testing.assert_close(banks["gate_up_blocks"][0][0, :, 1::2], w3.t())
    torch.testing.assert_close(banks["down_blocks"][0][0], w2.t())


def test_copy_fails_closed_on_wrong_shape_or_dtype():
    banks = {"gate_up_blocks": [torch.empty(1)]}
    with pytest.raises(ValueError, match="expected uint8"):
        weight._copy_expert_tensor(
            banks,
            layer=0,
            expert=0,
            proj="w1",
            kind="weight_packed",
            value=torch.empty(1, 1),
            hidden=64,
            intermediate=32,
        )


def test_official_keys_load_into_92_zero_based_moe_banks(tmp_path, monkeypatch):
    # Shrink only the numerical geometry; retain the official 1..92 layer map and
    # all six required tensors per expert so this exercises completeness checks.
    monkeypatch.setattr(weight, "get_tp_info", _tp1)
    tensors = {}
    for checkpoint_layer in range(1, 93):
        prefix = (
            f"language_model.model.layers.{checkpoint_layer}.block_sparse_moe.experts.0"
        )
        for proj, shape in (
            ("w1", (32, 32)),
            ("w3", (32, 32)),
            ("w2", (64, 16)),
        ):
            value = checkpoint_layer + {"w1": 1, "w2": 2, "w3": 3}[proj]
            tensors[f"{prefix}.{proj}.weight_packed"] = torch.full(
                shape, value % 256, dtype=torch.uint8
            )
        for proj, shape in (
            ("w1", (32, 2)),
            ("w3", (32, 2)),
            ("w2", (64, 1)),
        ):
            value = checkpoint_layer + {"w1": 4, "w2": 5, "w3": 6}[proj]
            tensors[f"{prefix}.{proj}.weight_scale"] = torch.full(
                shape, value % 256, dtype=torch.uint8
            )
    save_file(tensors, tmp_path / "model.safetensors")

    banks = weight.load_mxfp4_expert_banks(
        str(tmp_path), _config(), dtype=torch.bfloat16
    )
    assert len(banks["gate_up_blocks"]) == 92
    assert banks["gate_up_blocks"][0].shape == (1, 32, 64)
    assert banks["gate_up_blocks"][91][0, 0, 0].item() == 93  # layer 1/92 + w1 marker
    assert banks["gate_up_blocks"][91][0, 0, 1].item() == 95  # layer 92 + w3 marker
    assert banks["down_blocks"][0][0, 0, 0].item() == 3
    assert banks["gate_up_bias"][0] is banks["gate_up_bias"][91]
    assert banks["down_bias"][0] is banks["down_bias"][91]
    assert not torch.count_nonzero(banks["gate_up_bias"][0])


def test_loader_rejects_missing_and_unexpected_expert_tensors(tmp_path, monkeypatch):
    monkeypatch.setattr(weight, "get_tp_info", _tp1)
    bad = {
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight": torch.zeros(
            1, dtype=torch.uint8
        )
    }
    save_file(bad, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="unexpected Kimi-K3 expert tensor"):
        weight.load_mxfp4_expert_banks(str(tmp_path), _config(), dtype=torch.bfloat16)

    bad = {"language_model.model.norm.weight": torch.zeros(1)}
    save_file(bad, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="missing Kimi-K3 expert tensors"):
        weight.load_mxfp4_expert_banks(str(tmp_path), _config(), dtype=torch.bfloat16)
