from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.layers.quantization import set_quant_config
from freetoken.layers.quantization.configs.compressed_tensors import CompressedTensorsConfig
from freetoken.layers.quantization.configs.modelopt import ModelOptConfig
from freetoken.models.gemma4.weight import iter_weights, nvfp4_expert_spec


@pytest.fixture(scope="session", autouse=True)
def _tp_info():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _nvfp4(base: str, rows: int, cols: int, global_scale: float = 2.0):
    return {
        base + ".weight_packed": torch.randint(
            0, 256, (rows, cols // 2), dtype=torch.uint8
        ),
        base + ".weight_scale": torch.ones(
            rows, cols // 16, dtype=torch.float8_e4m3fn
        ),
        base + ".weight_global_scale": torch.tensor([global_scale]),
        base + ".input_global_scale": torch.tensor([2.0]),
    }


def test_mixed_compressed_tensors_dense_weights(tmp_path, monkeypatch):
    """Gemma 4 Unsloth exports mix FP8 attention with compressed-tensors
    NVFP4 shared MLPs; both roles must land in the buffers the model builds."""
    h, q, kv, intermediate = 16, 8, 4, 32
    layer = "model.language_model.layers.0"
    attn = layer + ".self_attn"
    mlp = layer + ".mlp"
    raw = {
        attn + ".q_proj.weight": torch.ones(q, h, dtype=torch.float8_e4m3fn),
        attn + ".q_proj.weight_scale": torch.full((q, 1), 2.0, dtype=torch.bfloat16),
        attn + ".k_proj.weight": torch.ones(kv, h, dtype=torch.float8_e4m3fn),
        attn + ".k_proj.weight_scale": torch.full((kv, 1), 3.0, dtype=torch.bfloat16),
        attn + ".v_proj.weight": torch.ones(kv, h, dtype=torch.float8_e4m3fn),
        attn + ".v_proj.weight_scale": torch.full((kv, 1), 4.0, dtype=torch.bfloat16),
        attn + ".o_proj.weight": torch.ones(h, q, dtype=torch.float8_e4m3fn),
        attn + ".o_proj.weight_scale": torch.full((h, 1), 5.0, dtype=torch.bfloat16),
        attn + ".k_scale": torch.tensor(1.0),
        attn + ".v_scale": torch.tensor(1.0),
    }
    raw |= _nvfp4(mlp + ".gate_proj", intermediate, h)
    raw |= _nvfp4(mlp + ".up_proj", intermediate, h)
    raw |= _nvfp4(mlp + ".down_proj", h, intermediate, global_scale=4.0)
    save_file(raw, tmp_path / "model.safetensors")

    import freetoken.models.gemma4.weight as weight

    config = SimpleNamespace(
        num_layers=1,
        dense_quant="none",
        attention_group_for_layer=lambda _layer: None,
    )
    monkeypatch.setattr(weight, "cached_load_hf_config", lambda _path: object())
    monkeypatch.setattr(weight, "parse_config", lambda _hf: config)

    loaded = dict(
        iter_weights(
            str(tmp_path),
            torch.device("cpu"),
            include_moe_experts=False,
            include_non_moe=True,
            include_vision=False,
        )
    )

    qkv = "model.layers.0.self_attn.qkv_proj"
    assert loaded[qkv + ".weight"].shape == (q + kv + kv, h)
    assert loaded[qkv + ".weight_scale"].shape == (q + kv + kv,)
    assert loaded[qkv + ".weight_scale"].dtype is torch.float32
    assert torch.equal(
        loaded[qkv + ".weight_scale"],
        torch.tensor([2.0] * q + [3.0] * kv + [4.0] * kv),
    )
    assert loaded["model.layers.0.self_attn.o_proj.weight_scale"].shape == (h,)
    assert not any(name.endswith((".k_scale", ".v_scale")) for name in loaded)

    gate_up = "model.layers.0.feed_forward.shared_mlp.gate_up_proj"
    assert loaded[gate_up + ".weight"].shape == (2 * intermediate, h // 2)
    assert loaded[gate_up + ".weight_scale"].shape == (2 * intermediate, h // 16)
    assert loaded[gate_up + ".weight_global"].shape == (2 * intermediate,)
    assert loaded[gate_up + ".weight_global"][0].item() == pytest.approx(0.5)
    assert loaded[gate_up + ".input_scale"].item() == pytest.approx(0.5)

    down = "model.layers.0.feed_forward.shared_mlp.down_proj"
    assert loaded[down + ".weight"].shape == (h, intermediate // 2)
    assert loaded[down + ".weight_scale"].shape == (h, intermediate // 16)
    assert loaded[down + ".weight_global"][0].item() == pytest.approx(0.25)
    assert not any("weight_packed" in name or "global_scale" in name for name in loaded)


def test_nvfp4_expert_spec_uses_checkpoint_dialect():
    ct = CompressedTensorsConfig(
        {
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "strategy": "tensor_group",
                        "group_size": 16,
                    },
                }
            }
        }
    )
    set_quant_config(ct)
    spec = nvfp4_expert_spec("unused", object())
    match = spec.key_pattern.match(
        "model.language_model.layers.2.experts.7.gate_proj.weight_packed"
    )
    assert match and match.group("kind") == "weight_packed"
    assert spec.kind_map == {
        "weight_packed": "weight",
        "weight_scale": "weight_scale",
        "weight_global_scale": "weight_scale_2",
    }
    assert spec.global_reciprocal
    assert spec.key_pattern.match(
        "model.language_model.layers.2.experts.7.gate_proj.input_global_scale"
    ) is None

    set_quant_config(ModelOptConfig({"quant_algo": "NVFP4"}))
    modelopt = nvfp4_expert_spec("unused", object())
    assert modelopt.kind_map is None
    assert not modelopt.global_reciprocal
    assert modelopt.key_pattern.match(
        "model.language_model.layers.2.experts.7.gate_proj.weight"
    )
