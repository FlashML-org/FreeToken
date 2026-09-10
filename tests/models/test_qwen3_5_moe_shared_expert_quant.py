"""Shared-expert quant detection for MoE checkpoints (issue #183).

modelopt MIXED_PRECISION checkpoints can quantize the routed experts and the
shared expert independently: Apodex-1.1-mini-NVFP4 ships NVFP4 experts with a
per-tensor FP8 shared expert. The old code assumed the shared expert matched
the experts (``dense_quant = "nvfp4"`` whenever experts are NVFP4), built an
``Nvfp4DenseColMerged`` module, and crashed at load with
``KeyError: 'model.layers.0.mlp.shared_expert.gate_up_proj.weight'`` because an
FP8 shared expert has no ``weight_scale_2`` / ``weight_global`` buffers.
"""

from __future__ import annotations

from types import SimpleNamespace

from freetoken.models.qwen3_5_moe.config import _shared_expert_quant


def _hf(quantization_config):
    return SimpleNamespace(quantization_config=quantization_config)


def test_no_quant_config_defaults_to_nvfp4():
    # Pure NVFP4 checkpoints ship without a quant_config accessor; the shared
    # expert is packed FP4 like the experts and stays native W4A16.
    assert _shared_expert_quant(SimpleNamespace()) == "nvfp4"


def test_modelopt_nvfp4_without_layer_map_defaults_to_nvfp4():
    cfg = _hf({"quant_algo": "NVFP4"})
    assert _shared_expert_quant(cfg) == "nvfp4"


def test_modelopt_mixed_fp8_shared_expert_is_dequantized():
    # Apodex-1.1-mini-NVFP4 layout: NVFP4 routed experts, per-tensor FP8
    # shared expert tagged explicitly in quantized_layers. Must NOT take the
    # native W4A16 path (that module needs weight_scale_2 / weight_global).
    cfg = _hf(
        {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.language_model.layers.0.mlp.experts": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                },
                "model.language_model.layers.0.mlp.shared_expert.gate_proj": {
                    "quant_algo": "FP8"
                },
                "model.language_model.layers.0.mlp.shared_expert.up_proj": {
                    "quant_algo": "FP8"
                },
                "model.language_model.layers.0.mlp.shared_expert.down_proj": {
                    "quant_algo": "FP8"
                },
            },
        }
    )
    assert _shared_expert_quant(cfg) == "none"


def test_modelopt_mixed_nvfp4_shared_expert_stays_native():
    cfg = _hf(
        {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.language_model.layers.0.mlp.shared_expert.gate_proj": {
                    "quant_algo": "W4A16_NVFP4"
                },
            },
        }
    )
    assert _shared_expert_quant(cfg) == "nvfp4"


def test_mixed_map_without_shared_expert_entry_defaults_to_nvfp4():
    # Per-layer map present but no shared_expert tag: keep the family default
    # (MoE NVFP4 checkpoints keep the shared expert native FP4).
    cfg = _hf(
        {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.language_model.layers.0.self_attn.q_proj": {
                    "quant_algo": "FP8"
                },
            },
        }
    )
    assert _shared_expert_quant(cfg) == "nvfp4"
