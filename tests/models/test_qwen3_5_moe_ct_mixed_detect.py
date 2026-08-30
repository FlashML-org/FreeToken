"""Detection of compressed-tensors mixed-precision checkpoints (NVFP4 experts +
128x128 block-FP8 dense side), e.g. kyaky/Qwen3.6-35B-A3B-Uncensored-NVFP4.

These are distinct from pure compressed-tensors NVFP4 checkpoints (where the
dense side is packed FP4 as well): they use llm-compressor config_groups with an
8-bit block strategy for attn/shared-expert and a 4-bit group_size=16
tensor_group strategy for the routed experts."""

from __future__ import annotations

from types import SimpleNamespace

from freetoken.models.qwen3_5_moe.config import _compressed_tensors_mixed_fp8block


def _hf(qc):
    return SimpleNamespace(quantization_config=qc)


_CT_MIXED = {
    "quant_method": "compressed-tensors",
    "format": "mixed-precision",
    "config_groups": {
        "group_0": {
            "weights": {"num_bits": 8, "strategy": "block", "block_structure": [128, 128]},
            "targets": ["re:.*shared_expert.*", "re:.*self_attn.*"],
        },
        "group_1": {
            "weights": {"num_bits": 4, "strategy": "tensor_group", "group_size": 16},
            "targets": ["re:.*mlp.experts.*"],
        },
    },
}

_CT_PURE_NVFP4 = {
    "quant_method": "compressed-tensors",
    "format": "nvfp4-pack-quantized",
    "config_groups": {
        "group_0": {
            "weights": {"num_bits": 4, "strategy": "tensor_group", "group_size": 16},
            "targets": ["re:.*"],
        },
    },
}

_CT_FP8_ONLY = {
    "quant_method": "compressed-tensors",
    "config_groups": {
        "group_0": {
            "weights": {"num_bits": 8, "strategy": "group", "group_size": 128},
            "targets": ["re:.*"],
        },
    },
}

_MODELOPT_MIXED = {
    "quant_algo": "MIXED_PRECISION",
    "quantized_layers": {
        "model.layers.0.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16},
        "model.layers.0.mlp.shared_expert.gate_proj": {"quant_algo": "FP8"},
    },
}


def test_ct_mixed_fp8block_detected():
    assert _compressed_tensors_mixed_fp8block(_hf(_CT_MIXED)) is True


def test_pure_ct_nvfp4_not_flagged():
    assert _compressed_tensors_mixed_fp8block(_hf(_CT_PURE_NVFP4)) is False


def test_ct_fp8_only_not_flagged():
    assert _compressed_tensors_mixed_fp8block(_hf(_CT_FP8_ONLY)) is False


def test_modelopt_mixed_not_matched():
    # modelopt MIXED_PRECISION (e.g. Apodex) has no config_groups -> handled elsewhere
    assert _compressed_tensors_mixed_fp8block(_hf(_MODELOPT_MIXED)) is False


def test_no_quant_config():
    assert _compressed_tensors_mixed_fp8block(SimpleNamespace()) is False
