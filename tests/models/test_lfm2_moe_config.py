"""lfm2_moe.parse_config against the released checkpoint configs.

Runs off the real config.json contents (LFM2.5-24B-A2B-FP8, verbatim) via RawConfigShim, so
the parse path under test is the one production hits; the bf16 export is the same dict minus
the keys that release adds (``tie_word_embeddings`` / the quantization block).
"""

from __future__ import annotations

import pytest

from freetoken.attention.base import AttnType
from freetoken.models.config import FullAttentionGroupConfig, ShortConvGroupConfig
from freetoken.models.lfm2_moe.config import parse_config
from freetoken.utils.hf import RawConfigShim

_LAYER_TYPES = ["conv", "conv", "full_attention"] + ["conv", "conv", "conv", "full_attention"] * 9 + ["conv"]

# verbatim from LiquidAI/LFM2.5-24B-A2B-FP8/config.json (layer_types expanded above)
_FP8_CONFIG = {
    "architectures": ["Lfm2MoeForCausalLM"],
    "bos_token_id": 124894,
    "conv_L_cache": 3,
    "conv_bias": False,
    "dtype": "bfloat16",
    "eos_token_id": 124900,
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 11776,
    "max_position_embeddings": 131072,
    "model_type": "lfm2_moe",
    "moe_intermediate_size": 1536,
    "norm_eps": 1e-05,
    "norm_topk_prob": True,
    "num_attention_heads": 32,
    "num_dense_layers": 2,
    "num_experts": 64,
    "num_experts_per_tok": 4,
    "num_hidden_layers": 40,
    "num_key_value_heads": 8,
    "pad_token_id": 124893,
    "quantization_config": {
        "config_groups": {
            "FP8_BLOCK": {
                "format": "float-quantized",
                "input_activations": {
                    "actorder": None, "block_structure": None, "dynamic": True,
                    "group_size": 128, "num_bits": 8, "observer": None, "observer_kwargs": {},
                    "scale_dtype": None, "strategy": "group", "symmetric": True, "type": "float",
                    "zp_dtype": None,
                },
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None, "block_structure": [128, 128], "dynamic": False,
                    "group_size": None, "num_bits": 8, "observer": None, "observer_kwargs": {},
                    "scale_dtype": None, "strategy": "block", "symmetric": True, "type": "float",
                    "zp_dtype": None,
                },
            }
        },
        "format": "float-quantized",
        "global_compression_ratio": None,
        "ignore": ["lm_head", "model.embed_tokens", "re:.*\\.gate$", "re:.*\\.(?:conv|short_conv)\\..*"],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "transform_config": {},
        "version": "0.19.0",
    },
    "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
    "routed_scaling_factor": 1.0,
    "tie_word_embeddings": True,
    "transformers_version": "5.12.1",
    "use_cache": True,
    "use_expert_bias": True,
    "vocab_size": 128000,
}


def _config(**overrides) -> RawConfigShim:
    data = {**_FP8_CONFIG, "layer_types": list(_LAYER_TYPES), **overrides}
    return RawConfigShim(data)


def test_group_partition_and_state_layers():
    cfg = parse_config(_config())
    conv = cfg.short_conv_group()
    full = next(g for g in cfg.attention_groups if isinstance(g, FullAttentionGroupConfig))
    assert isinstance(conv, ShortConvGroupConfig) and isinstance(full, FullAttentionGroupConfig)
    assert conv.name == "conv" and full.name == "full"
    assert conv.layer_ids == tuple(i for i, t in enumerate(_LAYER_TYPES) if t == "conv")
    assert full.layer_ids == tuple(i for i, t in enumerate(_LAYER_TYPES) if t == "full_attention")
    assert (len(conv.layer_ids), len(full.layer_ids)) == (30, 10)
    assert (conv.layer_ids[0], full.layer_ids[0]) == (0, 2)  # groups ordered by first layer id
    assert cfg.is_short_conv_layer(0) and not cfg.is_short_conv_layer(2)
    assert [cfg.attn_type_for_layer(i) for i in (0, 2)] == [AttnType.LINEAR, AttnType.FULL]
    # the conv layers are state-carrying but not recurrent: the pool serves them, the GDN
    # metadata consumers must not be told there is a recurrence
    assert cfg.has_state_layers and not cfg.has_linear_attention
    assert cfg.state_group() is conv
    # one paged-KV spec, for the full-attention group alone (the paged pool remaps to it)
    (spec,) = cfg.kv_cache_group_specs()
    assert (spec.name, spec.attn_type, spec.layer_ids) == ("full", AttnType.FULL, full.layer_ids)


def test_geometry_experts_and_rope():
    cfg = parse_config(_config())
    assert cfg.num_layers == 40
    assert (cfg.num_qo_heads, cfg.num_kv_heads, cfg.head_dim) == (32, 8, 64)
    assert cfg.hidden_size == 2048 and cfg.intermediate_size == 11776
    assert cfg.rms_norm_eps == 1e-5
    assert cfg.hidden_act == "silu"
    assert (cfg.num_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size) == (64, 4, 1536)
    assert cfg.first_k_dense_replace == 2  # HF: layer_idx < num_dense_layers
    assert cfg.num_moe_layers == 38
    assert cfg.norm_topk_prob and cfg.use_expert_bias
    assert cfg.routed_scaling_factor == 1.0
    assert cfg.tie_word_embeddings
    assert cfg.rotary_config.base == 10000000.0  # the checkpoint's rope_parameters
    assert cfg.rotary_config.scaling is None  # rope_type "default"
    assert cfg.rotary_config.rotary_dim == cfg.head_dim  # full-width, no partial rope


def test_expert_quant_tag_and_scale_block():
    """The FP8 export's block-fp8 experts, as the MoE kernels' format tag + scale block."""
    cfg = parse_config(_config())
    assert cfg.expert_quant == "fp8_block"
    assert cfg.weight_block_size == (128, 128)
    # the quantization_config does not cover the routers or the conv projections
    assert parse_config(_config(quantization_config=None)).expert_quant == "none"
    assert parse_config(_config(quantization_config=None)).weight_block_size is None


def test_tied_head_defaults_to_true_when_the_export_omits_the_key():
    """LFM2-8B-A1B ships no ``tie_word_embeddings`` and no ``lm_head`` tensor: the class default is tied."""
    data = {k: v for k, v in _FP8_CONFIG.items() if k != "tie_word_embeddings"}
    data["layer_types"] = list(_LAYER_TYPES)
    assert parse_config(RawConfigShim(data)).tie_word_embeddings
    assert not parse_config(_config(tie_word_embeddings=False)).tie_word_embeddings


def test_an_unknown_layer_type_is_refused():
    with pytest.raises(ValueError, match="unsupported lfm2_moe layer type 'sliding_attention'"):
        parse_config(_config(layer_types=["conv", "sliding_attention", "full_attention"]))


def test_a_conv_only_stack_is_refused():
    # the model's paged-KV spec comes from the full_attention layers: without one the pool
    # factory has no geometry to size
    with pytest.raises(ValueError, match="at least one full_attention layer"):
        parse_config(_config(layer_types=["conv", "conv", "conv"]))
