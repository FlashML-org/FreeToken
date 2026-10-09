"""mimo_v2 (MiMo v2.6 Flash) config parsing: hybrid groups, dual rope, MoE scalars.

Runs off a trimmed copy of the real checkpoint config.json via RawConfigShim (the
exact object cached_load_hf_config falls back to when transformers doesn't know
``mimo_v2`` yet), so the parse path under test is the one production hits.
"""

from __future__ import annotations

import pytest

from freetoken.attention.base import AttnType
from freetoken.models.config import (
    FullAttentionGroupConfig,
    SWAAttentionGroupConfig,
)
from freetoken.models.mimo_v2.config import parse_config
from freetoken.utils.hf import RawConfigShim

_NUM_LAYERS = 48
_FULL_IDS = (0, 5, 11, 17, 23, 29, 35, 41, 47)
_SWA_IDS = tuple(i for i in range(_NUM_LAYERS) if i not in _FULL_IDS)


def _hf_config(quantization_config: dict | None = None) -> RawConfigShim:
    # Trimmed from the MiMo v2.6 Flash config.json (reference/config_summary.txt).
    data: dict = {
        "architectures": ["MiMoV2ForCausalLM"],
        "model_type": "mimo_v2",
        "hidden_size": 4096,
        "intermediate_size": 16384,
        "num_hidden_layers": _NUM_LAYERS,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "swa_num_attention_heads": 64,
        "swa_num_key_value_heads": 8,
        "head_dim": 192,
        "swa_head_dim": 192,
        "v_head_dim": 128,
        "swa_v_head_dim": 128,
        "vocab_size": 152576,
        "hidden_act": "silu",
        "layernorm_epsilon": 1e-6,
        "max_position_embeddings": 1048576,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "attention_projection_layout": "fused_qkv",
        "attention_value_scale": 0.707,
        "attention_chunk_size": 128,
        "sliding_window": 128,
        "sliding_window_size": 128,
        "hybrid_layer_pattern": [0 if i in _FULL_IDS else 1 for i in range(_NUM_LAYERS)],
        "moe_layer_freq": [0] + [1] * (_NUM_LAYERS - 1),
        "rope_theta": 10000000.0,
        "swa_rope_theta": 10000.0,
        "partial_rotary_factor": 0.334,
        "rope_parameters": {
            "partial_rotary_factor": 0.334,
            "rope_theta": 10000000.0,
            "rope_type": "default",
            "type": "default",
        },
        "add_swa_attention_sink_bias": True,
        "add_full_attention_sink_bias": False,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "n_shared_experts": None,
        "moe_intermediate_size": 2048,
        "norm_topk_prob": True,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "n_group": 1,
        "topk_group": 1,
        "moe_router_dtype": "bfloat16",
        "routed_scaling_factor": None,
        "num_nextn_predict_layers": 3,
    }
    if quantization_config is not None:
        data["quantization_config"] = quantization_config
    return RawConfigShim(data)


_MODELOPT_MIXED_EXPERTS = {
    # The shipped release: modelopt MIXED_PRECISION listing ONLY the expert projections.
    "quant_method": "modelopt",
    "quant_algo": "MIXED_PRECISION",
    "group_size": 16,
    "kv_cache_quant_algo": None,
    "exclude_modules": [],
    "quantized_layers": {
        f"model.layers.{layer}.mlp.experts.{expert}.{proj}": {
            "quant_algo": "W4A16_NVFP4", "group_size": 16
        }
        for layer in range(1, _NUM_LAYERS)
        for expert in range(256)
        for proj in ("gate_proj", "up_proj", "down_proj")
    },
}


def test_attention_groups_split_by_hybrid_layer_pattern():
    cfg = parse_config(_hf_config())
    assert cfg.num_layers == _NUM_LAYERS
    full, swa = cfg.attention_groups  # layer 0 is full => full group lands first

    assert isinstance(full, FullAttentionGroupConfig)
    assert full.layer_ids == _FULL_IDS
    assert (full.num_kv_heads, full.head_dim) == (4, 192)
    assert full.rotary_config.base == 1e7

    assert isinstance(swa, SWAAttentionGroupConfig)
    assert swa.layer_ids == _SWA_IDS and len(swa.layer_ids) == 39
    assert (swa.num_kv_heads, swa.head_dim) == (8, 192)
    assert swa.sliding_window == 128  # EXACTLY 128 (engine mask k+W > q == HF inclusive)
    assert swa.rotary_config.base == 1e4

    assert cfg.has_swa_attention and cfg.has_hybrid_attention
    assert not cfg.is_swa_layer(0) and cfg.is_swa_layer(1)
    assert cfg.attn_type_for_layer(5) == AttnType.FULL
    assert cfg.attn_type_for_layer(1) == AttnType.SWA


def test_kv_cache_groups_are_192_wide_for_both_tiers():
    cfg = parse_config(_hf_config())
    assert cfg.kv_cache_groups() == [(len(_FULL_IDS), 4, 192), (len(_SWA_IDS), 8, 192)]
    specs = cfg.kv_cache_group_specs()
    assert [s.head_dim for s in specs] == [192, 192]  # V 128 is padded to 192 by the family


def test_dual_partial_rope():
    cfg = parse_config(_hf_config())
    full, swa = cfg.attention_groups
    # HF: rope_dim = int(head_dim * partial_rotary_factor) = int(192 * 0.334) = 64 (even)
    assert full.rotary_config.rotary_dim == 64
    assert swa.rotary_config.rotary_dim == 64
    assert full.rotary_config.head_dim == swa.rotary_config.head_dim == 192
    # rope_type "default" => no scaling (no yarn/proportional math anywhere)
    assert full.rotary_config.scaling is None and swa.rotary_config.scaling is None
    assert cfg.rotary_config is full.rotary_config


def test_moe_scalars_match_the_router_contract():
    cfg = parse_config(_hf_config(_MODELOPT_MIXED_EXPERTS))
    assert cfg.expert_quant == "nvfp4"
    assert cfg.num_experts == 256 and cfg.num_experts_per_tok == 8
    assert cfg.moe_intermediate_size == 2048 and cfg.intermediate_size == 16384
    assert cfg.first_k_dense_replace == 1  # moe_layer_freq = [0, 1 x 47]
    assert cfg.num_moe_layers == 47
    assert cfg.n_shared_experts == 0
    assert cfg.norm_topk_prob is True
    assert cfg.routed_scaling_factor == 1.0  # None => HF default 1.0
    assert (cfg.n_group, cfg.topk_group) == (1, 1)
    assert cfg.is_moe and not cfg.has_attn_bias
    assert cfg.attn_sm_scale == pytest.approx(192**-0.5)  # DEC-6: QK dim, not v_head_dim
    assert cfg.attention_value_scale == pytest.approx(0.707)
    assert cfg.rms_norm_eps == 1e-6
    assert cfg.vocab_size == 152576 and cfg.tie_word_embeddings is False
    assert cfg.hidden_act == "silu"


def test_expert_quant_without_a_quant_config():
    assert parse_config(_hf_config()).expert_quant == "none"


def test_value_scale_none_is_identity_default():
    data = _hf_config().to_dict()
    data.pop("attention_value_scale")
    assert parse_config(RawConfigShim(data)).attention_value_scale is None


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"moe_layer_freq": [0, 1, 0] + [1] * 45}, "contiguous sparse MoE suffix"),
        ({"add_full_attention_sink_bias": True}, "SWA layers only"),
        ({"swa_v_head_dim": 193}, "v_head_dim <= head_dim"),
        ({"swa_num_attention_heads": 32}, "same positive query-head count"),
        ({"sliding_window_size": 64}, "matching positive sliding_window"),
        ({"hidden_act": "gelu"}, "hidden_act='silu' only"),
    ],
)
def test_unsupported_topology_is_rejected_early(update, message):
    data = _hf_config().to_dict()
    data.update(update)
    with pytest.raises(ValueError, match=message):
        parse_config(RawConfigShim(data))


# Admission must reject group geometry that moe.py's _group_limited cannot execute:
# width-1 groups crash topk(2); eligible experts < top_k make a masked-out id win.
@pytest.mark.parametrize(
    ("update", "message"),
    [
        (
            {"n_routed_experts": 4, "n_group": 4, "topk_group": 1, "num_experts_per_tok": 1},
            "group width=1",
        ),
        (
            {"n_routed_experts": 8, "n_group": 4, "topk_group": 1, "num_experts_per_tok": 3},
            "num_experts_per_tok=3",
        ),
    ],
)
def test_unsupported_grouped_routing_is_rejected(update, message):
    data = _hf_config().to_dict()
    data.update(update)
    with pytest.raises(ValueError, match=message):
        parse_config(RawConfigShim(data))


def test_valid_grouped_routing_geometry_parses():
    data = _hf_config().to_dict()
    data.update({"n_routed_experts": 8, "n_group": 2, "topk_group": 1, "num_experts_per_tok": 2})
    cfg = parse_config(RawConfigShim(data))
    assert (cfg.n_group, cfg.topk_group, cfg.num_experts, cfg.num_experts_per_tok) == (2, 1, 8, 2)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"group_size": 32}, "group_size=16"),
        (
            {
                "quantized_layers": {
                    **_MODELOPT_MIXED_EXPERTS["quantized_layers"],
                    "model.layers.0.mlp.gate_proj": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
                }
            },
            "every routed expert projection",
        ),
        (
            {
                "quantized_layers": {
                    key: {"quant_algo": "NVFP4", "group_size": 16}
                    for key in _MODELOPT_MIXED_EXPERTS["quantized_layers"]
                }
            },
            "W4A16_NVFP4 group_size=16",
        ),
    ],
)
def test_modelopt_quantization_admission_is_explicit(update, message):
    quant = {**_MODELOPT_MIXED_EXPERTS, **update}
    with pytest.raises(ValueError, match=message):
        parse_config(_hf_config(quant))


def test_model_construction_rejects_tp_greater_than_one_before_building_layers(monkeypatch):
    from types import SimpleNamespace

    import freetoken.models.mimo_v2.model as mimo_model

    monkeypatch.setattr(mimo_model, "get_tp_info", lambda: SimpleNamespace(size=2))
    with pytest.raises(ValueError, match="tensor parallel size 1 only; got TP=2"):
        mimo_model.MiMoV2ForCausalLM(None)


def test_registry_and_aot_claim_mimo():
    from freetoken.kernel.aot_models import SUPPORTED_MODELS
    from freetoken.models.register import get_model_spec

    spec = get_model_spec("MiMoV2ForCausalLM")
    assert spec.module == "freetoken.models.mimo_v2"
    assert spec.model_cls == "MiMoV2ForCausalLM"

    entry = next(m for m in SUPPORTED_MODELS if m.architecture == "MiMoV2ForCausalLM")
    assert entry.hidden_size == 4096
    assert entry.kv_groups == ((4, 192), (8, 192))
    assert entry.top_k == 8
    assert entry.moe_intermediate_size == 2048
    assert "nvfp4" in entry.expert_formats
