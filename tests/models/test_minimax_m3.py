"""MiniMax-M3 config parsing, payload semantics and serving-mode resolution.

Drives ``minimax_m3.parse_config`` off a synthetic HF config shaped exactly like the
nvidia/MiniMax-M3-NVFP4 checkpoint's (multimodal wrapper: text tower in
``text_config``, quantization_config present) and checks every decision the plan
pinned: the BSA attention group (pool family + KV cost), the sparse indexer
geometry, the MoE/router knobs, the resolved quant modes and the env switches.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.attention.base import AttnType
from freetoken.models.minimax_m3.config import parse_config


class _Cfg:
    """Attribute-access shim over a dict (what AutoConfig hands parse_config)."""

    def __init__(self, data: dict):
        for k, v in data.items():
            setattr(self, k, _Cfg(v) if isinstance(v, dict) and k == "text_config" else v)


def _hf_config(num_layers: int = 60) -> _Cfg:
    dense = min(3, num_layers)
    moe_freq = [0] * dense + [1] * (num_layers - dense)
    return _Cfg(
        {
            "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
            "model_type": "minimax_m3_vl",
            "quantization_config": {"quant_algo": "MIXED_PRECISION", "quant_method": "modelopt"},
            "text_config": {
                "hidden_size": 6144,
                "intermediate_size": 3072,
                "dense_intermediate_size": 12288,
                "shared_intermediate_size": 3072,
                "num_hidden_layers": num_layers,
                "num_attention_heads": 64,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "vocab_size": 200064,
                "max_position_embeddings": 1048576,
                "rms_norm_eps": 1e-6,
                "use_gemma_norm": True,
                "attention_output_gate": False,
                "rope_theta": 5000000,
                "rotary_dim": 64,
                "partial_rotary_factor": 0.5,
                "hidden_act": "swigluoai",
                "swiglu_alpha": 1.702,
                "swiglu_limit": 7.0,
                "use_qk_norm": True,
                "qk_norm_type": "per_head",
                "num_local_experts": 128,
                "num_experts_per_tok": 4,
                "n_shared_experts": 1,
                "scoring_func": "sigmoid",
                "use_routing_bias": True,
                "routed_scaling_factor": 2.0,
                "moe_layer_freq": moe_freq,
                "tie_word_embeddings": False,
                "sparse_attention_config": {
                    "use_sparse_attention": True,
                    "sparse_index_dim": 128,
                    "sparse_num_index_heads": 4,
                    "sparse_topk_blocks": 16,
                    "sparse_block_size": 128,
                    "sparse_disable_index_value": moe_freq,
                    "sparse_score_type": "max",
                    "sparse_init_block": 0,
                    "sparse_local_block": 1,
                    "sparse_attention_freq": moe_freq,
                },
            },
        }
    )


def _hf_config_native(num_layers: int = 60) -> _Cfg:
    """The transformers >= 5 NATIVE minimax_m3_vl TextConfig shape: __post_init__
    pops rope_theta into rope_parameters, sparse_attention_config/moe_layer_freq
    into flat index_* keys + layer_types/mlp_layer_types, and force-sets
    hidden_act to "silu". parse_config must resolve both shapes identically
    (verified empirically against transformers 5.14's class)."""
    cfg = _hf_config(num_layers)
    text = cfg.text_config
    dense = min(3, num_layers)
    delattr(text, "rope_theta")
    text.rope_parameters = {
        "rope_theta": 5000000,
        "partial_rotary_factor": 0.5,
        "rope_type": "default",
    }
    text.hidden_act = "silu"
    delattr(text, "sparse_attention_config")
    text.layer_types = ["full_attention"] * dense + ["minimax_m3_sparse"] * (
        num_layers - dense
    )
    text.index_n_heads = 4
    text.index_head_dim = 128
    text.index_block_size = 128
    text.index_topk_blocks = 16
    text.index_local_blocks = 1
    delattr(text, "moe_layer_freq")
    text.mlp_layer_types = ["dense"] * dense + ["sparse"] * (num_layers - dense)
    return cfg


def test_parse_config_native_shape_matches_raw(monkeypatch):
    """The native shape silently produced rope base 10000, use_sparse=False and
    hidden_act='silu' before the dual-shape shim -- all three load-bearing reads
    must resolve identically to the raw-dict shape."""
    monkeypatch.delenv("FREETOKEN_M3_MAX_LAYERS", raising=False)
    monkeypatch.delenv("FREETOKEN_M3_SPARSE", raising=False)
    raw = parse_config(_hf_config())
    native = parse_config(_hf_config_native())

    assert native.rotary_config.base == raw.rotary_config.base == 5000000.0
    assert native.hidden_act == raw.hidden_act == "swigluoai"
    assert native.first_k_dense_replace == raw.first_k_dense_replace == 3
    a, b = native.m3_args, raw.m3_args
    assert a.use_sparse and a.sparse_layer_ids == b.sparse_layer_ids
    assert a.moe_layer_ids == b.moe_layer_ids
    assert (a.index_dim, a.num_index_heads, a.topk_blocks, a.block_size) == (
        b.index_dim, b.num_index_heads, b.topk_blocks, b.block_size,
    )
    assert (a.init_blocks, a.local_blocks) == (b.init_blocks, b.local_blocks)
    assert (a.rope_theta, a.rotary_dim) == (b.rope_theta, b.rotary_dim)


def test_parse_config_full_model(monkeypatch):
    monkeypatch.delenv("FREETOKEN_M3_MAX_LAYERS", raising=False)
    monkeypatch.delenv("FREETOKEN_M3_SPARSE", raising=False)
    cfg = parse_config(_hf_config())

    assert cfg.num_layers == 60
    assert (cfg.num_qo_heads, cfg.num_kv_heads, cfg.head_dim) == (64, 4, 128)
    assert cfg.rotary_config.rotary_dim == 64 and cfg.rotary_config.base == 5000000.0
    assert cfg.first_k_dense_replace == 3 and cfg.num_moe_layers == 57
    assert (cfg.num_experts, cfg.num_experts_per_tok) == (128, 4)
    assert cfg.moe_intermediate_size == 3072 and cfg.intermediate_size == 12288
    assert cfg.n_shared_experts == 1 and cfg.shared_expert_intermediate_size == 3072
    assert cfg.routed_scaling_factor == 2.0 and cfg.norm_topk_prob
    assert cfg.hidden_act == "swigluoai"
    assert cfg.hidden_act_alpha == pytest.approx(1.702)
    assert cfg.swiglu_limit == pytest.approx(7.0)
    assert cfg.expert_quant == "nvfp4"
    # Default env: MXFP8 dense native, bf16 lm_head.
    assert cfg.attn_quant == "mxfp8" and cfg.dense_quant == "mxfp8"
    assert cfg.lm_head_quant == "none"

    # One BSA group over all layers, index slab for the 57 sparse layers.
    (spec,) = cfg.kv_cache_group_specs()
    assert spec.attn_type == AttnType.BSA
    assert not spec.mla
    assert spec.index_head_dim == 128 and spec.num_index_layers == 57

    args = cfg.m3_args
    assert args.use_sparse and args.topk_blocks == 16 and args.block_size == 128
    assert (args.init_blocks, args.local_blocks) == (0, 1)
    assert args.num_index_heads == 4 and args.index_dim == 128
    assert args.sparse_layer_ids == tuple(range(3, 60))
    assert args.moe_layer_ids == tuple(range(3, 60))
    assert args.sparse_slot(3) == 0 and args.sparse_slot(59) == 56


def test_pool_family_and_kv_cost():
    cfg = parse_config(_hf_config())
    from freetoken.kvcache import resolve_pool_class
    from freetoken.kvcache.base import spec_kv_bytes_per_token
    from freetoken.kvcache.bsa_pool import BSAKVCache

    assert resolve_pool_class(cfg) is BSAKVCache

    class _Budget:
        model_config = cfg
        dtype = torch.bfloat16
        page_size = 128

        class tp_info:
            size = 1

    (spec,) = cfg.kv_cache_group_specs()
    # 60 layers x 2 slabs x 4 heads x 128 dims x 2 B + 57 index layers x 128 x 2 B
    assert spec_kv_bytes_per_token(spec, _Budget) == 60 * 2 * 4 * 128 * 2 + 57 * 128 * 2


def test_layer_cap_env(monkeypatch):
    monkeypatch.setenv("FREETOKEN_M3_MAX_LAYERS", "5")
    cfg = parse_config(_hf_config())
    assert cfg.num_layers == 5
    assert cfg.first_k_dense_replace == 3 and cfg.num_moe_layers == 2
    (spec,) = cfg.kv_cache_group_specs()
    assert spec.num_index_layers == 2
    assert cfg.m3_args.sparse_layer_ids[:2] == (3, 4)


def test_sparse_ablation_env(monkeypatch):
    monkeypatch.setenv("FREETOKEN_M3_SPARSE", "0")
    cfg = parse_config(_hf_config())
    (spec,) = cfg.kv_cache_group_specs()
    assert spec.attn_type == AttnType.FULL
    assert spec.index_head_dim == 0 and spec.num_index_layers == 0
    assert not cfg.m3_args.use_sparse

    from freetoken.kvcache import resolve_pool_class
    from freetoken.kvcache.mha_pool import MHAKVCache

    assert resolve_pool_class(cfg) is MHAKVCache


def test_mxfp8_ablation_env(monkeypatch):
    monkeypatch.setenv("FREETOKEN_M3_ATTN_MXFP8", "0")
    monkeypatch.setenv("FREETOKEN_M3_MLP_MXFP8", "0")
    cfg = parse_config(_hf_config())
    assert cfg.attn_quant == "none" and cfg.dense_quant == "none"


def test_registry_resolves_both_architectures():
    from freetoken.models.register import get_model_spec

    for arch in ("MiniMaxM3SparseForConditionalGeneration", "MiniMaxM3SparseForCausalLM"):
        spec = get_model_spec(arch)
        assert spec.module == "freetoken.models.minimax_m3"


def test_auto_backend_resolution():
    from freetoken.attention import attention_backend_info
    from freetoken.engine.engine import _required_attn_types, _resolve_auto_attention_backend

    cfg = parse_config(_hf_config())
    required = _required_attn_types(cfg)
    assert required == frozenset({AttnType.BSA})
    assert _resolve_auto_attention_backend(required) == "m3_sparse"
    assert attention_backend_info("m3_sparse").page_sizes == (128,)


def test_nvfp4_backend_restricted_to_triton_for_swigluoai():
    from freetoken.moe.nvfp4_backends import select_nvfp4_backend

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert select_nvfp4_backend(dev, 3072, "auto", activation="swigluoai") == "triton"
    with pytest.raises(RuntimeError):
        select_nvfp4_backend(dev, 3072, "marlin", activation="swigluoai")


def test_expert_source_spec_layer_to_bank():
    from freetoken.models.minimax_m3.weight import _EXPERT_KEY_RE, _NVFP4_SOURCE_SPEC

    cfg = parse_config(_hf_config())
    m = _EXPERT_KEY_RE.match(
        "language_model.model.layers.7.block_sparse_moe.experts.42.w1.weight_scale_2"
    )
    assert m and m.group("layer") == "7" and m.group("expert") == "42"
    assert m.group("proj") == "w1" and m.group("kind") == "weight_scale_2"
    assert _NVFP4_SOURCE_SPEC.proj_to_role == {"w1": "gate", "w3": "up", "w2": "down"}
    # Dense prefix leaves no bank holes; MoE layer i maps to bank i-3.
    assert _NVFP4_SOURCE_SPEC.layer_to_bank(3, cfg) == 0
    assert _NVFP4_SOURCE_SPEC.layer_to_bank(59, cfg) == 56
    assert _NVFP4_SOURCE_SPEC.layer_to_bank(0, cfg) is None
