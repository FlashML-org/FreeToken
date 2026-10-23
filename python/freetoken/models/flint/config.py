from __future__ import annotations

from dataclasses import replace
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
    detect_expert_quant,
)


def parse_config(hf: Any) -> ModelConfig:
    layers = int(hf.num_hidden_layers)
    types = tuple(hf.layer_types)
    if len(types) != layers or set(types) - {"full_attention", "sliding_attention"}:
        raise ValueError("Flint requires one full_attention/sliding_attention type per layer")
    full = tuple(i for i, kind in enumerate(types) if kind == "full_attention")
    swa = tuple(i for i, kind in enumerate(types) if kind == "sliding_attention")
    if set(hf.no_rope_layer_indices) != set(full):
        raise ValueError("Flint supports RoPE on sliding layers and NoPE on full layers only")
    dense = tuple(hf.mlp_only_layers)
    if dense != tuple(range(len(dense))) or len(dense) >= layers:
        raise ValueError("Flint dense MLP layers must form a leading contiguous prefix")
    if int(getattr(hf, "decoder_sparse_step", 1)) != 1:
        raise ValueError("Flint requires an MoE block in every layer after the dense prefix")
    if getattr(hf, "rope_scaling", None):
        raise NotImplementedError("Flint RoPE scaling is not supported")
    if hf.hidden_act != "silu" or hf.router_score_function != "sigmoid":
        raise ValueError("Flint requires SwiGLU experts and sigmoid routing")
    if hf.softmax_type != "learnable" or bool(getattr(hf, "attention_bias", False)):
        raise ValueError("Flint requires learned attention sinks and bias-free projections")
    window = int(hf.sliding_window)
    if window <= 0 or not full or not swa:
        raise ValueError("Flint requires full and sliding attention with a positive window")
    rope = RotaryConfig(
        head_dim=int(hf.head_dim), rotary_dim=int(hf.head_dim),
        max_position=int(hf.max_position_embeddings), base=float(hf.rope_theta), scaling=None,
    )
    return ModelConfig(
        num_layers=layers, num_qo_heads=int(hf.num_attention_heads),
        num_kv_heads=int(hf.num_key_value_heads), head_dim=int(hf.head_dim),
        hidden_size=int(hf.hidden_size), vocab_size=int(hf.vocab_size),
        intermediate_size=int(hf.intermediate_size), hidden_act="silu",
        rms_norm_eps=float(hf.rms_norm_eps),
        tie_word_embeddings=bool(hf.tie_word_embeddings), rotary_config=rope,
        num_experts=int(hf.num_experts), num_experts_per_tok=int(hf.num_experts_per_tok),
        moe_intermediate_size=int(hf.moe_intermediate_size),
        norm_topk_prob=bool(hf.norm_topk_prob),
        model_type="flint", architectures=list(hf.architectures), moe_enabled=True,
        expert_quant=detect_expert_quant(hf), first_k_dense_replace=len(dense),
        shared_expert_intermediate_size=int(hf.shared_expert_intermediate_size),
        use_qk_norm=True,
        attention_groups=(
            SWAAttentionGroupConfig(
                name="swa", layer_ids=swa, num_kv_heads=int(hf.num_key_value_heads),
                head_dim=int(hf.head_dim), rotary_config=rope, sliding_window=window,
            ),
            FullAttentionGroupConfig(
                name="full", layer_ids=full, num_kv_heads=int(hf.num_key_value_heads),
                head_dim=int(hf.head_dim), rotary_config=replace(rope, rotary_dim=0),
            ),
        ),
    )
