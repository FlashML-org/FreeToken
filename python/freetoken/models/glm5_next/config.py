"""Engine-facing configuration for GLM-5.3-Flash (``glm5_next``).

GLM-5.3 alternates three Kimi Delta Attention layers with one compressed DSA/MLA
layer and carries four manifold-constrained residual streams through every block.
This parser describes that hybrid layout for the runnable FreeToken implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)


@dataclass(frozen=True)
class Glm5NextArgs:
    hc_mult: int
    hc_eps: float
    hc_sinkhorn_iters: int
    linear_num_heads: int
    linear_head_dim: int
    linear_conv_kernel_dim: int
    linear_lower_bound: float | None
    kv_lora_rank: int
    q_lora_rank: int
    qk_head_dim: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int
    indexer_types: tuple[str, ...]
    layer_types: tuple[str, ...]
    mlp_layer_types: tuple[str, ...]


def _quant_get(hf_config: Any):
    quant = getattr(hf_config, "quantization_config", None)
    if quant is None:
        return None
    return (
        quant.get
        if isinstance(quant, dict)
        else (lambda k, d=None: getattr(quant, k, d))
    )


def _ignored(patterns: list[str], module_name: str) -> bool:
    return any(fnmatch(module_name, pattern) for pattern in patterns)


def _quant_modes(hf_config: Any) -> tuple[str, str, str, str, tuple[int, int] | None]:
    """Return expert/attention/dense/head formats plus optional block geometry."""
    get = _quant_get(hf_config)
    if get is None:
        return "none", "none", "none", "none", None

    algo = str(get("quant_algo") or get("quant_method") or "").lower()
    block = get("weight_block_size")
    if algo == "fp8" and block:
        block_size = tuple(int(x) for x in block)
        if block_size != (128, 128):
            raise ValueError(f"only 128x128 block-fp8 is supported, got {block_size}")
        # The official checkpoint's modules_to_not_convert keeps attention, KDA,
        # mHC, norms and the head resident in bf16; routed experts are block-fp8.
        return "fp8_block", "none", "none", "none", block_size

    if "fp4" not in algo:
        return algo or "none", "none", "none", "none", None

    ignore = list(get("ignore") or ())
    prefix = "model.language_model.layers.3"

    def mode(probe: str) -> str:
        return "none" if _ignored(ignore, probe) else "nvfp4"

    return (
        mode(f"{prefix}.mlp.experts.0.gate_proj"),
        mode(f"{prefix}.self_attn.q_proj"),
        mode("model.language_model.layers.0.mlp.gate_proj"),
        mode("lm_head"),
        None,
    )


def parse_config(hf_config: Any) -> ModelConfig:
    text = getattr(hf_config, "text_config", hf_config)
    layer_types = tuple(str(x) for x in text.layer_types)
    mlp_layer_types = tuple(str(x) for x in text.mlp_layer_types)
    if len(layer_types) != int(text.num_hidden_layers):
        raise ValueError("layer_types must contain one entry per decoder layer")
    if len(mlp_layer_types) != int(text.num_hidden_layers):
        raise ValueError("mlp_layer_types must contain one entry per decoder layer")

    linear_ids = tuple(
        i for i, kind in enumerate(layer_types) if kind == "linear_attention"
    )
    full_ids = tuple(
        i for i, kind in enumerate(layer_types) if kind == "deepseek_sparse_attention"
    )
    unknown = set(layer_types) - {"linear_attention", "deepseek_sparse_attention"}
    if unknown:
        raise ValueError(
            f"unsupported GLM-5.3 attention layer types: {sorted(unknown)}"
        )

    linear = text.linear_attn_config
    linear_get = (
        linear.get
        if isinstance(linear, dict)
        else lambda k, d=None: getattr(linear, k, d)
    )
    qk_rope = int(getattr(text, "qk_rope_head_dim", 0) or 0)
    qk_head = int(text.qk_head_dim)
    rotary = RotaryConfig(
        head_dim=qk_head,
        rotary_dim=qk_rope,
        max_position=int(text.max_position_embeddings),
        # The release sets rope_theta=null because qk_rope_head_dim is zero.  Keep a
        # harmless finite default so the generic cache schema remains well-formed.
        base=float(getattr(text, "rope_theta", None) or 10000.0),
        scaling=None,
    )
    indexer_types = tuple(str(x) for x in text.indexer_types)
    full_indexers = sum(1 for i in full_ids if indexer_types[i] == "full")

    groups = (
        LinearGatedDeltaGroupConfig(
            name="linear",
            layer_ids=linear_ids,
            num_key_heads=int(linear_get("num_heads")),
            num_value_heads=int(linear_get("num_heads")),
            key_head_dim=int(linear_get("head_dim")),
            value_head_dim=int(linear_get("head_dim")),
            conv_kernel_dim=int(linear_get("short_conv_kernel_size")),
            output_gate="sigmoid",
        ),
        FullAttentionGroupConfig(
            name="full",
            layer_ids=full_ids,
            num_kv_heads=1,
            head_dim=int(text.kv_lora_rank) + qk_rope,
            rotary_config=rotary,
            mla=True,
            index_head_dim=int(text.index_head_dim),
            num_index_layers=full_indexers,
            index_ratio=int(text.index_kpool),
        ),
    )
    expert_quant, attn_quant, dense_quant, lm_head_quant, block_size = _quant_modes(
        hf_config
    )
    args = Glm5NextArgs(
        hc_mult=int(text.hc_mult),
        hc_eps=float(text.hc_eps),
        hc_sinkhorn_iters=int(text.hc_sinkhorn_iters),
        linear_num_heads=int(linear_get("num_heads")),
        linear_head_dim=int(linear_get("head_dim")),
        linear_conv_kernel_dim=int(linear_get("short_conv_kernel_size")),
        linear_lower_bound=linear_get("gate_lower_bound"),
        kv_lora_rank=int(text.kv_lora_rank),
        q_lora_rank=int(text.q_lora_rank),
        qk_head_dim=qk_head,
        qk_nope_head_dim=int(text.qk_nope_head_dim),
        qk_rope_head_dim=qk_rope,
        v_head_dim=int(text.v_head_dim),
        index_n_heads=int(text.index_n_heads),
        index_head_dim=int(text.index_head_dim),
        index_topk=int(text.index_topk),
        index_kpool=int(text.index_kpool),
        indexer_types=indexer_types,
        layer_types=layer_types,
        mlp_layer_types=mlp_layer_types,
    )
    return ModelConfig(
        num_layers=int(text.num_hidden_layers),
        num_qo_heads=int(text.num_attention_heads),
        num_kv_heads=1,
        head_dim=int(text.kv_lora_rank) + qk_rope,
        hidden_size=int(text.hidden_size),
        vocab_size=int(text.vocab_size),
        intermediate_size=int(text.intermediate_size),
        hidden_act=str(text.hidden_act),
        rms_norm_eps=float(text.rms_norm_eps),
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=rotary,
        attention_groups=groups,
        num_experts=int(text.n_routed_experts),
        num_experts_per_tok=int(text.num_experts_per_tok),
        moe_intermediate_size=int(text.moe_intermediate_size),
        norm_topk_prob=bool(text.norm_topk_prob),
        model_type=str(getattr(hf_config, "model_type", "glm5_next")),
        architectures=list(
            getattr(hf_config, "architectures", ["Glm5NextForConditionalGeneration"])
        ),
        moe_enabled=True,
        first_k_dense_replace=int(text.first_k_dense_replace),
        n_shared_experts=int(text.n_shared_experts),
        routed_scaling_factor=float(text.routed_scaling_factor),
        n_group=int(text.n_group),
        topk_group=int(text.topk_group),
        swiglu_limit=float(text.swiglu_limit),
        attn_sm_scale=qk_head**-0.5,
        expert_quant=expert_quant,
        attn_quant=attn_quant,
        dense_quant=dense_quant,
        lm_head_quant=lm_head_quant,
        weight_block_size=block_size,
        glm5_args=args,
    )


__all__ = ["Glm5NextArgs", "parse_config"]
