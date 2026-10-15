"""LFM2-MoE (LiquidAI ``lfm2_moe``) config: depthwise short-conv + GQA hybrid, sigmoid-routed MoE.

``layer_types`` alternates ``conv`` (depthwise causal short conv, no paged KV, one conv-state
slab per layer) and ``full_attention`` (GQA with per-head q/k RMSNorm); the first
``num_dense_layers`` layers run a dense gated MLP, the rest a sigmoid-routed MoE with a
selection-only ``expert_bias``. The head is tied to the embedding by default.
"""

from __future__ import annotations

from typing import Any

from freetoken.layers.quantization import QuantConfig, QuantKind
from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    ShortConvGroupConfig,
)


def _expert_quant(hf_config: Any) -> tuple[str, tuple[int, int] | None]:
    """The routed experts' quant kind as the engine's format tag, with the scale block of block-fp8."""
    # detect_expert_quant tags every compressed-tensors export "compressed-tensors"; the
    # QuantConfig is the only route that yields the MoE kernels' format tags (fp8_block).
    scheme = QuantConfig.from_hf(hf_config).scheme_for_name(
        "model.layers.0.feed_forward.experts.0.w1"
    )
    if scheme is None:
        return "none", None
    return str(scheme.kind), scheme.weight.group if scheme.kind is QuantKind.FP8_BLOCK else None


def _layer_ids(hf: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(conv_ids, full_ids)`` from the checkpoint's ``layer_types``."""
    layer_types = list(hf.layer_types)
    conv_ids, full_ids = [], []
    for i, kind in enumerate(layer_types):
        if kind == "conv":
            conv_ids.append(i)
        elif kind == "full_attention":
            full_ids.append(i)
        else:
            raise ValueError(f"unsupported lfm2_moe layer type {kind!r}")
    if not full_ids:
        # the model's paged-KV spec is required: a conv-only stack is not a servable mixer mix
        raise ValueError("lfm2_moe needs at least one full_attention layer")
    return tuple(conv_ids), tuple(full_ids)


def parse_config(hf_config: Any) -> ModelConfig:
    head_dim = (
        getattr(hf_config, "head_dim", None)
        or hf_config.hidden_size // hf_config.num_attention_heads
    )
    num_kv_heads = getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)

    params = getattr(hf_config, "rope_parameters", None) or {}
    base = params.get("rope_theta", getattr(hf_config, "rope_theta", 1_000_000.0))
    rope_type = params.get("rope_type", "default")
    # avoid carrying the unhashable / non-scalar entries into get_rope's cache key
    scaling = (
        None
        if rope_type in (None, "default")
        else {k: v for k, v in params.items() if not isinstance(v, (list, dict))}
    )
    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=head_dim,
        max_position=hf_config.max_position_embeddings,
        base=base,
        scaling=scaling,
    )

    conv_ids, full_ids = _layer_ids(hf_config)
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=rotary,
    )
    conv_group = ShortConvGroupConfig(
        name="conv",
        layer_ids=conv_ids,
        conv_dim=hf_config.hidden_size,
        conv_kernel_dim=int(getattr(hf_config, "conv_L_cache", 3)),
        conv_bias=bool(getattr(hf_config, "conv_bias", False)),
    )
    # Order groups by their first layer id for deterministic iteration (the layer-type list
    # interleaves, so neither group owns a contiguous range).
    groups = tuple(
        sorted((g for g in (full_group, conv_group) if g.layer_ids), key=lambda g: g.layer_ids[0])
    )

    expert_quant, weight_block_size = _expert_quant(hf_config)

    return ModelConfig(
        num_layers=hf_config.num_hidden_layers,
        num_qo_heads=hf_config.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hf_config.hidden_size,
        vocab_size=hf_config.vocab_size,
        intermediate_size=hf_config.intermediate_size,
        hidden_act="silu",
        rms_norm_eps=hf_config.norm_eps,
        # LFM2-8B-A1B omits the key and ships no lm_head tensor: the class default is tied
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", True)),
        rotary_config=rotary,
        num_experts=hf_config.num_experts,
        num_experts_per_tok=hf_config.num_experts_per_tok,
        moe_intermediate_size=hf_config.moe_intermediate_size,
        norm_topk_prob=bool(getattr(hf_config, "norm_topk_prob", True)),
        routed_scaling_factor=float(getattr(hf_config, "routed_scaling_factor", 1.0)),
        use_expert_bias=bool(getattr(hf_config, "use_expert_bias", True)),
        first_k_dense_replace=int(getattr(hf_config, "num_dense_layers", 2)),
        moe_enabled=True,
        model_type=getattr(hf_config, "model_type", "lfm2_moe"),
        architectures=getattr(hf_config, "architectures", ["Lfm2MoeForCausalLM"]),
        attention_groups=groups,
        expert_quant=expert_quant,
        weight_block_size=weight_block_size,
    )


__all__ = ["parse_config"]
