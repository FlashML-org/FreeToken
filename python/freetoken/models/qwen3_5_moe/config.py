from __future__ import annotations

from typing import TYPE_CHECKING, Any

from freetoken.layers.quantization import QuantConfig, QuantKind
from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    mrope_layout_from_rope_params,
)
from freetoken.models.qwen3_vl.config import parse_vision_config

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


def _expert_quant(hf_config: Any, text: Any) -> tuple[str, tuple[int, int] | None]:
    """The routed experts' quant kind as the engine's format tag, with the scale block of block-fp8."""
    if not (getattr(text, "num_experts", 0) or 0):
        return "none", None
    # the engine reads this tag for its MoE strategy decisions; every module takes its own scheme from the QuantConfig when it is built
    scheme = QuantConfig.from_hf(hf_config).scheme_for_name("model.language_model.layers.0.mlp.experts.0.gate_proj")
    if scheme is None:
        return "none", None
    return str(scheme.kind), scheme.weight.group if scheme.kind is QuantKind.FP8_BLOCK else None


def _layer_types(text: Any) -> list[str]:
    layer_types = getattr(text, "layer_types", None)
    if layer_types is not None:
        return list(layer_types)
    # Fall back to full_attention_interval: every Nth layer (1-indexed) is full.
    interval = int(getattr(text, "full_attention_interval", 4))
    n = int(text.num_hidden_layers)
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(n)
    ]


def parse_config(hf_config: Any) -> ModelConfig:
    text = getattr(hf_config, "text_config", hf_config)

    head_dim = (
        getattr(text, "head_dim", None)
        or text.hidden_size // text.num_attention_heads
    )
    num_kv_heads = getattr(text, "num_key_value_heads", text.num_attention_heads)

    rope_params = getattr(text, "rope_parameters", None) or {}
    rope_theta = rope_params.get("rope_theta", getattr(text, "rope_theta", None))
    partial = (
        rope_params.get("partial_rotary_factor")
        or getattr(text, "partial_rotary_factor", None)
        or 1.0
    )
    rotary_dim = int(head_dim * partial)

    # For text-only with the default rope type, partial NeoX rope needs no scaling dict
    # (the mRoPE params reduce to standard partial rope for text). Avoid carrying the
    # unhashable ``mrope_section`` list into get_rope's cache key.
    rope_type = rope_params.get("rope_type", "default")
    rope_scaling = (
        None
        if rope_type in (None, "default")
        else {k: v for k, v in rope_params.items() if not isinstance(v, (list, dict))}
    )

    expert_quant, weight_block_size = _expert_quant(hf_config, text)

    # Dense variants (e.g. Qwen3.6-27B) report num_experts==0: route the decoder MLP through
    # the dense Qwen3_5DenseMLP instead of the MoE block.
    num_experts = getattr(text, "num_experts", 0) or 0
    moe_enabled = num_experts > 0

    layer_types = _layer_types(text)
    full_ids = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    linear_ids = tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")

    # 3-axis rope only with vision; text-only serving keeps the 1-D partial rope and the decode-graph layout
    vision_config = parse_vision_config(hf_config)
    full_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=text.max_position_embeddings,
        base=rope_theta,
        scaling=rope_scaling,
        mrope_section=(
            list(rope_params["mrope_section"])
            if vision_config is not None and "mrope_section" in rope_params
            else None
        ),
        mrope_layout=mrope_layout_from_rope_params(rope_params),
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=full_rotary,
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=text.linear_num_key_heads,
        num_value_heads=text.linear_num_value_heads,
        key_head_dim=text.linear_key_head_dim,
        value_head_dim=text.linear_value_head_dim,
        conv_kernel_dim=text.linear_conv_kernel_dim,
        output_gate="silu",
    )
    # Order groups by their first layer id for deterministic iteration.
    groups = tuple(
        sorted(
            (full_group, linear_group),
            key=lambda g: g.layer_ids[0] if g.layer_ids else 1 << 30,
        )
    )

    return ModelConfig(
        num_layers=text.num_hidden_layers,
        num_qo_heads=text.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=text.hidden_size,
        vocab_size=text.vocab_size,
        intermediate_size=getattr(text, "intermediate_size", 0),
        hidden_act=text.hidden_act,
        rms_norm_eps=text.rms_norm_eps,
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=full_rotary,
        num_experts=num_experts,
        num_experts_per_tok=getattr(text, "num_experts_per_tok", 0),
        moe_intermediate_size=getattr(text, "moe_intermediate_size", 0),
        shared_expert_intermediate_size=getattr(text, "shared_expert_intermediate_size", 0),
        norm_topk_prob=True,
        moe_enabled=moe_enabled,
        use_qk_norm=True,
        model_type=getattr(hf_config, "model_type", "qwen3_5_moe"),
        architectures=getattr(hf_config, "architectures", ["Qwen3_5MoeForConditionalGeneration"]),
        vision_config=vision_config,
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=groups,
        expert_quant=expert_quant,
        weight_block_size=weight_block_size,
    )


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    """Build the Qwen3.5 MoE runtime configuration from ``qwen35moe`` GGUF metadata.

    llama.cpp records the same hybrid decoder geometry as the official Hugging Face
    configuration, but expresses the Gated DeltaNet fields with its SSM vocabulary.
    This parser keeps that translation in one audited location.  It intentionally
    describes the model only: native Q4_K_M tensor loading and kernel dispatch are
    separate implementation milestones, so callers cannot mistake metadata parsing
    for a runnable GGUF path.
    """
    metadata = shim.metadata

    def value(key: str):
        """Read one required architecture-scoped GGUF value with a clear error."""
        full_key = f"qwen35moe.{key}"
        if full_key not in metadata:
            raise KeyError(f"missing GGUF metadata key {full_key}")
        return metadata[full_key]

    hidden_size = int(value("embedding_length"))
    head_dim = int(value("attention.key_length"))
    num_qo_heads = int(value("attention.head_count"))
    num_kv_heads = int(value("attention.head_count_kv"))
    linear_key_head_dim = int(value("ssm.state_size"))
    linear_value_head_dim = int(value("ssm.state_size"))
    linear_num_key_heads = int(value("ssm.group_count"))
    linear_inner_size = int(value("ssm.inner_size"))
    if linear_inner_size % linear_value_head_dim:
        raise ValueError(
            "qwen35moe.ssm.inner_size must divide exactly into value-head groups: "
            f"{linear_inner_size} / {linear_value_head_dim}"
        )
    linear_num_value_heads = linear_inner_size // linear_value_head_dim

    num_layers = int(value("block_count"))
    full_interval = int(value("full_attention_interval"))
    if full_interval <= 0:
        raise ValueError(f"invalid qwen35moe.full_attention_interval {full_interval}")
    layer_types = tuple(
        "full_attention" if (layer_index + 1) % full_interval == 0 else "linear_attention"
        for layer_index in range(num_layers)
    )
    full_ids = tuple(index for index, kind in enumerate(layer_types) if kind == "full_attention")
    linear_ids = tuple(index for index, kind in enumerate(layer_types) if kind == "linear_attention")

    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=int(value("rope.dimension_count")),
        max_position=int(value("context_length")),
        base=float(value("rope.freq_base")),
        scaling=None,
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=rotary,
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=linear_num_key_heads,
        num_value_heads=linear_num_value_heads,
        key_head_dim=linear_key_head_dim,
        value_head_dim=linear_value_head_dim,
        conv_kernel_dim=int(value("ssm.conv_kernel")),
        output_gate="silu",
    )

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        vocab_size=int(shim.vocab_size),
        intermediate_size=0,
        hidden_act="silu",
        rms_norm_eps=float(value("attention.layer_norm_rms_epsilon")),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        rotary_config=rotary,
        num_experts=int(value("expert_count")),
        num_experts_per_tok=int(value("expert_used_count")),
        moe_intermediate_size=int(value("expert_feed_forward_length")),
        shared_expert_intermediate_size=int(value("expert_shared_feed_forward_length")),
        norm_topk_prob=True,
        moe_enabled=True,
        use_qk_norm=True,
        model_type="qwen3_5_moe",
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        vision_config=None,
        attention_groups=(linear_group, full_group),
    )


__all__ = ["parse_config", "parse_gguf_config"]
