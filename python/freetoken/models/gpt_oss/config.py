from __future__ import annotations

from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
)


def _quant_method(hf_config: Any) -> str | None:
    quant_config = getattr(hf_config, "quantization_config", None) or {}
    if isinstance(quant_config, dict):
        return quant_config.get("quant_method")
    return getattr(quant_config, "quant_method", None)


def _rope_config(hf_config: Any, head_dim: int) -> RotaryConfig:
    rope_scaling = getattr(hf_config, "rope_scaling", None)
    rope_theta = getattr(hf_config, "rope_theta", None)
    rope_parameters = getattr(hf_config, "rope_parameters", None)
    if rope_theta is None and isinstance(rope_parameters, dict):
        rope_theta = rope_parameters.get("rope_theta")
    if rope_theta is None:
        rope_theta = 10000.0
    if rope_scaling is None and isinstance(rope_parameters, dict):
        rope_scaling = {
            key: value for key, value in rope_parameters.items() if key != "rope_theta"
        } or None
    return RotaryConfig(
        head_dim=head_dim,
        rotary_dim=head_dim,
        max_position=hf_config.max_position_embeddings,
        base=float(rope_theta),
        scaling=rope_scaling,
    )


def parse_config(hf_config: Any, model_path: str | None = None) -> ModelConfig:
    head_dim = getattr(hf_config, "head_dim", None) or (
        hf_config.hidden_size // hf_config.num_attention_heads
    )
    rotary_config = _rope_config(hf_config, head_dim)
    layer_types = tuple(getattr(hf_config, "layer_types", ()))
    if not layer_types:
        layer_types = tuple("full_attention" for _ in range(hf_config.num_hidden_layers))
    swa_layers = tuple(
        idx
        for idx, layer_type in enumerate(layer_types)
        if layer_type == "sliding_attention"
    )
    full_layers = tuple(
        idx
        for idx, layer_type in enumerate(layer_types)
        if layer_type != "sliding_attention"
    )
    num_kv_heads = getattr(
        hf_config,
        "num_key_value_heads",
        hf_config.num_attention_heads,
    )

    return ModelConfig(
        num_layers=hf_config.num_hidden_layers,
        num_qo_heads=hf_config.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hf_config.hidden_size,
        vocab_size=hf_config.vocab_size,
        intermediate_size=hf_config.intermediate_size,
        hidden_act=hf_config.hidden_act,
        rms_norm_eps=hf_config.rms_norm_eps,
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        rotary_config=rotary_config,
        num_experts=getattr(
            hf_config,
            "num_local_experts",
            getattr(hf_config, "num_experts", 0),
        ),
        num_experts_per_tok=getattr(hf_config, "num_experts_per_tok", 0),
        moe_intermediate_size=hf_config.intermediate_size,
        norm_topk_prob=True,
        model_type=getattr(hf_config, "model_type", "gpt_oss"),
        architectures=getattr(hf_config, "architectures", ["GptOssForCausalLM"]),
        moe_enabled=True,
        has_attn_bias=bool(getattr(hf_config, "attention_bias", True)),
        has_router_bias=True,
        moe_weight_format=_quant_method(hf_config),
        swiglu_limit=getattr(hf_config, "swiglu_limit", None),
        hidden_act_alpha=float(getattr(hf_config, "hidden_act_alpha", 1.702) or 1.702),
        attention_groups=(
            SWAAttentionGroupConfig(
                name="swa",
                layer_ids=swa_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_config=rotary_config,
                sliding_window=getattr(hf_config, "sliding_window", 0),
            ),
            FullAttentionGroupConfig(
                name="full",
                layer_ids=full_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_config=rotary_config,
            ),
        ),
    )


__all__ = ["parse_config"]
