from __future__ import annotations

from typing import Any

from freetoken.models.config import ModelConfig, RotaryConfig


def parse_config(hf_config: Any, model_path: str | None = None) -> ModelConfig:
    num_kv_heads = getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)
    head_dim = (
        getattr(hf_config, "head_dim", None)
        or hf_config.hidden_size // hf_config.num_attention_heads
    )
    rope_scaling = getattr(hf_config, "rope_scaling", None)
    rope_theta = getattr(hf_config, "rope_theta", None)
    if rope_theta is None and rope_scaling is not None:
        rope_theta = rope_scaling.get("rope_theta")
    if rope_theta is None:
        rope_theta = 10_000.0

    return ModelConfig(
        num_layers=hf_config.num_hidden_layers,
        num_qo_heads=hf_config.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hf_config.hidden_size,
        vocab_size=hf_config.vocab_size,
        intermediate_size=hf_config.intermediate_size,
        rms_norm_eps=hf_config.rms_norm_eps,
        rotary_config=RotaryConfig(
            head_dim=head_dim,
            rotary_dim=head_dim,
            max_position=hf_config.max_position_embeddings,
            base=rope_theta,
            scaling=rope_scaling,
        ),
        hidden_act=hf_config.hidden_act,
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        model_type=getattr(hf_config, "model_type", "qwen2"),
        architectures=getattr(hf_config, "architectures", ["Qwen2ForCausalLM"]),
    )


__all__ = ["parse_config"]
