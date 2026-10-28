from __future__ import annotations

from typing import Any

from freetoken.models.config import ModelConfig, RotaryConfig


def _gguf_field(metadata: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    raise KeyError(f"missing GGUF metadata key(s): {keys}")


def parse_gguf_config(hf_config: Any) -> ModelConfig:
    metadata = getattr(hf_config, "metadata", None)
    if metadata is None:
        return parse_config(hf_config)

    def field(*keys: str):
        return _gguf_field(metadata, *keys)

    text = type(
        "_GGUFTextConfig",
        (),
        {
            "hidden_size": int(field("qwen3_moe.embedding_length", "qwen3.embedding_length", "qwen.embedding_length")),
            "num_hidden_layers": int(field("qwen3_moe.block_count", "qwen3.block_count", "qwen.block_count")),
            "num_attention_heads": int(field("qwen3_moe.attention.head_count", "qwen3.attention.head_count", "qwen.attention.head_count")),
            "num_key_value_heads": int(field("qwen3_moe.attention.head_count_kv", "qwen3.attention.head_count_kv", "qwen.attention.head_count_kv", "qwen3_moe.attention.head_count")),
            "head_dim": int(field("qwen3_moe.attention.key_length", "qwen3.attention.key_length", "qwen.attention.key_length")),
            "max_position_embeddings": int(field("qwen3_moe.context_length", "qwen3.context_length", "qwen.context_length")),
            "hidden_act": "silu",
            "intermediate_size": int(field("qwen3_moe.feed_forward_length", "qwen3.feed_forward_length", "qwen.feed_forward_length")),
            "vocab_size": int(field("tokenizer.ggml.vocab_size", "vocab_size")),
            "rms_norm_eps": float(field("qwen3_moe.attention.layer_norm_rms_epsilon", "qwen3.attention.layer_norm_rms_epsilon", "qwen.attention.layer_norm_rms_epsilon")),
            "tie_word_embeddings": bool(getattr(hf_config, "tie_word_embeddings", False)),
            "rope_theta": float(field("qwen3_moe.rope.freq_base", "qwen3.rope.freq_base", "qwen.rope.freq_base", "10000000.0")),
            "rope_scaling": None,
            "model_type": "qwen3_moe",
            "architectures": ["Qwen3MoeForCausalLM"],
            "num_experts": int(field("qwen3_moe.expert_count", "qwen3.expert_count", "qwen.expert_count", "0")),
            "num_experts_per_tok": int(field("qwen3_moe.expert_used_count", "qwen3.expert_used_count", "qwen.expert_used_count", "0")),
            "moe_intermediate_size": int(field("qwen3_moe.expert_feed_forward_length", "qwen3.expert_feed_forward_length", "qwen.expert_feed_forward_length", "0")),
            "norm_topk_prob": False,
        },
    )()
    return parse_config(text)


def parse_config(hf_config: Any) -> ModelConfig:
    num_kv_heads = getattr(
        hf_config,
        "num_key_value_heads",
        hf_config.num_attention_heads,
    )
    head_dim = (
        getattr(hf_config, "head_dim", None)
        or hf_config.hidden_size // hf_config.num_attention_heads
    )
    rope_scaling = getattr(hf_config, "rope_scaling", None)
    rope_theta = getattr(hf_config, "rope_theta", None)
    if rope_theta is None and rope_scaling is not None:
        rope_theta = rope_scaling["rope_theta"]

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
        rotary_config=RotaryConfig(
            head_dim=head_dim,
            rotary_dim=head_dim,
            max_position=hf_config.max_position_embeddings,
            base=rope_theta,
            scaling=rope_scaling,
        ),
        num_experts=getattr(
            hf_config,
            "num_local_experts",
            getattr(hf_config, "num_experts", 0),
        ),
        num_experts_per_tok=getattr(hf_config, "num_experts_per_tok", 0),
        moe_intermediate_size=getattr(hf_config, "moe_intermediate_size", 0),
        norm_topk_prob=bool(getattr(hf_config, "norm_topk_prob", False)),
        model_type=getattr(hf_config, "model_type", "qwen3_moe"),
        architectures=getattr(hf_config, "architectures", ["Qwen3MoeForCausalLM"]),
    )


__all__ = ["parse_config"]
