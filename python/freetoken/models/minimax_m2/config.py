from __future__ import annotations

from typing import Any

from freetoken.models.config import ModelConfig, RotaryConfig, detect_expert_quant


def _rope_params(hf_config: Any) -> dict:
    """transformers>=5 nests rope settings in ``rope_parameters``; older configs use
    flat ``rope_theta`` / ``rope_scaling``."""
    params = getattr(hf_config, "rope_parameters", None)
    if isinstance(params, dict) and params:
        return dict(params)
    params = {"rope_theta": getattr(hf_config, "rope_theta", 10000.0)}
    scaling = getattr(hf_config, "rope_scaling", None)
    if isinstance(scaling, dict):
        params.update(scaling)
    return params


def parse_config(hf_config: Any, model_path: str | None = None) -> ModelConfig:
    head_dim = (
        getattr(hf_config, "head_dim", None)
        or hf_config.hidden_size // hf_config.num_attention_heads
    )
    num_kv_heads = getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)

    rope = _rope_params(hf_config)
    rope_theta = rope.get("rope_theta", 10000.0)
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    rope_scaling = None if rope_type in (None, "default") else rope

    # Partial RoPE: MiniMax-M2 rotates only the first ``rotary_dim`` dims of each head.
    rotary_dim = getattr(hf_config, "rotary_dim", None)
    if rotary_dim is None:
        partial = rope.get("partial_rotary_factor", getattr(hf_config, "partial_rotary_factor", 1.0))
        rotary_dim = int(head_dim * partial)

    # Experts use the dense ``intermediate_size`` (there is no separate moe size key).
    moe_intermediate_size = getattr(hf_config, "moe_intermediate_size", 0) or hf_config.intermediate_size

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
            rotary_dim=rotary_dim,
            max_position=hf_config.max_position_embeddings,
            base=rope_theta,
            scaling=rope_scaling,
        ),
        num_experts=getattr(hf_config, "num_local_experts", getattr(hf_config, "num_experts", 0)),
        num_experts_per_tok=hf_config.num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        norm_topk_prob=True,  # MiniMax always renormalizes the selected expert weights
        model_type=getattr(hf_config, "model_type", "minimax_m2"),
        architectures=getattr(hf_config, "architectures", ["MiniMaxM2ForCausalLM"]),
        moe_enabled=True,
        use_qk_norm=bool(getattr(hf_config, "use_qk_norm", False)),
        expert_quant=detect_expert_quant(hf_config),
    )


__all__ = ["parse_config"]
