from __future__ import annotations

from typing import Any

from freetoken.models.config import ModelConfig, RotaryConfig, detect_expert_quant


def _rope_params(hf_config: Any) -> dict:
    params = getattr(hf_config, "rope_parameters", None)
    if isinstance(params, dict) and params:
        return dict(params)
    params = {"rope_theta": getattr(hf_config, "rope_theta", 10000.0)}
    scaling = getattr(hf_config, "rope_scaling", None)
    if isinstance(scaling, dict):
        params.update(scaling)
    return params


def parse_config(hf_config: Any, model_path: str | None = None) -> ModelConfig:
    """Parse a HuggingFace ``Glm4MoeConfig`` (GLM-4.5/4.6/4.7) into FreeToken's
    :class:`ModelConfig`.

    GLM-4 MoE specifics handled here:
    - ``first_k_dense_replace`` leading dense MLP layers, then sparse MoE layers each with
      a shared expert (``n_shared_experts``) and a sigmoid + bias router scaled by
      ``routed_scaling_factor``.
    - Partial RoPE (``partial_rotary_factor`` 0.5 -> rotary_dim 64) and per-head qk-norm.
    - The trailing MTP layer (``num_nextn_predict_layers``) is dropped (we only run the
      ``num_hidden_layers`` main layers), so ``num_layers`` excludes it implicitly because
      HF counts it separately.
    """
    head_dim = (
        getattr(hf_config, "head_dim", None)
        or hf_config.hidden_size // hf_config.num_attention_heads
    )
    num_kv_heads = getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)

    rope = _rope_params(hf_config)
    rope_theta = rope.get("rope_theta", 10000.0)
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    rope_scaling = None if rope_type in (None, "default") else rope

    partial = rope.get(
        "partial_rotary_factor", getattr(hf_config, "partial_rotary_factor", 1.0)
    )
    rotary_dim = int(head_dim * partial)

    num_experts = (
        getattr(hf_config, "n_routed_experts", None)
        or getattr(hf_config, "num_local_experts", None)
        or getattr(hf_config, "num_experts", 0)
    )
    moe_intermediate_size = (
        getattr(hf_config, "moe_intermediate_size", 0) or hf_config.intermediate_size
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
        rotary_config=RotaryConfig(
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            max_position=hf_config.max_position_embeddings,
            base=rope_theta,
            scaling=rope_scaling,
        ),
        num_experts=num_experts,
        num_experts_per_tok=hf_config.num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        norm_topk_prob=bool(getattr(hf_config, "norm_topk_prob", True)),
        model_type=getattr(hf_config, "model_type", "glm4_moe"),
        architectures=getattr(hf_config, "architectures", ["Glm4MoeForCausalLM"]),
        moe_enabled=True,
        use_qk_norm=bool(getattr(hf_config, "use_qk_norm", False)),
        expert_quant=detect_expert_quant(hf_config),
        first_k_dense_replace=int(getattr(hf_config, "first_k_dense_replace", 0)),
        n_shared_experts=int(getattr(hf_config, "n_shared_experts", 0)),
        routed_scaling_factor=float(getattr(hf_config, "routed_scaling_factor", 1.0)),
        n_group=int(getattr(hf_config, "n_group", 1)),
        topk_group=int(getattr(hf_config, "topk_group", 1)),
        has_attn_bias=bool(getattr(hf_config, "attention_bias", False)),
    )


__all__ = ["parse_config"]
