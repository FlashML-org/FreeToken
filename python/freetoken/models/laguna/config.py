from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
    detect_compressed_tensors_nvfp4,
)


@dataclass(frozen=True)
class LagunaArgs:
    """Laguna-specific attention output gating (``modeling_laguna.py`` ``g_proj``).

    ``gating`` is ``"per-head"`` (one gate per head, broadcast over head_dim),
    ``True``/``"per-element"`` (one gate per channel) or ``False`` (no gating).
    """

    gating: str | bool
    gating_types: tuple[str, ...] | None = None


def parse_config(hf_config: Any) -> ModelConfig:
    # Laguna is not a multimodal wrapper; but keep text_config indirection
    # for parity with gemma4/qwen4_exp in case a future variant wraps.
    text = getattr(hf_config, "text_config", None)
    if text is not None:
        cfg = text
        top_architectures = getattr(hf_config, "architectures", None)
        top_cfg = hf_config
    else:
        cfg = hf_config
        top_architectures = getattr(cfg, "architectures", None)
        top_cfg = cfg

    head_dim = int(getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))
    max_position = int(getattr(cfg, "max_position_embeddings", 4096))
    rms_norm_eps = float(getattr(cfg, "rms_norm_eps", 1e-6))
    hidden_act = str(getattr(cfg, "hidden_act", "silu"))
    vocab_size = int(cfg.vocab_size)
    hidden_size = int(cfg.hidden_size)
    intermediate_size = int(getattr(cfg, "intermediate_size", 0))
    num_layers = int(cfg.num_hidden_layers)
    num_attention_heads = int(cfg.num_attention_heads)
    num_attention_heads_per_layer = getattr(cfg, "num_attention_heads_per_layer", None)
    if num_attention_heads_per_layer is not None:
        num_attention_heads_per_layer = tuple(int(x) for x in num_attention_heads_per_layer)
        assert len(num_attention_heads_per_layer) == num_layers, (
            f"num_attention_heads_per_layer len {len(num_attention_heads_per_layer)} != num_layers {num_layers}"
        )
        # ModelConfig.num_qo_heads sizes shared, layer-agnostic state: the Triton
        # backend's decode scratch and its CUDA-graph capture buffers
        # (attention/triton.py num_q_heads -> init_capture_graph). Laguna S 2.1's
        # config says 48 but its 36 SWA layers run 72 heads, and the captured buffer
        # is never resized on replay -- so it must be the MAX over layers, not the
        # nominal value. Per-layer projections read num_attention_heads_per_layer.
        num_qo_heads = max(num_attention_heads_per_layer)
    else:
        num_qo_heads = num_attention_heads
    tie_word_embeddings = bool(getattr(cfg, "tie_word_embeddings", False))
    sliding_window = getattr(cfg, "sliding_window", None)
    # Layer types: expected 1:3 full:swa pattern ("full_attention"/"sliding_attention").
    layer_types = getattr(cfg, "layer_types", None)
    if layer_types is None:
        layer_types = ["full_attention"] * num_layers
    layer_types = list(layer_types)
    assert len(layer_types) == num_layers

    # Rope: Laguna stores two dicts under rope_parameters.
    rope_parameters = getattr(cfg, "rope_parameters", None) or {}
    if not isinstance(rope_parameters, dict):
        rope_parameters = {}
    full_rope = rope_parameters.get("full_attention")
    swa_rope = rope_parameters.get("sliding_attention")
    # Fall back to outer rope_parameters when per-type not present (BF16 future etc.).
    if full_rope is None:
        full_rope = {k: v for k, v in rope_parameters.items() if k not in ("full_attention", "sliding_attention")}
        if not full_rope:
            full_rope = {"rope_type": "default", "rope_theta": 500000.0}
    if swa_rope is None:
        # Laguna without separate SWA rope (unlikely) — reuse full.
        swa_rope = full_rope

    # Full rope is YARN for S 2.1.
    full_rope_type = str(full_rope.get("rope_type", "default"))
    full_rope_theta = float(full_rope.get("rope_theta", 500000.0))
    full_partial = float(full_rope.get("partial_rotary_factor", 0.5))
    swa_rope_type = str(swa_rope.get("rope_type", "default"))
    swa_rope_theta = float(swa_rope.get("rope_theta", 10000.0))
    swa_partial = float(swa_rope.get("partial_rotary_factor", 1.0))

    # Build RotaryConfigs. Full uses YARN scaling dict, SWA uses default.
    # For YARN we carry the full dict (factor/beta/attention_factor...) but
    # only the scalar keys are hashable for get_rope cache key.
    def _rope_scaling(rope_dict: dict[str, Any], rope_type: str) -> dict[str, Any] | None:
        if rope_type == "default":
            return None
        # YARN needs the standard params; filter to scalar non-list values.
        # layers/rotary.py consumes rope_type/yarn params explicitly.
        scaling: dict[str, Any] = {"rope_type": rope_type}
        for k in ("factor", "beta_fast", "beta_slow", "original_max_position_embeddings",
                  "attention_factor", "truncate", "mscale", "mscale_all_dim"):
            if k in rope_dict:
                scaling[k] = rope_dict[k]
        # Also carry rope_theta via base field, not here.
        return scaling

    full_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=int(head_dim * full_partial),
        max_position=max_position,
        base=full_rope_theta,
        scaling=_rope_scaling(full_rope, full_rope_type),
    )
    swa_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=int(head_dim * swa_partial),
        max_position=max_position,
        base=swa_rope_theta,
        scaling=_rope_scaling(swa_rope, swa_rope_type),
    )

    # Split layer ids.
    full_ids = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    swa_ids = tuple(i for i, t in enumerate(layer_types) if t != "full_attention")
    # Canonical SWA type is "sliding_attention" but treat any non-full as SWA.
    if not full_ids:
        # Degenerate (all SWA) — still build two groups but full empty.
        full_ids = ()
    if not swa_ids:
        swa_ids = ()

    # Sliding window required when SWA layers exist.
    sw = int(sliding_window) if sliding_window is not None else 0
    if swa_ids and sw <= 0:
        # Default rescue: Laguna S 2.1 ships 512; keep a clear error if missing.
        raise ValueError("Laguna config has sliding_attention layers but no sliding_window")

    # Architecture bookkeeping.
    architectures = (
        top_architectures
        or getattr(cfg, "architectures", None)
        or ["LagunaForCausalLM"]
    )

    # MoE.
    num_experts = int(getattr(cfg, "num_experts", 0) or 0)
    num_experts_per_tok = int(getattr(cfg, "num_experts_per_tok", 0) or 0)
    moe_intermediate_size = int(getattr(cfg, "moe_intermediate_size", 0) or 0)
    shared_expert_intermediate_size = int(getattr(cfg, "shared_expert_intermediate_size", 0) or 0)
    norm_topk_prob = bool(getattr(cfg, "norm_topk_prob", True))
    # Laguna names it moe_routed_scaling_factor (2.5 for S 2.1); it multiplies the
    # routed-expert sum before the shared expert is added (modeling_laguna.py:255).
    # The repo convention folds it into topk_weights instead, which is equivalent.
    routed_scaling_factor = float(getattr(cfg, "moe_routed_scaling_factor", 1.0) or 1.0)
    moe_enabled = num_experts > 0 and num_experts_per_tok > 0
    # decoder_sparse_step + mlp_only_layers gating is handled in model.py (layer 0 dense).

    # Quant: Laguna NVFP4 on routed experts (compressed-tensors W4A16). Only layers
    # 1-39 are packed; 40-47 ship bf16 experts (the checkpoint's `ignore` list) and are
    # quantized to NVFP4 at conversion so all 47 layers share one bank layout.
    try:
        is_nvfp4 = detect_compressed_tensors_nvfp4(top_cfg)
    except ValueError:
        # An unsupported 4-bit float scheme (e.g. MXFP4 group_size 32) must surface
        # clearly instead of silently falling back to bf16.
        raise
    expert_quant = "nvfp4" if is_nvfp4 else "none"

    gating = getattr(cfg, "gating", "per-head")
    gating_types = getattr(cfg, "gating_types", None)
    if gating_types is not None:
        gating_types = tuple(str(x) for x in gating_types)

    # Laguna S 2.1: only layer 0 is dense (mlp_only_layers [0]).
    # Map to first_k_dense_replace so ModelConfig.num_moe_layers == 47.
    mlp_only_layers = getattr(cfg, "mlp_only_layers", None)
    if mlp_only_layers is not None and len(mlp_only_layers) > 0 and mlp_only_layers == [0]:
        first_k_dense_replace = 1
    else:
        first_k_dense_replace = int(getattr(cfg, "first_k_dense_replace", 0) or 0)

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        intermediate_size=intermediate_size,
        rms_norm_eps=rms_norm_eps,
        tie_word_embeddings=tie_word_embeddings,
        rotary_config=full_rotary,
        hidden_act=hidden_act,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
        model_type=getattr(cfg, "model_type", "laguna"),
        architectures=list(architectures),
        moe_enabled=moe_enabled,
        first_k_dense_replace=first_k_dense_replace,
        expert_quant=expert_quant,
        attn_quant="none",
        dense_quant="none",
        lm_head_quant="none",
        use_qk_norm=True,
        attn_sm_scale=None,  # 1/sqrt(head_dim) default
        attention_groups=(
            FullAttentionGroupConfig(
                name="full",
                layer_ids=full_ids,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_config=full_rotary,
            ),
            SWAAttentionGroupConfig(
                name="swa",
                layer_ids=swa_ids,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_config=swa_rotary,
                sliding_window=sw,
            ),
        ),
        num_attention_heads_per_layer=num_attention_heads_per_layer,
        laguna_args=LagunaArgs(gating=gating, gating_types=gating_types),
    )


__all__ = ["LagunaArgs", "parse_config"]
