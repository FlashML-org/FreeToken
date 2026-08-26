from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
    detect_expert_quant,
    vision_load_enabled,
)


@dataclass(frozen=True)
class VisionConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    patch_size: int
    position_embedding_size: int
    pooling_kernel_size: int
    rms_norm_eps: float
    rope_theta: float
    hidden_act: str
    standardize: bool
    use_clipped_linears: bool
    soft_tokens_per_image: int
    text_hidden_size: int


def _text_config(hf_config: Any) -> tuple[Any, list[str] | None, Any]:
    top_architectures = getattr(hf_config, "architectures", None)
    if hasattr(hf_config, "text_config") and hf_config.text_config is not None:
        return hf_config.text_config, top_architectures, hf_config
    return hf_config, top_architectures, hf_config


def _parse_vision_config(top_cfg: Any, text_hidden_size: int) -> VisionConfig | None:
    vc = getattr(top_cfg, "vision_config", None)
    if vc is None:
        return None
    # Vision is opt-in (default OFF). The vision tower + multimodal embedder are ~1 GiB of
    # resident, never-quantized (bf16) GPU weights that text-only serving never touches:
    # forward() ignores them and _merge_multimodal is a no-op without image embeds. Default
    # to text-only so that VRAM goes to KV / the expert cache; set FREETOKEN_LOAD_VISION=1
    # to load it. Returning None here is the single switch -- both the model build (model.py)
    # and weight loading (weight.py `include_vision = config.is_multimodal`) flow through
    # parse_config, so is_multimodal flips off consistently across both.
    if not vision_load_enabled():
        return None
    rope_params = getattr(vc, "rope_parameters", None) or {}
    act = getattr(vc, "hidden_activation", "gelu_pytorch_tanh")
    act = "gelu_tanh" if "tanh" in act else act
    num_heads = vc.num_attention_heads
    return VisionConfig(
        hidden_size=vc.hidden_size,
        num_layers=vc.num_hidden_layers,
        num_heads=num_heads,
        num_kv_heads=getattr(vc, "num_key_value_heads", num_heads),
        head_dim=getattr(vc, "head_dim", None) or vc.hidden_size // num_heads,
        intermediate_size=vc.intermediate_size,
        patch_size=vc.patch_size,
        position_embedding_size=vc.position_embedding_size,
        pooling_kernel_size=vc.pooling_kernel_size,
        rms_norm_eps=vc.rms_norm_eps,
        rope_theta=float(rope_params.get("rope_theta", 100.0)),
        hidden_act=act,
        standardize=bool(getattr(vc, "standardize", True)),
        use_clipped_linears=bool(getattr(vc, "use_clipped_linears", False)),
        soft_tokens_per_image=int(
            getattr(top_cfg, "vision_soft_tokens_per_image", None)
            or getattr(vc, "default_output_length", 280)
        ),
        text_hidden_size=text_hidden_size,
    )


def _attn_geometry(cfg: Any, layer_type: str, *, is_full: bool) -> tuple[int, int]:
    """``(head_dim, num_kv_heads)`` of one gemma-4 attention type.

    transformers >= 5.15 keeps both in ``per_layer_config`` and raises on a top-level read;
    older releases flattened them, the full layers' values under a ``global_*`` twin.
    """
    if getattr(cfg, "is_heterogeneous", False):
        layer_cfg = cfg.per_layer_config[layer_type]
        return layer_cfg.head_dim, layer_cfg.num_key_value_heads
    if not is_full:
        return cfg.head_dim, cfg.num_key_value_heads
    return (
        getattr(cfg, "global_head_dim", None) or cfg.head_dim,
        getattr(cfg, "num_global_key_value_heads", cfg.num_key_value_heads),
    )


def parse_config(hf_config: Any) -> ModelConfig:
    cfg, top_architectures, top_cfg = _text_config(hf_config)
    rope_params = cfg.rope_parameters
    swa_type, full_type = "sliding_attention", "full_attention"
    sliding_rope = rope_params[swa_type]
    full_rope = rope_params[full_type]
    sliding_head_dim, sliding_kv_heads = _attn_geometry(cfg, swa_type, is_full=False)
    full_head_dim, full_kv_heads = _attn_geometry(cfg, full_type, is_full=True)
    full_partial_rotary_factor = float(full_rope.get("partial_rotary_factor", 1.0))
    sliding_layer_ids = tuple(
        i for i, layer_type in enumerate(cfg.layer_types) if layer_type != full_type
    )
    full_layer_ids = tuple(
        i for i, layer_type in enumerate(cfg.layer_types) if layer_type == full_type
    )
    full_rotary_config = RotaryConfig(
        head_dim=full_head_dim,
        rotary_dim=int(full_head_dim * full_partial_rotary_factor),
        max_position=cfg.max_position_embeddings,
        base=float(full_rope["rope_theta"]),
        scaling={
            "rope_type": full_rope.get(
                "rope_type",
                "proportional" if full_partial_rotary_factor < 1.0 else "default",
            )
        },
    )
    swa_rotary_config = RotaryConfig(
        head_dim=sliding_head_dim,
        rotary_dim=sliding_head_dim,
        max_position=cfg.max_position_embeddings,
        base=float(sliding_rope["rope_theta"]),
        scaling=None,
    )
    architectures = (
        top_architectures
        or getattr(cfg, "architectures", None)
        or ["Gemma4ForConditionalGeneration"]
    )

    # Dense Gemma 4 checkpoints (gemma-4-31B-it, gemma-4-12B "Unified") carry no experts:
    # Gemma4UnifiedTextConfig omits these fields entirely, gemma4 leaves them None. Fall
    # back to 0 so moe_enabled -> is_moe -> False routes the dense feed-forward path.
    num_experts = getattr(cfg, "num_experts", None) or 0
    num_experts_per_tok = getattr(cfg, "top_k_experts", None) or 0
    moe_intermediate_size = getattr(cfg, "moe_intermediate_size", None) or 0

    # Dense modelopt-NVFP4 (e.g. nvidia/Gemma-4-31B-IT-NVFP4): only the dense MLP is W4A16
    # FP4; attn/lm_head/embeddings stay bf16 (modelopt ignore-list). Gated on `not moe_enabled`:
    # the sibling Gemma-4-26B-A4B-NVFP4 MoE checkpoint sets the same top-level quant_algo=NVFP4
    # for its routed experts while keeping shared_mlp bf16 -- do NOT drop this gate (nor extend
    # dense_quant to Gemma4MLP.shared_mlp) without re-deriving per-component quant from the
    # checkpoint's own ignore-list.
    expert_quant = detect_expert_quant(top_cfg)
    dense_quant = "nvfp4" if (expert_quant == "nvfp4" and not num_experts > 0) else "none"

    return ModelConfig(
        num_layers=cfg.num_hidden_layers,
        num_qo_heads=cfg.num_attention_heads,
        num_kv_heads=full_kv_heads,
        head_dim=full_head_dim,
        hidden_size=cfg.hidden_size,
        vocab_size=cfg.vocab_size,
        intermediate_size=cfg.intermediate_size,
        hidden_act="gelu_tanh",
        rms_norm_eps=cfg.rms_norm_eps,
        tie_word_embeddings=bool(getattr(cfg, "tie_word_embeddings", True)),
        rotary_config=full_rotary_config,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        norm_topk_prob=True,
        model_type=cfg.model_type,
        architectures=architectures,
        moe_enabled=num_experts > 0,
        expert_quant=expert_quant,
        dense_quant=dense_quant,
        use_qk_norm=True,
        attn_sm_scale=1.0,
        final_logit_softcapping=getattr(cfg, "final_logit_softcapping", None),
        embedding_scale=float(cfg.hidden_size) ** 0.5,
        vision_config=_parse_vision_config(top_cfg, cfg.hidden_size),
        image_token_id=getattr(top_cfg, "image_token_id", None),
        # Gemma-4 E-series (E2B/E4B): Per-Layer Embeddings + KV-layer sharing + double-wide MLP.
        # These keys are absent on the dense/MoE Gemma-4 line, so getattr defaults keep it off.
        per_layer_hidden_size=int(getattr(cfg, "hidden_size_per_layer_input", 0) or 0),
        per_layer_vocab_size=int(getattr(cfg, "vocab_size_per_layer_input", 0) or 0),
        num_kv_shared_layers=int(getattr(cfg, "num_kv_shared_layers", 0) or 0),
        use_double_wide_mlp=bool(getattr(cfg, "use_double_wide_mlp", False)),
        attention_groups=(
            FullAttentionGroupConfig(
                name="full",
                layer_ids=full_layer_ids,
                num_kv_heads=full_kv_heads,
                head_dim=full_head_dim,
                rotary_config=full_rotary_config,
                k_eq_v=bool(getattr(cfg, "attention_k_eq_v", False)),
            ),
            SWAAttentionGroupConfig(
                name="swa",
                layer_ids=sliding_layer_ids,
                num_kv_heads=sliding_kv_heads,
                head_dim=sliding_head_dim,
                rotary_config=swa_rotary_config,
                sliding_window=cfg.sliding_window,
            ),
        ),
    )


__all__ = ["VisionConfig", "parse_config"]
