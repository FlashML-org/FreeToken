"""Engine-facing config for DeepSeek-V4.1-Flash.

``parse_config`` maps the standard transformer fields into :class:`ModelConfig`, carries the full
:class:`DeepseekV41Args` in ``ModelConfig.dsv41_args`` for the model module, and declares the DSV41
attention group whose :class:`DSV41Geometry` prices the KV pool. The engine reconciles the runtime
knobs (``max_seq_len``, ``decoder_replay``) onto ``dsv41_args`` at config resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from freetoken.kvcache.dsv41_geometry import DSV41Geometry
from freetoken.models.config import DSV41AttentionGroupConfig, ModelConfig, RotaryConfig

from .args import DeepseekV41Args


@dataclass(frozen=True)
class VisionConfig:
    num_hidden_layers: int = 32
    hidden_size: int = 1024
    num_attention_heads: int = 16
    intermediate_size: int = 2816
    patch_size: int = 14
    rope_theta: float = 10000.0
    downsample_ratio: int = 3
    max_image_tokens: int = 1024
    min_pixels: int = 544 * 544
    max_wh_ratio: float | None = None


def dsv41_geometry(args: DeepseekV41Args) -> DSV41Geometry:
    return DSV41Geometry(
        n_layers=args.n_layers,
        head_dim=args.head_dim,
        index_head_dim=args.index_head_dim,
        window=args.window_size,
        compress_ratios=tuple(args.compress_ratios[: args.n_layers]),
        kv_source_layer_ids=args.backbone_kv_sources,
        # bounded replay recomputes the window before a prefix hit, which reads the window before that
        resume_windows=1 if args.decoder_replay == "exact" else 2,
        # bounded replay computes the decoder only over each request's last window: that KV is
        # request-private and lives in per-request rings, never in radix-shared pages
        private_window_layer_ids=() if args.decoder_replay == "exact" else tuple(range(args.decoder_start_layer, args.n_layers)),
    )


def parse_config(hf_config: Any) -> ModelConfig:
    args = DeepseekV41Args.from_hf(hf_config)
    text = getattr(hf_config, "text_config", None) or hf_config
    max_position = int(getattr(text, "max_position_embeddings", 0) or getattr(hf_config, "max_position_embeddings", 0) or 0)
    if max_position <= 0:
        max_position = int(args.original_seq_len * args.rope_factor)
    vc = getattr(hf_config, "vision_config", None)
    vision = VisionConfig(**{f.name: getattr(vc, f.name, f.default) for f in fields(VisionConfig)}) if vc is not None else None

    return ModelConfig(
        num_layers=args.n_layers,
        num_qo_heads=args.n_heads,
        num_kv_heads=1,  # one shared latent KV head (K == V)
        head_dim=args.head_dim,
        hidden_size=args.dim,
        vocab_size=args.vocab_size,
        intermediate_size=args.moe_inter_dim,
        hidden_act="silu",
        rms_norm_eps=args.norm_eps,
        tie_word_embeddings=False,
        rotary_config=RotaryConfig(
            head_dim=args.head_dim,
            rotary_dim=args.rope_head_dim,
            max_position=max_position,
            base=args.rope_theta,
            scaling={
                "rope_type": "yarn",
                "factor": args.rope_factor,
                "beta_fast": args.beta_fast,
                "beta_slow": args.beta_slow,
                "original_max_position_embeddings": args.original_seq_len,
            },
        ),
        num_experts=args.n_routed_experts,
        num_experts_per_tok=args.n_activated_experts,
        moe_intermediate_size=args.moe_inter_dim,
        norm_topk_prob=args.norm_topk_prob,
        n_shared_experts=args.n_shared_experts,
        routed_scaling_factor=args.route_scale,
        swiglu_limit=args.swiglu_limit,
        model_type="deepseek_v41",
        architectures=["DeepseekV41ForCausalLM"],
        moe_enabled=True,
        expert_quant="ds_fp4",
        attn_sm_scale=args.head_dim**-0.5,
        dsv41_args=args,
        vision_config=vision,
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=(
            DSV41AttentionGroupConfig(
                name="dsv41",
                layer_ids=tuple(range(args.n_layers)),
                num_kv_heads=1,
                head_dim=args.head_dim,
                sliding_window=args.window_size,
                geometry=dsv41_geometry(args),
            ),
        ),
    )


__all__ = ["parse_config", "dsv41_geometry"]
