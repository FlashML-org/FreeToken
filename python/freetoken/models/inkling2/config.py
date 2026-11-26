"""Inkling 2 Mini (``inkling2_loop``) hyperparameters.

A 24-layer MoE decoder whose middle layers ``[loop_start, loop_end)`` run ``num_passes``
times with shared weights. Every *visit* of a layer keeps its own KV cache and its own
short-convolution state, so the engine sees ``num_visits`` attention layers (48 for the
release) over ``num_layers`` decoder layers: KV layer id == visit index, in execution
order. Visits ``v`` with ``(v + 1) % 6 == 0`` are the global-attention ones, the rest
attend within a 1024-token window.

The attention has no rotary embedding: position enters through a hidden-state-conditioned
relative bias over the last ``rel_extent`` distances, short causal convolutions on K and V
and per-token sink logits. Semantics follow the Hugging Face reference
(``transformers`` PR #51050, ``models/inkling2_loop``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
    WindowStateSpec,
)

IMAGE_TOKEN_ID = 200054


@dataclass(frozen=True)
class Inkling2Args:
    hidden_size: int
    num_layers: int
    block_size: int  # decoder layers summed into one attention-residual block
    loop_start: int
    loop_end: int
    num_passes: int
    gate_rank: int
    swa_layers: frozenset[int]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    swa_num_heads: int
    swa_num_kv_heads: int
    swa_head_dim: int
    sliding_window: int
    d_rel: int
    rel_extent: int  # backward distances the full-attention bias covers (SWA: the window)
    log_scaling_n_floor: int
    log_scaling_alpha: float
    qk_clip: float
    rel_clip: float
    conv_kernel: int
    rms_norm_eps: float
    num_experts: int
    top_k: int
    moe_intermediate_size: int
    route_scale: float
    gate_softcap: float
    up_softcap: float
    final_logit_softcap: float
    vocab_size: int  # rows of the stored embed / unembed tables
    unpadded_vocab_size: int  # real tokens; the logits stop here

    @property
    def num_loop_layers(self) -> int:
        return self.loop_end - self.loop_start

    @property
    def num_visits(self) -> int:
        return self.num_layers + (self.num_passes - 1) * self.num_loop_layers

    def visit(self, layer: int, pass_idx: int = 0) -> int:
        """KV layer id of ``layer``'s ``pass_idx``-th execution."""
        if layer < self.loop_start:
            assert pass_idx == 0
            return layer
        if layer < self.loop_end:
            return self.loop_start + pass_idx * self.num_loop_layers + layer - self.loop_start
        assert pass_idx == 0
        return layer + (self.num_passes - 1) * self.num_loop_layers

    def visit_layers(self) -> list[int]:
        """Decoder layer of each visit, in execution order."""
        loop = list(range(self.loop_start, self.loop_end))
        return list(range(self.loop_start)) + loop * self.num_passes + list(range(self.loop_end, self.num_layers))

    def is_swa(self, layer: int) -> bool:
        return layer in self.swa_layers

    def is_loop(self, layer: int) -> bool:
        return self.loop_start <= layer < self.loop_end

    def is_block_boundary(self, layer: int) -> bool:
        return layer % self.block_size == 0


@dataclass(frozen=True)
class VisionConfig:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    patch_size: int  # native ViT patch side
    merge_size: int  # native patches per LM token side
    posemb_grid_size: int
    layer_norm_eps: float
    rope_theta: float
    projection_hidden_dim: int
    text_hidden_size: int

    @property
    def token_patch_side(self) -> int:
        """Pixel side of the square one LM image token covers (40 for the release)."""
        return self.patch_size * self.merge_size


def _text_config(hf_config: Any) -> Any:
    text = getattr(hf_config, "text_config", None)
    return text if text is not None else hf_config


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def parse_vision_config(hf_config: Any) -> VisionConfig | None:
    vc = getattr(hf_config, "vision_config", None)
    if vc is None:
        return None
    hidden = _get(vc, "hidden_size", 1024)
    rope = _get(vc, "rope_parameters", None) or {}
    if _get(vc, "hidden_act", "gelu") != "gelu":
        raise NotImplementedError(f"inkling2 vision activation {_get(vc, 'hidden_act')!r}; only gelu is implemented")
    if _get(vc, "projector_bias", False):
        raise NotImplementedError("inkling2 vision projector with biases is not implemented")
    if _get(vc, "temporal_patch_size", 2) != 2:
        raise NotImplementedError("inkling2 vision tower: only temporal_patch_size 2 (single images) is implemented")
    return VisionConfig(
        hidden_size=hidden,
        intermediate_size=int(_get(vc, "intermediate_size", None) or hidden * _get(vc, "mlp_ratio", 4.0)),
        num_layers=_get(vc, "num_hidden_layers", 23),
        num_heads=_get(vc, "num_attention_heads", 16),
        patch_size=_get(vc, "native_patch_size", None) or _get(vc, "patch_size", 20),
        merge_size=_get(vc, "adapter_merge_size", None) or _get(vc, "spatial_merge_size", 2),
        posemb_grid_size=_get(vc, "posemb_grid_size", 32),
        layer_norm_eps=_get(vc, "layer_norm_eps", 1e-5),
        rope_theta=float(rope.get("rope_theta", 10000.0)),
        projection_hidden_dim=_get(vc, "projection_hidden_dim", 4096),
        text_hidden_size=_get(vc, "text_hidden_size", 2304),
    )


def parse_args(text: Any) -> Inkling2Args:
    loop = _get(text, "loop_config", None) or {}
    if _get(loop, "is_varloop", False):
        raise NotImplementedError("the variable-loop research checkpoint is not served yet")
    num_layers = text.num_hidden_layers
    local = _get(text, "local_layer_ids", None)
    if local is None:
        layer_types = _get(text, "layer_types", None)
        if layer_types is None:
            layer_types = ["hybrid" if (i + 1) % 6 == 0 else "hybrid_sliding" for i in range(num_layers)]
        local = [i for i, t in enumerate(layer_types) if t == "hybrid_sliding"]
    args = Inkling2Args(
        hidden_size=text.hidden_size,
        num_layers=num_layers,
        block_size=_get(text, "n_layers_per_attn_res_block", 6),
        loop_start=_get(loop, "core_start", _get(text, "loop_layer_index_start", 6)),
        loop_end=_get(loop, "core_end", _get(text, "loop_layer_index_end", 18)),
        num_passes=_get(loop, "num_passes", _get(text, "loop_num_passes", 3)),
        gate_rank=_get(loop, "gate_rank", _get(text, "loop_gate_rank", 64)),
        swa_layers=frozenset(local),
        num_heads=text.num_attention_heads,
        num_kv_heads=text.num_key_value_heads,
        head_dim=text.head_dim,
        swa_num_heads=_get(text, "swa_num_attention_heads", text.num_attention_heads),
        swa_num_kv_heads=_get(text, "swa_num_key_value_heads", text.num_key_value_heads),
        swa_head_dim=_get(text, "swa_head_dim", text.head_dim),
        sliding_window=_get(text, "sliding_window_size", 1024),
        d_rel=_get(text, "d_rel", 16),
        rel_extent=_get(text, "rel_extent", 1024),
        log_scaling_n_floor=_get(text, "log_scaling_n_floor", 8192),
        log_scaling_alpha=float(_get(text, "log_scaling_alpha", 0.1)),
        qk_clip=float(_get(text, "qk_l2_clip_threshold", 22.627416997969522)),
        rel_clip=float(_get(text, "rel_l2_clip_threshold", 8.0)),
        conv_kernel=_get(text, "sconv_kernel_size", None) or _get(text, "conv_kernel_size", 4),
        rms_norm_eps=float(text.rms_norm_eps),
        num_experts=_get(text, "n_routed_experts", 128),
        top_k=text.num_experts_per_tok,
        moe_intermediate_size=_get(text, "intermediate_size", None) or text.moe_intermediate_size,
        route_scale=float(_get(text, "route_scale", 8.0)),
        gate_softcap=float(_get(text, "mlp_gate_softcap", 4.0)),
        up_softcap=float(_get(text, "mlp_up_softcap", 25.0)),
        final_logit_softcap=float(_get(text, "final_logit_softcapping", 30.0)),
        vocab_size=text.vocab_size,
        unpadded_vocab_size=_get(text, "unpadded_vocab_size", text.vocab_size),
    )
    if _get(text, "n_shared_experts", 0):
        raise NotImplementedError("inkling2 shared experts are not implemented")
    if _get(text, "q_bias", False) or _get(text, "o_bias", False):
        raise NotImplementedError("inkling2 attention biases are not implemented")
    if not _get(text, "use_gate_bias", True):
        raise NotImplementedError("inkling2 router without a selection bias is not implemented")
    if args.loop_start % args.block_size or args.loop_end % args.block_size or args.loop_start <= 0:
        raise ValueError(f"loop [{args.loop_start}, {args.loop_end}) must start after layer 0 on attention-residual block boundaries")
    if (args.swa_num_heads, args.swa_num_kv_heads, args.swa_head_dim) != (args.num_heads, args.num_kv_heads, args.head_dim):
        raise NotImplementedError("inkling2 sliding and full layers with different head geometry are not implemented")
    if args.d_rel != 16:
        raise NotImplementedError(f"inkling2 relative-bias rank {args.d_rel}; the kernels assume 16")
    return args


def parse_config(hf_config: Any) -> ModelConfig:
    text = _text_config(hf_config)
    args = parse_args(text)
    visit_layers = args.visit_layers()
    swa_ids = tuple(v for v, layer in enumerate(visit_layers) if args.is_swa(layer))
    full_ids = tuple(v for v, layer in enumerate(visit_layers) if not args.is_swa(layer))
    max_position = int(_get(text, "model_max_length", None) or _get(text, "max_position_embeddings", 1048576))
    # No rotary embedding anywhere; the table size only bounds the served sequence length.
    rotary = RotaryConfig(head_dim=args.head_dim, rotary_dim=args.head_dim, max_position=max_position, base=0.0, scaling=None)
    return ModelConfig(
        num_layers=args.num_layers,
        num_kv_layers=args.num_visits,
        num_qo_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        hidden_size=args.hidden_size,
        # the logits width; the embedding table keeps the stored (padded) rows
        vocab_size=args.unpadded_vocab_size,
        intermediate_size=args.moe_intermediate_size,
        rms_norm_eps=args.rms_norm_eps,
        rotary_config=rotary,
        hidden_act="softcap_swiglu",
        tie_word_embeddings=False,
        num_experts=args.num_experts,
        num_experts_per_tok=args.top_k,
        moe_intermediate_size=args.moe_intermediate_size,
        norm_topk_prob=True,
        model_type=getattr(hf_config, "model_type", "inkling2_loop"),
        architectures=list(getattr(hf_config, "architectures", None) or ["Inkling2LoopForConditionalGeneration"]),
        moe_enabled=True,
        expert_quant="nvfp4",
        routed_scaling_factor=args.route_scale,
        has_router_bias=True,
        final_logit_softcapping=args.final_logit_softcap,
        vision_config=parse_vision_config(hf_config),
        image_token_id=getattr(hf_config, "image_token_id", IMAGE_TOKEN_ID),
        attention_groups=(
            FullAttentionGroupConfig(
                name="full",
                layer_ids=full_ids,
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
                rotary_config=rotary,
            ),
            SWAAttentionGroupConfig(
                name="swa",
                layer_ids=swa_ids,
                num_kv_heads=args.swa_num_kv_heads,
                head_dim=args.swa_head_dim,
                rotary_config=rotary,
                sliding_window=args.sliding_window,
            ),
        ),
        inkling2_args=args,
        moe_layer_order=tuple(visit_layers) if args.num_passes > 1 else (),
        window_states=(conv_state_spec(args),),
        # its window KV serves only the resume points, not every cached token
        swa_full_tokens_ratio=0.1,
    )


CONV_STATE = "inkling2_conv"


def conv_state_widths(args: Inkling2Args) -> dict[str, int]:
    """Channels of each visit's four short convolutions, side by side in one state row."""
    kv = args.num_kv_heads * args.head_dim
    return {"k": kv, "v": kv, "attn": args.hidden_size, "mlp": args.hidden_size}


def conv_state_spec(args: Inkling2Args) -> WindowStateSpec:
    """Each visit's convolution inputs per token (bf16 projections, so bf16 is exact), one pool
    layer per visit; a token reads those of the ``K - 1`` tokens before it. A sequence starts from
    zeros."""
    return WindowStateSpec(
        name=CONV_STATE,
        shape=(sum(conv_state_widths(args).values()),),
        layer_ids=tuple(range(args.num_visits)),
        window=args.conv_kernel - 1,
        dtype=torch.bfloat16,
    )


__all__ = [
    "CONV_STATE", "IMAGE_TOKEN_ID", "Inkling2Args", "VisionConfig", "conv_state_spec", "conv_state_widths",
    "parse_args", "parse_config", "parse_vision_config",
]
