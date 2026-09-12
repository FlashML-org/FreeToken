from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    mrope_layout_from_rope_params,
)
from freetoken.models.qwen3.config import parse_config as parse_qwen3
from freetoken.models.qwen3_moe.config import parse_config as parse_qwen3_moe


@dataclass
class VisionConfig:
    hidden_size: int
    depth: int
    num_heads: int
    intermediate_size: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    num_position_embeddings: int
    out_hidden_size: int
    in_channels: int
    # ViT block indices whose merged features are added to the first decoder layers; () disables DeepStack
    deepstack_visual_indexes: tuple[int, ...] = ()


def parse_vision_config(hf_config: Any) -> VisionConfig | None:
    """None when the config carries no vision section, which is how a text-only engine asks for no tower and 1-D rope."""
    vc = getattr(hf_config, "vision_config", None)
    if vc is None:
        return None
    return VisionConfig(
        hidden_size=vc.hidden_size,
        depth=vc.depth,
        num_heads=vc.num_heads,
        intermediate_size=vc.intermediate_size,
        patch_size=vc.patch_size,
        temporal_patch_size=vc.temporal_patch_size,
        spatial_merge_size=vc.spatial_merge_size,
        num_position_embeddings=vc.num_position_embeddings,
        out_hidden_size=vc.out_hidden_size,
        in_channels=vc.in_channels,
        deepstack_visual_indexes=tuple(getattr(vc, "deepstack_visual_indexes", None) or ()),
    )


def parse_config(hf_config: Any) -> ModelConfig:
    """Qwen3-VL: a Qwen3 (dense or MoE) text tower under text_config plus the shared Qwen VL vision tower."""
    text = hf_config.text_config
    base = parse_qwen3_moe(text) if getattr(text, "num_experts", 0) else parse_qwen3(text)

    rope_params = text.rope_parameters
    rope_type = rope_params.get("rope_type", "default")
    # the default rope type needs no scaling dict, and the mrope_section list must stay out of get_rope's cache key
    rope_scaling = (
        None
        if rope_type in (None, "default")
        else {k: v for k, v in rope_params.items() if not isinstance(v, (list, dict))}
    )
    vision_config = parse_vision_config(hf_config)
    rotary = RotaryConfig(
        head_dim=base.head_dim,
        rotary_dim=base.head_dim,
        max_position=text.max_position_embeddings,
        base=rope_params["rope_theta"],
        scaling=rope_scaling,
        mrope_section=(
            list(rope_params["mrope_section"])
            if vision_config is not None and "mrope_section" in rope_params
            else None
        ),
        mrope_layout=mrope_layout_from_rope_params(rope_params),
    )
    # an explicit group carries the mrope rotary config, which model_is_mrope reads from the groups
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=tuple(range(base.num_layers)),
        num_kv_heads=base.num_kv_heads,
        head_dim=base.head_dim,
        rotary_config=rotary,
    )
    return dataclasses.replace(
        base,
        rotary_config=rotary,
        attention_groups=(full_group,),
        moe_enabled=bool(getattr(text, "num_experts", 0)),
        vision_config=vision_config,
        image_token_id=hf_config.image_token_id,
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        model_type=hf_config.model_type,
        architectures=list(hf_config.architectures),
    )


__all__ = ["VisionConfig", "parse_config", "parse_vision_config"]
