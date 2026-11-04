from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DFlashDraftConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    hidden_act: str
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    window: int
    block_size: int
    # the output of target layer i (0-based) feeds tap i, for every supported target
    target_layer_ids: tuple[int, ...]
    mask_token_id: int
    selector_top_k: int
    selector_rank: int
    conv_kernel_size: int
    conv_group_size: int
    num_target_layers: int


def parse_config(hf_config: Any) -> DFlashDraftConfig:
    """The draft's ``config.json``; only the DFlash 2 shape (non-causal, all sliding, conv + selector, default rope) is served."""
    archs = list(getattr(hf_config, "architectures", None) or [])
    if archs != ["DFlash2DraftModel"]:
        raise ValueError(f"expected a DFlash2DraftModel checkpoint, got architectures {archs}")
    dc = hf_config.dflash_config
    if not dc.get("selector_rank") or not dc.get("conv_kernel_size"):
        raise ValueError("only DFlash 2 drafts (dflash_config with selector_rank and conv_kernel_size) are supported")
    missing = [key for key in ("block_size", "target_layer_ids", "mask_token_id", "selector_top_k", "conv_group_size") if key not in dc]
    if missing:
        raise ValueError(f"dflash_config lacks {missing}")
    if getattr(hf_config, "is_causal", False):
        raise ValueError("only non-causal (is_causal false) DFlash drafts are supported")
    types = set(getattr(hf_config, "layer_types", None) or ["full_attention"])
    # the ring holds one window; a full-attention draft has no bound
    if types != {"sliding_attention"}:
        raise ValueError(f"only sliding_attention DFlash drafts are supported, got layer_types {sorted(types)}")
    if getattr(hf_config, "sliding_window", None) is None:
        raise ValueError("DFlash drafts need a sliding_window (use_sliding_window true)")
    hidden, head_dim = int(hf_config.hidden_size), int(hf_config.head_dim)
    rope = getattr(hf_config, "rope_parameters", None) or {"rope_theta": getattr(hf_config, "rope_theta", 10000.0)}
    if rope.get("rope_type", "default") != "default":
        raise ValueError(f"rope_type {rope.get('rope_type')!r} is not supported for DFlash drafts")
    rope_dim = rope.get("rope_dim") or head_dim
    factor = rope.get("partial_rotary_factor") or getattr(hf_config, "partial_rotary_factor", None) or 1.0
    neox = bool(getattr(hf_config, "rope_is_neox_style", getattr(hf_config, "is_neox_style", True)))
    # norm_rope_store rotates the whole head in NeoX halves
    if (int(rope_dim), float(factor), neox) != (head_dim, 1.0, True):
        raise ValueError(f"DFlash drafts need full NeoX rotary, got rope_dim {rope_dim}, partial_rotary_factor {factor}, neox style {neox}")
    # the draft forward does not apply these; ignoring them would only show up as lower acceptance
    identity = {"output_multiplier": (None, 1.0), "final_logit_softcapping": (None, 0.0), "attention_value_scale": (None, 1.0)}
    scaled = {key: dc[key] for key, values in identity.items() if dc.get(key) not in values}
    if scaled:
        raise ValueError(f"DFlash drafts with {scaled} in dflash_config are not supported")
    # tl.arange in the draft kernels needs powers of two
    sizes = {"selector_top_k": int(dc["selector_top_k"]), "selector_rank": int(dc["selector_rank"]), "head_dim": head_dim}
    odd = {key: v for key, v in sizes.items() if v < 1 or v & (v - 1)}
    if odd:
        raise ValueError(f"DFlash drafts need powers of two for {odd}")
    group = int(dc["conv_group_size"])
    if group < 1 or hidden % group:
        raise ValueError(f"conv_group_size {group} does not divide hidden_size {hidden}")
    taps = tuple(int(i) for i in dc["target_layer_ids"])
    # each tap fills its own column block of the feature buffer
    if not taps or len(set(taps)) != len(taps) or min(taps) < 0:
        raise ValueError(f"target_layer_ids {list(taps)} must be non-empty, distinct and >= 0")
    return DFlashDraftConfig(
        hidden_size=hidden,
        num_layers=int(hf_config.num_hidden_layers),
        num_heads=int(hf_config.num_attention_heads),
        num_kv_heads=int(hf_config.num_key_value_heads),
        head_dim=head_dim,
        intermediate_size=int(hf_config.intermediate_size),
        hidden_act=str(getattr(hf_config, "hidden_act", "silu")),
        vocab_size=int(hf_config.vocab_size),
        rms_norm_eps=float(getattr(hf_config, "rms_norm_eps", 1e-6)),
        rope_theta=float(rope["rope_theta"]),
        window=int(hf_config.sliding_window),
        block_size=int(dc["block_size"]),
        target_layer_ids=taps,
        mask_token_id=int(dc["mask_token_id"]),
        selector_top_k=sizes["selector_top_k"],
        selector_rank=sizes["selector_rank"],
        conv_kernel_size=int(dc["conv_kernel_size"]),
        conv_group_size=group,
        num_target_layers=int(hf_config.num_target_layers),
    )
