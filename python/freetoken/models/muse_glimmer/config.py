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

_SWA_TYPE = "sliding_attention"
_FULL_TYPE = "full_attention"
_VISION_LAYER_TYPES = {"window_attention", "full_attention"}


@dataclass(frozen=True)
class VisionConfig:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    layer_types: tuple[str, ...]
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    pos_emb_side: int
    layer_norm_eps: float
    rope_theta: float
    projector_hidden_size: int
    text_hidden_size: int
    text_rms_norm_eps: float

    @property
    def patch_dim(self) -> int:
        return self.temporal_patch_size * 3 * self.patch_size**2

    @property
    def out_hidden_size(self) -> int:
        return self.hidden_size * self.merge_size**2


def parse_vision_config(hf_config: Any) -> VisionConfig | None:
    """None when the config carries no vision section, which is how a text-only engine asks for no tower."""
    vc = getattr(hf_config, "vision_config", None)
    if vc is None:
        return None
    if vc.hidden_act != "gelu" or hf_config.projector_hidden_act != "gelu":
        raise NotImplementedError(
            f"muse_glimmer vision activations {vc.hidden_act!r} / {hf_config.projector_hidden_act!r}; only gelu is implemented"
        )
    if vc.pos_emb_height != vc.pos_emb_width:
        raise NotImplementedError("muse_glimmer vision tower: the position table must be square")
    layer_types = tuple(vc.layer_types)
    assert set(layer_types) <= _VISION_LAYER_TYPES, f"unknown vision layer types in {set(layer_types)}"
    rope_params = getattr(vc, "rope_parameters", None) or {}
    text = _text_config(hf_config)
    config = VisionConfig(
        hidden_size=vc.hidden_size,
        intermediate_size=vc.intermediate_size,
        num_layers=vc.num_hidden_layers,
        num_heads=vc.num_attention_heads,
        layer_types=layer_types,
        patch_size=vc.patch_size,
        temporal_patch_size=vc.patch_temporal,
        merge_size=vc.merge_size,
        pos_emb_side=vc.pos_emb_height,
        layer_norm_eps=vc.layer_norm_eps,
        rope_theta=float(rope_params.get("rope_theta", 10000.0)),
        projector_hidden_size=hf_config.projector_hidden_size,
        text_hidden_size=text.hidden_size,
        text_rms_norm_eps=text.rms_norm_eps,
    )
    assert hf_config.out_hidden_size == config.out_hidden_size, (
        f"adapter input {hf_config.out_hidden_size} != pixel-shuffled tower width {config.out_hidden_size}"
    )
    return config


def _text_config(hf_config: Any) -> Any:
    """Muse Glimmer ships as a multimodal wrapper (MuseGlimmerForConditionalGeneration):
    the text tower lives in ``text_config`` and the weights carry a ``language_model.``
    prefix."""
    text = getattr(hf_config, "text_config", None)
    return text if text is not None else hf_config


def _group_rope_theta(text: Any, layer_ids: tuple[int, ...], default: float) -> float:
    """Rope theta of one layer group. ``layer_rope_theta`` is the per-layer source of
    truth (0 marks the NoPE full-attention layers); absent, fall back to ``default``
    (the shared ``rope_parameters.rope_theta``)."""
    per_layer = getattr(text, "layer_rope_theta", None)
    if not per_layer:
        return default
    thetas = {float(per_layer[i]) for i in layer_ids}
    assert len(thetas) == 1, f"mixed rope thetas within one attention group: {thetas}"
    return thetas.pop()


def parse_config(hf_config: Any) -> ModelConfig:
    text = _text_config(hf_config)

    head_dim = getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads
    num_kv_heads = getattr(text, "num_key_value_heads", text.num_attention_heads)
    max_position = text.max_position_embeddings

    rope_params = getattr(text, "rope_parameters", None) or {}
    base_theta = float(rope_params.get("rope_theta", getattr(text, "rope_theta", 500000.0)))

    layer_types = list(text.layer_types)
    swa_ids = tuple(i for i, t in enumerate(layer_types) if t == _SWA_TYPE)
    full_ids = tuple(i for i, t in enumerate(layer_types) if t == _FULL_TYPE)
    assert len(swa_ids) + len(full_ids) == len(layer_types), (
        f"unknown layer types in {set(layer_types)}"
    )

    swa_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=head_dim,
        max_position=max_position,
        base=_group_rope_theta(text, swa_ids, base_theta),
        scaling=None,
    )
    # Full-attention layers are NoPE (layer_rope_theta == 0): base 0.0 is the marker the
    # attention module reads to skip rope entirely. It must never reach RotaryEmbedding
    # (0 ** x); MuseGlimmerAttention guards on it.
    full_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=head_dim,
        max_position=max_position,
        base=_group_rope_theta(text, full_ids, 0.0),
        scaling=None,
    )

    # The reference applies a weightless per-head RMSNorm to q and k, then multiplies q by
    # qk_scale_factor and runs standard 1/sqrt(head_dim) attention. q is not cached, so the
    # factor folds exactly into the softmax scale.
    qk_scale_factor = float(getattr(text, "qk_scale_factor", 1.0))
    attn_sm_scale = qk_scale_factor * head_dim**-0.5

    # RedHatAI/Muse-Glimmer-30B-NVFP4: every text-tower Linear (q/k/v/o + the attention
    # gate + the MLP) is compressed-tensors NVFP4 (W4A16); lm_head, embeddings, norms and
    # the (unserved) vision tower stay bf16.
    nvfp4 = detect_compressed_tensors_nvfp4(hf_config)
    quant = "nvfp4" if nvfp4 else "none"

    return ModelConfig(
        num_layers=text.num_hidden_layers,
        num_qo_heads=text.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=text.hidden_size,
        vocab_size=text.vocab_size,
        intermediate_size=text.intermediate_size,
        hidden_act=getattr(text, "hidden_activation", "silu"),
        rms_norm_eps=text.rms_norm_eps,
        post_norm_eps=float(getattr(text, "post_norm_eps", text.rms_norm_eps)),
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=swa_rotary,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        model_type=getattr(hf_config, "model_type", "muse_glimmer"),
        architectures=(
            getattr(hf_config, "architectures", None)
            or ["MuseGlimmerForConditionalGeneration"]
        ),
        use_qk_norm=True,
        attn_sm_scale=attn_sm_scale,
        final_logit_softcapping=getattr(text, "final_logit_softcapping", None),
        output_multiplier=getattr(text, "output_multiplier", None),
        vision_config=parse_vision_config(hf_config),
        image_token_id=getattr(hf_config, "image_token_id", None),
        attn_quant=quant,
        dense_quant=quant,
        lm_head_quant="none",
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
                sliding_window=text.sliding_window,
            ),
        ),
    )


__all__ = ["VisionConfig", "parse_config", "parse_vision_config"]
