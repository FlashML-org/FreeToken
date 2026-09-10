from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import torch

from freetoken.layers.quantization import QuantConfig
from freetoken.models.config import (
    fp8_dense_enabled,
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    SlotStateSpec,
)


@dataclass(frozen=True)
class Qwen4ExpArgs:
    """Qwen3.8-Flash-Next geometry beyond the generic ModelConfig fields (ModelConfig.qwen4_args)."""

    hidden_size: int
    # Hyper-connections: every layer reads/writes hc_count residual streams [T, hc_count*hidden].
    hc_count: int
    hc_lowrank: int
    # PLE n-gram embedding; layer ids are zero-based decoder layers.
    ple_layer_ids: Tuple[int, ...]
    ple_embed_dim: int
    ple_conv_kernel_size: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    make_ngram_vocab_size_divisible_by: int
    split_ngram_parts: int
    # n-gram hash windows never cross this token (the eos id); they restart after it.
    ngram_boundary_token_id: int
    # QSA indexer scoring geometry (the slab/ratio geometry lives on the attention group).
    index_n_heads: int
    index_kv_heads: int
    index_head_dim: int
    index_budget: int
    index_ratio: int

    @property
    def index_topk_blocks(self) -> int:
        return self.index_budget // self.index_ratio

    @property
    def num_ngram_heads(self) -> int:
        # one head group per n-gram order 2..ngram_size (Qwen3.8: 8 x 2-gram + 8 x 3-gram)
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def ngram_head_dim(self) -> int:
        return self.ple_embed_dim // self.num_ngram_heads

    @property
    def ple_conv_dilation(self) -> int:
        # HF Qwen4ExpTextPLELayer sets the depthwise conv dilation to ngram_size
        return self.ngram_size

    @property
    def ple_conv_state_len(self) -> int:
        return (self.ple_conv_kernel_size - 1) * self.ple_conv_dilation

    @property
    def ple_state_width(self) -> int:
        return self.hc_count * self.hidden_size


PLE_CONV_STATE = "ple_conv"
PLE_NGRAM_STATE = "ple_ngram_ctx"


def ple_slot_states(args: Qwen4ExpArgs) -> Tuple[SlotStateSpec, ...]:
    """Per-request PLE state riding the linear-state slots (see LinearStatePool.slot_states)."""
    if not args.ple_layer_ids:
        return ()
    return (
        # dilated-conv left context; replicated (not TP-sharded), model dtype
        SlotStateSpec(
            name=PLE_CONV_STATE,
            shape=(args.ple_state_width, args.ple_conv_state_len),
            layer_ids=args.ple_layer_ids,
        ),
        # last ngram_size-1 token ids, shared by every PLE layer; eos = hash boundary
        SlotStateSpec(
            name=PLE_NGRAM_STATE,
            shape=(args.ngram_size - 1,),
            dtype=torch.int32,
            fill_value=float(args.ngram_boundary_token_id),
        ),
    )


def _layer_types(text: Any) -> list[str]:
    layer_types = getattr(text, "layer_types", None)
    if layer_types is not None:
        # HF Qwen4ExpTextConfig rewrites full_attention to qwen_sparse_attention in __post_init__.
        return [
            "full_attention" if t == "qwen_sparse_attention" else t for t in layer_types
        ]
    # Fall back to full_attention_interval: every Nth layer (1-indexed) is full.
    interval = int(getattr(text, "full_attention_interval", 4))
    n = int(text.num_hidden_layers)
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(n)
    ]


def use_fp8_lmhead(config: ModelConfig) -> bool:
    """Whether to build/load the lm_head as load-time per-tensor FP8.

    The MODEL BUILDER and the WEIGHT READER must agree exactly: if one says yes and the
    other no, the state dict gains or loses ``lm_head.weight_scale`` and load fails. They
    used to test this separately and drifted. Conditions: the operator asked
    (FREETOKEN_FP8_LMHEAD=1), the head is untied (a tied head shares the bf16 embedding
    table), and the checkpoint declares no scheme OF ITS OWN for lm_head -- a quantized head
    goes through QuantConfig instead. Note it is the lm_head scheme that matters, not whether
    the checkpoint has a QuantConfig at all: this model ships NVFP4 experts with lm_head in
    the modelopt ignore list."""
    if not fp8_lmhead_enabled() or config.tie_word_embeddings:
        return False
    quant = getattr(config, "quant", None)
    return quant is None or quant.scheme_for("lm_head") is None


def parse_config(hf_config: Any) -> ModelConfig:
    text = getattr(hf_config, "text_config", hf_config)

    head_dim = (
        getattr(text, "head_dim", None)
        or text.hidden_size // text.num_attention_heads
    )
    num_kv_heads = getattr(text, "num_key_value_heads", text.num_attention_heads)

    rope_params = getattr(text, "rope_parameters", None) or {}
    rope_theta = rope_params.get("rope_theta", getattr(text, "rope_theta", None))
    partial = (
        rope_params.get("partial_rotary_factor")
        or getattr(text, "partial_rotary_factor", None)
        or 1.0
    )
    # int(), not round(): HF configuration_qwen4_exp truncates head_dim * partial.
    rotary_dim = int(head_dim * partial)

    # Text-only serving with the default rope type: the mRoPE sections reduce to standard
    # partial rope, and the unhashable ``mrope_section`` list must not reach get_rope's
    # cache key.
    rope_type = rope_params.get("rope_type", "default")
    rope_scaling = (
        None
        if rope_type in (None, "default")
        else {k: v for k, v in rope_params.items() if not isinstance(v, (list, dict))}
    )

    # FREETOKEN_FP8_DENSE=1: the bf16 attention / GDN projections are quantized at LOAD to
    # per-tensor e4m3 and served W8A8 through cuBLASLt (layers/fp8_dynamic.py). This is a
    # synthetic scheme for a checkpoint that ships those modules unquantized -- distinct from
    # the QuantConfig path, which serves what the checkpoint declares. The module builders
    # apply it only where the checkpoint has no scheme of its own, so the two never collide.
    attn_quant = "fp8_dynamic" if fp8_dense_enabled() else "none"
    layer_types = _layer_types(text)
    full_ids = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    linear_ids = tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")

    # the engine reads this flag for its MoE strategy decisions; every module takes its own scheme from the QuantConfig when it is built
    expert_scheme = QuantConfig.from_hf(hf_config).scheme_for_name("model.language_model.layers.0.mlp.experts.0.gate_proj")
    expert_quant = "none" if expert_scheme is None else str(expert_scheme.kind)

    # HF stores ple_layer_ids one-indexed (validated upstream as [1, num_layers]).
    ple_layer_ids = tuple(int(i) - 1 for i in (getattr(text, "ple_layer_ids", None) or ()))
    for lid in ple_layer_ids:
        if layer_types[lid] != "linear_attention":
            raise ValueError(f"PLE must sit on a linear_attention layer, got layer {lid}")

    full_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=text.max_position_embeddings,
        base=rope_theta,
        scaling=rope_scaling,
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=full_rotary,
        index_head_dim=int(text.indexer_head_dim),
        num_index_layers=len(full_ids),
        index_ratio=int(text.indexer_compress_ratio),
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=text.linear_num_key_heads,
        num_value_heads=text.linear_num_value_heads,
        key_head_dim=text.linear_key_head_dim,
        value_head_dim=text.linear_value_head_dim,
        conv_kernel_dim=text.linear_conv_kernel_dim,
        # HF resolves a null output_gate_type to hidden_act; mirror that instead of
        # stringifying None.
        output_gate=str(getattr(text, "output_gate_type", None) or text.hidden_act),
    )
    # Order groups by their first layer id for deterministic iteration.
    groups = tuple(
        sorted(
            (full_group, linear_group),
            key=lambda g: g.layer_ids[0] if g.layer_ids else 1 << 30,
        )
    )

    num_experts = int(getattr(text, "num_experts", 0) or 0)

    # HF accepts int | list here and uses the first entry (modeling_qwen4_exp Qwen4ExpTextNGramEmbedding)
    eos_token_id = text.eos_token_id
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0]

    qwen4_args = Qwen4ExpArgs(
        hidden_size=text.hidden_size,
        hc_count=int(text.hc_count),
        hc_lowrank=int(text.hc_lowrank),
        ple_layer_ids=ple_layer_ids,
        ple_embed_dim=int(text.ple_embed_dim),
        ple_conv_kernel_size=int(text.ple_conv_kernel_size),
        ngram_size=int(text.ngram_size),
        heads_per_ngram=int(text.heads_per_ngram),
        ngram_vocab_size_base=int(text.ngram_vocab_size_base),
        make_ngram_vocab_size_divisible_by=int(text.make_ngram_vocab_size_divisible_by),
        split_ngram_parts=int(text.split_ngram_parts),
        ngram_boundary_token_id=int(eos_token_id),
        index_n_heads=int(text.indexer_n_heads),
        index_kv_heads=int(text.indexer_kv_heads),
        index_head_dim=int(text.indexer_head_dim),
        index_budget=int(text.indexer_budget),
        index_ratio=int(text.indexer_compress_ratio),
    )

    return ModelConfig(
        num_layers=text.num_hidden_layers,
        num_qo_heads=text.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=text.hidden_size,
        vocab_size=text.vocab_size,
        intermediate_size=getattr(text, "intermediate_size", 0) or 0,
        hidden_act=text.hidden_act,
        rms_norm_eps=text.rms_norm_eps,
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=full_rotary,
        num_experts=num_experts,
        num_experts_per_tok=int(getattr(text, "num_experts_per_tok", 0) or 0),
        moe_intermediate_size=int(getattr(text, "moe_intermediate_size", 0) or 0),
        shared_expert_intermediate_size=int(
            getattr(text, "shared_expert_intermediate_size", 0) or 0
        ),
        # Absent from the shipped config; HF Qwen4ExpTextConfig defaults it True and the
        # Qwen3_5MoE block renormalizes unconditionally -- keep the two in agreement.
        norm_topk_prob=bool(getattr(text, "norm_topk_prob", True)),
        moe_enabled=num_experts > 0,
        use_qk_norm=True,
        model_type=getattr(hf_config, "model_type", "qwen4_exp"),
        architectures=getattr(hf_config, "architectures", ["Qwen4ExpForConditionalGeneration"]),
        vision_config=None,  # served text-only
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=groups,
        expert_quant=expert_quant,
        attn_quant=attn_quant,
        qwen4_args=qwen4_args,
        slot_states=ple_slot_states(qwen4_args),
    )


__all__ = ["PLE_CONV_STATE", "PLE_NGRAM_STATE", "Qwen4ExpArgs", "parse_config", "ple_slot_states"]
