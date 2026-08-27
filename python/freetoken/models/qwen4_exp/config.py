from __future__ import annotations

from typing import Any

from freetoken.models.config import (
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    QSAAttentionGroupConfig,
    RotaryConfig,
    detect_expert_quant,
)

from .args import Qwen4ExpArgs, Qwen4VisionConfig


def parse_config(hf_config: Any) -> ModelConfig:
    text = hf_config.text_config
    layer_types = list(text.layer_types)
    sparse_attention_types = {"full_attention", "qwen_sparse_attention"}
    unsupported = sorted(set(layer_types) - {"linear_attention", *sparse_attention_types})
    if unsupported:
        raise ValueError(f"Unsupported Qwen4-Exp layer types: {unsupported}")

    head_dim = int(text.head_dim)
    rope = text.rope_parameters
    rotary_dim = round(head_dim * float(rope.get("partial_rotary_factor", 1.0)))
    indexer_budget = int(text.indexer_budget)
    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=int(text.max_position_embeddings),
        base=float(rope["rope_theta"]),
        scaling=None,
    )

    full_ids = tuple(
        i for i, layer_type in enumerate(layer_types) if layer_type in sparse_attention_types
    )
    linear_ids = tuple(
        i for i, layer_type in enumerate(layer_types) if layer_type == "linear_attention"
    )
    groups = (
        LinearGatedDeltaGroupConfig(
            name="linear",
            layer_ids=linear_ids,
            num_key_heads=int(text.linear_num_key_heads),
            num_value_heads=int(text.linear_num_value_heads),
            key_head_dim=int(text.linear_key_head_dim),
            value_head_dim=int(text.linear_value_head_dim),
            conv_kernel_dim=int(text.linear_conv_kernel_dim),
            output_gate=True,
        ),
        QSAAttentionGroupConfig(
            name="qsa",
            layer_ids=full_ids,
            num_kv_heads=int(text.num_key_value_heads),
            head_dim=head_dim,
            rotary_config=rotary,
            index_num_heads=int(text.indexer_n_heads),
            index_num_kv_heads=int(text.indexer_kv_heads),
            index_head_dim=int(text.indexer_head_dim),
            index_token_budget=indexer_budget,
            index_compress_ratio=int(text.indexer_compress_ratio),
        ),
    )

    eos_token_id = text.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]
    qwen4_args = Qwen4ExpArgs(
        hc_count=int(text.hc_count),
        hc_lowrank=int(text.hc_lowrank),
        ple_layer_ids=tuple(int(layer_id) - 1 for layer_id in text.ple_layer_ids),
        ple_embed_dim=int(text.ple_embed_dim),
        ple_conv_kernel_size=int(text.ple_conv_kernel_size),
        ngram_size=int(text.ngram_size),
        heads_per_ngram=int(text.heads_per_ngram),
        ngram_vocab_size_base=int(text.ngram_vocab_size_base),
        split_ngram_parts=int(text.split_ngram_parts),
        eos_token_id=int(eos_token_id),
        indexer_n_heads=int(text.indexer_n_heads),
        indexer_kv_heads=int(text.indexer_kv_heads),
        indexer_head_dim=int(text.indexer_head_dim),
        indexer_budget=indexer_budget,
        indexer_compress_ratio=int(text.indexer_compress_ratio),
        output_gate_type=str(text.output_gate_type or text.hidden_act),
        mrope_section=tuple(int(value) for value in rope["mrope_section"]),
        mrope_interleaved=bool(rope.get("mrope_interleaved", False)),
    )
    if not qwen4_args.mrope_interleaved:
        raise ValueError("Qwen4-Exp requires interleaved MRoPE")
    if sum(qwen4_args.mrope_section) * 2 != rotary_dim:
        raise ValueError(
            "Qwen4-Exp mrope_section must cover the rotary dimension: "
            f"{qwen4_args.mrope_section} vs {rotary_dim}"
        )
    raw_vision = getattr(hf_config, "vision_config", None)
    vision_config = None
    if raw_vision is not None:
        vision_config = Qwen4VisionConfig(
            depth=int(raw_vision.depth),
            hidden_size=int(raw_vision.hidden_size),
            intermediate_size=int(raw_vision.intermediate_size),
            num_heads=int(raw_vision.num_heads),
            num_position_embeddings=int(raw_vision.num_position_embeddings),
            out_hidden_size=int(raw_vision.out_hidden_size),
            patch_size=int(raw_vision.patch_size),
            spatial_merge_size=int(raw_vision.spatial_merge_size),
            temporal_patch_size=int(raw_vision.temporal_patch_size),
            in_channels=int(raw_vision.in_channels),
            hidden_act=str(raw_vision.hidden_act),
            deepstack_visual_indexes=tuple(
                int(index) for index in raw_vision.deepstack_visual_indexes
            ),
        )

    quant = getattr(hf_config, "quantization_config", None)
    get_quant = (
        quant.get
        if isinstance(quant, dict)
        else (lambda key, default=None: getattr(quant, key, default))
    )
    detected_quant = detect_expert_quant(hf_config)
    if detected_quant == "fp8":
        raw_block_size = get_quant("weight_block_size")
        block_size = (
            tuple(int(value) for value in raw_block_size) if raw_block_size is not None else None
        )
        if block_size != (128, 128):
            raise ValueError(
                "Qwen4-Exp block-FP8 checkpoints require a 128x128 weight block size"
            )
        expert_quant = "fp8_block"
    elif detected_quant == "nvfp4":
        # RadixArk's ModelOpt checkpoint quantizes only the routed experts. The
        # attention, GDN, mHC, shared experts, router, embeddings, lm_head, and
        # vision tensors remain BF16; PLE remains its source FP8 format.
        block_size = None
        expert_quant = "nvfp4"
    else:
        raise ValueError(
            "Qwen4-Exp requires routed experts in 128x128 block-FP8 or ModelOpt NVFP4; "
            f"detected {detected_quant!r}"
        )

    return ModelConfig(
        num_layers=int(text.num_hidden_layers),
        num_qo_heads=int(text.num_attention_heads),
        num_kv_heads=int(text.num_key_value_heads),
        head_dim=head_dim,
        hidden_size=int(text.hidden_size),
        vocab_size=int(text.vocab_size),
        intermediate_size=int(getattr(text, "intermediate_size", 0) or 0),
        hidden_act=str(text.hidden_act),
        rms_norm_eps=float(text.rms_norm_eps),
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=rotary,
        num_experts=int(text.num_experts),
        num_experts_per_tok=int(text.num_experts_per_tok),
        moe_intermediate_size=int(text.moe_intermediate_size),
        shared_expert_intermediate_size=int(text.shared_expert_intermediate_size),
        # The released Qwen3.8-Flash-Next configs omit this older Qwen MoE
        # field. Omission means that the router weights are not renormalized.
        norm_topk_prob=bool(getattr(text, "norm_topk_prob", False)),
        model_type=str(hf_config.model_type),
        architectures=list(hf_config.architectures),
        moe_enabled=True,
        expert_quant=expert_quant,
        weight_block_size=block_size,
        # Only routed experts and PLE are FP8 in the official checkpoint. All
        # attention, hyper-connection, and shared-expert projections stay BF16.
        attn_quant="none",
        dense_quant="none",
        lm_head_quant="none",
        use_qk_norm=True,
        # Qwen3.8-Flash-Next is a VL checkpoint. Vision is part of this model,
        # not an optional text-only add-on.
        vision_config=vision_config,
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=groups,
        qwen4_args=qwen4_args,
        # PLE keeps per-request dilated-convolution state outside the generic
        # radix cache and performs mmap-backed CPU gathers during every forward.
        requires_naive_cache=True,
        supports_cuda_graph=False,
    )


__all__ = ["parse_config"]
