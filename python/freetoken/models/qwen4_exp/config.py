from __future__ import annotations

from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)

from .args import Qwen4ExpArgs


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
        # QSA selects every visible token through this point. FreeToken currently
        # serves that exact dense prefix and rejects longer requests.
        max_position=min(int(text.max_position_embeddings), indexer_budget),
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
        FullAttentionGroupConfig(
            name="full",
            layer_ids=full_ids,
            num_kv_heads=int(text.num_key_value_heads),
            head_dim=head_dim,
            rotary_config=rotary,
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
        indexer_budget=indexer_budget,
        indexer_compress_ratio=int(text.indexer_compress_ratio),
        output_gate_type=str(text.output_gate_type or text.hidden_act),
    )

    quant = hf_config.quantization_config
    if not isinstance(quant, dict):
        quant = quant.to_dict()
    method = str(quant.get("quant_method") or "").lower()
    algo = str(quant.get("quant_algo") or method).lower()
    if method == "fp8":
        block_size = tuple(int(value) for value in quant["weight_block_size"])
        if block_size != (128, 128):
            raise ValueError(f"Qwen4-Exp only supports 128x128 block-FP8, got {block_size}")
        expert_quant = "fp8_block"
    elif "fp4" in algo:
        # ModelOpt NVFP4 checkpoints use packed routed experts while leaving the
        # shared expert and other resident text weights in BF16.
        block_size = None
        expert_quant = "nvfp4"
    else:
        raise ValueError(
            "Qwen4-Exp requires a 128x128 block-FP8 or ModelOpt NVFP4 checkpoint, "
            f"got quant_method={method!r}, quant_algo={algo!r}"
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
        norm_topk_prob=bool(getattr(text, "norm_topk_prob", True)),
        model_type=str(hf_config.model_type),
        architectures=list(hf_config.architectures),
        moe_enabled=True,
        expert_quant=expert_quant,
        weight_block_size=block_size,
        # Only routed experts and PLE are quantized in the supported checkpoints.
        # Attention, hyper-connections, and shared-expert projections stay BF16.
        attn_quant="none",
        dense_quant="none",
        lm_head_quant="none",
        use_qk_norm=True,
        vision_config=None,
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=groups,
        qwen4_args=qwen4_args,
        # PLE keeps per-request dilated-convolution state outside the generic
        # radix cache, so prefix snapshots remain unsupported. Its host embedding
        # lookup is staged into a stable device buffer before CUDA-graph replay.
        requires_naive_cache=True,
        supports_cuda_graph=True,
    )


__all__ = ["parse_config"]
