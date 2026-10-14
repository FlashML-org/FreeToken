from __future__ import annotations

from typing import Any

from freetoken.layers.quantization import QuantConfig, QuantKind
from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    mrope_layout_from_rope_params,
)
from freetoken.models.qwen3_vl.config import parse_vision_config


def _expert_quant(hf_config: Any, text: Any) -> tuple[str, tuple[int, int] | None]:
    """The routed experts' quant kind as the engine's format tag, with the scale block of block-fp8."""
    if not (getattr(text, "num_experts", 0) or 0):
        return "none", None
    # the engine reads this tag for its MoE strategy decisions; every module takes its own scheme from the QuantConfig when it is built
    quant = QuantConfig.from_hf(hf_config)
    scheme = quant.scheme_for_name("model.layers.0.mlp.experts.0.gate_proj")
    if scheme is None:
        scheme = quant.scheme_for_name("model.language_model.layers.0.mlp.experts.0.gate_proj")
    if scheme is None:
        return "none", None
    return str(scheme.kind), scheme.weight.group if scheme.kind is QuantKind.FP8_BLOCK else None


def _layer_types(text: Any) -> list[str]:
    layer_types = getattr(text, "layer_types", None)
    if layer_types is not None:
        return list(layer_types)
    # Fall back to full_attention_interval: every Nth layer (1-indexed) is full.
    interval = int(getattr(text, "full_attention_interval", 4))
    n = int(text.num_hidden_layers)
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(n)
    ]


def _gguf_field(metadata: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    raise KeyError(f"missing GGUF metadata key(s): {keys}")


def _gguf_value(metadata: dict[str, Any], default: Any, *keys: str) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    return default


def parse_gguf_config(hf_config: Any) -> ModelConfig:
    metadata = getattr(hf_config, "metadata", None)
    if metadata is None:
        return parse_config(hf_config)

    def field(*keys: str):
        return _gguf_field(metadata, *keys)

    gguf_block_count = int(field("qwen35moe.block_count", "qwen3.block_count", "qwen3_5_moe.block_count", "qwen.block_count", "qwen3_moe.block_count"))
    nextn_layers = int(_gguf_value(metadata, 0, "qwen35moe.nextn_predict_layers", "qwen3.nextn_predict_layers", "qwen3_5_moe.nextn_predict_layers"))
    if nextn_layers < 0 or nextn_layers > gguf_block_count:
        raise ValueError(
            f"invalid GGUF NextN layer count {nextn_layers} for block_count={gguf_block_count}"
        )
    block_count = gguf_block_count - nextn_layers
    layer_types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(block_count)]
    model_path = getattr(hf_config, "model_path", None)
    tensor_names = None
    if model_path is not None:
        from freetoken.models.gguf.reader import gguf_tensor_names

        names = gguf_tensor_names(model_path)
        tensor_names = names
        layer_types = [
            "linear_attention"
            if (
                f"model.language_model.layers.{i}.linear_attn." in "\n".join(names)
                or f"blk.{i}.ssm_a" in names
            )
            else "full_attention"
            for i in range(block_count)
        ]
    shared_expert_size = _gguf_value(
        metadata,
        0,
        "qwen35moe.expert_shared_feed_forward_length",
        "qwen35moe.shared_expert_intermediate_size",
        "qwen3.shared_expert_intermediate_size",
        "qwen3_5_moe.shared_expert_intermediate_size",
        "qwen.shared_expert_intermediate_size",
        "qwen3_moe.shared_expert_intermediate_size",
    )
    if not shared_expert_size and model_path is not None:
        from freetoken.models.gguf.reader import iter_gguf_tensors

        for tensor in iter_gguf_tensors(model_path):
            if tensor.name.endswith("mlp.shared_expert.gate_proj.weight"):
                shared_expert_size = tensor.shape[0]
                break

    from freetoken.models.gguf.reader import GGML_NVFP4, iter_gguf_tensors

    has_nvfp4 = any(t.ggml_type == GGML_NVFP4 for t in iter_gguf_tensors(hf_config.model_path))
    data = type(
        "_GGUFTextConfig",
        (),
        {
            "hidden_size": int(field("qwen35moe.embedding_length", "qwen3.embedding_length", "qwen3_5_moe.embedding_length", "qwen.embedding_length", "qwen3_moe.embedding_length")),
            "num_hidden_layers": block_count,
            "num_attention_heads": int(field("qwen35moe.attention.head_count", "qwen3.attention.head_count", "qwen3_5_moe.attention.head_count", "qwen.attention.head_count", "qwen3_moe.attention.head_count")),
            "num_key_value_heads": int(field("qwen35moe.attention.head_count_kv", "qwen3.attention.head_count_kv", "qwen3_5_moe.attention.head_count_kv", "qwen.attention.head_count_kv", "qwen3_moe.attention.head_count_kv")),
            "head_dim": int(field("qwen35moe.attention.key_length", "qwen3.attention.key_length", "qwen3_5_moe.attention.key_length", "qwen.attention.key_length", "qwen3_moe.attention.key_length")),
            "max_position_embeddings": int(field("qwen35moe.context_length", "qwen3.context_length", "qwen3_5_moe.context_length", "qwen.context_length", "qwen3_moe.context_length")),
            "rms_norm_eps": float(field("qwen35moe.attention.layer_norm_rms_epsilon", "qwen3.attention.layer_norm_rms_epsilon", "qwen3_5_moe.attention.layer_norm_rms_epsilon", "qwen.attention.layer_norm_rms_epsilon", "qwen3_moe.attention.layer_norm_rms_epsilon")),
            "hidden_act": "silu",
            "intermediate_size": int(_gguf_value(metadata, shared_expert_size, "qwen35moe.feed_forward_length", "qwen35moe.expert_shared_feed_forward_length", "qwen3.feed_forward_length", "qwen3_5_moe.feed_forward_length", "qwen.feed_forward_length", "qwen3_moe.feed_forward_length")),
            "moe_intermediate_size": int(field("qwen35moe.expert_feed_forward_length", "qwen3.expert_feed_forward_length", "qwen3_5_moe.expert_feed_forward_length", "qwen.expert_feed_forward_length", "qwen3_moe.expert_feed_forward_length", "qwen3.feed_forward_length")),
            "num_experts": int(_gguf_value(metadata, 0, "qwen35moe.expert_count", "qwen3.expert_count", "qwen3_5_moe.expert_count", "qwen.expert_count", "qwen3_moe.expert_count")),
            "num_experts_per_tok": int(_gguf_value(metadata, 0, "qwen35moe.expert_used_count", "qwen3.expert_used_count", "qwen3_5_moe.expert_used_count", "qwen.expert_used_count", "qwen3_moe.expert_used_count")),
            "shared_expert_intermediate_size": int(shared_expert_size),
            "vocab_size": int(_gguf_value(metadata, len(metadata.get("tokenizer.ggml.tokens", [])), "tokenizer.ggml.vocab_size", "vocab_size")),
            "partial_rotary_factor": 0.25,
            "layer_types": layer_types,
            "rope_parameters": {"rope_theta": float(field("qwen35moe.rope.freq_base", "qwen3.rope.freq_base", "qwen3_5_moe.rope.freq_base", "qwen.rope.freq_base", "qwen3_moe.rope.freq_base", "10000000.0"))},
            "full_attention_interval": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "tie_word_embeddings": bool(getattr(hf_config, "tie_word_embeddings", False))
            and not (tensor_names is not None and "lm_head.weight" in tensor_names),
        },
    )
    top = type(
        "_GGUFConfig",
        (),
        {
            "text_config": data,
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "image_token_id": None,
            "video_token_id": None,
            "tie_word_embeddings": data.tie_word_embeddings,
            "num_experts": data.num_experts,
            "num_experts_per_tok": data.num_experts_per_tok,
            "moe_intermediate_size": data.moe_intermediate_size,
            "shared_expert_intermediate_size": data.shared_expert_intermediate_size,
            "vocab_size": data.vocab_size,
            "is_gguf": True,
            "gguf_model_path": hf_config.model_path,
            "gguf_block_count": gguf_block_count,
            "has_nvfp4": has_nvfp4,
        },
    )()
    top.gguf_tensor_types = {
        tensor.name: tensor.ggml_type
        for tensor in iter_gguf_tensors(hf_config.model_path)
    }
    return parse_config(top)


__all__ = ["parse_config", "parse_gguf_config"]


def parse_config(hf_config: Any) -> ModelConfig:
    text = getattr(hf_config, "text_config", hf_config)
    from freetoken.checkpoint.ftw import is_ftw_checkpoint

    ftw_path = getattr(hf_config, "_name_or_path", "")
    ftw_split_gdn = is_ftw_checkpoint(ftw_path)
    ftw_lm_head_nvfp4 = False
    if ftw_split_gdn:
        import json
        with open(f"{ftw_path}/freetoken_weight.json", encoding="utf-8") as handle:
            ftw_lm_head_nvfp4 = any(
                item.get("name") == "lm_head.weight" and item.get("dtype") == "uint8"
                for item in json.load(handle).get("tensors", ())
            )

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
    rotary_dim = int(head_dim * partial)

    # For text-only with the default rope type, partial NeoX rope needs no scaling dict
    # (the mRoPE params reduce to standard partial rope for text). Avoid carrying the
    # unhashable ``mrope_section`` list into get_rope's cache key.
    rope_type = rope_params.get("rope_type", "default")
    rope_scaling = (
        None
        if rope_type in (None, "default")
        else {k: v for k, v in rope_params.items() if not isinstance(v, (list, dict))}
    )

    expert_quant, weight_block_size = _expert_quant(hf_config, text)

    # Dense variants (e.g. Qwen3.6-27B) report num_experts==0: route the decoder MLP through
    # the dense Qwen3_5DenseMLP instead of the MoE block.
    num_experts = getattr(text, "num_experts", 0) or 0
    moe_enabled = num_experts > 0

    layer_types = _layer_types(text)
    full_ids = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    linear_ids = tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")

    # 3-axis rope only with vision; text-only serving keeps the 1-D partial rope and the decode-graph layout
    vision_config = parse_vision_config(hf_config)
    full_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=text.max_position_embeddings,
        base=rope_theta,
        scaling=rope_scaling,
        mrope_section=(
            list(rope_params["mrope_section"])
            if vision_config is not None and "mrope_section" in rope_params
            else None
        ),
        mrope_layout=mrope_layout_from_rope_params(rope_params),
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=full_rotary,
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=text.linear_num_key_heads,
        num_value_heads=text.linear_num_value_heads,
        key_head_dim=text.linear_key_head_dim,
        value_head_dim=text.linear_value_head_dim,
        conv_kernel_dim=text.linear_conv_kernel_dim,
        output_gate="silu",
    )
    # Order groups by their first layer id for deterministic iteration.
    groups = tuple(
        sorted(
            (full_group, linear_group),
            key=lambda g: g.layer_ids[0] if g.layer_ids else 1 << 30,
        )
    )

    return ModelConfig(
        num_layers=text.num_hidden_layers,
        num_qo_heads=text.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=text.hidden_size,
        vocab_size=text.vocab_size,
        intermediate_size=getattr(text, "intermediate_size", 0),
        hidden_act=text.hidden_act,
        rms_norm_eps=text.rms_norm_eps,
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=full_rotary,
        num_experts=num_experts,
        num_experts_per_tok=getattr(text, "num_experts_per_tok", 0),
        moe_intermediate_size=getattr(text, "moe_intermediate_size", 0),
        shared_expert_intermediate_size=getattr(text, "shared_expert_intermediate_size", 0),
        norm_topk_prob=True,
        moe_enabled=moe_enabled,
        use_qk_norm=True,
        model_type=getattr(hf_config, "model_type", "qwen3_5_moe"),
        architectures=getattr(hf_config, "architectures", ["Qwen3_5MoeForConditionalGeneration"]),
        vision_config=vision_config,
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=groups,
        expert_quant=("nvfp4" if getattr(hf_config, "has_nvfp4", False) else "q4_k") if getattr(hf_config, "is_gguf", False) and num_experts else expert_quant,
        weight_block_size=weight_block_size,
        # Native GGUF conversion is only valid for a GGUF config. HF/FTW NVFP4
        # checkpoints keep their regular quantization modules and FTW tensor names;
        # marking them q4_k makes the model allocate GGUF qweight buffers and causes
        # missing-key failures such as model.embed_tokens.qweight.
        moe_weight_format=(
            "nvfp4" if getattr(hf_config, "is_gguf", False) and getattr(hf_config, "has_nvfp4", False)
            else "q4_k" if getattr(hf_config, "is_gguf", False) else None
        ),
        gguf_model_path=getattr(
            hf_config, "gguf_model_path", getattr(hf_config, "model_path", None)
        ),
        gguf_block_count=getattr(hf_config, "gguf_block_count", None),
        gguf_tensor_types=getattr(hf_config, "gguf_tensor_types", None),
        ftw_split_gdn=ftw_split_gdn,
        ftw_lm_head_nvfp4=ftw_lm_head_nvfp4,
    )


__all__ = ["parse_config"]
