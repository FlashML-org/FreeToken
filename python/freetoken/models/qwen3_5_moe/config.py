from __future__ import annotations

from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    _nvfp4_global_reciprocal,
    detect_compressed_tensors_nvfp4,
    detect_expert_quant,
)


def _quant_accessor(hf_config: Any):
    """A ``get(key, default=None)`` accessor over the HF ``quantization_config`` (dict or
    object), or ``None`` when the model has no quant config."""
    quant = getattr(hf_config, "quantization_config", None)
    if quant is None:
        return None
    return quant.get if isinstance(quant, dict) else (lambda k, d=None: getattr(quant, k, d))


def _fp8_block_quant(hf_config: Any) -> tuple[str, tuple[int, int] | None]:
    """Detect DeepSeek-V3-style 128x128 block-fp8 from HF ``quantization_config``.

    Returns ``("fp8_block", (block_n, block_k))`` for a block-fp8 checkpoint (weights
    fp8-e4m3 + per-block ``weight_scale_inv``, dynamic activation), else ``("none", None)``.
    The quantization_config sits on the top-level hf_config (not ``text_config``).
    """
    get = _quant_accessor(hf_config)
    if get is None:
        return "none", None
    method = str(get("quant_method") or get("quant_algo") or "").lower()
    block = get("weight_block_size")
    if method == "fp8" and block:
        bs = tuple(int(x) for x in block)
        assert bs == (128, 128), f"only 128x128 block-fp8 is supported, got {bs}"
        return "fp8_block", bs
    return "none", None


def _has_moe_experts(hf_config: Any) -> bool:
    """True if the model declares any routed experts (``text_config.num_experts > 0``).
    Used to disambiguate dense compressed-tensors NVFP4 (Qwen3.6-27B) from MoE mixed-precision
    compressed-tensors (Ornith: FP8 attn/shared_expert + NVFP4 experts) so per-group quant
    detection isn't overridden by a blanket "everything is NVFP4" assumption."""
    text = getattr(hf_config, "text_config", hf_config)
    return (getattr(text, "num_experts", 0) or 0) > 0


def _expert_quant(hf_config: Any) -> str:
    """Quantization format of the *routed* experts (the only weights served from the
    offload cache). The nvidia/modelopt checkpoints are either plain NVFP4 (``quant_algo``
    ``NVFP4``) or ``MIXED_PRECISION`` (per-layer ``quantized_layers`` map); in the mixed
    case the routed experts carry their own ``W4A16_NVFP4``/``FP8`` algo. Dense quantized
    weights (attention/shared-expert/lm_head) are handled separately by dequant-at-load."""
    get = _quant_accessor(hf_config)
    if get is None:
        return "none"
    algo = str(get("quant_algo") or get("quant_method") or "").lower()
    if "fp4" in algo:
        return "nvfp4"
    if "mixed" in algo:
        layers = get("quantized_layers") or {}
        for name, spec in (layers.items() if isinstance(layers, dict) else []):
            if name.endswith(".mlp.experts") or ".mlp.experts." in name:
                expert_algo = str((spec or {}).get("quant_algo", "")).lower()
                if "fp4" in expert_algo:
                    return "nvfp4"
                if "fp8" in expert_algo:
                    return "fp8"
    if str(get("quant_method") or "").lower() == "compressed-tensors":
        # llm-compressor: the routed experts' group carries ``format: nvfp4-pack-quantized``.
        # ``detect_expert_quant`` (top-level) already returns "nvfp4" for this -- mirror it
        # here so the offload expert bank activates.
        if detect_expert_quant(hf_config) == "nvfp4":
            return "nvfp4"
    return "none"


# Detection now lives in models/config.py (shared with muse_glimmer); weight.py imports
# it under this name.
_compressed_tensors_nvfp4 = detect_compressed_tensors_nvfp4


def _lm_head_quant(hf_config: Any) -> str:
    """Whether the checkpoint stores ``lm_head`` as NVFP4. modelopt MIXED_PRECISION lists it in
    the per-layer ``quantized_layers`` map (``W4A16_NVFP4``); pure-NVFP4 checkpoints have no
    per-layer map and leave lm_head bf16. Returns ``"nvfp4"`` or ``"none"``."""
    get = _quant_accessor(hf_config)
    if get is None:
        return "none"
    layers = get("quantized_layers") or {}
    if not isinstance(layers, dict):
        return "none"
    for name, spec in layers.items():
        if name == "lm_head" or name.endswith(".lm_head"):
            if "fp4" in str((spec or {}).get("quant_algo", "")).lower():
                return "nvfp4"
    return "none"


def _dense_mlp_quant(hf_config: Any) -> str:
    """NVFP4 on the *dense* (non-MoE) decoder MLP. modelopt MIXED_PRECISION dense checkpoints
    (e.g. Qwen3.6-27B-NVFP4) list ``.mlp.{gate,up,down}_proj`` as ``W4A16_NVFP4`` in
    ``quantized_layers``; MoE checkpoints have ``.mlp.experts.*`` / ``.mlp.shared_expert.*``
    instead (covered by ``expert_quant``). ``endswith(".mlp.gate_proj")`` matches only the bare
    dense MLP -- not ``.mlp.shared_expert.gate_proj`` nor ``.mlp.experts.N.gate_proj``."""
    get = _quant_accessor(hf_config)
    if get is None:
        return "none"
    layers = get("quantized_layers") or {}
    if not isinstance(layers, dict):
        return "none"
    for name, spec in layers.items():
        if name.endswith((".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj")):
            if "fp4" in str((spec or {}).get("quant_algo", "")).lower():
                return "nvfp4"
    return "none"


def _attn_quant(hf_config: Any) -> str:
    """Per-tensor FP8 on the *dense* attention/GDN projections. The modelopt
    ``MIXED_PRECISION`` checkpoints tag ``self_attn.{q,k,v,o}_proj`` and
    ``linear_attn.{in_proj_qkv,in_proj_z,out_proj}`` with ``quant_algo`` ``FP8`` (fp8-e4m3
    weight + a scalar ``weight_scale``; W8A16). llm-compressor mixed-precision checkpoints
    (Ornith) use a per-tensor-FP8 ``config_groups`` entry targeting the same projections.
    Returns ``"fp8_pertensor"`` when present, else ``"none"``."""
    get = _quant_accessor(hf_config)
    if get is None:
        return "none"
    layers = get("quantized_layers") or {}
    if isinstance(layers, dict):
        for name, spec in layers.items():
            algo = str((spec or {}).get("quant_algo", "")).lower()
            if algo == "fp8" and (".self_attn." in name or ".linear_attn." in name):
                return "fp8_pertensor"
    # llm-compressor path: a config_group with format "float-quantized" and per-tensor
    # strategy (``group_size is None and strategy == "tensor"``) targeting ``.self_attn.``
    # or ``.linear_attn.`` is per-tensor FP8 (W8A16). Targets are regex strings
    # (``re:.*\.self_attn\.``), so the substring check uses the unescaped leaf path.
    for g in (get("config_groups") or {}).values():
        if not g:
            continue
        if not any((".self_attn" in t or "linear_attn" in t) for t in (g.get("targets") or [])):
            continue
        w = g.get("weights") or {}
        if str(w.get("type", "")).lower() != "float" or int(w.get("num_bits", 0) or 0) != 8:
            continue
        if w.get("group_size") is None and w.get("strategy") == "tensor":
            return "fp8_pertensor"
    return "none"


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

    expert_quant, weight_block_size = _fp8_block_quant(hf_config)
    if expert_quant == "none":
        expert_quant = _expert_quant(hf_config)  # nvfp4 / mixed-precision modelopt
    # Dense attention/GDN quant is independent of the routed experts (block-fp8 already
    # quantizes both, so only probe for per-tensor FP8 when experts aren't block-fp8).
    attn_quant = "none" if expert_quant == "fp8_block" else _attn_quant(hf_config)
    # Default: shared expert / dense MLP match the routed experts' quant (NVFP4 when experts
    # are NVFP4). The modelopt dense (non-MoE) checkpoint sets a per-projection override via
    # ``quantized_layers`` that ``_dense_mlp_quant`` picks up. llm-compressor mixed-precision
    # MoE (Ornith: NVFP4 experts + FP8 attn + FP8 shared_expert) is the one true exception:
    # its config_groups put the shared_expert in the FP8 group, so we explicitly read that
    # and override dense_quant to "none" -- the FP8 path is selected via attn_quant.
    if expert_quant == "nvfp4":
        mlp_q = _dense_mlp_quant(hf_config)
        dense_quant = mlp_q if mlp_q != "none" else "nvfp4"
        if _compressed_tensors_nvfp4(hf_config) and _has_moe_experts(hf_config):
            quant = getattr(hf_config, "quantization_config", None)
            get = quant.get if isinstance(quant, dict) else (lambda k, d=None: getattr(quant, k, d))
            for g in (get("config_groups") or {}).values():
                if not g:
                    continue
                if any("shared_expert" in t for t in (g.get("targets") or [])):
                    if str(g.get("format") or "").lower() == "float-quantized":
                        dense_quant = "none"
                    break
    else:
        dense_quant = _dense_mlp_quant(hf_config)
    lm_head_quant = _lm_head_quant(hf_config)

    # compressed-tensors NVFP4 *dense* (Qwen3.6-27B-NVFP4): every Linear is W4A16 NVFP4
    # (q/k/v/o, GDN out_proj, dense MLP). GDN in_proj_*, lm_head, norms stay bf16. Wire the
    # shared W4A16 kernels (attn_quant=="nvfp4" routes the attention/GDN linears through
    # them too). MoE compressed-tensors checkpoints (Ornith: FP8 attn + FP8 shared_expert
    # + NVFP4 experts) skip the override -- the per-group detectors above already chose
    # the right per-tensor quant for each block.
    if _compressed_tensors_nvfp4(hf_config) and not _has_moe_experts(hf_config):
        attn_quant = "nvfp4"
        dense_quant = "nvfp4"
        lm_head_quant = "none"

    # Dense variants (e.g. Qwen3.6-27B) report num_experts==0: route the decoder MLP through
    # the dense Qwen3_5DenseMLP instead of the MoE block.
    num_experts = getattr(text, "num_experts", 0) or 0
    moe_enabled = num_experts > 0

    layer_types = _layer_types(text)
    full_ids = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    linear_ids = tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")

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
        vision_config=None,  # text-only milestone
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=groups,
        expert_quant=expert_quant,
        weight_block_size=weight_block_size,
        attn_quant=attn_quant,
        dense_quant=dense_quant,
        lm_head_quant=lm_head_quant,
        nvfp4_global_reciprocal=_nvfp4_global_reciprocal(hf_config),
    )


__all__ = ["parse_config"]
