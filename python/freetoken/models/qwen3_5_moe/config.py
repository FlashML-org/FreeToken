from __future__ import annotations

from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    detect_compressed_tensors_nvfp4,
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
    return "none"


# Detection now lives in models/config.py (shared with muse_glimmer); weight.py imports
# it under this name.
_compressed_tensors_nvfp4 = detect_compressed_tensors_nvfp4


def _compressed_linear_storage(
    hf_config: Any,
) -> tuple[str, str, dict[int, str] | None, bool]:
    """Per-module dense-linear storage sniffed from the checkpoint's
    ``model.safetensors.index.json`` weight_map:
    ``(attention storage, dense-MLP fallback, per-layer dense-MLP overrides, lm_head
    is fp8)``.

    The attention storage is ``"nvfp4"`` (packed q/k/v/o + GDN out_proj linears),
    ``"fp8"`` (weight + scale siblings, e.g. unsloth's mixed export), or ``"none"``
    (bf16). The fallback is ``"nvfp4"`` when at least one dense-MLP layer is packed,
    else ``"none"``. The override map gives the storage of dense-MLP layers that are
    not packed NVFP4 (``"fp8"`` keeps them native W8A16; ``"bf16"`` dequantizes at
    load); ``None`` means every dense-MLP layer follows the fallback. When the index
    is unavailable (a hub id before download, or a single-file checkpoint) assume
    the official dense-NVFP4 layout (all packed, no overrides, bf16 lm_head).
    """
    import json
    import os
    import re

    path = getattr(hf_config, "name_or_path", None)
    if not isinstance(path, str) or not os.path.isdir(path):
        return "nvfp4", "nvfp4", None, False
    index = os.path.join(path, "model.safetensors.index.json")
    if not os.path.isfile(index):
        return "nvfp4", "nvfp4", None, False
    with open(index, encoding="utf-8") as f:
        weight_map = json.load(f).get("weight_map", {})
    keys = set(weight_map)
    attn_suffix_re = re.compile(r"\.(?:self_attn\.(?:q|k|v|o)_proj|linear_attn\.out_proj)\.")
    attn_bases = set()
    for k in keys:
        if attn_suffix_re.search(k) and k.endswith((".weight", ".weight_packed")):
            attn_bases.add(k.rsplit(".", 1)[0])
    attn_packed = any(b + ".weight_packed" in keys for b in attn_bases)
    attn_fp8 = any(b + ".weight_scale" in keys for b in attn_bases)
    attn = "nvfp4" if attn_packed else ("fp8" if attn_fp8 else "none")
    # Per-layer dense-MLP storage (order-independent): packed -> nvfp4, any scale
    # sibling -> fp8 (native W8A16), bare weight -> bf16. lm_head: fp8 when it ships a
    # scale sibling.
    mlp_key_re = re.compile(
        r"\.layers\.(\d+)\.mlp\.(?:gate|up|down)_proj\.(weight_packed|weight_scale|weight)$"
    )
    packed_layers: set[int] = set()
    scaled_layers: set[int] = set()
    plain_layers: set[int] = set()
    for k in keys:
        mm = mlp_key_re.search(k)
        if mm:
            layer = int(mm.group(1))
            suffix = mm.group(2)
            if suffix == "weight_packed":
                packed_layers.add(layer)
            elif suffix == "weight_scale":
                scaled_layers.add(layer)
            else:
                plain_layers.add(layer)
    lmhead_fp8 = any(k == "lm_head.weight_scale" or k.endswith(".lm_head.weight_scale") for k in keys)
    # No mlp.{gate,up,down}_proj keys at all: routed-expert MoE naming (shared_expert /
    # experts) -- keep the native assumption with no per-layer overrides.
    if not (packed_layers or scaled_layers or plain_layers):
        return attn, "nvfp4", None, lmhead_fp8
    all_layers = packed_layers | scaled_layers | plain_layers
    overrides = {
        layer: ("fp8" if layer in scaled_layers else "bf16")
        for layer in all_layers - packed_layers
    }
    return attn, "nvfp4" if packed_layers else "none", overrides or None, lmhead_fp8


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
    weight + a scalar ``weight_scale``; W8A16). Returns ``"fp8_pertensor"`` when present,
    else ``"none"`` (NVFP4 dense weights -- shared_expert/lm_head -- stay dequant-at-load)."""
    get = _quant_accessor(hf_config)
    if get is None:
        return "none"
    layers = get("quantized_layers") or {}
    if not isinstance(layers, dict):
        return "none"
    for name, spec in layers.items():
        algo = str((spec or {}).get("quant_algo", "")).lower()
        if algo == "fp8" and (".self_attn." in name or ".linear_attn." in name):
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
    # NVFP4 checkpoints store the dense MLP projections (shared_expert; dense non-MoE MLP) as
    # packed FP4 exactly like the routed experts -- independent of whether attention is FP8
    # (mixed) or bf16 (pure NVFP4). Keep them native FP4 (W4A16) whenever the experts are
    # NVFP4. The lm_head is detected separately (only the mixed checkpoint quantizes it).
    # MoE-NVFP4 keeps the shared_expert dense MLP native FP4 (expert_quant=="nvfp4"); a dense
    # (non-MoE) modelopt checkpoint instead tags the bare .mlp.{gate,up,down}_proj as NVFP4.
    dense_quant = "nvfp4" if expert_quant == "nvfp4" else _dense_mlp_quant(hf_config)
    lm_head_quant = _lm_head_quant(hf_config)

    # compressed-tensors NVFP4 (dense Qwen3.6-27B): the attention (q/k/v/o, GDN out_proj) AND
    # the dense MLP are W4A16 NVFP4; GDN in_proj_*, lm_head, norms stay bf16. Wire the shared
    # W4A16 kernels (attn_quant=="nvfp4" routes the attention/GDN linears through them too).
    # Mixed compressed-tensors exports (unsloth's dynamic per-module quant) store the
    # attention projections as weight-only FP8 (per-row scale) and some dense-MLP layers
    # as FP8: keep each group in its native storage (W8A16 attention, W4A16 packed MLP
    # layers, bf16 dequant for the rest) instead of dequantizing everything to bf16.
    dense_mlp_storage: dict[int, str] | None = None
    if _compressed_tensors_nvfp4(hf_config):
        attn_storage, dense_quant, dense_mlp_storage, lmhead_fp8 = _compressed_linear_storage(hf_config)
        attn_quant = (
            "fp8_pertensor" if attn_storage == "fp8"
            else "nvfp4" if attn_storage == "nvfp4"
            else "none"
        )
        # unsloth's mixed export stores lm_head as weight-only FP8 (per-row scale): keep
        # it native W8A16 (halves the ~2.5 GB bf16 lm_head); official layouts are bf16.
        lm_head_quant = "fp8_pertensor" if lmhead_fp8 else "none"

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
        dense_mlp_storage=dense_mlp_storage,
    )


__all__ = ["parse_config"]
