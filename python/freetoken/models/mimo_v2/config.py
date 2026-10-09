from __future__ import annotations

import math
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
)


def _rope_params(hf_config: Any) -> dict:
    top: dict = {}
    rope_theta = getattr(hf_config, "rope_theta", None)
    if rope_theta is not None:
        top["rope_theta"] = rope_theta
    partial = getattr(hf_config, "partial_rotary_factor", None)
    if partial is not None:
        top["partial_rotary_factor"] = partial
    scaling = getattr(hf_config, "rope_scaling", None)
    if isinstance(scaling, dict):
        top.update(scaling)

    params = getattr(hf_config, "rope_parameters", None)
    if isinstance(params, dict) and params:
        params = dict(params)
        for key, value in top.items():
            if key in params and params[key] != value:
                raise ValueError(
                    f"MiMo v2 rope config conflict: rope_parameters[{key!r}]={params[key]!r} "
                    f"disagrees with top-level {key}={value!r}; the two rope sources must agree"
                )
        return params
    if top:
        return top
    return {"rope_theta": 10000.0}


def _rotary_dim(head_dim: int, partial_rotary_factor: float) -> int:
    # HF MiMoV2Attention: self.rope_dim = int(head_dim * partial_rotary_factor), even enforced
    # (reference/modeling_mimo_v2.py:253-258; inv_freq spacing over that dim at :481-489).
    rotary_dim = int(head_dim * partial_rotary_factor)
    if rotary_dim % 2 != 0:
        raise ValueError(
            f"MiMo v2 rotary dimension must be even, got {rotary_dim} from "
            f"head_dim={head_dim} and partial_rotary_factor={partial_rotary_factor}"
        )
    return rotary_dim


def _get(config: Any, key: str, default: Any = None) -> Any:
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def _expert_quant(
    hf_config: Any, *, num_layers: int, first_moe_layer: int, num_experts: int
) -> str:
    quant = getattr(hf_config, "quantization_config", None)
    if quant is None:
        return "none"
    method = str(_get(quant, "quant_method") or "").lower()
    algo = str(_get(quant, "quant_algo") or "").upper()
    if method != "modelopt" or algo != "MIXED_PRECISION":
        raise ValueError(
            "MiMo v2 supports unquantized BF16 weights or ModelOpt MIXED_PRECISION "
            "W4A16_NVFP4 routed experts only"
        )
    if int(_get(quant, "group_size", 0) or 0) != 16:
        raise ValueError("MiMo v2 ModelOpt MIXED_PRECISION requires group_size=16")
    if _get(quant, "exclude_modules") or _get(quant, "ignore"):
        raise ValueError("MiMo v2 ModelOpt quantization must target routed experts explicitly")

    layers = _get(quant, "quantized_layers")
    if not isinstance(layers, dict):
        raise ValueError("MiMo v2 ModelOpt MIXED_PRECISION requires quantized_layers")
    expected = {
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj"
        for layer in range(first_moe_layer, num_layers)
        for expert in range(num_experts)
        for projection in ("gate", "up", "down")
    }
    if set(layers) != expected:
        missing = sorted(expected - set(layers))[:3]
        extra = sorted(set(layers) - expected)[:3]
        raise ValueError(
            "MiMo v2 ModelOpt quantized_layers must list every routed expert projection "
            f"and no other modules (missing={missing}, extra={extra})"
        )
    for name, entry in layers.items():
        if (
            str(_get(entry, "quant_algo") or "").upper() != "W4A16_NVFP4"
            or int(_get(entry, "group_size", 0) or 0) != 16
        ):
            raise ValueError(
                f"MiMo v2 ModelOpt target {name!r} must use W4A16_NVFP4 group_size=16"
            )
    return "nvfp4"


def _validate_config(hf_config: Any) -> None:
    num_layers = int(hf_config.num_hidden_layers)
    if num_layers < 2:
        raise ValueError("MiMo v2 requires at least one full-attention and one SWA layer")
    pattern = list(getattr(hf_config, "hybrid_layer_pattern", [0] * num_layers))
    if len(pattern) != num_layers or any(value not in (0, 1) for value in pattern):
        raise ValueError("MiMo v2 hybrid_layer_pattern must contain one 0/1 entry per layer")
    if 0 not in pattern or 1 not in pattern:
        raise ValueError("MiMo v2 requires at least one full-attention and one SWA layer")

    qo_heads = int(hf_config.num_attention_heads)
    swa_qo_heads = int(getattr(hf_config, "swa_num_attention_heads", qo_heads))
    if qo_heads <= 0 or swa_qo_heads != qo_heads:
        raise ValueError("MiMo v2 requires the same positive query-head count in full and SWA layers")
    full_dim = int(getattr(hf_config, "head_dim", None) or hf_config.hidden_size // qo_heads)
    swa_dim = int(getattr(hf_config, "swa_head_dim", None) or full_dim)
    full_kv = int(getattr(hf_config, "num_key_value_heads", qo_heads))
    swa_kv = int(getattr(hf_config, "swa_num_key_value_heads", full_kv))
    full_v = int(getattr(hf_config, "v_head_dim", None) or full_dim)
    swa_v = int(getattr(hf_config, "swa_v_head_dim", None) or full_v)
    for name, kv_heads, head_dim, v_dim in (
        ("full", full_kv, full_dim, full_v),
        ("SWA", swa_kv, swa_dim, swa_v),
    ):
        if kv_heads <= 0 or qo_heads % kv_heads:
            raise ValueError(f"MiMo v2 {name} query heads must divide evenly into positive KV heads")
        if head_dim <= 0 or not 0 < v_dim <= head_dim:
            raise ValueError(
                f"MiMo v2 {name} geometry requires 0 < v_head_dim <= head_dim, "
                f"got {v_dim} and {head_dim}"
            )

    window = int(getattr(hf_config, "sliding_window", 0) or 0)
    window_size = getattr(hf_config, "sliding_window_size", None)
    if window <= 0 or (window_size is not None and int(window_size) != window):
        raise ValueError("MiMo v2 requires matching positive sliding_window and sliding_window_size")
    if bool(getattr(hf_config, "add_full_attention_sink_bias", False)):
        raise ValueError("MiMo v2 supports attention sinks on SWA layers only")
    if not bool(getattr(hf_config, "add_swa_attention_sink_bias", False)):
        raise ValueError("MiMo v2 requires SWA attention sink bias")
    if getattr(hf_config, "attention_projection_layout", "fused_qkv") != "fused_qkv":
        raise ValueError("MiMo v2 supports fused_qkv attention projection layout only")
    if bool(getattr(hf_config, "attention_bias", False)):
        raise ValueError("MiMo v2 text-tower attention projections must be bias-free")
    if getattr(hf_config, "hidden_act", None) != "silu":
        raise ValueError("MiMo v2 supports hidden_act='silu' only")

    moe_freq = list(getattr(hf_config, "moe_layer_freq", [1] * num_layers))
    if len(moe_freq) != num_layers or any(value not in (0, 1) for value in moe_freq):
        raise ValueError("MiMo v2 moe_layer_freq must contain one 0/1 entry per layer")
    first_moe = next((i for i, value in enumerate(moe_freq) if value), num_layers)
    if first_moe == num_layers or any(value == 0 for value in moe_freq[first_moe:]):
        raise ValueError("MiMo v2 supports a dense prefix followed by a contiguous sparse MoE suffix")
    if int(getattr(hf_config, "n_shared_experts", 0) or 0) != 0:
        raise ValueError("MiMo v2 supports routed experts only; shared experts are unsupported")
    if getattr(hf_config, "scoring_func", "sigmoid") != "sigmoid":
        raise ValueError("MiMo v2 supports sigmoid MoE routing only")
    if getattr(hf_config, "topk_method", "noaux_tc") != "noaux_tc":
        raise ValueError("MiMo v2 supports noaux_tc MoE routing only")
    num_experts = int(getattr(hf_config, "n_routed_experts", 0) or 0)
    top_k = int(getattr(hf_config, "num_experts_per_tok", 0) or 0)
    n_group = int(getattr(hf_config, "n_group", 1))
    topk_group = int(getattr(hf_config, "topk_group", 1))
    if num_experts <= 0 or not 0 < top_k <= num_experts:
        raise ValueError("MiMo v2 requires 0 < num_experts_per_tok <= n_routed_experts")
    if n_group <= 0 or num_experts % n_group or not 0 < topk_group <= n_group:
        raise ValueError("MiMo v2 requires valid n_group/topk_group routing geometry")
    if n_group > 1:
        experts_per_group = num_experts // n_group
        # _group_limited topk(2) inside width-1 groups crashes, and eligible experts
        # (topk_group * width) below top_k make masked-out ids win with nonzero weights.
        if experts_per_group < 2 or top_k > topk_group * experts_per_group:
            raise ValueError(
                "MiMo v2 grouped routing requires group width >= 2 and num_experts_per_tok "
                f"<= topk_group * (n_routed_experts // n_group), got n_group={n_group}, "
                f"topk_group={topk_group}, group width={experts_per_group}, "
                f"num_experts_per_tok={top_k}"
            )

    rope = _rope_params(hf_config)
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    if rope_type not in (None, "default"):
        raise ValueError(f"MiMo v2 supports default RoPE only, got {rope_type!r}")
    for key, value in (
        ("rope_theta", rope.get("rope_theta", getattr(hf_config, "rope_theta", 10000.0))),
        ("swa_rope_theta", getattr(hf_config, "swa_rope_theta", None)),
    ):
        if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
            raise ValueError(f"MiMo v2 {key} must be finite and positive")
    partial = float(rope.get("partial_rotary_factor", getattr(hf_config, "partial_rotary_factor", 1.0)))
    if not math.isfinite(partial) or not 0 < partial <= 1:
        raise ValueError(f"MiMo v2 partial_rotary_factor must be in (0, 1], got {partial}")
    _rotary_dim(full_dim, partial)
    _rotary_dim(swa_dim, partial)

    scale = getattr(hf_config, "attention_value_scale", None)
    if scale is not None and (
        isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale)
    ):
        raise ValueError("MiMo v2 attention_value_scale must be a finite number or None")


def parse_config(hf_config: Any) -> ModelConfig:
    """Parse a HuggingFace ``MiMoV2Config`` into FreeToken's :class:`ModelConfig`.

    MiMo v2 specifics (reference/MIMO_CONTRACT.md SECTION 1):
    - hybrid full/SWA attention split by ``hybrid_layer_pattern`` (0 = full, 9 layers;
      1 = swa, 39 layers with ``sliding_window`` 128); head_dim 192 in both groups, V
      heads 128 in both (the family pads V to 192, Option B, in attention.py).
    - dual rope: theta ``rope_theta`` on full layers, ``swa_rope_theta`` on SWA layers,
      partial rotary factor 0.334 -> rotary_dim 64.
    - layer 0 is a dense SwiGLU MLP (``moe_layer_freq[0] == 0``); layers 1-47 are MoE
      with 256 routed experts, sigmoid + correction-bias (noaux_tc, n_group=1) routing,
      no shared experts and ``routed_scaling_factor`` defaulting to 1.0.
    - ``mtp``/vision/audio towers are dropped (text-only); ``attention_chunk_size`` has
      no kernel-math consumer (contract OQ-2) and is ignored.
    """
    _validate_config(hf_config)
    num_layers = hf_config.num_hidden_layers
    head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
    swa_head_dim = getattr(hf_config, "swa_head_dim", None) or head_dim
    # Per-group V width (modeling_mimo_v2.py:239-249): the full group reads ``v_head_dim``
    # (default head_dim), the SWA group ``swa_v_head_dim`` (default v_head_dim). The
    # release ships 128 on both; a checkpoint may ship any width (the tiny validation
    # ckpt uses 64), and the qkv/o_proj row math follows the config, never a constant.
    v_head_dim = getattr(hf_config, "v_head_dim", None) or head_dim
    swa_v_head_dim = getattr(hf_config, "swa_v_head_dim", None) or v_head_dim
    num_kv_heads = getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)
    swa_num_kv_heads = getattr(hf_config, "swa_num_key_value_heads", num_kv_heads)

    rope = _rope_params(hf_config)
    rope_theta = rope.get("rope_theta", getattr(hf_config, "rope_theta", 10000.0))
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    rope_scaling = None if rope_type in (None, "default") else rope
    partial = rope.get("partial_rotary_factor", getattr(hf_config, "partial_rotary_factor", 1.0))
    swa_rope_theta = getattr(hf_config, "swa_rope_theta", rope_theta)

    full_rotary_config = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=_rotary_dim(head_dim, float(partial)),
        max_position=hf_config.max_position_embeddings,
        base=float(rope_theta),
        scaling=rope_scaling,
    )
    swa_rotary_config = RotaryConfig(
        head_dim=swa_head_dim,
        rotary_dim=_rotary_dim(swa_head_dim, float(partial)),
        max_position=hf_config.max_position_embeddings,
        base=float(swa_rope_theta),
        scaling=rope_scaling,
    )

    pattern = list(getattr(hf_config, "hybrid_layer_pattern", [0] * num_layers))
    full_layer_ids = tuple(i for i, p in enumerate(pattern) if p == 0)
    swa_layer_ids = tuple(i for i, p in enumerate(pattern) if p != 0)

    moe_layer_freq = list(getattr(hf_config, "moe_layer_freq", [1] * num_layers))
    first_k_dense_replace = next((i for i, freq in enumerate(moe_layer_freq) if freq), len(moe_layer_freq))
    num_experts = int(getattr(hf_config, "n_routed_experts", 0) or 0)
    expert_quant = _expert_quant(
        hf_config,
        num_layers=num_layers,
        first_moe_layer=first_k_dense_replace,
        num_experts=num_experts,
    )

    routed_scaling_factor = getattr(hf_config, "routed_scaling_factor", None)
    # HF: `... if ... is not None else 1.0` (modeling_mimo_v2.py:141,183)

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=hf_config.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hf_config.hidden_size,
        vocab_size=hf_config.vocab_size,
        intermediate_size=hf_config.intermediate_size,
        hidden_act=hf_config.hidden_act,
        rms_norm_eps=getattr(hf_config, "layernorm_epsilon", getattr(hf_config, "rms_norm_eps", 1e-6)),
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        rotary_config=full_rotary_config,
        num_experts=num_experts,
        num_experts_per_tok=hf_config.num_experts_per_tok,
        moe_intermediate_size=hf_config.moe_intermediate_size,
        norm_topk_prob=bool(getattr(hf_config, "norm_topk_prob", True)),
        model_type=getattr(hf_config, "model_type", "mimo_v2"),
        architectures=getattr(hf_config, "architectures", ["MiMoV2ForCausalLM"]),
        moe_enabled=True,
        expert_quant=expert_quant,
        first_k_dense_replace=first_k_dense_replace,
        n_shared_experts=int(getattr(hf_config, "n_shared_experts", 0) or 0),
        routed_scaling_factor=1.0 if routed_scaling_factor is None else float(routed_scaling_factor),
        n_group=int(getattr(hf_config, "n_group", 1)),
        topk_group=int(getattr(hf_config, "topk_group", 1)),
        has_attn_bias=bool(getattr(hf_config, "attention_bias", False)),
        # sm_scale = head_dim**-0.5 = 192**-0.5 on BOTH groups (QK dim based, contract DEC-6)
        attn_sm_scale=float(head_dim) ** -0.5,
        attention_value_scale=(
            None if getattr(hf_config, "attention_value_scale", None) is None
            else float(hf_config.attention_value_scale)
        ),
        attention_groups=(
            FullAttentionGroupConfig(
                name="full",
                layer_ids=full_layer_ids,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_config=full_rotary_config,
                v_head_dim=v_head_dim,
            ),
            SWAAttentionGroupConfig(
                name="swa",
                layer_ids=swa_layer_ids,
                num_kv_heads=swa_num_kv_heads,
                head_dim=swa_head_dim,
                rotary_config=swa_rotary_config,
                sliding_window=hf_config.sliding_window,
                v_head_dim=swa_v_head_dim,
            ),
        ),
    )


__all__ = ["parse_config"]
