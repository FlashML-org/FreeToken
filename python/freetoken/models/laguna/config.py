from __future__ import annotations

import re
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
)
from freetoken.models.gguf.dequant import GGML_BF16, GGML_Q4_0


def _field(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _int4_scheme(hf_config: Any) -> dict:
    quant = _field(hf_config, "quantization_config") or {}
    groups = _field(quant, "config_groups", {}) or {}
    for group in groups.values():
        weights = _field(group, "weights", {}) or {}
        if (
            int(_field(weights, "num_bits", 0) or 0) == 4
            and str(_field(weights, "type", "")).lower() == "int"
        ):
            if (
                int(_field(weights, "group_size", 0) or 0) != 32
                or str(_field(weights, "strategy", "")).lower() != "group"
                or not bool(_field(weights, "symmetric", False))
            ):
                raise ValueError(
                    "Laguna compressed-tensors INT4 requires symmetric group-wise "
                    "weights with group_size=32"
                )
            return group
    raise ValueError(
        "Laguna safetensors support requires the compressed-tensors symmetric INT4 scheme"
    )


def _matches(patterns: list[str], module_name: str) -> bool:
    for pattern in patterns:
        if pattern.startswith("re:"):
            if re.fullmatch(pattern[3:], module_name):
                return True
        elif pattern in {"Linear", module_name}:
            return True
    return False


def _expert_types(hf_config: Any) -> tuple[tuple[int, int], ...]:
    """Per-MoE-layer storage type, including compressed-tensors ignore rules.

    The published Laguna-S INT4 checkpoint quantizes routed experts in layers 1--39
    and intentionally leaves layers 40--47 in BF16.  Carrying that distinction into
    the cache avoids silently requantizing the accuracy-sensitive tail.
    """
    scheme = _int4_scheme(hf_config)
    targets = list(_field(scheme, "targets", []) or [])
    quant = _field(hf_config, "quantization_config") or {}
    ignores = list(_field(quant, "ignore", []) or [])
    dense = len(list(_field(hf_config, "mlp_only_layers", [0]) or [0]))
    out = []
    for layer in range(dense, int(_field(hf_config, "num_hidden_layers"))):
        module = f"model.layers.{layer}.mlp.experts.0.gate_proj"
        is_int4 = _matches(targets, module) and not _matches(ignores, module)
        qtype = GGML_Q4_0 if is_int4 else GGML_BF16
        out.append((qtype, qtype))
    return tuple(out)


def _rope_config(params: dict, *, head_dim: int, max_position: int) -> RotaryConfig:
    partial = float(params.get("partial_rotary_factor", 1.0))
    rope_type = str(params.get("rope_type", "default"))
    scaling = None
    if rope_type != "default":
        scaling = {
            key: value
            for key, value in params.items()
            if key not in {"rope_theta", "partial_rotary_factor"}
        }
    return RotaryConfig(
        head_dim=head_dim,
        rotary_dim=int(head_dim * partial),
        max_position=max_position,
        base=float(params.get("rope_theta", 10000.0)),
        scaling=scaling,
    )


def parse_config(hf_config: Any) -> ModelConfig:
    num_layers = int(hf_config.num_hidden_layers)
    head_dim = int(hf_config.head_dim)
    max_position = int(hf_config.max_position_embeddings)
    head_counts = tuple(int(v) for v in hf_config.num_attention_heads_per_layer)
    layer_types = tuple(hf_config.layer_types)
    if len(head_counts) != num_layers or len(layer_types) != num_layers:
        raise ValueError("Laguna per-layer attention metadata length does not match num_hidden_layers")

    rope = hf_config.rope_parameters
    full_rotary = _rope_config(rope["full_attention"], head_dim=head_dim, max_position=max_position)
    swa_rotary = _rope_config(rope["sliding_attention"], head_dim=head_dim, max_position=max_position)
    full_layers = tuple(i for i, kind in enumerate(layer_types) if kind == "full_attention")
    swa_layers = tuple(i for i, kind in enumerate(layer_types) if kind == "sliding_attention")
    if len(full_layers) + len(swa_layers) != num_layers:
        raise ValueError(f"unsupported Laguna layer_types: {sorted(set(layer_types))}")

    dense_layers = tuple(int(i) for i in getattr(hf_config, "mlp_only_layers", [0]))
    if dense_layers != tuple(range(len(dense_layers))):
        raise ValueError("FreeToken requires Laguna dense MLP layers to be a leading prefix")

    types = _expert_types(hf_config)
    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=max(head_counts),
        num_qo_heads_per_layer=head_counts,
        num_kv_heads=int(hf_config.num_key_value_heads),
        head_dim=head_dim,
        hidden_size=int(hf_config.hidden_size),
        vocab_size=int(hf_config.vocab_size),
        intermediate_size=int(hf_config.intermediate_size),
        rms_norm_eps=float(hf_config.rms_norm_eps),
        rotary_config=full_rotary,
        hidden_act=str(getattr(hf_config, "hidden_act", "silu")),
        tie_word_embeddings=bool(hf_config.tie_word_embeddings),
        num_experts=int(hf_config.num_experts),
        num_experts_per_tok=int(hf_config.num_experts_per_tok),
        moe_intermediate_size=int(hf_config.moe_intermediate_size),
        shared_expert_intermediate_size=int(hf_config.shared_expert_intermediate_size),
        n_shared_experts=1,
        norm_topk_prob=bool(hf_config.norm_topk_prob),
        routed_scaling_factor=float(hf_config.moe_routed_scaling_factor),
        first_k_dense_replace=len(dense_layers),
        use_qk_norm=True,
        model_type="laguna",
        architectures=list(hf_config.architectures),
        moe_enabled=True,
        attention_groups=(
            FullAttentionGroupConfig(
                name="full",
                layer_ids=full_layers,
                num_kv_heads=int(hf_config.num_key_value_heads),
                head_dim=head_dim,
                rotary_config=full_rotary,
            ),
            SWAAttentionGroupConfig(
                name="swa",
                layer_ids=swa_layers,
                num_kv_heads=int(hf_config.num_key_value_heads),
                head_dim=head_dim,
                rotary_config=swa_rotary,
                sliding_window=int(hf_config.sliding_window),
            ),
        ),
        expert_quant="laguna_int4",
        moe_weight_format="laguna_int4",
        gguf_expert_types=types,
    )


__all__ = ["parse_config"]
