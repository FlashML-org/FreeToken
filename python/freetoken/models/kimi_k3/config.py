"""Engine-facing, fail-closed config for the Kimi-K3 text tower."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)

from .args import load_args


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _text_config(hf_config: Any) -> Any:
    return _value(hf_config, "text_config") or hf_config


def _quant_contract(value: Any) -> tuple[int, str, int, str, str]:
    return (
        int(_value(value, "num_bits", 0) or 0),
        str(_value(value, "type", "")).lower(),
        int(_value(value, "group_size", 0) or 0),
        str(_value(value, "strategy", "")).lower(),
        str(_value(value, "scale_dtype", "")).lower(),
    )


def _development_ignores(text: Any) -> set[str]:
    linear = _value(text, "linear_attn_config") or {}
    kda_layers = {int(x) - 1 for x in (_value(linear, "kda_layers", ()) or ())}
    full_layers = {int(x) - 1 for x in (_value(linear, "full_attn_layers", ()) or ())}
    num_layers = int(_value(text, "num_hidden_layers", 0) or 0)
    ignores = {
        *(
            f"vision_tower.encoder.blocks.{layer}.{suffix}"
            for layer in range(2)
            for suffix in ("mlp.fc0", "mlp.fc1", "wqkv", "wo")
        ),
        "mm_projector.proj.0",
        "mm_projector.proj.2",
        "language_model.model.output_attn_res_proj",
        "language_model.lm_head",
    }
    kda_suffixes = (
        "q_proj",
        "k_proj",
        "v_proj",
        "f_a_proj",
        "f_b_proj",
        "b_proj",
        "g_proj",
        "o_norm",
        "o_proj",
    )
    mla_suffixes = (
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
        "g_proj",
    )
    for layer in range(num_layers):
        prefix = f"language_model.model.layers.{layer}"
        suffixes = kda_suffixes if layer in kda_layers else mla_suffixes
        if layer not in kda_layers | full_layers:
            raise ValueError("Kimi-K3 development attention map is incomplete")
        ignores.update(f"{prefix}.self_attn.{suffix}" for suffix in suffixes)
        ignores.update((f"{prefix}.self_attention_res_proj", f"{prefix}.mlp_res_proj"))
        if layer == 0:
            ignores.update(
                f"{prefix}.mlp.{part}_proj" for part in ("gate", "up", "down")
            )
        else:
            ignores.add(f"{prefix}.block_sparse_moe.gate")
    return ignores


def detect_kimi_mxfp4(config: Any) -> str:
    """Accept only the two audited compressed-tensors Kimi MXFP4 layouts."""
    text = _text_config(config)
    quant = _value(config, "quantization_config") or _value(text, "quantization_config")
    if not quant:
        return "none"
    method = str(_value(quant, "quant_method", "")).lower()
    fmt = str(_value(quant, "format", "")).lower()
    groups = _value(quant, "config_groups", {}) or {}
    if method != "compressed-tensors" or fmt != "mxfp4-pack-quantized":
        raise ValueError(
            f"unsupported Kimi-K3 quantization: method={method!r}, format={fmt!r}"
        )
    if not isinstance(groups, Mapping) or len(groups) != 1:
        raise ValueError("Kimi-K3 support requires exactly one MXFP4 config group")
    group = next(iter(groups.values()))
    if str(_value(group, "format", "")).lower() != "mxfp4-pack-quantized":
        raise ValueError(
            "Kimi-K3 MXFP4 group format does not match the public checkpoint"
        )
    targets = list(_value(group, "targets", ()) or ())
    weights = _value(group, "weights", {}) or {}
    contract = _quant_contract(weights)
    if contract != (4, "float", 32, "group", "torch.uint8"):
        raise ValueError(f"unsupported Kimi-K3 MXFP4 geometry: {contract!r}")
    if _value(group, "output_activations") is not None:
        raise ValueError("Kimi-K3 W4A16 support requires unquantized outputs")

    # The public checkpoint is deliberately mixed precision: only the routed
    # experts are MXFP4.  Accepting a future export that quantizes attention,
    # shared experts, the dense prefix, or the vocabulary head would construct
    # BF16 modules for packed tensors and fail only after a multi-terabyte
    # download.  Verify the exact exclusion contract up front instead.
    production_ignores = {
        r"re:.*self_attn.*",
        r"re:.*shared_experts.*",
        r"re:.*mlp\.(gate|up|gate_up|down)_proj.*",
        r"re:.*lm_head.*",
        r"re:.*vision_tower.*",
        r"re:.*mm_projector.*",
    }
    ignores = set(_value(quant, "ignore", ()) or ())
    input_activations = _value(group, "input_activations")
    if targets == ["Linear"]:
        if input_activations is not None:
            raise ValueError("Kimi-K3 W4A16 support requires unquantized activations")
        if ignores != production_ignores:
            raise ValueError(
                "Kimi-K3 MXFP4 ignore contract differs from the public routed-expert-only checkpoint"
            )
        return "mxfp4"

    # The 0.40B development checkpoint stores every non-router MoE projection
    # in MXFP4 and records dynamic group-32 activation quantization. FreeToken's
    # W4A16 kernels consume the same packed weights with BF16 activations.
    if targets != [r"re:.*block_sparse_moe.*"]:
        raise ValueError(f"unsupported Kimi-K3 MXFP4 targets: {targets!r}")
    activation_contract = _quant_contract(input_activations or {})
    if activation_contract != (4, "float", 32, "group", "torch.uint8") or not bool(
        _value(input_activations, "dynamic", False)
    ):
        raise ValueError(
            f"unsupported Kimi-K3 activation quantization: {activation_contract!r}"
        )
    if ignores != _development_ignores(text):
        raise ValueError(
            "Kimi-K3 MXFP4 ignore contract differs from the audited development checkpoint"
        )
    return "mxfp4"


def parse_config(hf_config: Any) -> ModelConfig:
    text = _text_config(hf_config)
    args = load_args(text)
    expert_quant = detect_kimi_mxfp4(hf_config)
    first_dense = int(_value(text, "first_k_dense_replace", 0))
    if first_dense != 1:
        raise ValueError(
            f"Kimi-K3 support requires one dense prefix layer, got {first_dense}"
        )
    if str(_value(text, "hidden_act")) != "situ":
        raise ValueError("Kimi-K3 support implements SiTU activation only")
    if str(_value(text, "moe_router_activation_func")) != "sigmoid":
        raise ValueError("Kimi-K3 support implements sigmoid expert routing only")
    if str(_value(text, "topk_method")) != "noaux_tc":
        raise ValueError("Kimi-K3 support implements noaux_tc expert selection only")

    # Kimi's MLA is explicitly NoPE. The dimensions still describe the raw q/k
    # split and latent cache; rotary_dim=0 prevents accidental position transforms.
    no_rope = RotaryConfig(
        head_dim=args.qk_head_dim,
        rotary_dim=0,
        max_position=int(_value(text, "max_position_embeddings")),
        base=10000.0,
        scaling=None,
    )
    groups = tuple(
        sorted(
            (
                LinearGatedDeltaGroupConfig(
                    name="kda",
                    layer_ids=args.kda_layer_ids,
                    num_key_heads=args.kda_num_heads,
                    num_value_heads=args.kda_num_heads,
                    key_head_dim=args.kda_head_dim,
                    value_head_dim=args.kda_head_dim,
                    conv_kernel_dim=args.kda_conv_kernel,
                    output_gate=True,
                ),
                FullAttentionGroupConfig(
                    name="mla",
                    layer_ids=args.mla_layer_ids,
                    num_kv_heads=1,
                    head_dim=args.mla_latent_dim,
                    rotary_config=no_rope,
                    mla=True,
                ),
            ),
            key=lambda group: group.layer_ids[0],
        )
    )
    return ModelConfig(
        num_layers=args.num_layers,
        num_qo_heads=args.num_heads,
        num_kv_heads=1,
        head_dim=args.mla_latent_dim,
        hidden_size=args.hidden_size,
        vocab_size=int(_value(text, "vocab_size")),
        intermediate_size=int(_value(text, "intermediate_size")),
        hidden_act="situ",
        rms_norm_eps=float(_value(text, "rms_norm_eps")),
        tie_word_embeddings=bool(_value(text, "tie_word_embeddings", False)),
        rotary_config=no_rope,
        attention_groups=groups,
        num_experts=int(_value(text, "num_experts")),
        num_experts_per_tok=int(_value(text, "num_experts_per_token")),
        moe_intermediate_size=int(_value(text, "moe_intermediate_size")),
        norm_topk_prob=bool(_value(text, "moe_renormalize")),
        model_type=str(_value(hf_config, "model_type", "kimi_k3")),
        architectures=list(
            _value(hf_config, "architectures", ["KimiK3ForConditionalGeneration"])
        ),
        supports_hybrid_radix=False,
        moe_enabled=True,
        expert_quant=expert_quant,
        moe_weight_format="mxfp4" if expert_quant == "mxfp4" else None,
        first_k_dense_replace=first_dense,
        n_shared_experts=int(_value(text, "num_shared_experts", 0)),
        shared_expert_intermediate_size=(
            int(_value(text, "num_shared_experts", 0))
            * int(_value(text, "moe_intermediate_size"))
        ),
        routed_scaling_factor=float(_value(text, "routed_scaling_factor", 1.0)),
        n_group=int(_value(text, "num_expert_group", 1)),
        topk_group=int(_value(text, "topk_group", 1)),
        has_router_bias=True,
        attn_sm_scale=args.qk_head_dim**-0.5,
        vision_config=None,
        kimi_k3_args=args,
    )


__all__ = ["detect_kimi_mxfp4", "parse_config"]
