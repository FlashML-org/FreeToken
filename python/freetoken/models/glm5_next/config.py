"""Engine-facing config for GLM-5.3-Flash (``glm5_next``).

Two attention groups, ordered by first layer id:

* ``linear`` -- 34 KDA layers (``LinearGatedDeltaGroupConfig`` with
  ``variant="kda"``). KDA's state geometry coincides with GDN's (conv width
  ``2*K*dk + V*dv`` == q|k|v at ``H*d`` each; recurrent ``[H, d, d]``), so the
  existing ``LinearStatePool`` shapes serve it unchanged.
* ``full`` -- 11 DSA layers (``FullAttentionGroupConfig`` with ``mla=True`` and
  the indexer dims). The MLA is NoPE (``qk_rope_head_dim == 0``): the latent row
  is bare ckv (512), and the indexer K slab is kpool-compressed
  (``index_kpool=4`` -> tokens/4 stored entries; the spec's ``index_ratio``
  drives both the pool factory and the KV cost model).

MoE routing mirrors glm_moe_dsa (sigmoid ``noaux_tc``, shared expert, routed
scaling 2.5) with 288 routed experts and a clamped SwiGLU
(``swiglu_limit=10``). Everything model-specific beyond ``ModelConfig`` rides in
``glm5_args`` (``Glm5NextArgs``).

Resident-weight quantization (attn/dense/lm_head fp8-at-load, the GLM-5.2
bandwidth trick) is OFF by default: the NVFP4 exports deliberately quantize only
the routed experts (attention / shared expert / lm_head ship bf16 behind the
quantization_config ignore list), and a serving engine must not override the
checkpoint author's precision decision silently -- the checkpoint is served
as-is, like vLLM/sglang. FREETOKEN_GLM5_ATTN_FP8=1 / FREETOKEN_GLM5_MLP_FP8=1
opt into the W8A16 requantization (measured decode 36 -> 45.5 tok/s on the
hybrid reference setup; the win needs CUDA graphs -- launch-bound eager decode
gets slower).
"""

from __future__ import annotations

import os
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    detect_expert_quant,
)

from .args import load_args

# Load-time W8A16 fp8 for resident weights: default OFF, env opt-in (see module
# docstring for the rationale and measured numbers). attn covers the KDA
# in_proj_qkv/o_proj + DSA projections (the precision-sensitive b|f_a|g_a gate
# slice stays bf16, see kda.py); mlp covers dense MLPs, shared expert, lm_head.
_ATTN_FP8 = os.getenv("FREETOKEN_GLM5_ATTN_FP8", "0") != "0"
_MLP_FP8 = os.getenv("FREETOKEN_GLM5_MLP_FP8", "0") != "0"


def _dsa_on(args, dsa_layer_ids) -> bool:
    """DSA serving switch, resolved ONCE into the attention-group spec (the pool
    factory, KV cost model, and backend read the spec, never the env)."""
    return (
        len(dsa_layer_ids) > 0
        and args.index_topk > 0
        and args.index_head_dim > 0
        and os.getenv("FREETOKEN_GLM5_DSA", "1") != "0"
    )


def parse_config(hf_config: Any) -> ModelConfig:
    args = load_args(hf_config)
    text = getattr(hf_config, "text_config", hf_config)

    num_layers = len(args.layer_types)
    # Dev/testing only: cap the layer count so the forward path / KV / offload cache
    # can be exercised without the full ~175 GB of experts. Unset in normal use.
    _cap = os.environ.get("FREETOKEN_GLM5_MAX_LAYERS")
    if _cap:
        num_layers = min(num_layers, int(_cap))

    kda_ids = tuple(i for i in args.kda_layer_ids if i < num_layers)
    dsa_ids = tuple(i for i in args.dsa_layer_ids if i < num_layers)

    # NoPE: the main attention carries no rotary dims (rotary_dim == 0); rope
    # survives only in the indexer geometry (args.rope_theta / interleave).
    rotary_config = RotaryConfig(
        head_dim=args.qk_head_dim,
        rotary_dim=args.qk_rope_head_dim,
        max_position=args.max_position,
        base=args.rope_theta,
        scaling=None,
    )
    latent_dim = args.latent_dim  # 512: bare ckv, no kpe rows

    dsa_on = _dsa_on(args, dsa_ids)
    # Each DSA layer owns its indexer ("full" in indexer_types); count only the
    # layers that actually exist under a dev layer cap.
    num_index_layers = (
        sum(
            1
            for i in dsa_ids
            if i < len(args.indexer_types) and args.indexer_types[i] == "full"
        )
        if dsa_on
        else 0
    )

    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=kda_ids,
        # KDA: one qk-sized and one v-sized head set (H=64, d=128 each). Mapping onto
        # the GDN field names keeps LinearStatePool's conv-width formula exact:
        # 2*K*dk + V*dv = (q|k) + v = 3 * H * d.
        num_key_heads=args.linear_num_heads,
        num_value_heads=args.linear_num_heads,
        key_head_dim=args.linear_head_dim,
        value_head_dim=args.linear_head_dim,
        conv_kernel_dim=args.linear_conv_kernel_dim,
        output_gate="sigmoid",  # KDA o_norm gates with sigmoid (kda.py)
        variant="kda",
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=dsa_ids,
        num_kv_heads=1,  # single shared MLA latent
        head_dim=latent_dim,
        rotary_config=rotary_config,
        mla=True,
        index_head_dim=args.index_head_dim if dsa_on else 0,
        num_index_layers=num_index_layers,
        index_ratio=(args.index_kpool if dsa_on and args.index_kpool_compress else 1),
    )
    groups = tuple(
        sorted(
            (linear_group, full_group),
            key=lambda g: g.layer_ids[0] if g.layer_ids else 1 << 30,
        )
    )

    # The MLP layout is a dense prefix + sparse tail; ModelConfig models exactly that
    # via first_k_dense_replace, so assert the checkpoint matches before collapsing.
    mlp_types = args.mlp_layer_types[:num_layers]
    first_dense = next(
        (i for i, t in enumerate(mlp_types) if t == "sparse"), len(mlp_types)
    )
    assert all(t == "dense" for t in mlp_types[:first_dense]) and all(
        t == "sparse" for t in mlp_types[first_dense:]
    ), f"mlp_layer_types is not a dense-prefix layout: {mlp_types}"

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=args.num_heads,
        num_kv_heads=1,
        head_dim=latent_dim,
        hidden_size=args.hidden_size,
        vocab_size=text.vocab_size,
        intermediate_size=text.intermediate_size,
        # hidden_act stands proxy for the EXPERT activation everywhere the engine
        # gates on it (NVFP4 backend selection, CPU-executor capability): GLM-5.3
        # experts and dense MLPs both run clamped SwiGLU when swiglu_limit is set,
        # even though the HF config still says "silu". Passing "silu" through would
        # let auto-selection repack experts into the marlin/b12x silu-only epilogue.
        hidden_act=(
            "swiglu_clamp" if args.swiglu_limit is not None else text.hidden_act
        ),
        rms_norm_eps=args.norm_eps,
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=rotary_config,
        attention_groups=groups,
        num_experts=(
            getattr(text, "n_routed_experts", None) or getattr(text, "num_experts", 0)
        ),
        num_experts_per_tok=(
            getattr(text, "num_experts_per_tok", None)
            or getattr(text, "num_experts_per_token", 0)
        ),
        moe_intermediate_size=getattr(text, "moe_intermediate_size", 0)
        or text.intermediate_size,
        norm_topk_prob=bool(getattr(text, "norm_topk_prob", True)),
        model_type=getattr(hf_config, "model_type", "glm5_next"),
        architectures=getattr(
            hf_config, "architectures", ["Glm5NextForConditionalGeneration"]
        ),
        moe_enabled=True,
        expert_quant=detect_expert_quant(hf_config),
        first_k_dense_replace=first_dense,
        n_shared_experts=int(getattr(text, "n_shared_experts", 0) or 0),
        routed_scaling_factor=float(getattr(text, "routed_scaling_factor", 1.0)),
        n_group=int(getattr(text, "n_group", 1) or 1),
        topk_group=int(getattr(text, "topk_group", 1) or 1),
        attn_sm_scale=args.qk_head_dim**-0.5,
        has_attn_bias=bool(getattr(text, "attention_bias", False)),
        swiglu_limit=args.swiglu_limit,
        attn_quant="fp8_pertensor" if _ATTN_FP8 else "none",
        dense_quant="fp8_pertensor" if _MLP_FP8 else "none",
        lm_head_quant="fp8_pertensor" if _MLP_FP8 else "none",
        vision_config=None,  # text-only milestone; model.visual.* weights are dropped
        image_token_id=getattr(hf_config, "image_token_id", None),
        glm5_args=args,
    )


__all__ = ["parse_config"]
