"""Engine-facing config for GLM-5.2 (``glm_moe_dsa``).

MLA runs latent-KV: the MLA/DSA pool (``kvcache/dsa_pool.py``) stores a single
``kv_lora_rank + qk_rope_head_dim`` latent row per token (the model absorbs ``kv_b``
into Q and onto the output; the attention group carries ``mla=True`` plus the DSA
index dims, and the pool factory / KV cost model / engine gate all key off that
spec), and the all-Triton ``dsa`` backend attends over it (gathered-KV sparse MLA
kernel, identity selection for the dense regimes).
MoE routing knobs mirror the glm4_moe path (sigmoid ``noaux_tc`` router, shared
expert, routed scaling). The DSA indexer geometry rides in ``glm_dsa_args``.

Resident-weight quantization is resolved HERE (from the FREETOKEN_GLM_*_FP8 env
switches, default on) into the standard ``ModelConfig`` fields -- ``attn_quant`` /
``dense_quant`` / ``lm_head_quant`` = ``"fp8_pertensor"`` or ``"none"`` -- and every
consumer (the module constructors and the weight loader) reads those fields, so the
resolved config is the single record of what the served weights actually are.
"""

from __future__ import annotations

import os
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    detect_expert_quant,
)

from .args import load_args

# W8A16 fp8 (per-row scale, quantized at load) for the resident weights; decode is
# weight-bandwidth bound and the freed VRAM densifies the expert cache. =0 restores
# checkpoint-faithful bf16. NB: these resolve into ModelConfig at parse time and change
# what iter_weights yields -- an FTW checkpoint converted under one setting must be
# served under the same one (see weight.py).
_ATTN_FP8 = os.getenv("FREETOKEN_GLM_ATTN_FP8", "1") != "0"
_MLP_FP8 = os.getenv("FREETOKEN_GLM_MLP_FP8", "1") != "0"


def _dsa_on(args, num_layers: int) -> bool:
    """DSA serving switch, resolved ONCE here into the attention-group spec (the pool
    factory, the KV cost model, and the backend all read the spec, never the env)."""
    return (
        bool(args.indexer_types[:num_layers])
        and args.index_topk > 0
        and args.index_head_dim > 0
        and os.getenv("FREETOKEN_GLM_DSA", "1") != "0"
    )


def parse_config(hf_config: Any, model_path: str | None = None) -> ModelConfig:
    args = load_args(hf_config)
    # Latent-KV MLA: the paged pool stores a single "head" = ckv (kv_lora_rank) | kpe
    # (qk_rope_head_dim); the model absorbs kv_b into Q/O and attends with flashinfer MLA.
    latent_dim = args.kv_lora_rank + args.qk_rope_head_dim  # 576

    # Dev/testing only: cap the layer count (e.g. FREETOKEN_GLM_DSA_MAX_LAYERS=5 -> 3 dense
    # + 2 MoE layers) so the forward path / KV / offload cache can be exercised without
    # pinning the full ~407 GB of experts. Unset in normal use.
    num_layers = hf_config.num_hidden_layers
    _cap = os.environ.get("FREETOKEN_GLM_DSA_MAX_LAYERS")
    if _cap:
        num_layers = min(num_layers, int(_cap))

    rotary_config = RotaryConfig(
        head_dim=args.qk_head_dim,
        rotary_dim=args.qk_rope_head_dim,
        max_position=args.max_position,
        base=args.rope_theta,
        scaling=None,  # rope_type "default"; interleaved rope applied inside the model
    )
    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=args.num_heads,
        num_kv_heads=1,  # single shared MLA latent
        head_dim=latent_dim,
        hidden_size=hf_config.hidden_size,
        vocab_size=hf_config.vocab_size,
        intermediate_size=hf_config.intermediate_size,
        hidden_act=hf_config.hidden_act,
        rms_norm_eps=hf_config.rms_norm_eps,
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        rotary_config=rotary_config,
        # MLA stores one latent per token (mla=True: single-slab latent pool, 1x cost
        # model). The DSA index dims ride the same spec: the pool factory sizes the
        # index-key slab and the cost model budgets its bytes/token from these fields,
        # so they can never disagree. FREETOKEN_GLM_DSA=0 zeroes them (dense ablation:
        # plain MLAKVCache, no index slab; the backend serves everything through the
        # Triton kernel's identity-selection path).
        attention_groups=(
            FullAttentionGroupConfig(
                name="full",
                layer_ids=tuple(range(num_layers)),
                num_kv_heads=1,
                head_dim=latent_dim,
                rotary_config=rotary_config,
                mla=True,
                index_head_dim=args.index_head_dim if _dsa_on(args, num_layers) else 0,
                num_index_layers=(
                    sum(1 for t in args.indexer_types[:num_layers] if t == "full")
                    if _dsa_on(args, num_layers)
                    else 0
                ),
            ),
        ),
        num_experts=(
            getattr(hf_config, "n_routed_experts", None)
            or getattr(hf_config, "num_experts", 0)
        ),
        num_experts_per_tok=hf_config.num_experts_per_tok,
        moe_intermediate_size=getattr(hf_config, "moe_intermediate_size", 0)
        or hf_config.intermediate_size,
        norm_topk_prob=bool(getattr(hf_config, "norm_topk_prob", True)),
        model_type=getattr(hf_config, "model_type", "glm_moe_dsa"),
        architectures=getattr(hf_config, "architectures", ["GlmMoeDsaForCausalLM"]),
        moe_enabled=True,
        expert_quant=detect_expert_quant(hf_config),
        first_k_dense_replace=int(getattr(hf_config, "first_k_dense_replace", 0)),
        n_shared_experts=int(getattr(hf_config, "n_shared_experts", 0)),
        routed_scaling_factor=float(getattr(hf_config, "routed_scaling_factor", 1.0)),
        n_group=int(getattr(hf_config, "n_group", 1)),
        topk_group=int(getattr(hf_config, "topk_group", 1)),
        attn_sm_scale=args.qk_head_dim**-0.5,
        has_attn_bias=bool(getattr(hf_config, "attention_bias", False)),
        # Resident-weight quant modes (see the module docstring): recorded here so the
        # resolved config describes the served weights; the constructors and
        # iter_weights read these fields, never the env directly.
        attn_quant="fp8_pertensor" if _ATTN_FP8 else "none",
        dense_quant="fp8_pertensor" if _MLP_FP8 else "none",
        lm_head_quant="fp8_pertensor" if _MLP_FP8 else "none",
        glm_dsa_args=args,
    )


__all__ = ["parse_config"]
