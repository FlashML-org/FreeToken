"""Engine-facing config for GLM-5.3-Flash (``glm5_next``).

Hybrid attention resolved into two groups:
  * full layers (``full_attn_layers``) -> MLA + DSA. We derive a ``GlmMoeDsaArgs`` and put it
    on ``ModelConfig.glm_dsa_args`` so the glm_moe_dsa pool / KV cost model / MLA-DSA attention
    serve them UNCHANGED. GLM-5.3 MLA is NoPE (``qk_rope_head_dim == 0``): the latent row is
    just ``kv_lora_rank`` wide and the rope is a no-op.
  * linear layers (``kda_layers``) -> KDA gated-delta (``LinearGatedDeltaGroupConfig``, the
    same linear_state pool the qwen3.5 GDN path uses; the KDA-specific gating is applied in
    the model module, not the pool).

The mHC (hyper-connection) + KDA + VRAM/host split payload rides on the new
``ModelConfig.glm5_args`` (Glm5NextArgs).

Precision (quality-first): the first milestone keeps every resident (non-expert) weight in
BF16 for a clean quality baseline. FREETOKEN_GLM5_ATTN_FP8 / _MLP_FP8 turn on the non-expert
FP8 optimization later; lm_head and the MLA-latent projections deliberately stay BF16 per the
sensitivity analysis, so they have their own switch (default off).
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
from freetoken.models.glm_moe_dsa.args import GlmMoeDsaArgs

from .args import Glm5NextArgs, load_args

# Non-expert FP8: default OFF (bf16 quality baseline). Turn on after the bf16 run is the A/B
# reference. lm_head stays bf16 even when MLP_FP8 is on (logits / rare-word sensitivity).
_ATTN_FP8 = os.getenv("FREETOKEN_GLM5_ATTN_FP8", "0") != "0"
_VISION = os.getenv("FREETOKEN_GLM5_VISION", "0") != "0"
_MLP_FP8 = os.getenv("FREETOKEN_GLM5_MLP_FP8", "0") != "0"
_LMHEAD_FP8 = os.getenv("FREETOKEN_GLM5_LMHEAD_FP8", "0") != "0"


def _derive_dsa_args(a: Glm5NextArgs) -> GlmMoeDsaArgs:
    """GlmMoeDsaArgs for the full MLA+DSA layers -- reuses the glm_moe_dsa machinery verbatim."""
    return GlmMoeDsaArgs(
        hidden_size=a.hidden_size,
        num_heads=a.num_heads,
        q_lora_rank=a.q_lora_rank,
        kv_lora_rank=a.kv_lora_rank,
        qk_nope_head_dim=a.qk_nope_head_dim,
        qk_rope_head_dim=a.qk_rope_head_dim,
        v_head_dim=a.v_head_dim,
        norm_eps=a.norm_eps,
        rope_theta=a.rope_theta,
        rope_interleave=True,
        indexer_rope_interleave=a.indexer_rope_interleave,
        max_position=a.max_position,
        index_n_heads=a.index_n_heads,
        index_head_dim=a.index_head_dim,
        index_topk=a.index_topk,
        indexer_types=a.indexer_types,
        index_kpool=a.index_kpool,
        index_kpool_always_select_tail=a.index_kpool_always_select_tail,
    )



def _vision_cfg_dict(hf_config) -> dict | None:
    """Raw vision_config as a plain dict (the tower port indexes by key). Env-gated:
    default OFF keeps the text-only boot byte-identical."""
    if not _VISION:
        return None
    vc = getattr(hf_config, "vision_config", None)
    if vc is None or isinstance(vc, dict):
        return vc
    if hasattr(vc, "to_dict"):
        return vc.to_dict()
    return dict(vars(vc))

def parse_config(hf_config: Any) -> ModelConfig:
    a = load_args(hf_config)

    # Dev cap (FREETOKEN_GLM5_MAX_LAYERS=5 -> 3 dense + 2 MoE) to exercise the forward path /
    # KV / offload without pinning the full 163 GiB of experts. Unset in normal use.
    num_layers = a.num_layers
    cap = os.environ.get("FREETOKEN_GLM5_MAX_LAYERS")
    if cap:
        num_layers = min(num_layers, int(cap))

    dsa_args = _derive_dsa_args(a)
    if a.mtp_enabled:
        import dataclasses as _dc

        dsa_args = _dc.replace(dsa_args, indexer_types=a.indexer_types + ("full",))
    latent_dim = a.kv_lora_rank + a.qk_rope_head_dim  # 512 (NoPE: qk_rope_head_dim == 0)
    dsa_on = (
        a.index_topk > 0 and a.index_head_dim > 0 and os.getenv("FREETOKEN_GLM5_DSA", "1") != "0"
    )

    full_ids = tuple(i for i in a.full_layer_ids if i < num_layers)
    kda_ids = tuple(i for i in a.kda_layer_ids if i < num_layers)
    # MTP layer 45: one more MLA/DSA layer sharing the paged latent/index slabs (its own
    # dense slab row via the layer_ids remap). Only when serving the full stack.
    mtp_on = a.mtp_enabled and num_layers == a.num_layers
    if mtp_on:
        full_ids = full_ids + (a.mtp_layer_id,)

    full_rotary = RotaryConfig(
        head_dim=a.qk_head_dim,
        rotary_dim=a.qk_rope_head_dim,
        max_position=a.max_position,
        base=a.rope_theta,
        scaling=None,
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=1,  # single shared MLA latent
        head_dim=latent_dim,
        rotary_config=full_rotary,
        mla=True,
        index_head_dim=a.index_head_dim if dsa_on else 0,
        num_index_layers=(
            sum(1 for i in full_ids if dsa_args.indexer_types[i] == "full")
            if dsa_on
            else 0
        ),
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=kda_ids,
        num_key_heads=a.kda.num_heads,
        num_value_heads=a.kda.num_heads,
        key_head_dim=a.kda.head_dim,
        value_head_dim=a.kda.head_dim,
        conv_kernel_dim=a.kda.short_conv_kernel_size,
        output_gate=True,
    )
    groups = tuple(
        sorted(
            (full_group, linear_group),
            key=lambda g: g.layer_ids[0] if g.layer_ids else 1 << 30,
        )
    )

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=a.num_heads,
        num_kv_heads=1,
        head_dim=latent_dim,
        hidden_size=a.hidden_size,
        vocab_size=a.vocab_size,
        intermediate_size=a.intermediate_size,
        rms_norm_eps=a.norm_eps,
        rotary_config=full_rotary,
        hidden_act=a.hidden_act,
        tie_word_embeddings=a.tie_word_embeddings,
        num_experts=a.n_routed_experts,
        num_experts_per_tok=a.num_experts_per_tok,
        moe_intermediate_size=a.moe_intermediate_size,
        norm_topk_prob=a.norm_topk_prob,
        model_type=getattr(hf_config, "model_type", "glm5_next"),
        architectures=getattr(hf_config, "architectures", ["Glm5NextForConditionalGeneration"]),
        moe_enabled=True,
        expert_quant=detect_expert_quant(hf_config) or "nvfp4",
        first_k_dense_replace=a.first_k_dense_replace,
        n_shared_experts=a.n_shared_experts,
        routed_scaling_factor=a.routed_scaling_factor,
        n_group=a.n_group,
        topk_group=a.topk_group,
        attn_sm_scale=a.qk_head_dim**-0.5,
        swiglu_limit=(a.swiglu_limit or None),
        has_attn_bias=False,
        vision_config=_vision_cfg_dict(hf_config),
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=groups,
        attn_quant="fp8_pertensor" if _ATTN_FP8 else "none",
        dense_quant="fp8_pertensor" if _MLP_FP8 else "none",
        lm_head_quant="fp8_pertensor" if _LMHEAD_FP8 else "none",
        glm_dsa_args=dsa_args,
        glm5_args=a,
    )


__all__ = ["parse_config"]
