"""GLM-5.3-Flash (``glm5_next``) hyperparameters.

GLM-5.3-Flash is a hybrid-attention MoE with manifold-constrained Hyper-Connections:
  * 45 layers, ``layer_types`` = 34 KDA (linear_attention) + 11 MLA+DSA (deepseek_sparse).
  * MLA is NoPE here (``qk_rope_head_dim == 0``, ``mla_use_nope``); DSA adds a Lightning
    indexer with a k-pool compressor (``index_kpool``).
  * KDA = gated-delta linear attention (``linear_attn_config``): 64 heads x 128 head_dim,
    a short conv (kernel 4) and a lower-bounded forget gate.
  * mHC (``hc_mult`` = 4 residual streams, sinkhorn iters 20) -- identical knobs to DSV4,
    so the DSV4 hyper-connection machinery is reused verbatim.
  * MoE: 288 routed / top-8 / 1 shared, first 3 layers dense; sigmoid noaux_tc router.

Fields live under ``config.text_config`` (multimodal wrapper); we serve text-only. This
payload is stashed on ``ModelConfig.glm_dsa_args`` (DSA/MLA dims, opaque to the engine) plus
``ModelConfig.glm5_kda`` / hyper-connection fields carried on ModelConfig directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Tuple


def _parse_resident_env() -> Tuple[int, ...]:
    """FREETOKEN_GLM5_RESIDENT_LAYERS: comma list and/or a-b ranges, e.g. "17-24" or
    "3,17-20". Empty/unset = no resident layers (pure offload)."""
    raw = os.environ.get("FREETOKEN_GLM5_RESIDENT_LAYERS", "").strip()
    if not raw:
        return ()
    out = []
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return tuple(sorted(set(out)))


def _text(hf_config: Any) -> Any:
    """GLM-5.3 puts the language tower under text_config; tolerate a flat config too."""
    return getattr(hf_config, "text_config", None) or hf_config


@dataclass(frozen=True)
class KdaArgs:
    """Gated-delta (KDA) linear-attention dims for the 34 linear layers."""
    num_heads: int
    head_dim: int
    short_conv_kernel_size: int
    gate_lower_bound: float


@dataclass(frozen=True)
class Glm5NextArgs:
    # ---- basic ----
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    vocab_size: int
    intermediate_size: int          # dense MLP (first_k_dense_replace layers)
    hidden_act: str
    norm_eps: float
    max_position: int
    tie_word_embeddings: bool
    # ---- hyper-connections (mHC) -- same knobs as DSV4 ----
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    # ---- MLA (11 full layers) ----
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    mla_use_nope: bool
    rope_theta: float
    # ---- DSA indexer (Lightning + k-pool compress) ----
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int
    index_kpool_always_select_tail: bool
    index_kpool_compress: bool
    indexer_rope_interleave: bool
    indexer_types: Tuple[str, ...]
    # ---- KDA (34 linear layers) ----
    kda: KdaArgs
    # ---- hybrid layout (explicit layer-id lists from linear_attn_config) ----
    layer_types: Tuple[str, ...]
    kda_layer_ids: Tuple[int, ...]
    full_layer_ids: Tuple[int, ...]
    # ---- MoE ----
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    moe_intermediate_size: int
    first_k_dense_replace: int
    routed_scaling_factor: float
    n_group: int
    topk_group: int
    norm_topk_prob: bool
    scoring_func: str
    topk_method: str
    swiglu_limit: float
    # Hotness-driven VRAM/host split: these MoE layers keep their FULL packed expert
    # banks on the GPU (ResidentNvfp4Experts) and are skipped by the host bank builder.
    resident_layer_ids: Tuple[int, ...] = ()
    # KDA fp8: quantize in_proj_qkv/o_proj of the 34 KDA layers to fp8-e4m3 per-row
    # at load (FREETOKEN_GLM5_KDA_FP8=1). ~4.5GB less VRAM read per decode step.
    kda_fp8: bool = False
    # MTP self-speculative decoding: serve the checkpoint's layer-45 MTP block
    # (FREETOKEN_GLM5_MTP=1). Its MLA/DSA attention gets its own DSA-pool slab slot.
    mtp_enabled: bool = False
    mtp_layer_id: int = 45

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def is_kda_layer(self, layer_id: int) -> bool:
        return layer_id in self.kda_layer_ids

    def is_full_layer(self, layer_id: int) -> bool:
        return layer_id in self.full_layer_ids

    def is_dense_layer(self, layer_id: int) -> bool:
        return layer_id < self.first_k_dense_replace


def load_args(hf_config: Any) -> Glm5NextArgs:
    t = _text(hf_config)
    lac = getattr(t, "linear_attn_config", None) or {}
    if not isinstance(lac, dict):  # transformers may wrap it in an object
        lac = {k: getattr(lac, k) for k in ("num_heads", "head_dim", "short_conv_kernel_size",
               "gate_lower_bound", "kda_layers", "full_attn_layers") if hasattr(lac, k)}
    rope = getattr(t, "rope_parameters", None) or {}
    rope_theta = float(rope.get("rope_theta", getattr(t, "rope_theta", 10000.0)))
    layer_types = tuple(getattr(t, "layer_types", ()) or ())
    kda_ids = tuple(lac.get("kda_layers", [])
                    or [i for i, x in enumerate(layer_types) if x == "linear_attention"])
    full_ids = tuple(lac.get("full_attn_layers", [])
                     or [i for i, x in enumerate(layer_types) if x != "linear_attention"])
    return Glm5NextArgs(
        hidden_size=t.hidden_size,
        num_layers=t.num_hidden_layers,
        num_heads=t.num_attention_heads,
        num_kv_heads=int(getattr(t, "num_key_value_heads", t.num_attention_heads)),
        vocab_size=t.vocab_size,
        intermediate_size=t.intermediate_size,
        hidden_act=getattr(t, "hidden_act", "silu"),
        norm_eps=t.rms_norm_eps,
        max_position=int(getattr(t, "max_position_embeddings", 1048576)),
        tie_word_embeddings=bool(getattr(t, "tie_word_embeddings", False)),
        hc_mult=int(getattr(t, "hc_mult", 4)),
        hc_sinkhorn_iters=int(getattr(t, "hc_sinkhorn_iters", 20)),
        hc_eps=float(getattr(t, "hc_eps", 1e-6)),
        q_lora_rank=int(t.q_lora_rank),
        kv_lora_rank=int(t.kv_lora_rank),
        qk_nope_head_dim=int(getattr(t, "qk_nope_head_dim", 0)),
        qk_rope_head_dim=int(getattr(t, "qk_rope_head_dim", 0)),
        v_head_dim=int(getattr(t, "v_head_dim", 0)),
        mla_use_nope=bool(getattr(t, "mla_use_nope", False)),
        rope_theta=rope_theta,
        index_n_heads=int(getattr(t, "index_n_heads", 0)),
        index_head_dim=int(getattr(t, "index_head_dim", 0)),
        index_topk=int(getattr(t, "index_topk", 0)),
        index_kpool=int(getattr(t, "index_kpool", 1)),
        index_kpool_always_select_tail=bool(getattr(t, "index_kpool_always_select_tail", True)),
        index_kpool_compress=bool(getattr(t, "index_kpool_compress", True)),
        indexer_rope_interleave=bool(getattr(t, "indexer_rope_interleave", True)),
        indexer_types=tuple(getattr(t, "indexer_types", ()) or ()),
        kda=KdaArgs(
            num_heads=int(lac.get("num_heads", t.num_attention_heads)),
            head_dim=int(lac.get("head_dim", 128)),
            short_conv_kernel_size=int(lac.get("short_conv_kernel_size", 4)),
            gate_lower_bound=float(lac.get("gate_lower_bound", -5.0)),
        ),
        layer_types=layer_types,
        kda_layer_ids=kda_ids,
        full_layer_ids=full_ids,
        n_routed_experts=int(getattr(t, "n_routed_experts", 0)),
        num_experts_per_tok=int(t.num_experts_per_tok),
        n_shared_experts=int(getattr(t, "n_shared_experts", 0)),
        moe_intermediate_size=int(getattr(t, "moe_intermediate_size", 0) or t.intermediate_size),
        first_k_dense_replace=int(getattr(t, "first_k_dense_replace", 0)),
        routed_scaling_factor=float(getattr(t, "routed_scaling_factor", 1.0)),
        n_group=int(getattr(t, "n_group", 1)),
        topk_group=int(getattr(t, "topk_group", 1)),
        norm_topk_prob=bool(getattr(t, "norm_topk_prob", True)),
        scoring_func=str(getattr(t, "scoring_func", "sigmoid")),
        topk_method=str(getattr(t, "topk_method", "noaux_tc")),
        swiglu_limit=float(getattr(t, "swiglu_limit", 0.0)),
        resident_layer_ids=_parse_resident_env(),
        kda_fp8=os.environ.get("FREETOKEN_GLM5_KDA_FP8", "0") == "1",
        mtp_enabled=os.environ.get("FREETOKEN_GLM5_MTP", "0") == "1",
    )


__all__ = ["Glm5NextArgs", "KdaArgs", "load_args"]
