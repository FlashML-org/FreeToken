"""Engine-facing config for MiniMax-M3 (``minimax_m3``).

The checkpoint is a multimodal wrapper (``model_type=minimax_m3_vl``): the text tower
lives in ``text_config`` and the weights carry a ``language_model.`` prefix. FreeToken
serves the text tower; the ViT vision stack (``vision_tower.`` / projector) is skipped
at load like the other VL checkpoints' (``VISION_KEY_PREFIXES``) -- multimodal input
is future work, so ``ModelConfig.vision_config`` stays None here.

Attention is GQA with block-sparse selection on the trailing layers: parse_config
declares ONE full-attention group over all layers carrying ``mla=False`` plus the
index dims, which resolves to ``AttnType.BSA`` -- the pool factory builds
``BSAKVCache`` (paged GQA K/V + the index-key slab), the KV cost model budgets the
slab from the same spec, and the ``m3_sparse`` backend serves both the dense leading
layers and the sparse layers. ``FREETOKEN_M3_SPARSE=0`` zeroes the index dims (dense
ablation: plain MHAKVCache, every layer attends its whole history through a generic
FULL backend). ``page_size`` is pinned to the 128-token sparse block by the backend's
``page_sizes`` declaration, so one KV page == one sparse block.

Resident-weight quantization is resolved HERE (from the FREETOKEN_M3_*_MXFP8 env
switches, default on) into the standard ``ModelConfig`` fields -- ``attn_quant`` /
``dense_quant`` = ``"mxfp8"`` or ``"none"`` -- and every consumer (the module
constructors and the weight loader) reads those fields, so the resolved config is the
single record of what the served weights actually are. The routed experts are NVFP4
(same ModelOpt layout as MiniMax-M2 / GLM) and always live in the offload cache;
their swigluoai activation restricts the NVFP4 GEMM backend to the Triton kernels
(``select_nvfp4_backend``). lm_head / embeddings are BF16 in the checkpoint.
"""

from __future__ import annotations

import os
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
)

from .args import load_args

def _text_config(hf_config: Any) -> Any:
    return getattr(hf_config, "text_config", None) or hf_config


# House rule: an env switch that changes WHAT IS SERVED must leave a server-log
# trace. parse_config runs a few times per process (engine + weight loader), so
# each resolved-mode line is logged once per distinct value.
_LOGGED_MODES: set = set()


def _log_mode_once(key: str, message: str) -> None:
    if key in _LOGGED_MODES:
        return
    _LOGGED_MODES.add(key)
    from freetoken.utils import init_logger

    init_logger(__name__).info(message)


def parse_config(hf_config: Any, model_path: str | None = None) -> ModelConfig:
    text = _text_config(hf_config)

    num_layers = text.num_hidden_layers
    # Dev/testing only: cap the layer count (e.g. FREETOKEN_M3_MAX_LAYERS=5 -> 3 dense
    # + 2 MoE layers) so the forward path / KV / offload cache can be exercised without
    # pinning the full ~230 GB of experts. Unset in normal use (GLM-5.2 precedent).
    _cap = os.environ.get("FREETOKEN_M3_MAX_LAYERS")
    if _cap and int(_cap) < num_layers:  # a cap >= the real count changes nothing
        num_layers = int(_cap)
        _log_mode_once(
            f"cap={num_layers}",
            f"FREETOKEN_M3_MAX_LAYERS: serving a TRUNCATED model "
            f"({num_layers}/{text.num_hidden_layers} layers) -- dev/testing only, "
            "outputs are garbage; unset the env for real serving.",
        )

    sparse_enabled = os.getenv("FREETOKEN_M3_SPARSE", "1") != "0"
    if not sparse_enabled:
        _log_mode_once(
            "sparse=0",
            "FREETOKEN_M3_SPARSE=0: serving the DENSE-attention ablation (plain "
            "MHA pool, no indexer; different math and outputs from the shipped "
            "block-sparse model).",
        )
    args = load_args(text, num_layers, sparse_enabled=sparse_enabled)

    # W8A16 MXFP8 (the checkpoint's native dense quantization) for the resident
    # weights; decode is weight-bandwidth bound and the freed VRAM densifies the
    # expert cache. =0 dequantizes to bf16 at load (bring-up / ablation). Read at
    # PARSE time so the resolved config is the single record of the served weights
    # -- an FTW checkpoint converted under one setting must be served under the
    # same one (see weight.py) -- and logged HERE so the FTW serve path (which
    # never runs iter_weights) still leaves a serve-time record of the modes.
    attn_mxfp8 = os.getenv("FREETOKEN_M3_ATTN_MXFP8", "1") != "0"
    mlp_mxfp8 = os.getenv("FREETOKEN_M3_MLP_MXFP8", "1") != "0"
    _log_mode_once(
        f"quant={attn_mxfp8}/{mlp_mxfp8}",
        f"MiniMax-M3 resident quant: attn={'mxfp8' if attn_mxfp8 else 'none'} "
        f"dense={'mxfp8' if mlp_mxfp8 else 'none'} lm_head=none "
        "(FREETOKEN_M3_ATTN_MXFP8/FREETOKEN_M3_MLP_MXFP8; an FTW checkpoint "
        "converted under one setting must be served under the same one).",
    )

    # Leading dense-FFN layers: M3's moe_layer_freq is a contiguous 0-prefix; the
    # offload cache and the generic num_moe_layers arithmetic rely on that shape.
    first_k_dense = num_layers - len(args.moe_layer_ids)
    assert args.moe_layer_ids == tuple(range(first_k_dense, num_layers)), (
        "MiniMax-M3 expects a contiguous dense prefix in moe_layer_freq, got "
        f"{args.moe_layer_ids}"
    )

    rotary_config = RotaryConfig(
        head_dim=args.head_dim,
        rotary_dim=args.rotary_dim,
        max_position=args.max_position,
        base=args.rope_theta,
        scaling=None,  # plain NeoX partial rope, no scaling
    )

    # Routed experts: NVFP4 per the modelopt MIXED_PRECISION quantized_layers map
    # (checked structurally by the bank loader's shape/dtype asserts). The generic
    # detect_expert_quant would report "mixed_precision", which names the checkpoint,
    # not the experts -- so the expert format is pinned here (qwen3_5_moe precedent).
    quant = getattr(hf_config, "quantization_config", None)
    expert_quant = "nvfp4" if quant is not None else "none"

    # The activation is model-constant swigluoai: assert instead of consume.
    # Raw checkpoints declare "swigluoai"; the native transformers TextConfig
    # force-normalizes hidden_act to "silu", which would send the backend
    # selection down a silu-only path.
    declared_act = getattr(text, "hidden_act", "swigluoai")
    assert declared_act in ("swigluoai", "silu"), (
        f"MiniMax-M3 support implements swigluoai only, got hidden_act={declared_act!r}"
    )

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        hidden_size=args.hidden_size,
        vocab_size=text.vocab_size,
        intermediate_size=args.dense_intermediate_size,
        hidden_act="swigluoai",  # pinned; see the declared_act assert above
        rms_norm_eps=text.rms_norm_eps,
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=rotary_config,
        # One group over all layers: GQA K/V everywhere, plus the index-key slab for
        # the sparse layers (mla=False + index dims -> AttnType.BSA -> BSAKVCache).
        # The same spec drives the pool factory and the KV cost model, so they can
        # never disagree. FREETOKEN_M3_SPARSE=0 zeroes the index dims (dense ablation:
        # plain MHAKVCache, generic FULL backends serve every layer).
        attention_groups=(
            FullAttentionGroupConfig(
                name="full",
                layer_ids=tuple(range(num_layers)),
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
                rotary_config=rotary_config,
                mla=False,
                index_head_dim=args.index_dim if args.use_sparse else 0,
                num_index_layers=(
                    sum(1 for lid in args.sparse_layer_ids if lid < num_layers)
                    if args.use_sparse
                    else 0
                ),
            ),
        ),
        num_experts=int(getattr(text, "num_local_experts", 0)),
        num_experts_per_tok=text.num_experts_per_tok,
        # Routed experts use the text tower's intermediate_size (3072); the dense
        # MLP width (12288) rides ModelConfig.intermediate_size above.
        moe_intermediate_size=text.intermediate_size,
        norm_topk_prob=True,  # MiniMax always renormalizes the selected expert weights
        model_type=getattr(hf_config, "model_type", "minimax_m3_vl"),
        architectures=getattr(
            hf_config, "architectures", ["MiniMaxM3SparseForConditionalGeneration"]
        ),
        moe_enabled=True,
        expert_quant=expert_quant,
        first_k_dense_replace=first_k_dense,
        n_shared_experts=int(getattr(text, "n_shared_experts", 0)),
        shared_expert_intermediate_size=args.shared_intermediate_size,
        routed_scaling_factor=float(getattr(text, "routed_scaling_factor", 1.0)),
        has_router_bias=bool(getattr(text, "use_routing_bias", False)),
        use_qk_norm=bool(getattr(text, "use_qk_norm", True)),
        swiglu_limit=args.swiglu_limit,
        hidden_act_alpha=args.swiglu_alpha,
        # Resident-weight quant modes (see the module docstring): recorded here so the
        # resolved config describes the served weights; the constructors and
        # iter_weights read these fields, never the env directly. lm_head stays bf16
        # (the checkpoint excludes it from quantization).
        attn_quant="mxfp8" if attn_mxfp8 else "none",
        dense_quant="mxfp8" if mlp_mxfp8 else "none",
        lm_head_quant="none",
        m3_args=args,
    )


__all__ = ["parse_config"]
