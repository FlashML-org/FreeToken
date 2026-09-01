"""GLM-5.3-Flash (``glm5_next``) hyperparameters.

GLM-5.3-Flash is the first hybrid GLM: 34 of 45 decoder layers run KDA linear
attention (Kimi-Delta-Attention: per-channel gated delta rule with separate q/k/v
short convolutions) and the remaining 11 run DeepSeek-sparse attention -- MLA
latent KV with a Lightning indexer whose K cache is *pool-compressed*
(``index_kpool`` tokens fold into one stored entry). The main attention is NoPE
(``qk_rope_head_dim == 0``, ``mla_use_nope``): no rotary embedding on Q/K at all;
positional information enters only through the indexer's pool-compression APE.
The residual stream is widened by mHC (Manifold-Constrained Hyper-Connections,
``hc_mult`` streams mixed by a Sinkhorn-projected matrix around every sublayer).

This payload carries everything the model module needs beyond the generic
``ModelConfig`` fields; it is stashed on ``ModelConfig.glm5_args`` (opaque to the
engine). Field spellings follow the checkpoint's ``config.json`` (transformers
5.16 ``Glm5NextTextConfig``); ``load_args`` folds the checkpoint aliases
(``hc_mult``, ``hc_sinkhorn_iters``, ``mla_use_nope``, the nested
``linear_attn_config`` dict) the same way vLLM's config class does, so a future
flattened checkpoint keeps loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

# Layer-type strings used by the checkpoint's ``layer_types`` field.
KDA_LAYER = "linear_attention"
DSA_LAYER = "deepseek_sparse_attention"


@dataclass(frozen=True)
class Glm5NextArgs:
    hidden_size: int
    num_heads: int
    # ---- MLA (the 11 "deepseek_sparse_attention" layers) ----
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int  # 0: NoPE -- main attention carries no rotary dims
    v_head_dim: int
    mla_nope: bool
    norm_eps: float
    max_position: int
    # ---- DSA indexer ----
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    indexer_types: Tuple[str, ...]
    indexer_rope_interleave: bool
    # kpool compression: every ``index_kpool`` indexer-K entries pool into one
    # stored entry (softmax(gate + APE)-weighted sum); top-k selects pools
    # (select_k = index_topk // index_kpool) and ``always_select_tail``
    # force-includes the in-progress tail pool.
    index_kpool: int
    index_kpool_compress: bool
    index_kpool_always_select_tail: bool
    # ---- KDA linear attention (the 34 "linear_attention" layers) ----
    linear_num_heads: int
    linear_head_dim: int
    linear_conv_kernel_dim: int
    linear_lower_bound: float
    # ---- per-layer layout ----
    layer_types: Tuple[str, ...]
    mlp_layer_types: Tuple[str, ...]
    # ---- mHC (Manifold-Constrained Hyper-Connections) ----
    mhc: bool
    mhc_num_residual_streams: int
    hc_eps: float
    mhc_sinkhorn_iterations: int
    mhc_tau: float
    mhc_post_mult_value: float
    mhc_no_norm_weight: bool
    # ---- misc ----
    swiglu_limit: float | None
    rope_theta: float  # indexer-side only; the main attention is NoPE

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def latent_dim(self) -> int:
        """Width of one MLA latent row in the paged pool: ckv | kpe."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def kda_layer_ids(self) -> Tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == KDA_LAYER)

    @property
    def dsa_layer_ids(self) -> Tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == DSA_LAYER)

    def is_kda_layer(self, layer_id: int) -> bool:
        return self.layer_types[layer_id] == KDA_LAYER


def _require_mhc(mhc: bool) -> bool:
    """The decoder layer wires the hyper-connection tensors unconditionally
    (model.py touches hc_attn_fn on every forward), so an mhc=False checkpoint
    would die with an AttributeError mid-forward -- refuse it at parse until a
    real one exists to implement against."""
    if not mhc:
        raise NotImplementedError(
            "glm5_next requires mhc=True (manifold-constrained hyper-connections); "
            "an mhc=False checkpoint has no implementation yet."
        )
    return mhc


def _get(cfg: Any, name: str, default: Any = None) -> Any:
    return getattr(cfg, name, default)


def load_args(hf_config: Any) -> Glm5NextArgs:
    """Build ``Glm5NextArgs`` from the checkpoint config (nested ``text_config`` or a
    flat text-only config). Raw checkpoint spellings win; the vLLM-normalized names
    are accepted as fallbacks."""
    text = _get(hf_config, "text_config", hf_config)

    num_layers = int(text.num_hidden_layers)
    layer_types = tuple(_get(text, "layer_types", ()) or ())
    if not layer_types:
        raise ValueError("glm5_next config is missing layer_types")
    if len(layer_types) != num_layers:
        raise ValueError(
            f"layer_types has {len(layer_types)} entries for {num_layers} layers"
        )
    unknown = sorted(set(layer_types) - {KDA_LAYER, DSA_LAYER})
    if unknown:
        raise ValueError(f"unsupported layer_types entries: {unknown}")

    mlp_layer_types = tuple(_get(text, "mlp_layer_types", ()) or ())
    if not mlp_layer_types:
        # Older-schema fallback (mirrors vLLM): derive from first_k_dense_replace.
        first_dense = int(_get(text, "first_k_dense_replace", 0) or 0)
        mlp_layer_types = ("dense",) * first_dense + ("sparse",) * (
            num_layers - first_dense
        )

    # Checkpoint alias folding (transformers 5.16 spellings first).
    mla_nope = _get(text, "mla_use_nope", _get(text, "mla_nope", False))
    hc_mult = _get(text, "hc_mult", _get(text, "mhc_num_residual_streams", 4))
    hc_sinkhorn = _get(
        text, "hc_sinkhorn_iters", _get(text, "mhc_sinkhorn_iterations", 20)
    )

    # KDA head geometry ships as the nested ``linear_attn_config`` dict; a future
    # flattened schema would carry vLLM-style ``linear_*`` top-level fields.
    linear_cfg = _get(text, "linear_attn_config", None)
    if linear_cfg is not None and not isinstance(linear_cfg, dict):
        linear_cfg = {
            k: getattr(linear_cfg, k)
            for k in (
                "num_heads",
                "head_dim",
                "short_conv_kernel_size",
                "gate_lower_bound",
            )
            if hasattr(linear_cfg, k)
        }
    linear_cfg = linear_cfg or {}
    linear_num_heads = int(
        linear_cfg.get("num_heads", _get(text, "linear_num_heads", 0))
    )
    linear_head_dim = int(linear_cfg.get("head_dim", _get(text, "linear_head_dim", 0)))
    linear_conv = int(
        linear_cfg.get(
            "short_conv_kernel_size", _get(text, "linear_conv_kernel_dim", 4)
        )
    )
    linear_lower_bound = float(
        linear_cfg.get("gate_lower_bound", _get(text, "linear_lower_bound", -5.0))
    )
    if KDA_LAYER in layer_types and (linear_num_heads <= 0 or linear_head_dim <= 0):
        raise ValueError("glm5_next config is missing the KDA linear_attn_config dims")

    # The main attention is NoPE; rope exists only on the indexer side. The
    # checkpoint ships no rope_theta -- fall back to the transformers default.
    rope = _get(text, "rope_parameters", None) or {}
    rope_theta = float(rope.get("rope_theta", _get(text, "rope_theta", 10000.0)))

    swiglu_limit = _get(text, "swiglu_limit", None)

    return Glm5NextArgs(
        hidden_size=int(text.hidden_size),
        num_heads=int(text.num_attention_heads),
        q_lora_rank=int(text.q_lora_rank),
        kv_lora_rank=int(text.kv_lora_rank),
        qk_nope_head_dim=int(text.qk_nope_head_dim),
        qk_rope_head_dim=int(text.qk_rope_head_dim),
        v_head_dim=int(text.v_head_dim),
        mla_nope=bool(mla_nope),
        norm_eps=float(text.rms_norm_eps),
        max_position=int(text.max_position_embeddings),
        index_n_heads=int(_get(text, "index_n_heads", 0) or 0),
        index_head_dim=int(_get(text, "index_head_dim", 0) or 0),
        index_topk=int(_get(text, "index_topk", 0) or 0),
        indexer_types=tuple(_get(text, "indexer_types", ()) or ()),
        indexer_rope_interleave=bool(_get(text, "indexer_rope_interleave", False)),
        index_kpool=int(_get(text, "index_kpool", 1) or 1),
        index_kpool_compress=bool(_get(text, "index_kpool_compress", False)),
        index_kpool_always_select_tail=bool(
            _get(text, "index_kpool_always_select_tail", False)
        ),
        linear_num_heads=linear_num_heads,
        linear_head_dim=linear_head_dim,
        linear_conv_kernel_dim=linear_conv,
        linear_lower_bound=linear_lower_bound,
        layer_types=layer_types,
        mlp_layer_types=mlp_layer_types,
        mhc=_require_mhc(bool(_get(text, "mhc", False))),
        mhc_num_residual_streams=int(hc_mult),
        hc_eps=float(_get(text, "hc_eps", 1e-6)),
        mhc_sinkhorn_iterations=int(hc_sinkhorn),
        mhc_tau=float(_get(text, "mhc_tau", 0.05)),
        mhc_post_mult_value=float(_get(text, "mhc_post_mult_value", 2.0)),
        mhc_no_norm_weight=bool(_get(text, "mhc_no_norm_weight", False)),
        swiglu_limit=(None if swiglu_limit is None else float(swiglu_limit)),
        rope_theta=rope_theta,
    )


__all__ = ["Glm5NextArgs", "load_args", "KDA_LAYER", "DSA_LAYER"]
