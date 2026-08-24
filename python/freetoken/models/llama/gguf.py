"""Llama GGUF adapter: build the FreeToken ``ModelConfig`` from GGUF metadata and load
the native block-quantized weights.

Mirrors ``gemma4/gguf.py`` but for the standard *dense* llama architecture -- there is
no MoE (routed experts / offload cache), no Per-Layer Embeddings, and no MatFormer, so
this is a much smaller adapter. ``parse_gguf_config`` maps ``llama.<key>`` KV metadata
to the same ``ModelConfig`` ``llama.config.parse_config`` would build from a HF config;
``iter_gguf_weights`` maps GGUF tensor names to the FreeToken llama weight names keeping
the projections GGUF-block-quantized; ``convert_llama_to_gguf`` swaps the dense
``nn.Linear``/embedding for the native-quant ``GGUFLinear``/``GGUFEmbedding`` ops.

Two things differ from the gemma4 template and are llama-GGUF specific (both flagged in
code): the checkpoint is q4_K_M, so (1) the projections carry *mixed* per-tensor K-quant
types (Q4_K, with Q6_K on some attn_v/ffn_down), read per-tensor rather than hardcoded,
and (2) because q/k/v within a layer can have different types, they are kept as separate
packed ``GGUFLinear`` sub-projections whose outputs are concatenated (weight-level
fusion, as gemma4 does for its uniform Q4_0 GGUF, would need one common packed type).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.layers import BaseOP
from freetoken.models.config import ModelConfig, RotaryConfig
from freetoken.models.gguf.dequant import dequantize

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


# --------------------------------------------------------------------------------------
# GGUF tensor name  <->  FreeToken llama module base name.
# --------------------------------------------------------------------------------------
# q/k/v and gate/up are kept as separate packed sub-projections (see module docstring),
# so each maps to its own module base under the fused ``qkv_proj`` / ``gate_up_proj``
# container; their outputs are concatenated at forward time.
_LAYER_QUANT_MAP = {
    "attn_q.weight": "self_attn.qkv_proj.q_proj",
    "attn_k.weight": "self_attn.qkv_proj.k_proj",
    "attn_v.weight": "self_attn.qkv_proj.v_proj",
    "attn_output.weight": "self_attn.o_proj",
    "ffn_gate.weight": "mlp.gate_up_proj.gate_proj",
    "ffn_up.weight": "mlp.gate_up_proj.up_proj",
    "ffn_down.weight": "mlp.down_proj",
}
# Per-layer 1:1 norm tensors (gguf suffix -> module-relative name). Tiny F32 -> bf16.
_LAYER_NORM_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
}


def _quant_base_for(name: str) -> str | None:
    """FreeToken module base of a *packed* GGUF weight, or ``None`` (norms / skipped).

    Shared by ``parse_gguf_config`` (records each base's ggml type) and
    ``iter_gguf_weights`` (yields ``{base}.qweight``) so the two never disagree.
    """
    if name == "token_embd.weight":
        return "model.embed_tokens"
    if name == "output.weight":  # untied LM head (absent when embeddings are tied)
        return "lm_head"
    if not name.startswith("blk."):
        return None
    parts = name.split(".", 2)
    layer = parts[1]
    suffix = parts[2]
    rel = _LAYER_QUANT_MAP.get(suffix)
    return f"model.layers.{layer}.{rel}" if rel is not None else None


# --------------------------------------------------------------------------------------
# Config.
# --------------------------------------------------------------------------------------


def _rope_scaling(shim: "GgufConfigShim", head_dim: int) -> dict | None:
    """Recover llama3 long-context rope scaling from the baked ``rope_freqs.weight``.

    llama.cpp does not write llama3 rope params as KV metadata; like it does for gemma4's
    partial rope, it bakes the per-frequency smoothing into a ``rope_freqs.weight`` divisor
    tensor (ggml applies ``theta / rope_freqs[j]``). The tensor's max entry is exactly the
    scaling ``factor`` (the low-frequency dims are set to ``factor``); the remaining llama3
    params are Llama-3.x's fixed constants (low_freq 1, high_freq 4, original context 8192),
    read from KV only if a converter happened to emit them. FreeToken's ``"llama3"`` rope
    branch reconstructs the identical smoothed frequencies from these (verified to ~1e-10).

    Returns ``None`` (plain rope) when no ``rope_freqs.weight`` is present -- a Llama-2-style
    checkpoint, an un-scaled model, or a metadata-only GGUF (no tensor table).
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors

    m = shim.metadata
    for t in iter_gguf_tensors(shim.model_path):
        if t.name != "rope_freqs.weight":
            continue
        # F32 tensor: the packed bytes ARE the values (same trick as gemma4._full_rotary_dim).
        freqs = t.packed().reshape(-1).view(torch.float32)
        factor = float(freqs.max().item())
        if factor <= 1.0 + 1e-6:
            return None  # no long-context scaling baked in
        return {
            "rope_type": "llama3",
            "factor": factor,
            "low_freq_factor": float(m.get("llama.rope.scaling.low_freq_factor", 1.0)),
            "high_freq_factor": float(m.get("llama.rope.scaling.high_freq_factor", 4.0)),
            "original_max_position_embeddings": int(
                m.get("llama.rope.scaling.original_context_length", 8192)
            ),
        }
    return None


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    m = shim.metadata

    def g(key: str):
        val = m.get(f"llama.{key}")
        if val is None:
            raise KeyError(f"missing GGUF metadata key llama.{key}")
        return val

    def g_opt(key, default=None):
        val = m.get(f"llama.{key}")
        return default if val is None else val

    num_layers = int(g("block_count"))
    hidden = int(g("embedding_length"))
    num_qo_heads = int(g("attention.head_count"))
    num_kv_heads = int(g_opt("attention.head_count_kv", num_qo_heads))
    # llama is uniform GQA (unlike gemma4's per-layer kv array); a single scalar.
    if isinstance(num_kv_heads, (list, tuple)):
        num_kv_heads = int(num_kv_heads[0])
    head_dim = int(g_opt("attention.key_length", 0) or (hidden // num_qo_heads))
    rotary_dim = int(g_opt("rope.dimension_count", 0) or head_dim)
    # feed_forward_length is a scalar for llama; guard the (unused) list form defensively.
    ffn = g("feed_forward_length")
    intermediate = int(ffn[0] if isinstance(ffn, (list, tuple)) else ffn)

    max_pos = int(g("context_length"))
    rope_theta = float(g_opt("rope.freq_base", 10_000.0))
    rope_scaling = _rope_scaling(shim, head_dim)
    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=max_pos,
        base=rope_theta,
        scaling=rope_scaling,
    )

    # Per-tensor packed quant types (q4_K_M mixes Q4_K / Q6_K), keyed by module base so the
    # module-swap can size each GGUFLinear/GGUFEmbedding buffer to its real type. Empty for a
    # metadata-only GGUF (no tensor table) -> convert_llama_to_gguf raises with a clear note.
    from freetoken.models.gguf.reader import iter_gguf_tensors

    quant_types: dict[str, int] = {}
    for t in iter_gguf_tensors(shim.model_path):
        base = _quant_base_for(t.name)
        if base is not None:
            quant_types[base] = int(t.ggml_type)

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden,
        vocab_size=int(shim.vocab_size),
        intermediate_size=intermediate,
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        rotary_config=rotary,
        hidden_act="silu",
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        model_type="llama",
        architectures=list(shim.architectures),
        # GGUF sentinel: is_gguf_model() -> True (matches gemma4's use of this field).
        moe_weight_format="q4_0",
        gguf_quant_types=quant_types or None,
    )


# --------------------------------------------------------------------------------------
# Weight loading: GGUF tensor names -> FreeToken llama module params.
# --------------------------------------------------------------------------------------


def _to_bf16(t) -> torch.Tensor:
    """Dequantize a GgufTensor (F32/F16) to a dense bf16 tensor of its torch shape."""
    flat = dequantize(t.packed().reshape(-1), t.ggml_type, torch.bfloat16)
    return flat.reshape(t.shape)


def _unpermute_qk_rows(packed: torch.Tensor, n_head: int) -> torch.Tensor:
    """Invert llama.cpp's Q/K output-row permutation on the packed (still-quantized) bytes.

    convert_hf_to_gguf permutes ``attn_q``/``attn_k`` rows ``(n_head, 2, half) ->
    (n_head, half, 2)`` so ggml's adjacent-pair rope reproduces HF's split-half rope.
    FreeToken applies NeoX (split-half) rope, so it needs the original HF row order --
    otherwise q/k are rotated on the wrong dim pairs and attention is scrambled (coherent
    tokens leak through but generation degenerates). K-quant packs each output row as an
    independent contiguous byte segment (``packed`` is ``(out_features, row_bytes)``), so we
    restore the HF order by reordering whole rows -- the per-row quant blocks are untouched.
    ``n_head`` is the query head count for ``attn_q`` and the KV head count for ``attn_k``
    (llama.cpp permutes K by ``n_head_kv``).
    """
    out_features, row_bytes = packed.shape
    half = out_features // n_head // 2
    return (
        packed.reshape(n_head, half, 2, row_bytes)
        .transpose(1, 2)
        .contiguous()
        .reshape(out_features, row_bytes)
    )


def _require_tp1(what: str) -> None:
    """GGUF quant layers are not tensor-parallel sharded; reject TP>1 with a clear error
    (mirrors the gemma4 GGUF loader's TP=1 restriction)."""
    from freetoken.distributed import get_tp_info

    if get_tp_info().size > 1:
        raise NotImplementedError(
            f"llama GGUF {what} currently supports TP=1 only "
            "(GGUF quant layers are not tensor-parallel sharded)."
        )


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (param_name, tensor) for every llama param.

    Quantized projections (attention q/k/v/o, MLP gate/up/down) and the token embedding
    stay in their native packed block layout and are yielded as ``.qweight`` (uint8);
    the RMSNorms dequantize to bf16. Llama is dense, so there are no experts to skip and
    ``include_moe_experts`` is irrelevant. Unlike gemma4, q/k/v and gate/up are NOT fused
    at the weight level (their K-quant types can differ) -- each projection is emitted
    standalone into its ``qkv_proj`` / ``gate_up_proj`` container sub-module.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors, load_gguf_metadata

    if not include_non_moe:
        return
    _require_tp1("weight loading")

    # Q/K row counts for the llama.cpp rope permutation (see _unpermute_qk_rows).
    meta = load_gguf_metadata(model_path)
    n_qo_heads = int(meta["llama.attention.head_count"])
    n_kv_heads = int(meta.get("llama.attention.head_count_kv", n_qo_heads))

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "token_embd.weight":
            yield "model.embed_tokens.qweight", t.packed()
            continue
        if name == "output_norm.weight":
            yield "model.norm.weight", _to_bf16(t)
            continue
        if name == "output.weight":
            # Untied LM head. convert_llama_to_gguf rejects the untied case up front, so
            # this is unreachable for a tied checkpoint (Llama-3.2-1B/3B); kept for parity.
            yield "lm_head.qweight", t.packed()
            continue
        if name == "rope_freqs.weight":
            continue  # rope frequencies recomputed in-engine (see _rope_scaling)
        if not name.startswith("blk."):
            raise ValueError(f"unmapped llama GGUF tensor: {name}")

        parts = name.split(".", 2)
        layer, suffix = parts[1], parts[2]
        base = f"model.layers.{layer}"
        if suffix in _LAYER_NORM_MAP:
            yield f"{base}.{_LAYER_NORM_MAP[suffix]}", _to_bf16(t)
            continue
        quant_base = _quant_base_for(name)
        if quant_base is None:
            raise ValueError(f"unmapped llama GGUF tensor: {name}")
        packed = t.packed()
        if suffix == "attn_q.weight":
            packed = _unpermute_qk_rows(packed, n_qo_heads)
        elif suffix == "attn_k.weight":
            packed = _unpermute_qk_rows(packed, n_kv_heads)
        yield f"{quant_base}.qweight", packed


# --------------------------------------------------------------------------------------
# Model layer swap: dense bf16 Linear/Embedding -> native GGUF-quant ops.
# --------------------------------------------------------------------------------------


def is_gguf_model(config: ModelConfig) -> bool:
    """True when the model was parsed from a GGUF checkpoint (native-quant path)."""
    return getattr(config, "moe_weight_format", None) == "q4_0"


class GGUFFusedProj(BaseOP):
    """A column-parallel fused projection whose packed sub-projections may carry different
    GGUF quant types.

    A q4_K_M checkpoint bumps some projections to higher precision (Q6_K attn_v), so q/k/v
    within a layer need not share a quant type. Weight-level fusion (as gemma4 does for its
    uniform Q4_0 GGUF, concatenating packed rows) requires one common ``row_bytes``, so
    instead we hold one ``GGUFLinear`` per part and concatenate their OUTPUTS -- exactly
    what a fused ``LinearQKVMerged`` / ``LinearColParallelMerged`` forward produces, at the
    cost of one extra small GEMV per part.
    """

    def __init__(self, parts: list[tuple[str, BaseOP]]):
        for attr, lin in parts:
            setattr(self, attr, lin)
        self._part_attrs = [attr for attr, _ in parts]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([getattr(self, a).forward(x) for a in self._part_attrs], dim=-1)


class GGUFTiedLMHead:
    """Tied LM head over a native GGUF embedding table (logits via ggml matmul).

    Holds only a reference to the GGUF embedding (no params of its own -> empty
    state_dict), mirroring ``ParallelLMHead`` with ``tie_word_embeddings``. TP=1 only.
    Same shape as gemma4's, minus the softcap (llama has none).
    """

    def __init__(self, embedding, quant_type: int):
        self._embedding = embedding
        self._quant_type = quant_type

    def state_dict(self, *, prefix: str = "", result=None):
        return result if result is not None else {}

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False):
        state_dict.pop(f"{prefix}.weight", None)
        state_dict.pop(f"{prefix}.bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx
        from freetoken.layers.gguf import fused_mul_mat_gguf

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        return fused_mul_mat_gguf(x, self._embedding.qweight, self._quant_type)


def convert_llama_to_gguf(model, config: ModelConfig) -> None:
    """In place: replace llama's dense projections + embedding with native GGUF ops.

    Every attention (q/k/v/o) and MLP (gate/up/down) projection becomes a ``GGUFLinear``
    sized to that tensor's real K-quant type (read from ``config.gguf_quant_types``); the
    token embedding becomes a ``GGUFEmbedding`` (and, tied, the LM head). RMSNorms stay
    dense bf16 (F32 in the GGUF). q/k/v and gate/up live in ``GGUFFusedProj`` containers
    that concatenate outputs, so mixed per-projection quant types are fine.
    """
    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear

    types = config.gguf_quant_types
    if not types:
        raise NotImplementedError(
            "llama GGUF conversion needs per-tensor quant types "
            "(config.gguf_quant_types); a metadata-only / FTW-converted GGUF that strips "
            "the tensor table is not supported."
        )

    def qtype(base: str) -> int:
        try:
            return types[base]
        except KeyError as exc:
            raise KeyError(f"no GGUF quant type recorded for {base}") from exc

    def lin(in_features: int, out_features: int, base: str) -> GGUFLinear:
        return GGUFLinear(in_features, out_features, qtype(base), has_bias=False)

    hidden = config.hidden_size
    qo_dim = config.num_qo_heads * config.head_dim
    kv_dim = config.num_kv_heads * config.head_dim
    inter = config.intermediate_size

    inner = model.model
    embed_type = qtype("model.embed_tokens")
    embed = GGUFEmbedding(
        num_embeddings=config.vocab_size,
        embedding_dim=hidden,
        quant_type=embed_type,
        embed_scale=config.embedding_scale,  # None for llama
    )
    inner.embed_tokens = embed

    for layer_id, layer in enumerate(inner.layers.op_list):
        base = f"model.layers.{layer_id}"
        attn = layer.self_attn
        qkv = f"{base}.self_attn.qkv_proj"
        attn.qkv_proj = GGUFFusedProj(
            [
                ("q_proj", lin(hidden, qo_dim, f"{qkv}.q_proj")),
                ("k_proj", lin(hidden, kv_dim, f"{qkv}.k_proj")),
                ("v_proj", lin(hidden, kv_dim, f"{qkv}.v_proj")),
            ]
        )
        attn.o_proj = lin(qo_dim, hidden, f"{base}.self_attn.o_proj")

        mlp = layer.mlp
        gu = f"{base}.mlp.gate_up_proj"
        mlp.gate_up_proj = GGUFFusedProj(
            [
                ("gate_proj", lin(hidden, inter, f"{gu}.gate_proj")),
                ("up_proj", lin(hidden, inter, f"{gu}.up_proj")),
            ]
        )
        mlp.down_proj = lin(inter, hidden, f"{base}.mlp.down_proj")

    if config.tie_word_embeddings:
        model.lm_head = GGUFTiedLMHead(embed, embed_type)
    else:
        # Untied head would need a GGUFLinear-backed lm_head that also does the prefill
        # last-token gather ParallelLMHead does; Llama-3.2-1B/3B tie embeddings, so this is
        # only reachable for 3.1-8B-class GGUFs.
        raise NotImplementedError(
            "untied llama GGUF lm_head is not yet supported (the target Llama-3.2 GGUFs "
            "tie word embeddings)."
        )


__all__ = [
    "parse_gguf_config",
    "iter_gguf_weights",
    "convert_llama_to_gguf",
    "is_gguf_model",
]
