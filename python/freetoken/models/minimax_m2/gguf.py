"""MiniMax-M2 from a llama.cpp ``minimax-m2`` GGUF: config from the GGUF metadata, tensors by name; the routed experts load as GGUF expert banks (``freetoken.models.gguf.experts``).

The llama.cpp converter (conversion/minimax.py) only stacks the w1 / w2 / w3 experts: both sides run NeoX rope, so q/k carry no rope permute, and the RMSNorm weights are the HF ones.
"""

from __future__ import annotations

import re
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.models.gguf.reader import GgufTensorInfo, gguf_tensor_index
from freetoken.models.gguf.tensors import float_tensor, fused, matrix

from .config import parse_config

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim

_ARCH = "minimax-m2"
_LAYER_LEAVES = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "block_sparse_moe.gate": "ffn_gate_inp",
    "block_sparse_moe.experts.0.w1": "ffn_gate_exps",
    "block_sparse_moe.experts.0.w3": "ffn_up_exps",
    "block_sparse_moe.experts.0.w2": "ffn_down_exps",
}
_QKV_LEAVES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
_TOP_LEAVES = {"model.embed_tokens": "token_embd", "lm_head": "output"}
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_SIGMOID_GATING = 2


def to_gguf(name: str, tensors: dict[str, int]) -> tuple[str, ...]:
    """Checkpoint module name -> the GGUF tensors (without ``.weight``) that store it; empty for modules the GGUF has no matrix for."""
    if name in _TOP_LEAVES:
        return (_TOP_LEAVES[name],)
    m = _LAYER_RE.match(name)
    if m is None:
        return ()
    blk, leaf = f"blk.{m.group(1)}", m.group(2)
    if leaf in _QKV_LEAVES and f"{blk}.attn_qkv.weight" in tensors:
        return (f"{blk}.attn_qkv",)
    tensor = _LAYER_LEAVES.get(leaf)
    return (f"{blk}.{tensor}",) if tensor is not None else ()


def _check_supported(shim: GgufConfigShim) -> None:
    if any(k.startswith("prism.hadamard.") for k in shim.metadata):
        raise NotImplementedError(f"{shim.model_path}: Hadamard-rotated (prism) GGUF weights are not supported")
    index = gguf_tensor_index(shim.model_path)
    scaled = sorted(n for n in index if n.endswith((".scale", ".input_scale")))
    if scaled:
        raise NotImplementedError(f"{shim.model_path}: GGUF tensors with per-tensor scales are not supported (e.g. {scaled[0]})")
    # a metadata-only GGUF (an FTW dir) has no tensor table to check
    if index and "blk.0.exp_probs_b.bias" not in index:
        raise NotImplementedError(f"{shim.model_path}: converted by an older llama.cpp that names the router bias exp_probs_b, which current llama.cpp refuses too")


def _hf_config(shim: GgufConfigShim) -> SimpleNamespace:
    """The HF config fields ``parse_config`` reads, from the GGUF metadata."""
    md = shim.metadata

    def kv(key: str):
        return md[f"{_ARCH}.{key}"]

    _check_supported(shim)
    gating = int(md.get(f"{_ARCH}.expert_gating_func", 0))
    if gating != _SIGMOID_GATING:
        raise NotImplementedError(f"{shim.model_path}: expert gating function {gating} is not supported; MiniMax-M2 routes with a sigmoid (2)")
    scaling = md.get(f"{_ARCH}.rope.scaling.type", "none")
    if scaling != "none":
        raise NotImplementedError(f"{shim.model_path}: rope scaling {scaling!r} is not supported")
    # a list holds one size per layer
    expert_ff = kv("expert_feed_forward_length")
    if isinstance(expert_ff, list):
        if len(set(expert_ff)) != 1:
            raise NotImplementedError(f"{shim.model_path}: per-layer expert sizes {sorted(set(expert_ff))} are not supported")
        expert_ff = expert_ff[0]
    return SimpleNamespace(
        model_type="minimax_m2",
        architectures=list(shim.architectures),
        hidden_act="silu",
        hidden_size=int(kv("embedding_length")),
        num_hidden_layers=int(kv("block_count")),
        num_attention_heads=int(kv("attention.head_count")),
        num_key_value_heads=int(kv("attention.head_count_kv")),
        head_dim=int(kv("attention.key_length")),
        rotary_dim=int(kv("rope.dimension_count")),
        rope_theta=float(kv("rope.freq_base")),
        max_position_embeddings=int(kv("context_length")),
        rms_norm_eps=float(kv("attention.layer_norm_rms_epsilon")),
        vocab_size=int(shim.vocab_size),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        intermediate_size=int(expert_ff),
        num_local_experts=int(kv("expert_count")),
        num_experts_per_tok=int(kv("expert_used_count")),
        # llama.cpp always builds the per-layer q/k norms of this arch
        use_qk_norm=True,
    )


def parse_gguf_config(shim: GgufConfigShim) -> ModelConfig:
    return replace(parse_config(_hf_config(shim)), expert_quant="gguf")


def iter_gguf_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """The dense state dict from the GGUF; the routed experts load as GGUF expert banks (``freetoken.models.gguf.experts``)."""
    if not include_non_moe:
        return
    from freetoken.utils import cached_load_hf_config

    hf = _hf_config(cached_load_hf_config(model_path))
    index = gguf_tensor_index(model_path)
    hidden = hf.hidden_size
    q_rows, kv_rows = hf.num_attention_heads * hf.head_dim, hf.num_key_value_heads * hf.head_dim

    def t(name: str) -> GgufTensorInfo:
        return index[name if name in index else f"{name}.weight"]

    yield from fused("model.embed_tokens", [t("token_embd")], [matrix(t("token_embd"))], hidden)
    if not hf.tie_word_embeddings:
        yield from fused("lm_head", [t("output")], [matrix(t("output"))], hidden)
    yield "model.norm.weight", float_tensor(t("output_norm"))

    for layer in range(hf.num_hidden_layers):
        blk, pre = f"blk.{layer}", f"model.layers.{layer}"
        yield f"{pre}.input_layernorm.weight", float_tensor(t(f"{blk}.attn_norm"))
        yield f"{pre}.post_attention_layernorm.weight", float_tensor(t(f"{blk}.ffn_norm"))
        sa = f"{pre}.self_attn"
        # convert_hf_to_gguf.py --fuse-qkv stores q, k and v as one attn_qkv matrix
        if f"{blk}.attn_qkv.weight" in index:
            qkv = [t(f"{blk}.attn_qkv")] * 3
            parts = list(matrix(qkv[0]).split([q_rows, kv_rows, kv_rows]))
        else:
            qkv = [t(f"{blk}.attn_{p}") for p in ("q", "k", "v")]
            parts = [matrix(p) for p in qkv]
        yield from fused(f"{sa}.qkv_proj", qkv, parts, hidden)
        yield f"{sa}.q_norm.weight", float_tensor(t(f"{blk}.attn_q_norm"))
        yield f"{sa}.k_norm.weight", float_tensor(t(f"{blk}.attn_k_norm"))
        out = t(f"{blk}.attn_output")
        yield from fused(f"{sa}.o_proj", [out], [matrix(out)], int(out.ne[0]))
        moe = f"{pre}.block_sparse_moe"
        yield f"{moe}.gate.weight", float_tensor(t(f"{blk}.ffn_gate_inp"))
        yield f"{moe}.e_score_correction_bias", float_tensor(t(f"{blk}.exp_probs_b.bias"))


__all__ = ["iter_gguf_weights", "parse_gguf_config", "to_gguf"]
