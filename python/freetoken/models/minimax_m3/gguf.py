"""MiniMax-M3 text from a llama.cpp ``minimax-m3`` GGUF: config from the GGUF metadata, tensors by name; the routed experts load as GGUF expert banks (``freetoken.models.gguf.experts``).

The llama.cpp converter (conversion/minimax.py) stacks the w1 / w2 / w3 experts, stores the indexer projections as F32, and adds 1 to every RMSNorm weight so llama.cpp can run plain RMSNorm; the loader subtracts the 1 for this family's (1 + w) norms.
Both sides run NeoX rope, so q/k carry no rope permute.
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

_ARCH = "minimax-m3"
_LAYER_LEAVES = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "self_attn.index_q_proj": "indexer.q_proj",
    "self_attn.index_k_proj": "indexer.k_proj",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
    "block_sparse_moe.gate": "ffn_gate_inp",
    "block_sparse_moe.shared_experts.gate_proj": "ffn_gate_shexp",
    "block_sparse_moe.shared_experts.up_proj": "ffn_up_shexp",
    "block_sparse_moe.shared_experts.down_proj": "ffn_down_shexp",
    "block_sparse_moe.experts.0.w1": "ffn_gate_exps",
    "block_sparse_moe.experts.0.w3": "ffn_up_exps",
    "block_sparse_moe.experts.0.w2": "ffn_down_exps",
}
_TOP_LEAVES = {"model.embed_tokens": "token_embd", "lm_head": "output"}
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_SIGMOID_GATING = 2
# llama.cpp hard-codes these for the arch (src/llama-graph.cpp LLM_FFN_SWIGLU_OAI_MOE)
_SWIGLU_ALPHA, _SWIGLU_LIMIT = 1.702, 7.0


def to_gguf(name: str, tensors: dict[str, int]) -> tuple[str, ...]:
    """Checkpoint module name -> the GGUF tensors (without ``.weight``) that store it; empty for modules the GGUF has no matrix for."""
    if name in _TOP_LEAVES:
        return (_TOP_LEAVES[name],)
    m = _LAYER_RE.match(name)
    if m is None:
        return ()
    tensor = _LAYER_LEAVES.get(m.group(2))
    return (f"blk.{m.group(1)}.{tensor}",) if tensor is not None else ()


def _check_supported(shim: GgufConfigShim) -> None:
    if any(k.startswith("prism.hadamard.") for k in shim.metadata):
        raise NotImplementedError(f"{shim.model_path}: Hadamard-rotated (prism) GGUF weights are not supported")
    scaled = sorted(n for n in gguf_tensor_index(shim.model_path) if n.endswith((".scale", ".input_scale")))
    if scaled:
        raise NotImplementedError(f"{shim.model_path}: GGUF tensors with per-tensor scales are not supported (e.g. {scaled[0]})")


def _hf_config(shim: GgufConfigShim) -> SimpleNamespace:
    """The HF text config fields ``parse_config`` reads, from the GGUF metadata."""
    md = shim.metadata

    def kv(key: str):
        return md[f"{_ARCH}.{key}"]

    _check_supported(shim)
    if f"{_ARCH}.attention.indexer.head_count" not in md:
        raise NotImplementedError(f"{shim.model_path}: converted by an older llama.cpp without the sparse-attention indexer, which current llama.cpp refuses too")
    gating = int(kv("expert_gating_func"))
    if gating != _SIGMOID_GATING:
        raise NotImplementedError(f"{shim.model_path}: expert gating function {gating} is not supported; MiniMax-M3 routes with a sigmoid (2)")
    if not md.get(f"{_ARCH}.expert_weights_norm", False):
        raise NotImplementedError(f"{shim.model_path}: routing without renormalized expert weights is not supported")
    scaling = md.get(f"{_ARCH}.rope.scaling.type", "none")
    if scaling != "none":
        raise NotImplementedError(f"{shim.model_path}: rope scaling {scaling!r} is not supported")
    expert_ff = kv("expert_feed_forward_length")
    if isinstance(expert_ff, list):
        if len(set(expert_ff)) != 1:
            raise NotImplementedError(f"{shim.model_path}: per-layer expert sizes {sorted(set(expert_ff))} are not supported")
        expert_ff = expert_ff[0]
    num_layers = int(kv("block_count"))
    dense = int(md.get(f"{_ARCH}.leading_dense_block_count", 0))
    # llama.cpp runs sparse attention and the MoE FFN on every layer after the leading dense ones
    sparse = [0] * dense + [1] * (num_layers - dense)
    return SimpleNamespace(
        model_type="minimax_m3",
        architectures=list(shim.architectures),
        hidden_act="swigluoai",
        swiglu_alpha=_SWIGLU_ALPHA,
        swiglu_limit=_SWIGLU_LIMIT,
        hidden_size=int(kv("embedding_length")),
        num_hidden_layers=num_layers,
        num_attention_heads=int(kv("attention.head_count")),
        num_key_value_heads=int(kv("attention.head_count_kv")),
        head_dim=int(kv("attention.key_length")),
        rotary_dim=int(kv("rope.dimension_count")),
        rope_theta=float(kv("rope.freq_base")),
        max_position_embeddings=int(kv("context_length")),
        rms_norm_eps=float(kv("attention.layer_norm_rms_epsilon")),
        vocab_size=int(shim.vocab_size),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        dense_intermediate_size=int(kv("feed_forward_length")),
        intermediate_size=int(expert_ff),
        shared_intermediate_size=int(expert_ff),
        n_shared_experts=int(kv("expert_shared_count")),
        num_local_experts=int(kv("expert_count")),
        num_experts_per_tok=int(kv("expert_used_count")),
        # llama.cpp skips the scale when the key is absent or 0
        routed_scaling_factor=float(md.get(f"{_ARCH}.expert_weights_scale", 0.0)) or 1.0,
        use_routing_bias=True,
        moe_layer_freq=sparse,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": sparse,
            "sparse_num_index_heads": int(kv("attention.indexer.head_count")),
            "sparse_index_dim": int(kv("attention.indexer.key_length")),
            "sparse_topk_blocks": int(kv("attention.indexer.top_k")),
            "sparse_block_size": int(kv("attention.indexer.block_size")),
            "sparse_local_block": int(kv("attention.indexer.local_blocks")),
            "sparse_init_block": 0,
            "sparse_score_type": "max",
        },
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
    """The dense text state dict from the GGUF; the routed experts load as GGUF expert banks (``freetoken.models.gguf.experts``)."""
    if not include_non_moe:
        return
    from freetoken.utils import cached_load_hf_config

    config = parse_gguf_config(cached_load_hf_config(model_path))
    args = config.m3_args
    index = gguf_tensor_index(model_path)
    hidden = config.hidden_size

    def t(name: str) -> GgufTensorInfo:
        return index[name if name in index else f"{name}.weight"]

    def norm(name: str) -> torch.Tensor:
        return float_tensor(t(name)).float() - 1.0

    yield from fused("model.embed_tokens", [t("token_embd")], [matrix(t("token_embd"))], hidden)
    if not config.tie_word_embeddings:
        yield from fused("lm_head", [t("output")], [matrix(t("output"))], hidden)
    yield "model.norm.weight", norm("output_norm")

    for layer in range(config.num_layers):
        blk, pre = f"blk.{layer}", f"model.layers.{layer}"
        yield f"{pre}.input_layernorm.weight", norm(f"{blk}.attn_norm")
        yield f"{pre}.post_attention_layernorm.weight", norm(f"{blk}.ffn_norm")
        sa = f"{pre}.self_attn"
        qkv = [t(f"{blk}.attn_{p}") for p in ("q", "k", "v")]
        yield from fused(f"{sa}.qkv_proj", qkv, [matrix(p) for p in qkv], hidden)
        out = t(f"{blk}.attn_output")
        yield from fused(f"{sa}.o_proj", [out], [matrix(out)], int(out.ne[0]))
        yield f"{sa}.q_norm.weight", norm(f"{blk}.attn_q_norm")
        yield f"{sa}.k_norm.weight", norm(f"{blk}.attn_k_norm")
        if args.is_sparse_layer(layer):
            iq, ik = t(f"{blk}.indexer.q_proj"), t(f"{blk}.indexer.k_proj")
            yield from fused(f"{sa}.index_qk_proj", [iq, ik], [matrix(iq), matrix(ik)], hidden)
            yield f"{sa}.index_q_norm.weight", norm(f"{blk}.indexer.q_norm")
            yield f"{sa}.index_k_norm.weight", norm(f"{blk}.indexer.k_norm")
        if layer in args.moe_layer_ids:
            moe = f"{pre}.block_sparse_moe"
            yield f"{moe}.gate.weight", float_tensor(t(f"{blk}.ffn_gate_inp")).float()
            yield f"{moe}.e_score_correction_bias", float_tensor(t(f"{blk}.exp_probs_b.bias")).float()
            mlp, parts = f"{moe}.shared_experts", [t(f"{blk}.ffn_{p}_shexp") for p in ("gate", "up", "down")]
        else:
            mlp, parts = f"{pre}.mlp", [t(f"{blk}.ffn_{p}") for p in ("gate", "up", "down")]
        gate, up, down = parts
        yield from fused(f"{mlp}.gate_up_proj", [gate, up], [matrix(gate), matrix(up)], hidden)
        yield from fused(f"{mlp}.down_proj", [down], [matrix(down)], int(down.ne[0]))


__all__ = ["iter_gguf_weights", "parse_gguf_config", "to_gguf"]
