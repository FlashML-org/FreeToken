"""Qwen3 MoE from a llama.cpp ``qwen3moe`` GGUF: config from the GGUF metadata, tensors by name; the routed experts load as GGUF expert banks (``freetoken.models.gguf.experts``).

The llama.cpp converter (conversion/qwen.py) stores these tensors unchanged: both sides run NeoX rope, so q/k carry no rope permute, and the RMSNorm weights are the HF ones.
"""

from __future__ import annotations

import re
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.models.gguf.reader import GgufTensorInfo, gguf_tensor_index
from freetoken.models.gguf.tensors import check_scales, float_tensor, fused, matrix

from .config import parse_config

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim

_ARCH = "qwen3moe"
_LAYER_LEAVES = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate": "ffn_gate_inp",
    "mlp.experts.0.gate_proj": "ffn_gate_exps",
    "mlp.experts.0.up_proj": "ffn_up_exps",
    "mlp.experts.0.down_proj": "ffn_down_exps",
}
_QKV_LEAVES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
_TOP_LEAVES = {"model.embed_tokens": "token_embd", "lm_head": "output"}
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


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
    if leaf in ("mlp.experts.0.gate_proj", "mlp.experts.0.up_proj") and f"{blk}.ffn_gate_up_exps.weight" in tensors:
        return (f"{blk}.ffn_gate_up_exps",)
    tensor = _LAYER_LEAVES.get(leaf)
    return (f"{blk}.{tensor}",) if tensor is not None else ()


def _check_supported(shim: GgufConfigShim) -> None:
    if any(k.startswith("prism.hadamard.") for k in shim.metadata):
        raise NotImplementedError(f"{shim.model_path}: Hadamard-rotated (prism) GGUF weights are not supported")
    check_scales(shim.model_path)


def _hf_config(shim: GgufConfigShim) -> SimpleNamespace:
    """The HF config fields ``parse_config`` reads, from the GGUF metadata."""
    md = shim.metadata

    def kv(key: str):
        return md[f"{_ARCH}.{key}"]

    _check_supported(shim)
    rope = {"rope_theta": float(kv("rope.freq_base")), "rope_type": "default"}
    scaling = md.get(f"{_ARCH}.rope.scaling.type", "none")
    if scaling == "yarn":
        rope.update(
            rope_type="yarn",
            factor=float(kv("rope.scaling.factor")),
            original_max_position_embeddings=int(kv("rope.scaling.original_context_length")),
        )
    elif scaling != "none":
        raise NotImplementedError(f"{shim.model_path}: rope scaling {scaling!r} is not supported")
    key_length = md.get(f"{_ARCH}.attention.key_length")
    return SimpleNamespace(
        model_type="qwen3_moe",
        architectures=list(shim.architectures),
        hidden_act="silu",
        hidden_size=int(kv("embedding_length")),
        num_hidden_layers=int(kv("block_count")),
        num_attention_heads=int(kv("attention.head_count")),
        num_key_value_heads=int(kv("attention.head_count_kv")),
        head_dim=int(key_length) if key_length is not None else None,
        rope_scaling=rope,
        max_position_embeddings=int(kv("context_length")),
        rms_norm_eps=float(kv("attention.layer_norm_rms_epsilon")),
        vocab_size=int(shim.vocab_size),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        intermediate_size=int(kv("feed_forward_length")),
        num_experts=int(kv("expert_count")),
        num_experts_per_tok=int(kv("expert_used_count")),
        moe_intermediate_size=int(kv("expert_feed_forward_length")),
        # llama.cpp always renormalizes the top-k weights of a qwen3moe router
        norm_topk_prob=True,
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
        qkv = [t(f"{blk}.attn_qkv")] if f"{blk}.attn_qkv.weight" in index else [t(f"{blk}.attn_{p}") for p in ("q", "k", "v")]
        yield from fused(f"{sa}.qkv_proj", qkv, [matrix(p) for p in qkv], hidden)
        yield f"{sa}.q_norm.weight", float_tensor(t(f"{blk}.attn_q_norm"))
        yield f"{sa}.k_norm.weight", float_tensor(t(f"{blk}.attn_k_norm"))
        out = t(f"{blk}.attn_output")
        yield from fused(f"{sa}.o_proj", [out], [matrix(out)], int(out.ne[0]))
        yield f"{pre}.mlp.gate.weight", float_tensor(t(f"{blk}.ffn_gate_inp"))


__all__ = ["iter_gguf_weights", "parse_gguf_config", "to_gguf"]
