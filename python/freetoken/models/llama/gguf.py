"""Llama and Mistral from a llama.cpp ``llama`` GGUF: config from the GGUF metadata, tensors by name.

llama.cpp runs interleaved rope on this arch, so its converter (conversion/llama.py) permutes the rows of each q and k head; the loader undoes that, and the model keeps HF's NeoX rope.
The RMSNorm weights are the HF ones.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.models.gguf.reader import GgufTensorInfo, gguf_tensor_index
from freetoken.models.gguf.tensors import check_scales, float_tensor, fused, matrix

from .config import parse_config

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim

_ARCH = "llama"
_LAYER_LEAVES = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
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
    tensor = _LAYER_LEAVES.get(leaf)
    return (f"{blk}.{tensor}",) if tensor is not None else ()


def _check_supported(shim: GgufConfigShim) -> None:
    if any(k.startswith("prism.hadamard.") for k in shim.metadata):
        raise NotImplementedError(f"{shim.model_path}: Hadamard-rotated (prism) GGUF weights are not supported")
    check_scales(shim.model_path)
    names = sorted(gguf_tensor_index(shim.model_path))
    # the llama arch also carries Mixtral experts and optional biases, which the llama model does not have
    extra = [n for n in names if n.endswith(".bias") or ".ffn_gate_inp." in n]
    if extra:
        raise NotImplementedError(f"{shim.model_path}: llama GGUF tensors with biases or MoE experts are not supported (e.g. {extra[0]})")


def _rope_scaling(shim: GgufConfigShim, base: float, head_dim: int) -> dict | None:
    """The HF rope scaling of the file: none, or Llama 3's, which the converter stores only as the ``rope_freqs`` divisors."""
    md = shim.metadata
    scaling = md.get(f"{_ARCH}.rope.scaling.type", "none")
    # llama.cpp scales linearly by rope.scaling.factor (or the older rope.scale_linear) when the type key is absent
    factor = float(md.get(f"{_ARCH}.rope.scaling.factor", md.get(f"{_ARCH}.rope.scale_linear", 0.0)))
    attn_factor = float(md.get(f"{_ARCH}.rope.scaling.attn_factor", 1.0))
    if scaling not in ("none", "linear") or factor not in (0.0, 1.0) or attn_factor != 1.0:
        raise NotImplementedError(f"{shim.model_path}: rope scaling {scaling!r} (factor {factor}, attn_factor {attn_factor}) is not supported")
    index = gguf_tensor_index(shim.model_path)
    if not index:
        raise NotImplementedError(f"{shim.model_path}: a metadata-only llama GGUF has no rope_freqs tensor to tell Llama 3 rope from plain rope")
    info = index.get("rope_freqs.weight")
    if info is None:
        return None
    from freetoken.layers.rotary import _rope_frequencies

    divisors = float_tensor(info).float()
    # Llama 3.1-3.3 differ only in factor
    llama3 = {
        "rope_type": "llama3",
        "factor": float(divisors.max()),
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }
    plain, _ = _rope_frequencies(head_dim, head_dim, base, None)
    scaled, _ = _rope_frequencies(head_dim, head_dim, base, llama3)
    if not torch.allclose(plain / scaled, divisors, rtol=1e-5):
        raise NotImplementedError(f"{shim.model_path}: rope_freqs does not match Llama 3 rope scaling")
    return llama3


def _hf_config(shim: GgufConfigShim) -> SimpleNamespace:
    """The HF config fields ``parse_config`` reads, from the GGUF metadata."""
    md = shim.metadata

    def kv(key: str):
        return md[f"{_ARCH}.{key}"]

    _check_supported(shim)
    hidden, heads = int(kv("embedding_length")), int(kv("attention.head_count"))
    # Mistral-7B files carry no key_length
    head_dim = int(md.get(f"{_ARCH}.attention.key_length", hidden // heads))
    q = gguf_tensor_index(shim.model_path).get("blk.0.attn_q.weight")
    # early Mistral-Nemo conversions lack key_length, so hidden / heads gives the wrong head_dim; llama.cpp rejects them too
    if q is not None and q.ne[1] != heads * head_dim:
        raise NotImplementedError(f"{shim.model_path}: attn_q has {q.ne[1]} rows, not head_count x head_dim = {heads * head_dim}")
    if int(md.get(f"{_ARCH}.rope.dimension_count", head_dim)) != head_dim:
        raise NotImplementedError(f"{shim.model_path}: partial rope is not supported")
    base = float(md.get(f"{_ARCH}.rope.freq_base", 10000.0))
    return SimpleNamespace(
        model_type="llama",
        architectures=list(shim.architectures),
        hidden_act="silu",
        hidden_size=hidden,
        num_hidden_layers=int(kv("block_count")),
        num_attention_heads=heads,
        num_key_value_heads=int(md.get(f"{_ARCH}.attention.head_count_kv", heads)),
        head_dim=head_dim,
        rope_theta=base,
        rope_scaling=_rope_scaling(shim, base, head_dim),
        max_position_embeddings=int(kv("context_length")),
        rms_norm_eps=float(kv("attention.layer_norm_rms_epsilon")),
        vocab_size=int(shim.vocab_size),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        intermediate_size=int(kv("feed_forward_length")),
    )


def parse_gguf_config(shim: GgufConfigShim) -> ModelConfig:
    return parse_config(_hf_config(shim))


def _unpermute(rows: torch.Tensor, heads: int) -> torch.Tensor:
    """Undo the converter's rope permute of one q or k matrix (``LlamaModel.permute``); it moves whole rows, so packed ggml rows work too."""
    return rows.reshape(heads, rows.shape[0] // heads // 2, 2, *rows.shape[1:]).swapaxes(1, 2).reshape(rows.shape)


def iter_gguf_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """The state dict from the GGUF."""
    if not include_non_moe:
        return
    from freetoken.utils import cached_load_hf_config

    hf = _hf_config(cached_load_hf_config(model_path))
    index = gguf_tensor_index(model_path)
    hidden, heads, kv_heads = hf.hidden_size, hf.num_attention_heads, hf.num_key_value_heads
    q_rows, kv_rows = heads * hf.head_dim, kv_heads * hf.head_dim

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
            q, k, v = matrix(qkv[0]).split([q_rows, kv_rows, kv_rows])
        else:
            qkv = [t(f"{blk}.attn_{p}") for p in ("q", "k", "v")]
            q, k, v = (matrix(p) for p in qkv)
        yield from fused(f"{sa}.qkv_proj", qkv, [_unpermute(q, heads), _unpermute(k, kv_heads), v], hidden)
        out = t(f"{blk}.attn_output")
        yield from fused(f"{sa}.o_proj", [out], [matrix(out)], int(out.ne[0]))
        mlp = f"{pre}.mlp"
        gate, up, down = (t(f"{blk}.ffn_{p}") for p in ("gate", "up", "down"))
        yield from fused(f"{mlp}.gate_up_proj", [gate, up], [matrix(gate), matrix(up)], hidden)
        yield from fused(f"{mlp}.down_proj", [down], [matrix(down)], int(down.ne[0]))


__all__ = ["iter_gguf_weights", "parse_gguf_config", "to_gguf"]
