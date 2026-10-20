"""Muse Glimmer text from a llama.cpp ``muse-glimmer`` GGUF: config from the GGUF metadata, tensors by name.

The converter (conversion/muse_glimmer.py) adds 1 to the four norms of each block, permutes the rows of each q and k head for llama.cpp's interleaved rope, and writes constant q/k norms that carry ``qk_scale_factor``.
The loader subtracts the 1, undoes the permute, and reads ``qk_scale_factor`` back from the q norm.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.models.gguf.reader import GgufTensorInfo, gguf_tensor_index
from freetoken.models.gguf.tensors import float_tensor, fused, matrix

from .config import parse_config

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim

_ARCH = "muse-glimmer"
_LAYER_LEAVES = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.gate_proj": "attn_gate",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}
_QKV_LEAVES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
_TOP_LEAVES = {"model.embed_tokens": "token_embd", "lm_head": "output"}
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
# checkpoint norm -> GGUF norm; the GGUF stores each as weight + 1
_SHIFTED_NORMS = {
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "post_attention_norm",
    "pre_feedforward_layernorm": "ffn_norm",
    "post_feedforward_layernorm": "post_ffw_norm",
}


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
    index = gguf_tensor_index(shim.model_path)
    if not index:
        raise NotImplementedError(f"{shim.model_path}: a metadata-only muse-glimmer GGUF has no q norm to read qk_scale_factor from")
    if any(k.startswith("prism.hadamard.") for k in shim.metadata):
        raise NotImplementedError(f"{shim.model_path}: Hadamard-rotated (prism) GGUF weights are not supported")
    scaled = sorted(n for n in index if n.endswith((".scale", ".input_scale")))
    if scaled:
        raise NotImplementedError(f"{shim.model_path}: GGUF tensors with per-tensor scales are not supported (e.g. {scaled[0]})")


def _qk_scale_factor(model_path: str, num_layers: int) -> float:
    """The constant the converter wrote into every q norm; the k norms are all ones."""
    index = gguf_tensor_index(model_path)
    scale = float(float_tensor(index["blk.0.attn_q_norm.weight"])[0])
    for layer in range(num_layers):
        q, k = (float_tensor(index[f"blk.{layer}.attn_{p}_norm.weight"]).float() for p in ("q", "k"))
        if not (torch.all(q == scale) and torch.all(k == 1.0)):
            raise NotImplementedError(f"{model_path}: the q/k norms of layer {layer} are not the converter's constant qk_scale_factor / ones")
    return scale


def _hf_config(shim: GgufConfigShim) -> SimpleNamespace:
    """The HF text config fields ``parse_config`` reads, from the GGUF metadata."""
    md = shim.metadata

    def kv(key: str):
        return md[f"{_ARCH}.{key}"]

    _check_supported(shim)
    num_layers = int(kv("block_count"))
    # a bool per layer, or llama.cpp's period: every n-th layer is full attention, and 0 means none is
    pattern = md.get(f"{_ARCH}.attention.sliding_window_pattern", 4)
    if isinstance(pattern, list):
        swa = [bool(p) for p in pattern]
        if len(swa) != num_layers:
            raise ValueError(f"{shim.model_path}: sliding_window_pattern has {len(swa)} entries for {num_layers} layers")
    else:
        swa = [int(pattern) == 0 or i % int(pattern) < int(pattern) - 1 for i in range(num_layers)]
    base = float(kv("rope.freq_base"))
    swa_base = float(md.get(f"{_ARCH}.rope.freq_base_swa", base))
    return SimpleNamespace(
        model_type="muse_glimmer",
        architectures=list(shim.architectures),
        hidden_activation="silu",
        hidden_size=int(kv("embedding_length")),
        num_hidden_layers=num_layers,
        num_attention_heads=int(kv("attention.head_count")),
        num_key_value_heads=int(kv("attention.head_count_kv")),
        head_dim=int(kv("attention.key_length")),
        intermediate_size=int(kv("feed_forward_length")),
        max_position_embeddings=int(kv("context_length")),
        rms_norm_eps=float(kv("attention.layer_norm_rms_epsilon")),
        # llama.cpp hard-codes the post-norm eps and NoPE on the full layers (src/models/muse-glimmer.cpp)
        post_norm_eps=1e-8,
        layer_types=["sliding_attention" if s else "full_attention" for s in swa],
        layer_rope_theta=[swa_base if s else 0.0 for s in swa],
        sliding_window=int(kv("attention.sliding_window")),
        qk_scale_factor=_qk_scale_factor(shim.model_path, num_layers),
        # llama.cpp's default when the key is absent; 0 turns the cap off
        final_logit_softcapping=float(md.get(f"{_ARCH}.final_logit_softcapping", 30.0)) or None,
        output_multiplier=float(kv("logit_scale")),
        vocab_size=int(shim.vocab_size),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
    )


def parse_gguf_config(shim: GgufConfigShim) -> ModelConfig:
    return parse_config(_hf_config(shim))


def _unpermute(rows: torch.Tensor, heads: int) -> torch.Tensor:
    """Undo the converter's rope permute of one q or k matrix (``_unpermute_for_rope``); it moves whole rows, so packed ggml rows work too."""
    return rows.reshape(heads, rows.shape[0] // heads // 2, 2, *rows.shape[1:]).swapaxes(1, 2).reshape(rows.shape)


def iter_gguf_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """The text state dict from the GGUF."""
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
        for norm, tensor in _SHIFTED_NORMS.items():
            yield f"{pre}.{norm}.weight", float_tensor(t(f"{blk}.{tensor}")).float() - 1.0
        sa = f"{pre}.self_attn"
        # convert_hf_to_gguf.py --fuse-qkv stores q, k and v as one attn_qkv matrix
        if f"{blk}.attn_qkv.weight" in index:
            qkv = [t(f"{blk}.attn_qkv")] * 3
            q, k, v = matrix(qkv[0]).split([q_rows, kv_rows, kv_rows])
        else:
            qkv = [t(f"{blk}.attn_{p}") for p in ("q", "k", "v")]
            q, k, v = (matrix(p) for p in qkv)
        yield from fused(f"{sa}.qkv_proj", qkv, [_unpermute(q, heads), _unpermute(k, kv_heads), v], hidden)
        gate = t(f"{blk}.attn_gate")
        yield from fused(f"{sa}.gate_proj", [gate], [matrix(gate)], hidden)
        out = t(f"{blk}.attn_output")
        yield from fused(f"{sa}.o_proj", [out], [matrix(out)], int(out.ne[0]))
        mlp = f"{pre}.mlp"
        gate, up, down = (t(f"{blk}.ffn_{p}") for p in ("gate", "up", "down"))
        yield from fused(f"{mlp}.gate_up_proj", [gate, up], [matrix(gate), matrix(up)], hidden)
        yield from fused(f"{mlp}.down_proj", [down], [matrix(down)], int(down.ne[0]))


__all__ = ["iter_gguf_weights", "parse_gguf_config", "to_gguf"]
