"""GLM-4.5 / 4.6 / 4.7 from a llama.cpp ``glm4moe`` GGUF: config from the GGUF metadata, tensors by name; the routed experts load as GGUF expert banks (``freetoken.models.gguf.experts``).

The llama.cpp converter (conversion/glm.py) only renames the tensors and stacks the experts: both sides run NeoX rope on the first ``rope.dimension_count`` dims of a head, so q/k carry no rope permute.
Files that keep the MTP layer append it as the last block, which is not loaded.
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

_ARCH = "glm4moe"
_LAYER_LEAVES = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
    "mlp.gate": "ffn_gate_inp",
    "mlp.shared_experts.gate_proj": "ffn_gate_shexp",
    "mlp.shared_experts.up_proj": "ffn_up_shexp",
    "mlp.shared_experts.down_proj": "ffn_down_shexp",
    "mlp.experts.0.gate_proj": "ffn_gate_exps",
    "mlp.experts.0.up_proj": "ffn_up_exps",
    "mlp.experts.0.down_proj": "ffn_down_exps",
}
_TOP_LEAVES = {"model.embed_tokens": "token_embd", "lm_head": "output"}
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
# llama.cpp's LLAMA_EXPERT_GATING_FUNC_TYPE_*: 0 (unset) means sigmoid for this arch
_SIGMOID_GATING = (0, 2)


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
    md = shim.metadata
    if any(k.startswith("prism.hadamard.") for k in md):
        raise NotImplementedError(f"{shim.model_path}: Hadamard-rotated (prism) GGUF weights are not supported")
    names = sorted(gguf_tensor_index(shim.model_path))
    if not names:
        raise NotImplementedError(f"{shim.model_path}: a metadata-only glm4moe GGUF has no tensor table to tell whether the attention has q/k norms and biases")
    if "blk.0.attn_q.weight" not in names:
        raise NotImplementedError(f"{shim.model_path}: no decoder layer tensors (blk.0.attn_q), e.g. an MTP-only draft GGUF")
    scaled = [n for n in names if n.endswith((".scale", ".input_scale"))]
    if scaled:
        raise NotImplementedError(f"{shim.model_path}: GGUF tensors with per-tensor scales are not supported (e.g. {scaled[0]})")
    fused_qkv = [n for n in names if ".attn_qkv." in n]
    if fused_qkv:
        raise NotImplementedError(f"{shim.model_path}: a fused attn_qkv glm4moe GGUF is not supported (e.g. {fused_qkv[0]})")
    gating = int(md.get(f"{_ARCH}.expert_gating_func", 0))
    if gating not in _SIGMOID_GATING:
        raise NotImplementedError(f"{shim.model_path}: expert gating function {gating} is not supported, GLM routes with sigmoid")
    scaling = md.get(f"{_ARCH}.rope.scaling.type", "none")
    if scaling != "none":
        raise NotImplementedError(f"{shim.model_path}: rope scaling {scaling!r} is not supported")


def _hf_config(shim: GgufConfigShim) -> SimpleNamespace:
    """The HF config fields ``parse_config`` reads, from the GGUF metadata."""
    md = shim.metadata

    def kv(key: str):
        return md[f"{_ARCH}.{key}"]

    _check_supported(shim)
    index = gguf_tensor_index(shim.model_path)
    head_dim = int(kv("attention.key_length"))
    return SimpleNamespace(
        model_type="glm4_moe",
        architectures=list(shim.architectures),
        hidden_act="silu",
        hidden_size=int(kv("embedding_length")),
        # block_count also counts the MTP layers a GGUF may carry after the decoder
        num_hidden_layers=int(kv("block_count")) - int(md.get(f"{_ARCH}.nextn_predict_layers", 0)),
        num_attention_heads=int(kv("attention.head_count")),
        num_key_value_heads=int(kv("attention.head_count_kv")),
        head_dim=head_dim,
        rope_theta=float(kv("rope.freq_base")),
        partial_rotary_factor=int(kv("rope.dimension_count")) / head_dim,
        max_position_embeddings=int(kv("context_length")),
        rms_norm_eps=float(kv("attention.layer_norm_rms_epsilon")),
        vocab_size=int(shim.vocab_size),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        intermediate_size=int(kv("feed_forward_length")),
        n_routed_experts=int(kv("expert_count")),
        num_experts_per_tok=int(kv("expert_used_count")),
        moe_intermediate_size=int(kv("expert_feed_forward_length")),
        n_shared_experts=int(kv("expert_shared_count")),
        first_k_dense_replace=int(md.get(f"{_ARCH}.leading_dense_block_count", 0)),
        # llama.cpp neither renormalizes nor scales the router weights when the keys are absent
        norm_topk_prob=bool(md.get(f"{_ARCH}.expert_weights_norm", False)),
        routed_scaling_factor=float(md.get(f"{_ARCH}.expert_weights_scale", 0.0)) or 1.0,
        n_group=max(1, int(md.get(f"{_ARCH}.expert_group_count", 1))),
        topk_group=max(1, int(md.get(f"{_ARCH}.expert_group_used_count", 1))),
        # GLM-4.5-Air has no q/k norms; the converter writes no flag for them or for the qkv biases
        use_qk_norm="blk.0.attn_q_norm.weight" in index,
        attention_bias="blk.0.attn_q.bias" in index,
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

    def t(name: str) -> GgufTensorInfo:
        return index[name if name in index else f"{name}.weight"]

    def linear(module: str, tensor: str) -> Iterator[tuple[str, torch.Tensor]]:
        info = t(tensor)
        yield from fused(module, [info], [matrix(info)], int(info.ne[0]))
        if f"{tensor}.bias" in index:
            yield f"{module}.bias", float_tensor(index[f"{tensor}.bias"])

    yield from linear("model.embed_tokens", "token_embd")
    if not hf.tie_word_embeddings:
        yield from linear("lm_head", "output")
    yield "model.norm.weight", float_tensor(t("output_norm"))

    for layer in range(hf.num_hidden_layers):
        blk, pre = f"blk.{layer}", f"model.layers.{layer}"
        yield f"{pre}.input_layernorm.weight", float_tensor(t(f"{blk}.attn_norm"))
        yield f"{pre}.post_attention_layernorm.weight", float_tensor(t(f"{blk}.post_attention_norm"))
        sa = f"{pre}.self_attn"
        for proj, tensor in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")):
            yield from linear(f"{sa}.{proj}", f"{blk}.{tensor}")
        if hf.use_qk_norm:
            yield f"{sa}.q_norm.weight", float_tensor(t(f"{blk}.attn_q_norm"))
            yield f"{sa}.k_norm.weight", float_tensor(t(f"{blk}.attn_k_norm"))
        mlp = f"{pre}.mlp"
        if layer < hf.first_k_dense_replace:
            for proj in ("gate", "up", "down"):
                yield from linear(f"{mlp}.{proj}_proj", f"{blk}.ffn_{proj}")
            continue
        yield f"{mlp}.gate.weight", float_tensor(t(f"{blk}.ffn_gate_inp"))
        yield f"{mlp}.e_score_correction_bias", float_tensor(t(f"{blk}.exp_probs_b.bias"))
        for proj in ("gate", "up", "down"):
            yield from linear(f"{mlp}.shared_experts.{proj}_proj", f"{blk}.ffn_{proj}_shexp")


__all__ = ["iter_gguf_weights", "parse_gguf_config", "to_gguf"]
