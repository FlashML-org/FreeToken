"""gpt-oss GGUFs: the llama.cpp ``gpt-oss`` arch served by the family's own HF classes with every tensor in its packed ggml type.

The converter (conversion/gpt_oss.py) writes the attention projections fused as ``attn_qkv`` (``--fuse-qkv``; separate ``attn_q``/``attn_k``/``attn_v`` without), the attention sinks as ``attn_sinks.weight``, and a per-expert bias for each of gate / up / down - the only GGUF family whose experts carry biases, so they load as two dense tensors the ggml kernel adds around the gated activation. The MXFP4 releases (ggml-org) repack the routed experts into ggml MXFP4, scales ahead of the blocks in llama.cpp's nibble order, which the ggml kernels read as packed rows.

The GGUF carries no per-layer attention types: every gpt-oss release alternates ``sliding_attention``/``full_attention`` from layer 0 (the HF configs' ``layer_types``), which is what this loader derives from ``attention.sliding_window``. The swiglu's alpha (1.702) and limit (7) are family constants llama.cpp also hardcodes; the GGUF stores neither.
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

# gpt-oss swiglu constants, as llama.cpp's swiglu_oai op and every HF config ship them
_ACT_ALPHA = 1.702
_ACT_LIMIT = 7.0

_LAYER_LEAVES = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
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
    tensor = _LAYER_LEAVES.get(leaf)
    return (f"{blk}.{tensor}",) if tensor is not None else ()


def _hf_config(shim: GgufConfigShim) -> SimpleNamespace:
    """The HF config fields ``parse_config`` reads, from the GGUF metadata."""
    md = shim.metadata
    index = gguf_tensor_index(shim.model_path)
    arch = shim.model_type  # the GGUF architecture: gpt-oss

    def kv(key: str):
        return md[f"{arch}.{key}"]

    check_scales(shim.model_path)
    scaling = md.get(f"{arch}.rope.scaling.type", "none")
    if scaling == "yarn":
        # every release's rope_scaling; the betas default to llama.cpp's when a file omits them
        rope_scaling = {
            "rope_type": "yarn",
            "factor": float(kv("rope.scaling.factor")),
            "original_max_position_embeddings": int(kv("rope.scaling.original_context_length")),
            "beta_fast": float(md.get(f"{arch}.rope.scaling.yarn_beta_fast", 32.0)),
            "beta_slow": float(md.get(f"{arch}.rope.scaling.yarn_beta_slow", 1.0)),
        }
    elif scaling == "none":
        rope_scaling = None
    else:
        raise NotImplementedError(f"{shim.model_path}: rope scaling {scaling!r} is not supported")
    num_layers = int(kv("block_count"))
    return SimpleNamespace(
        model_type="gpt_oss",
        architectures=list(shim.architectures),
        hidden_act="silu",
        hidden_size=int(kv("embedding_length")),
        num_hidden_layers=num_layers,
        num_attention_heads=int(kv("attention.head_count")),
        num_key_value_heads=int(kv("attention.head_count_kv")),
        head_dim=int(kv("attention.key_length")),
        rope_theta=float(kv("rope.freq_base")),
        rope_scaling=rope_scaling,
        max_position_embeddings=int(kv("context_length")),
        rms_norm_eps=float(kv("attention.layer_norm_rms_epsilon")),
        vocab_size=int(shim.vocab_size),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        intermediate_size=int(kv("expert_feed_forward_length")),
        num_local_experts=int(kv("expert_count")),
        num_experts_per_tok=int(kv("expert_used_count")),
        # the GGUF has no per-layer KV: every release alternates sliding/full from layer 0
        layer_types=["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(num_layers)],
        sliding_window=int(kv("attention.sliding_window")),
        attention_bias="blk.0.attn_output.bias" in index,
        swiglu_limit=_ACT_LIMIT,
        hidden_act_alpha=_ACT_ALPHA,
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

    config = parse_gguf_config(cached_load_hf_config(model_path))
    index = gguf_tensor_index(model_path)
    hidden = config.hidden_size

    def t(name: str) -> GgufTensorInfo:
        return index[name if name in index else f"{name}.weight"]

    def bias(name: str) -> torch.Tensor:
        return float_tensor(index[f"{name}.bias"])

    yield from fused("model.embed_tokens", [t("token_embd")], [matrix(t("token_embd"))], hidden)
    if not config.tie_word_embeddings:
        yield from fused("lm_head", [t("output")], [matrix(t("output"))], hidden)
    yield "model.norm.weight", float_tensor(t("output_norm"))

    for layer in range(config.num_layers):
        blk, pre = f"blk.{layer}", f"model.layers.{layer}"
        yield f"{pre}.input_layernorm.weight", float_tensor(t(f"{blk}.attn_norm"))
        yield f"{pre}.post_attention_layernorm.weight", float_tensor(t(f"{blk}.post_attention_norm"))
        sa = f"{pre}.self_attn"
        # convert_hf_to_gguf.py --fuse-qkv stores q, k and v as one attn_qkv matrix
        qkv = [t(f"{blk}.attn_qkv")] if f"{blk}.attn_qkv.weight" in index else [t(f"{blk}.attn_{p}") for p in ("q", "k", "v")]
        yield from fused(f"{sa}.qkv_proj", qkv, [matrix(p) for p in qkv], hidden)
        if f"{blk}.attn_qkv.bias" in index:
            yield f"{sa}.qkv_proj.bias", bias(f"{blk}.attn_qkv")
        elif f"{blk}.attn_q.bias" in index:
            yield f"{sa}.qkv_proj.bias", torch.cat([bias(f"{blk}.attn_{p}") for p in ("q", "k", "v")])
        out = t(f"{blk}.attn_output")
        yield from fused(f"{sa}.o_proj", [out], [matrix(out)], int(out.ne[0]))
        if f"{blk}.attn_output.bias" in index:
            yield f"{sa}.o_proj.bias", bias(f"{blk}.attn_output")
        yield f"{sa}.sinks", float_tensor(t(f"{blk}.attn_sinks"))
        mlp = f"{pre}.mlp"
        yield f"{mlp}.router.weight", float_tensor(t(f"{blk}.ffn_gate_inp"))
        if f"{blk}.ffn_gate_inp.bias" in index:
            yield f"{mlp}.router.bias", bias(f"{blk}.ffn_gate_inp")
        # the per-expert gate/up/down biases ride the expert banks (models/gguf/experts.py),
        # following each expert through the slot cache like its weights do


__all__ = ["iter_gguf_weights", "parse_gguf_config", "to_gguf"]
