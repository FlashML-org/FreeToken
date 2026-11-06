from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Iterator

import torch

from freetoken.models.config import ModelConfig
from freetoken.models.gguf.config import GgufConfigShim
from freetoken.models.gguf.reader import gguf_tensor_index
from freetoken.models.gguf.tensors import check_scales, float_tensor, fused, matrix
from freetoken.utils import cached_load_hf_config

from .config import parse_config

_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_LEAVES = {
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
    "self_attn.g_proj": "attn_gate", "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
}


def to_gguf(name: str, tensors: dict[str, int]) -> tuple[str, ...]:
    top = {"model.embed_tokens": "token_embd", "lm_head": "output"}
    if name in top:
        return (top[name],)
    match = _LAYER.match(name)
    leaf = _LEAVES.get(match[2]) if match else None
    return (f"blk.{match[1]}.{leaf}",) if leaf else ()


def _clamp(value) -> float:
    # llama.cpp's converter writes one clamp per layer; the first community GGUFs carry a scalar; absent or 0 is no clamp
    values = set(value) if isinstance(value, list) else {value}
    if len(values) != 1:
        raise NotImplementedError("Ricursive GGUF per-layer swiglu_clamp values must all match")
    return float(values.pop())


def _hf_config(shim: GgufConfigShim) -> SimpleNamespace:
    md = shim.metadata

    def kv(name):
        return md[f"ricursive.{name}"]

    if md.get("ricursive.rope.scaling.type", "none") != "none":
        raise NotImplementedError("Ricursive GGUF RoPE scaling is not supported")
    check_scales(shim.model_path)
    layers = int(kv("block_count"))
    # llama.cpp's default when the pattern is absent: every fifth layer is global, as in the checkpoint config
    pattern = md.get("ricursive.attention.sliding_window_pattern")
    if pattern is None:
        pattern = [i % 5 != 4 for i in range(layers)]
    scale = md.get("ricursive.embedding_scale")
    return SimpleNamespace(
        architectures=list(shim.architectures), num_hidden_layers=layers,
        hidden_size=int(kv("embedding_length")), intermediate_size=int(kv("feed_forward_length")),
        num_attention_heads=int(kv("attention.head_count")), num_key_value_heads=int(kv("attention.head_count_kv")),
        head_dim=int(kv("attention.key_length")), vocab_size=shim.vocab_size,
        max_position_embeddings=int(kv("context_length")),
        rms_norm_eps=float(kv("attention.layer_norm_rms_epsilon")), rope_theta=float(kv("rope.freq_base")),
        layer_types=["sliding_attention" if v else "full_attention" for v in pattern],
        sliding_window=int(kv("attention.sliding_window")), embed_scale=float(scale) if scale else None,
        ffn_clamp_threshold=_clamp(md.get("ricursive.swiglu_clamp", 0.0)), tie_word_embeddings=shim.tie_word_embeddings,
        hidden_act="silu", use_gated_attention=True, gate_full_precision=True, use_valres=True,
    )


def parse_gguf_config(shim: GgufConfigShim) -> ModelConfig:
    return parse_config(_hf_config(shim))


def iter_gguf_weights(
    model_path: str, device: torch.device, *, include_moe_experts: bool, include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    if not include_non_moe:
        return
    hf = _hf_config(cached_load_hf_config(model_path))
    index = gguf_tensor_index(model_path)

    def t(name):
        return index[name if name in index else f"{name}.weight"]

    def linear(module, names):
        infos = [t(n) for n in names]
        yield from fused(module, infos, [matrix(v) for v in infos], int(infos[0].ne[0]))

    yield from linear("model.embed_tokens", ["token_embd"])
    if not hf.tie_word_embeddings:
        yield from linear("lm_head", ["output"])
    yield "model.norm.weight", float_tensor(t("output_norm"))
    for layer in range(hf.num_hidden_layers):
        blk, pre = f"blk.{layer}", f"model.layers.{layer}"
        for norm, tensor in (
            ("input_layernorm", "attn_norm"), ("post_attention_layernorm", "post_attention_norm"),
            ("pre_feedforward_layernorm", "ffn_norm"), ("post_feedforward_layernorm", "post_ffw_norm"),
            ("self_attn.q_norm", "attn_q_norm"), ("self_attn.k_norm", "attn_k_norm"),
        ):
            yield f"{pre}.{norm}.weight", float_tensor(t(f"{blk}.{tensor}"))
        sa = f"{pre}.self_attn"
        yield from linear(f"{sa}.qk_proj", [f"{blk}.attn_q", f"{blk}.attn_k"])
        yield from linear(f"{sa}.v_proj", [f"{blk}.attn_v"])
        yield from linear(f"{sa}.g_proj", [f"{blk}.attn_gate"])
        yield from linear(f"{sa}.o_proj", [f"{blk}.attn_output"])
        yield f"{sa}.value_res_blend", torch.sigmoid(float_tensor(t(f"{blk}.attn_vres_lambda")).float())
        yield from linear(f"{pre}.mlp.gate_up_proj", [f"{blk}.ffn_gate", f"{blk}.ffn_up"])
        yield from linear(f"{pre}.mlp.down_proj", [f"{blk}.ffn_down"])
