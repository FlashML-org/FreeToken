from __future__ import annotations

import re
from dataclasses import replace
from types import SimpleNamespace
from typing import Iterator

import torch

from freetoken.models.gguf.reader import gguf_tensor_index
from freetoken.models.gguf.config import GgufConfigShim
from freetoken.models.config import ModelConfig
from freetoken.models.gguf.tensors import check_scales, float_tensor, fused, matrix
from freetoken.utils import cached_load_hf_config

from .config import parse_config

_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_LEAVES = {
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
    "mlp.gate": "ffn_gate_inp",
    **{f"mlp.{p}_proj": f"ffn_{p}" for p in ("gate", "up", "down")},
    **{f"mlp.shared_expert.{p}_proj": f"ffn_{p}_shexp" for p in ("gate", "up", "down")},
    **{f"mlp.experts.0.{p}_proj": f"ffn_{p}_exps" for p in ("gate", "up", "down")},
}


def to_gguf(name: str, tensors: dict[str, int]) -> tuple[str, ...]:
    top = {"model.embed_tokens": "token_embd", "lm_head": "output"}
    if name in top:
        return (top[name],)
    match = _LAYER.match(name)
    if match is None:
        return ()
    leaf = _LEAVES.get(match[2])
    return (f"blk.{match[1]}.{leaf}",) if leaf else ()


def _hf_config(shim: GgufConfigShim) -> SimpleNamespace:
    md = shim.metadata
    def kv(name):
        return md[f"flint.{name}"]
    if md.get("flint.rope.scaling.type", "none") != "none":
        raise NotImplementedError("Flint GGUF RoPE scaling is not supported")
    if int(kv("expert_gating_func")) != 2 or float(kv("expert_weights_scale")) != 1.0:
        raise NotImplementedError("Flint GGUF requires unscaled sigmoid routing")
    check_scales(shim.model_path)
    swa = tuple(kv("attention.sliding_window_pattern"))
    rope = tuple(kv("attention.rope_pattern"))
    if len(rope) != int(kv("block_count")):
        raise ValueError("Flint GGUF requires one RoPE flag per layer")
    return SimpleNamespace(
        architectures=list(shim.architectures), model_type="flint",
        num_hidden_layers=int(kv("block_count")), hidden_size=int(kv("embedding_length")),
        intermediate_size=int(kv("feed_forward_length")),
        num_attention_heads=int(kv("attention.head_count")),
        num_key_value_heads=int(kv("attention.head_count_kv")),
        head_dim=int(kv("attention.key_length")), vocab_size=shim.vocab_size,
        max_position_embeddings=int(kv("context_length")),
        rms_norm_eps=float(kv("attention.layer_norm_rms_epsilon")),
        rope_theta=float(kv("rope.freq_base")),
        layer_types=["sliding_attention" if v else "full_attention" for v in swa],
        no_rope_layer_indices=[i for i, v in enumerate(rope) if not v],
        mlp_only_layers=list(range(int(kv("leading_dense_block_count")))),
        num_experts=int(kv("expert_count")), num_experts_per_tok=int(kv("expert_used_count")),
        moe_intermediate_size=int(kv("expert_feed_forward_length")),
        shared_expert_intermediate_size=int(kv("expert_shared_feed_forward_length")),
        norm_topk_prob=bool(kv("expert_weights_norm")),
        sliding_window=int(kv("attention.sliding_window")),
        hidden_act="silu", router_score_function="sigmoid", softmax_type="learnable",
        tie_word_embeddings=shim.tie_word_embeddings,
    )


def parse_gguf_config(shim: GgufConfigShim) -> ModelConfig:
    return replace(parse_config(_hf_config(shim)), expert_quant="gguf")


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
        yield f"{pre}.input_layernorm.weight", float_tensor(t(f"{blk}.attn_norm"))
        yield f"{pre}.post_attention_layernorm.weight", float_tensor(t(f"{blk}.ffn_norm"))
        sa = f"{pre}.self_attn"
        yield from linear(f"{sa}.qkv_proj", [f"{blk}.attn_{p}" for p in ("q", "k", "v")])
        yield from linear(f"{sa}.o_proj", [f"{blk}.attn_output"])
        for p in ("q", "k"):
            yield f"{sa}.{p}_norm.weight", float_tensor(t(f"{blk}.attn_{p}_norm"))
        yield f"{sa}.softmax_offset", float_tensor(t(f"{blk}.attn_sinks")).float()
        mlp = f"{pre}.mlp"
        if layer in hf.mlp_only_layers:
            yield from linear(f"{mlp}.gate_up_proj", [f"{blk}.ffn_{p}" for p in ("gate", "up")])
            yield from linear(f"{mlp}.down_proj", [f"{blk}.ffn_down"])
        else:
            yield f"{mlp}.gate.weight", float_tensor(t(f"{blk}.ffn_gate_inp"))
            bias = index.get(f"{blk}.exp_probs_b.bias")
            yield f"{mlp}.gate.expert_bias", (
                float_tensor(bias).float() if bias is not None
                else torch.zeros(hf.num_experts, device=device, dtype=torch.float32)
            )
            yield from linear(f"{mlp}.shared_expert.gate_up_proj", [f"{blk}.ffn_{p}_shexp" for p in ("gate", "up")])
            yield from linear(f"{mlp}.shared_expert.down_proj", [f"{blk}.ffn_down_shexp"])
