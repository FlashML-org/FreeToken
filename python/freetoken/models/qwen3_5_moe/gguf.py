"""Qwen3.5 / 3.6 / 3.8 dense and MoE from a llama.cpp ``qwen35`` / ``qwen35moe`` GGUF: config from the GGUF metadata, tensor names, and the converter's transforms undone.

The llama.cpp converter (conversion/qwen.py) stores ``A_log`` as ``-exp(A_log)``, the squeezed conv kernels, and the GDN value heads in tiled order (k-head minor). Its zero-centred norms are stored as ``w + 1``; the loader subtracts the 1 in fp32 and the norms add it back in fp32 at runtime. Row reorders are undone on the packed bytes; ``ssm_out`` keeps its tiled columns and the GDN layer permutes its input instead.
"""

from __future__ import annotations

import re
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.layers.quantization.ggml_types import is_quantized
from freetoken.models.gguf.reader import GgufTensorInfo, gguf_tensor_index
from freetoken.models.gguf.tensors import check_scales, float_tensor, fused, matrix

from .config import parse_config

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim

_LAYER_LEAVES: dict[str, tuple[str, ...]] = {
    "linear_attn.in_proj_qkv": ("attn_qkv",),
    "linear_attn.in_proj_z": ("attn_gate",),
    "linear_attn.in_proj_b": ("ssm_beta",),
    "linear_attn.in_proj_a": ("ssm_alpha",),
    "linear_attn.out_proj": ("ssm_out",),
    "self_attn.q_proj": ("attn_q",),
    "self_attn.k_proj": ("attn_k",),
    "self_attn.v_proj": ("attn_v",),
    "self_attn.o_proj": ("attn_output",),
    "mlp.gate_proj": ("ffn_gate",),
    "mlp.up_proj": ("ffn_up",),
    "mlp.down_proj": ("ffn_down",),
    "mlp.gate": ("ffn_gate_inp",),
    "mlp.shared_expert_gate": ("ffn_gate_inp_shexp",),
    "mlp.shared_expert.gate_proj": ("ffn_gate_shexp",),
    "mlp.shared_expert.up_proj": ("ffn_up_shexp",),
    "mlp.shared_expert.down_proj": ("ffn_down_shexp",),
    "mlp.experts.0.gate_proj": ("ffn_gate_exps",),
    "mlp.experts.0.up_proj": ("ffn_up_exps",),
    "mlp.experts.0.down_proj": ("ffn_down_exps",),
}
_TOP_LEAVES: dict[str, tuple[str, ...]] = {
    "model.language_model.embed_tokens": ("token_embd",),
    "lm_head": ("output",),
}
_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")


def to_gguf(name: str, tensors: dict[str, int]) -> tuple[str, ...]:
    """Checkpoint module name -> the GGUF tensors (without ``.weight``) that store it; empty for modules the GGUF has no matrix for."""
    if name in _TOP_LEAVES:
        return _TOP_LEAVES[name]
    m = _LAYER_RE.match(name)
    if m is None:
        return ()
    blk, leaf = f"blk.{m.group(1)}", m.group(2)
    if leaf in ("mlp.experts.0.gate_proj", "mlp.experts.0.up_proj") and f"{blk}.ffn_gate_up_exps.weight" in tensors:
        return (f"{blk}.ffn_gate_up_exps",)
    return tuple(f"{blk}.{t}" for t in _LAYER_LEAVES.get(leaf, ()))


def _check_supported(shim: GgufConfigShim) -> None:
    if any(k.startswith("prism.hadamard.") for k in shim.metadata):
        raise NotImplementedError(f"{shim.model_path}: Hadamard-rotated (prism) GGUF weights are not supported")
    check_scales(shim.model_path)


def parse_gguf_config(shim: GgufConfigShim) -> ModelConfig:
    md = shim.metadata
    arch = shim.model_type

    def kv(key: str):
        return md[f"{arch}.{key}"]

    _check_supported(shim)
    head_dim = int(kv("attention.key_length"))
    n_v = int(kv("ssm.time_step_rank"))
    # block_count also counts the MTP layers a GGUF may carry after the decoder
    num_layers = int(kv("block_count")) - int(md.get(f"{arch}.nextn_predict_layers", 0))
    recurrent = md.get(f"{arch}.attention.recurrent_layers")
    if recurrent is not None:
        layer_types = ["linear_attention" if r else "full_attention" for r in list(recurrent)[:num_layers]]
    else:
        interval = int(kv("full_attention_interval"))
        layer_types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(num_layers)]
    moe = arch == "qwen35moe"
    text = SimpleNamespace(
        hidden_act="silu",
        hidden_size=int(kv("embedding_length")),
        num_hidden_layers=num_layers,
        layer_types=layer_types,
        num_attention_heads=int(kv("attention.head_count")),
        num_key_value_heads=int(kv("attention.head_count_kv")),
        head_dim=head_dim,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": float(kv("rope.freq_base")),
            "partial_rotary_factor": int(kv("rope.dimension_count")) / head_dim,
            "mrope_section": [int(s) for s in kv("rope.dimension_sections")][:3],
            "mrope_interleaved": True,
        },
        max_position_embeddings=int(kv("context_length")),
        rms_norm_eps=float(kv("attention.layer_norm_rms_epsilon")),
        vocab_size=int(shim.vocab_size),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        intermediate_size=0 if moe else int(kv("feed_forward_length")),
        num_experts=int(kv("expert_count")) if moe else 0,
        num_experts_per_tok=int(kv("expert_used_count")) if moe else 0,
        moe_intermediate_size=int(kv("expert_feed_forward_length")) if moe else 0,
        shared_expert_intermediate_size=int(kv("expert_shared_feed_forward_length")) if moe else 0,
        linear_num_key_heads=int(kv("ssm.group_count")),
        linear_num_value_heads=n_v,
        linear_key_head_dim=int(kv("ssm.state_size")),
        linear_value_head_dim=int(kv("ssm.inner_size")) // n_v,
        linear_conv_kernel_dim=int(kv("ssm.conv_kernel")),
    )
    hf_like = SimpleNamespace(
        text_config=text,
        model_type="qwen3_5_moe" if moe else "qwen3_5",
        architectures=list(shim.architectures),
        vision_config=None,
    )
    config = parse_config(hf_like)
    return replace(config, expert_quant="gguf") if moe else config


# ----------------------------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------------------------

def _untile(t: torch.Tensor, num_k_heads: int, head_dim: int, dim: int = 0) -> torch.Tensor:
    """Tiled value heads (k-head minor) back to grouped (k-head major) along ``dim``."""
    n = t.shape[dim]
    r = n // (num_k_heads * head_dim)
    idx = torch.arange(n).view(r, num_k_heads, head_dim).transpose(0, 1).reshape(n)
    return t.index_select(dim, idx)


def iter_gguf_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """The dense state dict from the GGUF; the routed experts load as offload banks (``freetoken.models.gguf.experts``)."""
    if not include_non_moe:
        return
    from freetoken.utils import cached_load_hf_config

    config = parse_gguf_config(cached_load_hf_config(model_path))
    index = gguf_tensor_index(model_path)
    g = config.linear_attention_group()
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
        yield f"{pre}.post_attention_layernorm.weight", norm(f"{blk}.post_attention_norm")
        if config.is_linear_layer(layer):
            la = f"{pre}.linear_attn"
            nk, dk = g.num_key_heads, g.key_head_dim
            qkv, gate = t(f"{blk}.attn_qkv"), t(f"{blk}.attn_gate")
            qkv_rows = matrix(qkv)
            v_start = 2 * nk * dk
            qkv_rows = torch.cat([qkv_rows[:v_start], _untile(qkv_rows[v_start:], nk, g.value_head_dim)])
            z_rows = _untile(matrix(gate), nk, g.value_head_dim)
            beta_info, alpha_info = t(f"{blk}.ssm_beta"), t(f"{blk}.ssm_alpha")
            beta, alpha = (_untile(matrix(i), nk, 1) for i in (beta_info, alpha_info))
            if is_quantized(qkv.ggml_type) or is_quantized(gate.ggml_type):
                yield from fused(f"{la}.in_proj_qkvz", [qkv, gate], [qkv_rows, z_rows], hidden)
                yield from fused(f"{la}.in_proj_ba", [beta_info, alpha_info], [beta, alpha], hidden)
            else:
                yield from fused(f"{la}.in_proj", [qkv, gate, beta_info, alpha_info], [qkv_rows, z_rows, beta, alpha], hidden)
            conv = float_tensor(t(f"{blk}.ssm_conv1d")).float()
            conv = torch.cat([conv[:v_start], _untile(conv[v_start:], nk, g.value_head_dim)])
            yield f"{la}.conv1d.weight", conv.unsqueeze(1)
            yield f"{la}.dt_bias", _untile(float_tensor(t(f"{blk}.ssm_dt.bias")).float(), nk, 1)
            yield f"{la}.A_log", _untile(torch.log(-float_tensor(t(f"{blk}.ssm_a")).float()), nk, 1)
            yield f"{la}.norm.weight", float_tensor(t(f"{blk}.ssm_norm"))
            yield from fused(f"{la}.out_proj", [t(f"{blk}.ssm_out")], [matrix(t(f"{blk}.ssm_out"))], g.num_value_heads * g.value_head_dim)
        else:
            sa = f"{pre}.self_attn"
            parts = [t(f"{blk}.attn_{p}") for p in ("q", "k", "v")]
            yield from fused(f"{sa}.qkv_proj", parts, [matrix(p) for p in parts], hidden)
            out = t(f"{blk}.attn_output")
            yield from fused(f"{sa}.o_proj", [out], [matrix(out)], int(out.ne[0]))
            yield f"{sa}.q_norm.weight", norm(f"{blk}.attn_q_norm")
            yield f"{sa}.k_norm.weight", norm(f"{blk}.attn_k_norm")

        mlp = f"{pre}.mlp"
        if config.moe_enabled:
            yield f"{mlp}.gate.weight", float_tensor(t(f"{blk}.ffn_gate_inp"))
            sg, su, sd = (t(f"{blk}.ffn_{p}_shexp") for p in ("gate", "up", "down"))
            yield from fused(f"{mlp}.shared_expert.gate_up_proj", [sg, su], [matrix(sg), matrix(su)], hidden)
            yield from fused(f"{mlp}.shared_expert.down_proj", [sd], [matrix(sd)], config.shared_expert_intermediate_size)
            yield f"{mlp}.shared_expert_gate.weight", float_tensor(t(f"{blk}.ffn_gate_inp_shexp")).reshape(1, -1)
        else:
            fg, fu, fd = (t(f"{blk}.ffn_{p}") for p in ("gate", "up", "down"))
            yield from fused(f"{mlp}.gate_up_proj", [fg, fu], [matrix(fg), matrix(fu)], hidden)
            yield from fused(f"{mlp}.down_proj", [fd], [matrix(fd)], config.intermediate_size)


__all__ = ["iter_gguf_weights", "parse_gguf_config", "to_gguf"]
