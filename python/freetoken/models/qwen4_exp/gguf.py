"""Qwen3.8-Flash-Next from a llama.cpp ``qwen4exp`` GGUF: config from the GGUF metadata, tensor names, and the converter's transforms undone.

The llama.cpp converter (conversion/qwen.py, conversion/qwen4exp.py) stores every zero-centred norm as ``w + 1``, ``A_log`` as ``-exp(A_log)``, the squeezed conv kernels, the indexer projection split into q and k, and the GDN value heads in tiled order (k-head minor). Row reorders are undone on the packed bytes; ``ssm_out`` keeps its tiled columns and the GDN layer permutes its input instead (``Qwen4ExpArgs.gdn_out_heads_tiled``).
"""

from __future__ import annotations

import re
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.layers.quantization.ggml_types import is_quantized
from freetoken.models.gguf.mmproj import find_mmproj, log_mmproj
from freetoken.models.gguf.reader import GgufTensorInfo, gguf_tensor_index
from freetoken.models.gguf.tensors import check_scales, float_tensor, fused, matrix
from freetoken.models.qwen3_vl.gguf import gguf_mm_config, iter_mmproj_weights

from .config import parse_config

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim

_ARCH = "qwen4exp"

# HF config fields the GGUF metadata does not carry (Qwen3.8-Flash-Next's own config.json values)
_FAMILY_DEFAULTS = dict(
    hidden_act="silu",
    output_gate_type="sigmoid",
    indexer_kv_heads=1,
    ngram_vocab_size_base=20000000,
    make_ngram_vocab_size_divisible_by=128,
    split_ngram_parts=128,
    norm_topk_prob=True,
)

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
    "self_attn.indexer.index_q_proj": ("indexer.q_proj",),
    "self_attn.indexer.index_k_proj": ("indexer.k_proj",),
    "mlp.gate": ("ffn_gate_inp",),
    "mlp.shared_expert_gate": ("ffn_gate_inp_shexp",),
    "mlp.shared_expert.gate_proj": ("ffn_gate_shexp",),
    "mlp.shared_expert.up_proj": ("ffn_up_shexp",),
    "mlp.shared_expert.down_proj": ("ffn_down_shexp",),
    "mlp.experts.0.gate_proj": ("ffn_gate_exps",),
    "mlp.experts.0.up_proj": ("ffn_up_exps",),
    "mlp.experts.0.down_proj": ("ffn_down_exps",),
    "attn_hyper_connection.input_mix_weight_down": ("hc_attn_down",),
    "attn_hyper_connection.block_inject_weight": ("hc_attn_inject",),
    "attn_hyper_connection.input_mix_weight_up": ("hc_attn_up",),
    "mlp_hyper_connection.input_mix_weight_down": ("hc_ffn_down",),
    "mlp_hyper_connection.block_inject_weight": ("hc_ffn_inject",),
    "mlp_hyper_connection.input_mix_weight_up": ("hc_ffn_up",),
    "ple.key_proj": ("ple_key",),
    "ple.value_proj": ("ple_value",),
}
_TOP_LEAVES: dict[str, tuple[str, ...]] = {
    "model.language_model.embed_tokens": ("token_embd",),
    "lm_head": ("output",),
    "model.language_model.hyper_connection_mixer.input_mix_weight_down": ("output_hc_down",),
    "model.language_model.hyper_connection_mixer.input_mix_weight_up": ("output_hc_up",),
}
_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")


def to_gguf(name: str, tensors: dict[str, int]) -> tuple[str, ...]:
    """Checkpoint module name -> the GGUF tensors (without ``.weight``) that store it; empty for modules the GGUF has no matrix for."""
    if name in _TOP_LEAVES:
        return _TOP_LEAVES[name]
    m = _LAYER_RE.match(name)
    if m is None:
        return ()
    return tuple(f"blk.{m.group(1)}.{leaf}" for leaf in _LAYER_LEAVES.get(m.group(2), ()))


def parse_gguf_config(shim: GgufConfigShim) -> ModelConfig:
    # NVFP4 weight scales are unvalidated on this arch, so refuse them instead of risking wrong output
    check_scales(shim.model_path, nvfp4=False)
    # a metadata-only GGUF (an FTW dir's source_metadata.gguf) has no tensor table to check
    index = gguf_tensor_index(shim.model_path)
    if index and "blk.0.hc_attn_down.weight" not in index:
        raise NotImplementedError(f"{shim.model_path}: no decoder layer tensors (blk.0.hc_attn_down), e.g. an MTP-only draft GGUF")
    md = shim.metadata

    def kv(key: str):
        return md[f"{_ARCH}.{key}"]

    head_dim = int(kv("attention.key_length"))
    sections = [int(s) for s in kv("rope.dimension_sections")][:3]
    ratios = [int(r) for r in kv("attention.compress_ratios")]
    n_v = int(kv("ssm.time_step_rank"))
    text = SimpleNamespace(
        **_FAMILY_DEFAULTS,
        hidden_size=int(kv("embedding_length")),
        # block_count also counts the MTP layers a GGUF may carry after the decoder
        num_hidden_layers=int(kv("block_count")) - int(md.get(f"{_ARCH}.nextn_predict_layers", 0)),
        num_attention_heads=int(kv("attention.head_count")),
        num_key_value_heads=int(kv("attention.head_count_kv")),
        head_dim=head_dim,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": float(kv("rope.freq_base")),
            "partial_rotary_factor": int(kv("rope.dimension_count")) / head_dim,
            "mrope_section": sections,
            "mrope_interleaved": True,
        },
        max_position_embeddings=int(kv("context_length")),
        full_attention_interval=int(kv("full_attention_interval")),
        rms_norm_eps=float(kv("attention.layer_norm_rms_epsilon")),
        vocab_size=int(shim.vocab_size),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        num_experts=int(kv("expert_count")),
        num_experts_per_tok=int(kv("expert_used_count")),
        moe_intermediate_size=int(kv("expert_feed_forward_length")),
        shared_expert_intermediate_size=int(kv("expert_shared_feed_forward_length")),
        linear_num_key_heads=int(kv("ssm.group_count")),
        linear_num_value_heads=n_v,
        linear_key_head_dim=int(kv("ssm.state_size")),
        linear_value_head_dim=int(kv("ssm.inner_size")) // n_v,
        linear_conv_kernel_dim=int(kv("ssm.conv_kernel")),
        hc_count=int(kv("hyper_connection.count")),
        hc_lowrank=int(kv("hyper_connection.low_rank")),
        indexer_n_heads=int(kv("attention.indexer.head_count")),
        indexer_head_dim=int(kv("attention.indexer.key_length")),
        indexer_budget=int(kv("attention.indexer.top_k")),
        indexer_compress_ratio=max(ratios),
        ple_layer_ids=[int(i) + 1 for i in kv("ple.layers")],
        # the GGUF key is one n-gram head's row; HF counts all (ngram_size - 1) * heads_per_ngram heads
        ple_embed_dim=int(kv("embedding_length_per_layer_input")) * (int(kv("ple.ngram_size")) - 1) * int(kv("ple.heads_per_ngram")),
        ple_conv_kernel_size=int(kv("ple.conv_kernel")),
        ngram_size=int(kv("ple.ngram_size")),
        heads_per_ngram=int(kv("ple.heads_per_ngram")),
        eos_token_id=int(kv("ple.eos_token_id")),
    )
    # with an mmproj, the id the image processor places; ple.image_token_id names the same token for text-only files
    image_token_id = getattr(shim, "image_token_id", None)
    if image_token_id is None:
        image_token_id = md.get(f"{_ARCH}.ple.image_token_id")
    hf_like = SimpleNamespace(
        text_config=text,
        model_type="qwen4_exp",
        architectures=list(shim.architectures),
        image_token_id=None if image_token_id is None else int(image_token_id),
        vision_config=getattr(shim, "vision_config", None),
    )
    config = parse_config(hf_like)
    return replace(config, expert_quant="gguf", qwen4_args=replace(config.qwen4_args, gdn_out_heads_tiled=True))


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
    vision = include_vision and config.is_multimodal
    log_mmproj(model_path, vision)
    index = gguf_tensor_index(model_path)
    md = cached_load_hf_config(model_path).metadata
    g = config.linear_attention_group()
    hidden = config.hidden_size

    def t(name: str) -> GgufTensorInfo:
        return index[name if name in index else f"{name}.weight"]

    def norm(name: str) -> torch.Tensor:
        return float_tensor(t(name)).float() - 1.0

    def hc(prefix: str, src: str) -> Iterator[tuple[str, torch.Tensor]]:
        down, inject, up = (t(f"{src}_{p}") for p in ("down", "inject", "up"))
        yield from fused(f"{prefix}.input_mix_weight_down_block_inject", [down, inject], [matrix(down), matrix(inject)],
                         hidden * config.qwen4_args.hc_count, pad_to=16)
        yield from fused(f"{prefix}.input_mix_weight_up", [up], [matrix(up)], config.qwen4_args.hc_lowrank)
        yield f"{prefix}.hc_norm.weight", norm(f"{src}_norm")

    yield from fused("model.embed_tokens", [t("token_embd")], [matrix(t("token_embd"))], hidden)
    yield from fused("lm_head", [t("output")], [matrix(t("output"))], hidden)
    for proj, k in (("down", hidden * config.qwen4_args.hc_count), ("up", config.qwen4_args.hc_lowrank)):
        info = t(f"output_hc_{proj}")
        yield from fused(f"model.hyper_connection_mixer.input_mix_weight_{proj}", [info], [matrix(info)], k)
    yield "model.hyper_connection_mixer.hc_norm.weight", norm("output_hc_norm")

    for layer in range(config.num_layers):
        blk, pre = f"blk.{layer}", f"model.layers.{layer}"
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
            iq, ik = t(f"{blk}.indexer.q_proj"), t(f"{blk}.indexer.k_proj")
            yield from fused(f"{sa}.indexer.index_qk_proj", [iq, ik], [matrix(iq), matrix(ik)], hidden)
            yield f"{sa}.indexer.q_layernorm.weight", norm(f"{blk}.indexer.q_norm")
            yield f"{sa}.indexer.k_layernorm.weight", norm(f"{blk}.indexer.k_norm")

        mlp = f"{pre}.mlp"
        yield f"{mlp}.gate.weight", float_tensor(t(f"{blk}.ffn_gate_inp"))
        sg, su, sd = (t(f"{blk}.ffn_{p}_shexp") for p in ("gate", "up", "down"))
        yield from fused(f"{mlp}.shared_expert.gate_up_proj", [sg, su], [matrix(sg), matrix(su)], hidden)
        yield from fused(f"{mlp}.shared_expert.down_proj", [sd], [matrix(sd)], config.shared_expert_intermediate_size)
        yield f"{mlp}.shared_expert_gate.weight", float_tensor(t(f"{blk}.ffn_gate_inp_shexp")).reshape(1, -1)
        yield from hc(f"{pre}.attn_hyper_connection", f"{blk}.hc_attn")
        yield from hc(f"{pre}.mlp_hyper_connection", f"{blk}.hc_ffn")

        if layer in config.qwen4_args.ple_layer_ids:
            ple = f"{pre}.ple"
            for proj in ("key", "value"):
                info = t(f"{blk}.ple_{proj}")
                yield from fused(f"{ple}.{proj}_proj", [info], [matrix(info)], config.qwen4_args.ple_embed_dim)
            for n in ("key", "query", "conv"):
                yield f"{ple}.norm_{n}.weight", norm(f"{blk}.ple_norm_{n}")
            yield f"{ple}.conv1d.weight", float_tensor(t(f"{blk}.ple_conv1d")).float().unsqueeze(1)
            yield f"{ple}.ple_embedding.layer_multipliers", torch.tensor([int(x) for x in md[f"{_ARCH}.ple.layer_multipliers"]], dtype=torch.int64)
            yield f"{ple}.ple_embedding.ngram_heads_vocab_sizes", torch.tensor([int(x) for x in md[f"{_ARCH}.ple.head_vocab_sizes"]], dtype=torch.int64)
            yield f"{ple}.ple_embedding.ngram_heads_offsets", torch.tensor([int(x) for x in md[f"{_ARCH}.ple.head_offsets"]], dtype=torch.int64)

    if vision:
        yield from iter_mmproj_weights(find_mmproj(model_path))


def ple_table_info(model_path: str) -> GgufTensorInfo:
    return gguf_tensor_index(model_path)["per_layer_token_embd.weight"]


__all__ = ["gguf_mm_config", "iter_gguf_weights", "parse_gguf_config", "ple_table_info", "to_gguf"]
