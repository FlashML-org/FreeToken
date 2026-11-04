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
from freetoken.models.gguf.mmproj import find_mmproj, log_mmproj
from freetoken.models.gguf.reader import GgufTensorInfo, gguf_tensor_index
from freetoken.models.gguf.tensors import check_scales, float_tensor, fused, matrix
from freetoken.models.qwen3_vl.gguf import gguf_mm_config, iter_mmproj_weights

from .config import parse_config

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.moe.expert_banks import ExpertBanks
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
# the MTP draft's layer reads the block after the trunk: the same leaves under mtp.layers.0
_MTP_LAYER_RE = re.compile(r"^mtp\.layers\.0\.(.+)$")


def _block_tensors(blk: str, leaf: str, tensors: dict[str, int]) -> tuple[str, ...]:
    """The GGUF tensors (without ``.weight``) of one block's module leaf; stacked experts fuse gate|up when the file stores them fused."""
    if leaf in ("mlp.experts.0.gate_proj", "mlp.experts.0.up_proj") and f"{blk}.ffn_gate_up_exps.weight" in tensors:
        return (f"{blk}.ffn_gate_up_exps",)
    return tuple(f"{blk}.{t}" for t in _LAYER_LEAVES.get(leaf, ()))


def _nextn_blk(tensors: dict[str, int]) -> str | None:
    """The block whose ``nextn`` tensors the MTP draft reads (``blk.64``), or None in a file converted without them."""
    return next((name.split(".nextn.")[0] for name in tensors if ".nextn.eh_proj.weight" in name), None)


def to_gguf(name: str, tensors: dict[str, int]) -> tuple[str, ...]:
    """Checkpoint module name -> the GGUF tensors (without ``.weight``) that store it; empty for modules the GGUF has no matrix for."""
    if name in _TOP_LEAVES:
        return _TOP_LEAVES[name]
    if name == "mtp.fc":
        nextn = _nextn_blk(tensors)
        return () if nextn is None else (f"{nextn}.nextn.eh_proj",)
    m = _MTP_LAYER_RE.match(name)
    if m is not None:
        nextn = _nextn_blk(tensors)
        return () if nextn is None else _block_tensors(nextn, m.group(1), tensors)
    m = _LAYER_RE.match(name)
    if m is None:
        return ()
    return _block_tensors(f"blk.{m.group(1)}", m.group(2), tensors)


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
        vision_config=getattr(shim, "vision_config", None),
        image_token_id=getattr(shim, "image_token_id", None),
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


def _iter_block(
    t,
    norm,
    blk: str,
    pre: str,
    config: ModelConfig,
    layer_id: int,
) -> Iterator[tuple[str, torch.Tensor]]:
    """One decoder block (attention, MLP) as ``{pre}.*`` state-dict entries; the trunk loop and the nextn draft reader share it."""
    hidden = config.hidden_size
    yield f"{pre}.input_layernorm.weight", norm(f"{blk}.attn_norm")
    yield f"{pre}.post_attention_layernorm.weight", norm(f"{blk}.post_attention_norm")
    if config.is_linear_layer(layer_id):
        g = config.linear_attention_group()
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
        yield from _iter_block(t, norm, f"blk.{layer}", f"model.layers.{layer}", config, layer)

    if vision:
        yield from iter_mmproj_weights(find_mmproj(model_path))


# ----------------------------------------------------------------------------------------------
# The trunk's nextn (MTP) block: llama.cpp keeps it after the last decoder layer
# ----------------------------------------------------------------------------------------------

def nextn_block_index(model_path: str) -> int:
    """The trunk block that carries the nextn tensors; parse_gguf_config subtracts the same count from block_count."""
    from freetoken.models.gguf.reader import gguf_architecture, load_gguf_metadata

    md = load_gguf_metadata(model_path)
    arch = gguf_architecture(model_path)
    nextn = int(md.get(f"{arch}.nextn_predict_layers", 0))
    if nextn == 0:
        raise ValueError(
            f"{model_path}: no MTP (nextn) block to draft from: the trunk was converted without one "
            "(llama.cpp --no-mtp), or the file is a Strata-style MTP pack, which is not a draft "
            "source for this family"
        )
    if nextn > 1:
        raise ValueError(f"{model_path}: {nextn} nextn layers, but the MTP draft reads exactly one")
    return int(md[f"{arch}.block_count"]) - nextn


def iter_nextn_weights(model_path: str, config: ModelConfig) -> Iterator[tuple[str, torch.Tensor]]:
    """The nextn block as Qwen3_5MTPDraft names it: the block's tensors under the draft's state-dict names, its routed experts excluded (they load as resident banks)."""
    blk = f"blk.{nextn_block_index(model_path)}"
    index = gguf_tensor_index(model_path)
    if f"{blk}.nextn.eh_proj.weight" not in index:
        raise ValueError(f"{model_path}: {blk} has no nextn tensors")

    def t(name: str) -> GgufTensorInfo:
        key = name if name in index else f"{name}.weight"
        if key not in index:
            raise ValueError(f"{model_path}: the MTP (nextn) block lacks {key}; reconvert with a current llama.cpp")
        return index[key]

    def norm(name: str) -> torch.Tensor:
        return float_tensor(t(name)).float() - 1.0

    eh = t(f"{blk}.nextn.eh_proj")
    yield from fused("fc", [eh], [matrix(eh)], 2 * config.hidden_size)
    yield "pre_fc_norm_embedding.weight", norm(f"{blk}.nextn.enorm")
    yield "pre_fc_norm_hidden.weight", norm(f"{blk}.nextn.hnorm")
    yield "norm.weight", norm(f"{blk}.nextn.shared_head_norm")
    # the draft's one layer is full attention, whatever the trunk mixes around it
    yield from _iter_block(t, norm, blk, "layers.0", config, 0)


def nextn_expert_banks(model_path: str, device: torch.device) -> "ExpertBanks":
    """The nextn block's stacked experts as the draft's one resident ggml bank set."""
    from freetoken.models.gguf.experts import load_gguf_expert_banks

    return load_gguf_expert_banks(model_path, 1, first_layer=nextn_block_index(model_path), device=device)


def nextn_quant(model_path: str) -> "GGUFConfig":
    """The draft's quant over its own file's tensor table: the trunk's when the draft is the
    trunk's nextn block, the sidecar's when it is an mtp-*.gguf at another ggml mix."""
    from freetoken.models.gguf.config import GGUF_ARCH_TO_REGISTRY
    from freetoken.models.gguf.reader import gguf_architecture
    from freetoken.models.register import checkpoint_quant_config, get_model_spec

    # the HF and GGUF rows share one model class, so the target's class name resolves to the
    # HF row and its from_hf branch; the file's own architecture names the GGUF row instead
    spec = get_model_spec(GGUF_ARCH_TO_REGISTRY[gguf_architecture(model_path)])
    return checkpoint_quant_config(model_path, SimpleNamespace(model_path=model_path), spec)


__all__ = ["gguf_mm_config", "iter_gguf_weights", "iter_nextn_weights", "nextn_block_index", "nextn_expert_banks", "parse_gguf_config", "to_gguf"]
