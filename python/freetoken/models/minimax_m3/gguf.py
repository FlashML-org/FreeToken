"""MiniMax-M3 from a llama.cpp ``minimax-m3`` GGUF: config from the GGUF metadata, tensors by name; the routed experts load as GGUF expert banks (``freetoken.models.gguf.experts``).

The llama.cpp converter (conversion/minimax.py) stacks the w1 / w2 / w3 experts, stores the indexer projections as F32, and adds 1 to every RMSNorm weight so llama.cpp can run plain RMSNorm; the loader subtracts the 1 for this family's (1 + w) norms.
Both sides run NeoX rope, so q/k carry no rope permute.

A ``minimax_m3`` mmproj beside the text file adds the vision tower under ``vision_tower.``. The converter splits the Conv3d patch embedding into its two temporal halves (``v.patch_embd.weight`` and ``.weight.1``) and reorders the rows of each ViT head's q / k from HF's 3-axis rope layout [t h w | t h w | pad] to [t t | h h | w w | pad]; both are undone at load, every other tower tensor is stored in HF layout.
"""

from __future__ import annotations

import re
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
import torch

from freetoken.models.gguf.mmproj import Mmproj, find_mmproj, log_mmproj, mmproj_tensor
from freetoken.models.gguf.reader import (
    GgufTensorInfo,
    gguf_tensor_index,
    load_gguf_metadata,
)
from freetoken.models.gguf.tensors import float_tensor, fused, matrix

from .config import IMAGE_END_ID, IMAGE_START_ID, parse_config

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim

_ARCH = "minimax-m3"
_LAYER_LEAVES = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "self_attn.index_q_proj": "indexer.q_proj",
    "self_attn.index_k_proj": "indexer.k_proj",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
    "block_sparse_moe.gate": "ffn_gate_inp",
    "block_sparse_moe.shared_experts.gate_proj": "ffn_gate_shexp",
    "block_sparse_moe.shared_experts.up_proj": "ffn_up_shexp",
    "block_sparse_moe.shared_experts.down_proj": "ffn_down_shexp",
    "block_sparse_moe.experts.0.w1": "ffn_gate_exps",
    "block_sparse_moe.experts.0.w3": "ffn_up_exps",
    "block_sparse_moe.experts.0.w2": "ffn_down_exps",
}
_TOP_LEAVES = {"model.embed_tokens": "token_embd", "lm_head": "output"}
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_SIGMOID_GATING = 2
# llama.cpp hard-codes these for the arch (src/llama-graph.cpp LLM_FFN_SWIGLU_OAI_MOE)
_SWIGLU_ALPHA, _SWIGLU_LIMIT = 1.702, 7.0


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
    if any(k.startswith("prism.hadamard.") for k in shim.metadata):
        raise NotImplementedError(f"{shim.model_path}: Hadamard-rotated (prism) GGUF weights are not supported")
    scaled = sorted(n for n in gguf_tensor_index(shim.model_path) if n.endswith((".scale", ".input_scale")))
    if scaled:
        raise NotImplementedError(f"{shim.model_path}: GGUF tensors with per-tensor scales are not supported (e.g. {scaled[0]})")


def _hf_config(shim: GgufConfigShim) -> SimpleNamespace:
    """The HF text config fields ``parse_config`` reads, from the GGUF metadata."""
    md = shim.metadata

    def kv(key: str):
        return md[f"{_ARCH}.{key}"]

    _check_supported(shim)
    if f"{_ARCH}.attention.indexer.head_count" not in md:
        raise NotImplementedError(f"{shim.model_path}: converted by an older llama.cpp without the sparse-attention indexer, which current llama.cpp refuses too")
    gating = int(kv("expert_gating_func"))
    if gating != _SIGMOID_GATING:
        raise NotImplementedError(f"{shim.model_path}: expert gating function {gating} is not supported; MiniMax-M3 routes with a sigmoid (2)")
    if not md.get(f"{_ARCH}.expert_weights_norm", False):
        raise NotImplementedError(f"{shim.model_path}: routing without renormalized expert weights is not supported")
    scaling = md.get(f"{_ARCH}.rope.scaling.type", "none")
    if scaling != "none":
        raise NotImplementedError(f"{shim.model_path}: rope scaling {scaling!r} is not supported")
    expert_ff = kv("expert_feed_forward_length")
    if isinstance(expert_ff, list):
        if len(set(expert_ff)) != 1:
            raise NotImplementedError(f"{shim.model_path}: per-layer expert sizes {sorted(set(expert_ff))} are not supported")
        expert_ff = expert_ff[0]
    num_layers = int(kv("block_count"))
    dense = int(md.get(f"{_ARCH}.leading_dense_block_count", 0))
    # llama.cpp runs sparse attention and the MoE FFN on every layer after the leading dense ones
    sparse = [0] * dense + [1] * (num_layers - dense)
    return SimpleNamespace(
        model_type="minimax_m3",
        architectures=list(shim.architectures),
        # filled by a minimax_m3 mmproj (gguf_mm_config); a text-only process nulls vision_config on its copy
        vision_config=getattr(shim, "vision_config", None),
        projector_hidden_size=getattr(shim, "projector_hidden_size", None),
        image_token_id=getattr(shim, "image_token_id", None),
        hidden_act="swigluoai",
        swiglu_alpha=_SWIGLU_ALPHA,
        swiglu_limit=_SWIGLU_LIMIT,
        hidden_size=int(kv("embedding_length")),
        num_hidden_layers=num_layers,
        num_attention_heads=int(kv("attention.head_count")),
        num_key_value_heads=int(kv("attention.head_count_kv")),
        head_dim=int(kv("attention.key_length")),
        rotary_dim=int(kv("rope.dimension_count")),
        rope_theta=float(kv("rope.freq_base")),
        max_position_embeddings=int(kv("context_length")),
        rms_norm_eps=float(kv("attention.layer_norm_rms_epsilon")),
        vocab_size=int(shim.vocab_size),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        dense_intermediate_size=int(kv("feed_forward_length")),
        intermediate_size=int(expert_ff),
        shared_intermediate_size=int(expert_ff),
        n_shared_experts=int(kv("expert_shared_count")),
        num_local_experts=int(kv("expert_count")),
        num_experts_per_tok=int(kv("expert_used_count")),
        # llama.cpp skips the scale when the key is absent or 0
        routed_scaling_factor=float(md.get(f"{_ARCH}.expert_weights_scale", 0.0)) or 1.0,
        use_routing_bias=True,
        moe_layer_freq=sparse,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": sparse,
            "sparse_num_index_heads": int(kv("attention.indexer.head_count")),
            "sparse_index_dim": int(kv("attention.indexer.key_length")),
            "sparse_topk_blocks": int(kv("attention.indexer.top_k")),
            "sparse_block_size": int(kv("attention.indexer.block_size")),
            "sparse_local_block": int(kv("attention.indexer.local_blocks")),
            "sparse_init_block": 0,
            "sparse_score_type": "max",
        },
    )


def parse_gguf_config(shim: GgufConfigShim) -> ModelConfig:
    return replace(parse_config(_hf_config(shim)), expert_quant="gguf")


# ----------------------------------------------------------------------------------------------
# Vision tower (a minimax_m3 mmproj)
# ----------------------------------------------------------------------------------------------

_PROJECTOR = "minimax_m3"
_IMAGE_TOKEN = "]<]image[>["
_WRAPPERS = {"]<]start of image[>[": IMAGE_START_ID, "]<]end of image[>[": IMAGE_END_ID}
# no mmproj key carries the rope theta; llama.cpp hardcodes the HF value too (clip.cpp)
_ROPE_THETA = 10000.0
# MiniMaxM3VLImageProcessor's own size: the HF preprocessor_config.json stores a stale [672, 672] that the class replaces with this
_IMAGE_SIZE = {"shortest_edge": 3136, "longest_edge": 451584}
_PATCH = ("v.patch_embd.weight", "v.patch_embd.weight.1")
_TOWER = "vision_tower"

_TOP = {
    "v.pre_ln.weight": "vision_model.pre_layrnorm.weight",
    "v.pre_ln.bias": "vision_model.pre_layrnorm.bias",
    "mm.1.weight": "multi_modal_projector.linear_1.weight",
    "mm.1.bias": "multi_modal_projector.linear_1.bias",
    "mm.2.weight": "multi_modal_projector.linear_2.weight",
    "mm.2.bias": "multi_modal_projector.linear_2.bias",
    "mm.merger.fc1.weight": "patch_merge_mlp.linear_1.weight",
    "mm.merger.fc1.bias": "patch_merge_mlp.linear_1.bias",
    "mm.merger.fc2.weight": "patch_merge_mlp.linear_2.weight",
    "mm.merger.fc2.bias": "patch_merge_mlp.linear_2.bias",
}
_BLOCK = {
    "attn_out": "self_attn.out_proj",
    "ln1": "layer_norm1",
    "ln2": "layer_norm2",
    "ffn_up": "mlp.fc1",
    "ffn_down": "mlp.fc2",
}
_QKV = ("attn_q", "attn_k", "attn_v")
_BLOCK_RE = re.compile(r"^v\.blk\.(\d+)\.(\w+)\.(weight|bias)$")


def _f32_value(v: float) -> float:
    """The shortest decimal of a float32 metadata value: 1e-05, as the HF config writes it, not 9.99999975e-06."""
    return float(str(np.float32(v)))


def gguf_mm_config(text_md: dict[str, Any], mmproj: Mmproj, vocab_size: int, *, model_path: str = "") -> tuple[dict[str, Any], dict[str, Any]]:
    """The HF config.json keys and the preprocessor_config.json of the tower in ``mmproj``, for the text GGUF whose metadata is ``text_md``."""
    if mmproj.projector != _PROJECTOR:
        raise ValueError(f"mmproj {mmproj.path} is a {mmproj.projector!r} tower; MiniMax-M3 needs {_PROJECTOR!r}")
    md, index = load_gguf_metadata(mmproj.path), gguf_tensor_index(mmproj.path)

    def kv(key: str):
        return md[f"clip.vision.{key}"]

    text_hidden = int(text_md[f"{_ARCH}.embedding_length"])
    if int(kv("projection_dim")) != text_hidden:
        raise ValueError(f"mmproj {mmproj.path} projects to {int(kv('projection_dim'))} but the text model is {text_hidden} wide")
    tokens = text_md["tokenizer.ggml.tokens"]
    if _IMAGE_TOKEN not in tokens:
        raise ValueError(f"the text GGUF tokenizer has no {_IMAGE_TOKEN} token")
    # the processor hardcodes the wrapper ids; a tokenizer with them elsewhere would fail per request instead of here
    for token, want in _WRAPPERS.items():
        got = tokens.index(token) if token in tokens else None
        if got != want:
            raise ValueError(f"the text GGUF tokenizer has {token} at id {got}, not the family's {want}")
    # exactly the tensor names llama.cpp loads, both patch-embedding halves included; other spellings (mm.merge.*) fail there too
    unexpected = [n for n in index if n not in _PATCH and n not in _TOP and not ((m := _BLOCK_RE.match(n)) and m[2] in (*_BLOCK, *_QKV))]
    missing = [n for n in (*_PATCH, *_TOP) if n not in index]
    if unexpected or missing:
        raise ValueError(f"mmproj {mmproj.path} is not in the minimax_m3 tensor layout (unexpected {unexpected[:3]}, missing {missing[:3]})")
    merge = int(kv("spatial_merge_size"))
    vision_config = {
        # llama.cpp's minimax_m3 graph runs erf-GELU whatever clip.use_gelu says
        "hidden_act": "gelu",
        "hidden_size": int(kv("embedding_length")),
        "num_hidden_layers": int(kv("block_count")),
        "num_attention_heads": int(kv("attention.head_count")),
        "intermediate_size": int(kv("feed_forward_length")),
        # ggml order [kw, kh, in, out]
        "num_channels": int(index[_PATCH[0]].ne[2]),
        "patch_size": int(kv("patch_size")),
        "temporal_patch_size": 2,
        "spatial_merge_size": merge,
        "layer_norm_eps": _f32_value(kv("attention.layer_norm_epsilon")),
        "rope_theta": _ROPE_THETA,
    }
    hf_fields = {
        "vision_config": vision_config,
        # ggml order [in, out]
        "projector_hidden_size": int(index["mm.1.weight"].ne[1]),
        "image_token_id": tokens.index(_IMAGE_TOKEN),
    }
    preprocessor = {
        "image_processor_type": "MiniMaxM3VLImageProcessor",
        "size": dict(_IMAGE_SIZE),
        "patch_size": vision_config["patch_size"],
        "temporal_patch_size": 2,
        "merge_size": merge,
        "image_mean": [float(v) for v in kv("image_mean")],
        "image_std": [float(v) for v in kv("image_std")],
    }
    return hf_fields, preprocessor


def _hf_qk_rows(head_dim: int) -> torch.Tensor:
    """The mmproj row of each HF q / k row within one head (the inverse of conversion/minimax.py ``_permute_vit_qk``)."""
    band = (2 * (head_dim // 2) // 3) // 2
    stored = []
    for axis in range(3):
        stored += [*range(axis * band, (axis + 1) * band), *range((3 + axis) * band, (4 + axis) * band)]
    stored += range(6 * band, head_dim)
    return torch.argsort(torch.tensor(stored))


def iter_mmproj_weights(mmproj: Mmproj) -> Iterator[tuple[str, torch.Tensor]]:
    """The tower state dict under ``vision_tower.``, named as the HF checkpoint loads (q / k / v fused)."""
    md, index = load_gguf_metadata(mmproj.path), gguf_tensor_index(mmproj.path)
    heads = int(md["clip.vision.attention.head_count"])
    rows = _hf_qk_rows(int(md["clip.vision.embedding_length"]) // heads)

    def hf_qk(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(heads, rows.numel(), -1)[:, rows].reshape(t.shape)

    yield f"{_TOWER}.vision_model.embeddings.patch_embedding.weight", torch.stack([mmproj_tensor(index[n]) for n in _PATCH], dim=2)
    for name, info in index.items():
        if name in _PATCH:
            continue
        if name in _TOP:
            yield f"{_TOWER}.{_TOP[name]}", mmproj_tensor(info)
            continue
        # gguf_mm_config refused any other name
        layer, part, kind = _BLOCK_RE.match(name).groups()
        dst = f"{_TOWER}.vision_model.encoder.layers.{layer}"
        if part in _BLOCK:
            yield f"{dst}.{_BLOCK[part]}.{kind}", mmproj_tensor(info)
        elif part == "attn_q":
            q, k, v = (mmproj_tensor(index[f"v.blk.{layer}.{p}.{kind}"]) for p in _QKV)
            yield f"{dst}.self_attn.qkv.{kind}", torch.cat([hf_qk(q), hf_qk(k), v])


def iter_gguf_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """The dense state dict from the GGUF (and the tower from its mmproj); the routed experts load as GGUF expert banks (``freetoken.models.gguf.experts``)."""
    if not include_non_moe:
        return
    from freetoken.utils import cached_load_hf_config

    config = parse_gguf_config(cached_load_hf_config(model_path))
    vision = include_vision and config.is_multimodal
    log_mmproj(model_path, vision)
    args = config.m3_args
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
        blk, pre = f"blk.{layer}", f"model.layers.{layer}"
        yield f"{pre}.input_layernorm.weight", norm(f"{blk}.attn_norm")
        yield f"{pre}.post_attention_layernorm.weight", norm(f"{blk}.ffn_norm")
        sa = f"{pre}.self_attn"
        qkv = [t(f"{blk}.attn_{p}") for p in ("q", "k", "v")]
        yield from fused(f"{sa}.qkv_proj", qkv, [matrix(p) for p in qkv], hidden)
        out = t(f"{blk}.attn_output")
        yield from fused(f"{sa}.o_proj", [out], [matrix(out)], int(out.ne[0]))
        yield f"{sa}.q_norm.weight", norm(f"{blk}.attn_q_norm")
        yield f"{sa}.k_norm.weight", norm(f"{blk}.attn_k_norm")
        if args.is_sparse_layer(layer):
            iq, ik = t(f"{blk}.indexer.q_proj"), t(f"{blk}.indexer.k_proj")
            yield from fused(f"{sa}.index_qk_proj", [iq, ik], [matrix(iq), matrix(ik)], hidden)
            yield f"{sa}.index_q_norm.weight", norm(f"{blk}.indexer.q_norm")
            yield f"{sa}.index_k_norm.weight", norm(f"{blk}.indexer.k_norm")
        if layer in args.moe_layer_ids:
            moe = f"{pre}.block_sparse_moe"
            yield f"{moe}.gate.weight", float_tensor(t(f"{blk}.ffn_gate_inp")).float()
            yield f"{moe}.e_score_correction_bias", float_tensor(t(f"{blk}.exp_probs_b.bias")).float()
            mlp, parts = f"{moe}.shared_experts", [t(f"{blk}.ffn_{p}_shexp") for p in ("gate", "up", "down")]
        else:
            mlp, parts = f"{pre}.mlp", [t(f"{blk}.ffn_{p}") for p in ("gate", "up", "down")]
        gate, up, down = parts
        yield from fused(f"{mlp}.gate_up_proj", [gate, up], [matrix(gate), matrix(up)], hidden)
        yield from fused(f"{mlp}.down_proj", [down], [matrix(down)], int(down.ne[0]))

    if vision:
        yield from iter_mmproj_weights(find_mmproj(model_path))


__all__ = ["gguf_mm_config", "iter_gguf_weights", "iter_mmproj_weights", "parse_gguf_config", "to_gguf"]
