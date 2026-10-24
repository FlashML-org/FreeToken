"""Qwen3-VL GGUFs: the text trunks of llama.cpp ``qwen3vl`` / ``qwen3vlmoe`` files, and the Qwen VL vision tower from a ``qwen3vl_merger`` mmproj (the HF config keys, the image processor config, and the tower weights).

The text trunk runs without its vision side: no tower and plain rope instead of interleaved mrope. llama.cpp's IMROPE keeps the standard full-spectrum theta (ggml.h: the dim index is not reset per section), so equal (t, h, w) text positions reduce to NeoX rope and q/k carry no permute, like the ``qwen3`` / ``qwen3moe`` GGUFs. The routed experts stack as ``ffn_*_exps`` because the converter splits HF's packed ``gate_up_proj`` / ``down_proj`` experts (conversion/qwen3vl.py).

llama.cpp splits the Conv3d patch embedding into one tensor per temporal slice (``v.patch_embd.weight`` and ``.weight.1``); everything else is stored in HF layout.
"""

from __future__ import annotations

import re
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, TYPE_CHECKING, Iterator

import torch

from freetoken.models.gguf.mmproj import Mmproj, mmproj_tensor
from freetoken.models.gguf.reader import GgufTensorInfo, gguf_tensor_index, load_gguf_metadata
from freetoken.models.gguf.tensors import check_scales, float_tensor, fused, matrix

from .config import parse_config

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim

# ----------------------------------------------------------------------------------------------
# Vision tower (a qwen3vl_merger mmproj)
# ----------------------------------------------------------------------------------------------

_PROJECTOR = "qwen3vl_merger"
_IMAGE_TOKEN = "<|image_pad|>"
# preprocessor_config.json of the HF releases; the mmproj does not store the pixel limits
_IMAGE_SIZE = {"shortest_edge": 65536, "longest_edge": 16777216}

_TOP = {
    "v.patch_embd.bias": "patch_embed.proj.bias",
    "v.position_embd.weight": "pos_embed.weight",
    "v.post_ln.weight": "merger.norm.weight",
    "v.post_ln.bias": "merger.norm.bias",
    "mm.0.weight": "merger.linear_fc1.weight",
    "mm.0.bias": "merger.linear_fc1.bias",
    "mm.2.weight": "merger.linear_fc2.weight",
    "mm.2.bias": "merger.linear_fc2.bias",
}
_BLOCK = {
    "attn_qkv": "attn.qkv",
    "attn_out": "attn.proj",
    "ffn_up": "mlp.linear_fc1",
    "ffn_down": "mlp.linear_fc2",
    "ln1": "norm1",
    "ln2": "norm2",
}
_BLOCK_RE = re.compile(r"^v\.blk\.(\d+)\.(\w+)\.(weight|bias)$")
_PATCH = ("v.patch_embd.weight", "v.patch_embd.weight.1")


def gguf_mm_config(text_md: dict[str, Any], mmproj: Mmproj, vocab_size: int, *, model_path: str = "") -> tuple[dict[str, Any], dict[str, Any]]:
    """The HF config.json keys and the preprocessor_config.json of the tower in ``mmproj``, for the text GGUF whose metadata is ``text_md``."""
    if mmproj.projector != _PROJECTOR:
        raise ValueError(f"mmproj {mmproj.path} is a {mmproj.projector!r} tower; Qwen VL needs {_PROJECTOR!r}")
    md, index = load_gguf_metadata(mmproj.path), gguf_tensor_index(mmproj.path)
    if any(md.get("clip.vision.is_deepstack_layers") or []):
        raise ValueError(f"mmproj {mmproj.path} is a Qwen3-VL tower with DeepStack layers; Qwen3.5 and later read none")

    def kv(key: str):
        return md[f"clip.vision.{key}"]

    arch = text_md["general.architecture"]
    text_hidden = int(text_md[f"{arch}.embedding_length"])
    if int(kv("projection_dim")) != text_hidden:
        raise ValueError(f"mmproj {mmproj.path} projects to {int(kv('projection_dim'))} but the text model is {text_hidden} wide")
    tokens = text_md["tokenizer.ggml.tokens"]
    if _IMAGE_TOKEN not in tokens:
        raise ValueError(f"the text GGUF tokenizer has no {_IMAGE_TOKEN} token")
    temporal = 2 if _PATCH[1] in index else 1
    vision_config = {
        "hidden_size": int(kv("embedding_length")),
        "depth": int(kv("block_count")),
        "num_heads": int(kv("attention.head_count")),
        "intermediate_size": int(kv("feed_forward_length")),
        "patch_size": int(kv("patch_size")),
        "temporal_patch_size": temporal,
        "spatial_merge_size": int(kv("spatial_merge_size")),
        "num_position_embeddings": int(index["v.position_embd.weight"].ne[1]),
        "out_hidden_size": text_hidden,
        # ggml order [kw, kh, in, out]
        "in_channels": int(index[_PATCH[0]].ne[2]),
        "deepstack_visual_indexes": [],
    }
    hf_fields = {
        "vision_config": vision_config,
        "image_token_id": tokens.index(_IMAGE_TOKEN),
        "text_config": {
            "vocab_size": vocab_size,
            "rope_parameters": {"mrope_section": [int(s) for s in text_md[f"{arch}.rope.dimension_sections"]][:3]},
        },
    }
    preprocessor = {
        "image_processor_type": "Qwen2VLImageProcessor",
        "size": dict(_IMAGE_SIZE),
        "patch_size": vision_config["patch_size"],
        "temporal_patch_size": temporal,
        "merge_size": vision_config["spatial_merge_size"],
        "image_mean": [float(v) for v in kv("image_mean")],
        "image_std": [float(v) for v in kv("image_std")],
    }
    return hf_fields, preprocessor


def _name(gguf_name: str) -> str:
    if gguf_name in _TOP:
        return _TOP[gguf_name]
    if (m := _BLOCK_RE.match(gguf_name)) and m[2] in _BLOCK:
        return f"blocks.{m[1]}.{_BLOCK[m[2]]}.{m[3]}"
    raise ValueError(f"unexpected tensor {gguf_name!r} in a {_PROJECTOR} mmproj")


def iter_mmproj_weights(mmproj: Mmproj) -> Iterator[tuple[str, torch.Tensor]]:
    """The tower state dict under ``visual.``, named as the HF checkpoints load."""
    index = gguf_tensor_index(mmproj.path)
    slices = [mmproj_tensor(index[n]) for n in _PATCH if n in index]
    yield "visual.patch_embed.proj.weight", torch.stack(slices, dim=2)
    for gguf_name, info in index.items():
        if gguf_name not in _PATCH:
            yield f"visual.{_name(gguf_name)}", mmproj_tensor(info)


# ----------------------------------------------------------------------------------------------
# Text trunk (a qwen3vl / qwen3vlmoe GGUF)
# ----------------------------------------------------------------------------------------------

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
    if leaf in ("mlp.experts.0.gate_proj", "mlp.experts.0.up_proj") and f"{blk}.ffn_gate_up_exps.weight" in tensors:
        return (f"{blk}.ffn_gate_up_exps",)
    tensor = _LAYER_LEAVES.get(leaf)
    return (f"{blk}.{tensor}",) if tensor is not None else ()


def _check_supported(shim: GgufConfigShim) -> None:
    if any(k.startswith("prism.hadamard.") for k in shim.metadata):
        raise NotImplementedError(f"{shim.model_path}: Hadamard-rotated (prism) GGUF weights are not supported")
    check_scales(shim.model_path)


def _hf_config(shim: GgufConfigShim) -> SimpleNamespace:
    """The HF config fields ``parse_config`` reads, from the GGUF metadata."""
    md = shim.metadata
    arch = shim.model_type  # the GGUF architecture: qwen3vl or qwen3vlmoe
    moe = arch.endswith("moe")

    def kv(key: str):
        return md[f"{arch}.{key}"]

    _check_supported(shim)
    rope = {"rope_type": "default", "rope_theta": float(kv("rope.freq_base"))}
    scaling = md.get(f"{arch}.rope.scaling.type", "none")
    if scaling == "yarn":
        rope.update(
            rope_type="yarn",
            factor=float(kv("rope.scaling.factor")),
            original_max_position_embeddings=int(kv("rope.scaling.original_context_length")),
        )
    elif scaling != "none":
        raise NotImplementedError(f"{shim.model_path}: rope scaling {scaling!r} is not supported")
    key_length = md.get(f"{arch}.attention.key_length")
    text = SimpleNamespace(
        model_type="qwen3_moe" if moe else "qwen3",
        architectures=list(shim.architectures),
        hidden_act="silu",
        hidden_size=int(kv("embedding_length")),
        num_hidden_layers=int(kv("block_count")),
        num_attention_heads=int(kv("attention.head_count")),
        num_key_value_heads=int(kv("attention.head_count_kv")),
        head_dim=int(key_length) if key_length is not None else None,
        rope_scaling=None if scaling == "none" else rope,
        rope_parameters=rope,
        max_position_embeddings=int(kv("context_length")),
        rms_norm_eps=float(kv("attention.layer_norm_rms_epsilon")),
        vocab_size=int(shim.vocab_size),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        intermediate_size=int(kv("feed_forward_length")),
    )
    if moe:
        text.num_experts = int(kv("expert_count"))
        text.num_experts_per_tok = int(kv("expert_used_count"))
        text.moe_intermediate_size = int(kv("expert_feed_forward_length"))
        # llama.cpp always renormalizes the top-k weights of a qwen3moe-style router
        text.norm_topk_prob = True
    else:
        text.num_experts = 0
    # text only: no vision_config, so the wrapper builds no tower and keeps 1-D rope
    return SimpleNamespace(
        text_config=text,
        model_type="qwen3_vl",
        architectures=list(shim.architectures),
        image_token_id=None,
        vision_config=None,
        tie_word_embeddings=bool(shim.tie_word_embeddings),
    )


def parse_gguf_config(shim: GgufConfigShim) -> ModelConfig:
    config = parse_config(_hf_config(shim))
    return replace(config, expert_quant="gguf") if config.num_experts else config


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

    text = _hf_config(cached_load_hf_config(model_path)).text_config
    index = gguf_tensor_index(model_path)
    hidden = text.hidden_size

    def t(name: str) -> GgufTensorInfo:
        return index[name if name in index else f"{name}.weight"]

    yield from fused("model.embed_tokens", [t("token_embd")], [matrix(t("token_embd"))], hidden)
    if not text.tie_word_embeddings:
        yield from fused("lm_head", [t("output")], [matrix(t("output"))], hidden)
    yield "model.norm.weight", float_tensor(t("output_norm"))

    for layer in range(text.num_hidden_layers):
        blk, pre = f"blk.{layer}", f"model.layers.{layer}"
        yield f"{pre}.input_layernorm.weight", float_tensor(t(f"{blk}.attn_norm"))
        yield f"{pre}.post_attention_layernorm.weight", float_tensor(t(f"{blk}.ffn_norm"))
        sa = f"{pre}.self_attn"
        # convert_hf_to_gguf.py --fuse-qkv stores q, k and v as one attn_qkv matrix
        qkv = [t(f"{blk}.attn_qkv")] if f"{blk}.attn_qkv.weight" in index else [t(f"{blk}.attn_{p}") for p in ("q", "k", "v")]
        yield from fused(f"{sa}.qkv_proj", qkv, [matrix(p) for p in qkv], hidden)
        yield f"{sa}.q_norm.weight", float_tensor(t(f"{blk}.attn_q_norm"))
        yield f"{sa}.k_norm.weight", float_tensor(t(f"{blk}.attn_k_norm"))
        out = t(f"{blk}.attn_output")
        yield from fused(f"{sa}.o_proj", [out], [matrix(out)], int(out.ne[0]))
        mlp = f"{pre}.mlp"
        if text.num_experts:
            yield f"{mlp}.gate.weight", float_tensor(t(f"{blk}.ffn_gate_inp"))
        else:
            gate, up, down = (t(f"{blk}.ffn_{p}") for p in ("gate", "up", "down"))
            yield from fused(f"{mlp}.gate_up_proj", [gate, up], [matrix(gate), matrix(up)], hidden)
            yield from fused(f"{mlp}.down_proj", [down], [matrix(down)], int(down.ne[0]))


__all__ = ["gguf_mm_config", "iter_gguf_weights", "iter_mmproj_weights", "parse_gguf_config", "to_gguf"]
