"""The Qwen VL vision tower from a llama.cpp ``qwen3vl_merger`` mmproj: the HF config keys, the image processor config, and the tower weights.

llama.cpp splits the Conv3d patch embedding into one tensor per temporal slice (``v.patch_embd.weight`` and ``.weight.1``); everything else is stored in HF layout.
"""

from __future__ import annotations

import re
from typing import Any, Iterator

import torch

from freetoken.models.gguf.mmproj import Mmproj, mmproj_tensor
from freetoken.models.gguf.reader import gguf_tensor_index, load_gguf_metadata

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


def gguf_mm_config(text_md: dict[str, Any], mmproj: Mmproj, vocab_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
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


__all__ = ["gguf_mm_config", "iter_mmproj_weights"]
