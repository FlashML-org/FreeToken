"""Muse Glimmer text from a llama.cpp ``muse-glimmer`` GGUF: config from the GGUF metadata, tensors by name.

The converter (conversion/muse_glimmer.py) adds 1 to the four norms of each block, permutes the rows of each q and k head for llama.cpp's interleaved rope, and writes constant q/k norms that carry ``qk_scale_factor``.
The loader subtracts the 1, undoes the permute, and reads ``qk_scale_factor`` back from the q norm.

A ``muse-glimmer`` mmproj beside the text file adds the vision tower (images only): its metadata and the family constants below become the HF vision keys and the image processor config, and its tensors load under ``vision_tower.``.
The converter permutes the vision q/k rows the same way (undone at load) and sums the two temporal slabs of the patch embedding into one; the loader splits the sum back over both slabs, which is exact for an image (one frame in both slabs) but not for video.
"""

from __future__ import annotations

import math
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
import torch

from freetoken.models.gguf.mmproj import Mmproj, find_mmproj, log_mmproj, mmproj_tensor
from freetoken.models.gguf.reader import GgufTensorInfo, gguf_tensor_index, load_gguf_metadata
from freetoken.models.gguf.tensors import float_tensor, fused, matrix

from .config import IMAGE_END_ID, IMAGE_START_ID, parse_config

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim

_ARCH = "muse-glimmer"
_LAYER_LEAVES = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.gate_proj": "attn_gate",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}
_QKV_LEAVES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
_TOP_LEAVES = {"model.embed_tokens": "token_embd", "lm_head": "output"}
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
# checkpoint norm -> GGUF norm; the GGUF stores each as weight + 1
_SHIFTED_NORMS = {
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "post_attention_norm",
    "pre_feedforward_layernorm": "ffn_norm",
    "post_feedforward_layernorm": "post_ffw_norm",
}


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


def _check_supported(shim: GgufConfigShim) -> None:
    index = gguf_tensor_index(shim.model_path)
    if not index:
        raise NotImplementedError(f"{shim.model_path}: a metadata-only muse-glimmer GGUF has no q norm to read qk_scale_factor from")
    if any(k.startswith("prism.hadamard.") for k in shim.metadata):
        raise NotImplementedError(f"{shim.model_path}: Hadamard-rotated (prism) GGUF weights are not supported")
    scaled = sorted(n for n in index if n.endswith((".scale", ".input_scale")))
    if scaled:
        raise NotImplementedError(f"{shim.model_path}: GGUF tensors with per-tensor scales are not supported (e.g. {scaled[0]})")


def _qk_scale_factor(model_path: str, num_layers: int) -> float:
    """The constant the converter wrote into every q norm; the k norms are all ones."""
    index = gguf_tensor_index(model_path)
    scale = float(float_tensor(index["blk.0.attn_q_norm.weight"])[0])
    for layer in range(num_layers):
        q, k = (float_tensor(index[f"blk.{layer}.attn_{p}_norm.weight"]).float() for p in ("q", "k"))
        if not (torch.all(q == scale) and torch.all(k == 1.0)):
            raise NotImplementedError(f"{model_path}: the q/k norms of layer {layer} are not the converter's constant qk_scale_factor / ones")
    return scale


def _hf_config(shim: GgufConfigShim) -> SimpleNamespace:
    """The HF text config fields ``parse_config`` reads, from the GGUF metadata."""
    md = shim.metadata

    def kv(key: str):
        return md[f"{_ARCH}.{key}"]

    _check_supported(shim)
    num_layers = int(kv("block_count"))
    # a bool per layer, or llama.cpp's period: every n-th layer is full attention, and 0 means none is
    pattern = md.get(f"{_ARCH}.attention.sliding_window_pattern", 4)
    if isinstance(pattern, list):
        swa = [bool(p) for p in pattern]
        if len(swa) != num_layers:
            raise ValueError(f"{shim.model_path}: sliding_window_pattern has {len(swa)} entries for {num_layers} layers")
    else:
        swa = [int(pattern) == 0 or i % int(pattern) < int(pattern) - 1 for i in range(num_layers)]
    base = float(kv("rope.freq_base"))
    swa_base = float(md.get(f"{_ARCH}.rope.freq_base_swa", base))
    return SimpleNamespace(
        model_type="muse_glimmer",
        architectures=list(shim.architectures),
        hidden_activation="silu",
        hidden_size=int(kv("embedding_length")),
        num_hidden_layers=num_layers,
        num_attention_heads=int(kv("attention.head_count")),
        num_key_value_heads=int(kv("attention.head_count_kv")),
        head_dim=int(kv("attention.key_length")),
        intermediate_size=int(kv("feed_forward_length")),
        max_position_embeddings=int(kv("context_length")),
        rms_norm_eps=float(kv("attention.layer_norm_rms_epsilon")),
        # llama.cpp hard-codes the post-norm eps and NoPE on the full layers (src/models/muse-glimmer.cpp)
        post_norm_eps=1e-8,
        layer_types=["sliding_attention" if s else "full_attention" for s in swa],
        layer_rope_theta=[swa_base if s else 0.0 for s in swa],
        sliding_window=int(kv("attention.sliding_window")),
        qk_scale_factor=_qk_scale_factor(shim.model_path, num_layers),
        # llama.cpp's default when the key is absent; 0 turns the cap off
        final_logit_softcapping=float(md.get(f"{_ARCH}.final_logit_softcapping", 30.0)) or None,
        output_multiplier=float(kv("logit_scale")),
        vocab_size=int(shim.vocab_size),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        # filled by a muse-glimmer mmproj; a text-only process nulls vision_config on its copy
        vision_config=getattr(shim, "vision_config", None),
        image_token_id=getattr(shim, "image_token_id", None),
        projector_hidden_act=getattr(shim, "projector_hidden_act", None),
        projector_hidden_size=getattr(shim, "projector_hidden_size", None),
        out_hidden_size=getattr(shim, "out_hidden_size", None),
    )


def parse_gguf_config(shim: GgufConfigShim) -> ModelConfig:
    return parse_config(_hf_config(shim))


def _unpermute(rows: torch.Tensor, heads: int) -> torch.Tensor:
    """Undo the converter's rope permute of one q or k matrix (``_unpermute_for_rope``); it moves whole rows, so packed ggml rows work too."""
    return rows.reshape(heads, rows.shape[0] // heads // 2, 2, *rows.shape[1:]).swapaxes(1, 2).reshape(rows.shape)


# ----------------------------------------------------------------------------------------------
# Vision tower (a muse-glimmer mmproj)
# ----------------------------------------------------------------------------------------------

_PROJECTOR = "muse-glimmer"
_IMAGE_TOKEN = "<|patch|>"
# not stored in the mmproj; llama.cpp hardcodes the same values (clip.cpp) and they match the HF release
_TEMPORAL_PATCH = 2
_ROPE_THETA = 10000.0
_MAX_IMAGE_TOKENS = 4096
# window attention except every 4th block and the last one
_FULL_EVERY = 4

_TOP = {
    "v.position_embd.weight": "patch_embedder.position_embedding_table.weight",
    "v.pre_ln.weight": "ln_pre.weight",
    "v.pre_ln.bias": "ln_pre.bias",
    "v.post_ln.weight": "ln_post.weight",
    "v.post_ln.bias": "ln_post.bias",
    "mm.0.weight": "adapter.fc1.weight",
    "mm.1.weight": "adapter.fc2.weight",
    "mm.2.weight": "projection.weight",
}
_BLOCK = {
    "attn_out": "attn.proj",
    "ffn_up": "mlp.fc1",
    "ffn_down": "mlp.fc2",
    "ln1": "norm1",
    "ln2": "norm2",
}


def gguf_mm_config(text_md: dict[str, Any], mmproj: Mmproj, vocab_size: int, *, model_path: str = "") -> tuple[dict[str, Any], dict[str, Any]]:
    """The HF config.json keys and the preprocessor_config.json of the tower in ``mmproj``, for the text GGUF whose metadata is ``text_md``."""
    if mmproj.projector != _PROJECTOR:
        raise ValueError(f"mmproj {mmproj.path} is a {mmproj.projector!r} tower; Muse-Glimmer needs {_PROJECTOR!r}")
    md, index = load_gguf_metadata(mmproj.path), gguf_tensor_index(mmproj.path)

    def kv(key: str):
        return md[f"clip.vision.{key}"]

    text_hidden = int(text_md[f"{_ARCH}.embedding_length"])
    if int(kv("projection_dim")) != text_hidden:
        raise ValueError(f"mmproj {mmproj.path} projects to {int(kv('projection_dim'))} but the text model is {text_hidden} wide")
    tokens = text_md["tokenizer.ggml.tokens"]
    if _IMAGE_TOKEN not in tokens:
        raise ValueError(f"the text GGUF tokenizer has no {_IMAGE_TOKEN} token")
    # the processor hardcodes the wrapper ids; a tokenizer that moved them would fail per request instead of here
    for token, want in (("<|image_start|>", IMAGE_START_ID), ("<|image_end|>", IMAGE_END_ID)):
        if token not in tokens or tokens.index(token) != want:
            raise ValueError(f"the text GGUF tokenizer does not carry {token} at the family's id {want}")
    hidden, patch, merge = int(kv("embedding_length")), int(kv("patch_size")), int(kv("spatial_merge_size"))
    # ggml order [kw, kh, in, out]: one summed slab, not the HF [out, temporal * in * kh * kw]
    if tuple(index["v.patch_embd.weight"].ne) != (patch, patch, 3, hidden):
        raise ValueError(f"mmproj {mmproj.path}: v.patch_embd.weight {tuple(index['v.patch_embd.weight'].ne)} is not the summed [p, p, 3, hidden] layout")
    # ggml order [hidden, rows]: a square table of side x side positions
    rows = int(index["v.position_embd.weight"].ne[1])
    side = math.isqrt(rows)
    if side * side != rows:
        raise ValueError(f"mmproj {mmproj.path}: the position table has {rows} rows, not a square")
    layers = int(kv("block_count"))
    vision_config = {
        "hidden_act": "gelu",
        "hidden_size": hidden,
        "intermediate_size": int(kv("feed_forward_length")),
        "num_hidden_layers": layers,
        "num_attention_heads": int(kv("attention.head_count")),
        "layer_types": ["full_attention" if (i + 1) % _FULL_EVERY == 0 or i == layers - 1 else "window_attention" for i in range(layers)],
        "patch_size": patch,
        "patch_temporal": _TEMPORAL_PATCH,
        "merge_size": merge,
        "pos_emb_height": side,
        "pos_emb_width": side,
        # stored as f32; the decimal is the HF value
        "layer_norm_eps": float(str(np.float32(kv("attention.layer_norm_epsilon")))),
        "rope_parameters": {"rope_theta": _ROPE_THETA, "rope_type": "default"},
    }
    # ggml order [in, out]: the adapter reads the pixel-shuffled width and writes the projector width
    adapter = index["mm.0.weight"].ne
    hf_fields = {
        "vision_config": vision_config,
        "image_token_id": tokens.index(_IMAGE_TOKEN),
        "projector_hidden_act": "gelu",
        "projector_hidden_size": int(adapter[1]),
        "out_hidden_size": int(adapter[0]),
    }
    preprocessor = {
        "image_processor_type": "MuseGlimmerImageProcessor",
        "do_convert_rgb": True,
        "do_resize": True,
        "resample": 1,  # LANCZOS
        "do_rescale": True,
        "rescale_factor": 1 / 255,
        "do_normalize": True,
        "image_mean": [float(v) for v in kv("image_mean")],
        "image_std": [float(v) for v in kv("image_std")],
        "patch_size": patch,
        "temporal_patch_size": _TEMPORAL_PATCH,
        "merge_size": merge,
        "max_image_tokens": _MAX_IMAGE_TOKENS,
    }
    return hf_fields, preprocessor


def iter_mmproj_weights(mmproj: Mmproj) -> Iterator[tuple[str, torch.Tensor]]:
    """The tower state dict under ``vision_tower.``, named as the HF checkpoint loads."""
    index = gguf_tensor_index(mmproj.path)
    heads = int(load_gguf_metadata(mmproj.path)["clip.vision.attention.head_count"])
    n_blocks = 1 + max(int(n.split(".")[2]) for n in index if n.startswith("v.blk."))

    # the converter stores W0 + W1 of the two temporal slabs, which is all an image needs (its frame fills both); the bf16 hi + lo split carries that f32 sum without rounding
    summed = mmproj_tensor(index["v.patch_embd.weight"]).float().flatten(1)
    hi = summed.bfloat16()
    lo = (summed - hi.float()).bfloat16()
    yield "vision_tower.patch_embedder.patch_embedding.weight", torch.cat([hi, lo], dim=1)
    for gguf_name, name in _TOP.items():
        yield f"vision_tower.{name}", mmproj_tensor(index[gguf_name])

    consumed = {"v.patch_embd.weight", *_TOP}
    for layer in range(n_blocks):
        blk, dst = f"v.blk.{layer}", f"vision_tower.layers.{layer}"
        for part in ("weight", "bias"):
            q, k, v = (mmproj_tensor(index[f"{blk}.attn_{p}.{part}"]) for p in "qkv")
            yield f"{dst}.attn.qkv.{part}", torch.cat([_unpermute(q, heads), _unpermute(k, heads), v], dim=0)
            consumed.update(f"{blk}.attn_{p}.{part}" for p in "qkv")
            for src, name in _BLOCK.items():
                yield f"{dst}.{name}.{part}", mmproj_tensor(index[f"{blk}.{src}.{part}"])
                consumed.add(f"{blk}.{src}.{part}")

    leftovers = sorted(set(index) - consumed)
    if leftovers:
        raise ValueError(f"mmproj {mmproj.path}: unexpected tensors {leftovers[:5]}")


def iter_gguf_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """The state dict from the GGUF, and the vision tower from its mmproj."""
    if not include_non_moe:
        return
    from freetoken.utils import cached_load_hf_config

    hf = _hf_config(cached_load_hf_config(model_path))
    vision = include_vision and hf.vision_config is not None
    log_mmproj(model_path, vision)
    index = gguf_tensor_index(model_path)
    hidden, heads, kv_heads = hf.hidden_size, hf.num_attention_heads, hf.num_key_value_heads
    q_rows, kv_rows = heads * hf.head_dim, kv_heads * hf.head_dim

    def t(name: str) -> GgufTensorInfo:
        return index[name if name in index else f"{name}.weight"]

    yield from fused("model.embed_tokens", [t("token_embd")], [matrix(t("token_embd"))], hidden)
    if not hf.tie_word_embeddings:
        yield from fused("lm_head", [t("output")], [matrix(t("output"))], hidden)
    yield "model.norm.weight", float_tensor(t("output_norm"))

    for layer in range(hf.num_hidden_layers):
        blk, pre = f"blk.{layer}", f"model.layers.{layer}"
        for norm, tensor in _SHIFTED_NORMS.items():
            yield f"{pre}.{norm}.weight", float_tensor(t(f"{blk}.{tensor}")).float() - 1.0
        sa = f"{pre}.self_attn"
        # convert_hf_to_gguf.py --fuse-qkv stores q, k and v as one attn_qkv matrix
        if f"{blk}.attn_qkv.weight" in index:
            qkv = [t(f"{blk}.attn_qkv")] * 3
            q, k, v = matrix(qkv[0]).split([q_rows, kv_rows, kv_rows])
        else:
            qkv = [t(f"{blk}.attn_{p}") for p in ("q", "k", "v")]
            q, k, v = (matrix(p) for p in qkv)
        yield from fused(f"{sa}.qkv_proj", qkv, [_unpermute(q, heads), _unpermute(k, kv_heads), v], hidden)
        gate = t(f"{blk}.attn_gate")
        yield from fused(f"{sa}.gate_proj", [gate], [matrix(gate)], hidden)
        out = t(f"{blk}.attn_output")
        yield from fused(f"{sa}.o_proj", [out], [matrix(out)], int(out.ne[0]))
        mlp = f"{pre}.mlp"
        gate, up, down = (t(f"{blk}.ffn_{p}") for p in ("gate", "up", "down"))
        yield from fused(f"{mlp}.gate_up_proj", [gate, up], [matrix(gate), matrix(up)], hidden)
        yield from fused(f"{mlp}.down_proj", [down], [matrix(down)], int(down.ne[0]))

    if vision:
        yield from iter_mmproj_weights(find_mmproj(model_path))


__all__ = ["gguf_mm_config", "iter_gguf_weights", "iter_mmproj_weights", "parse_gguf_config", "to_gguf"]
