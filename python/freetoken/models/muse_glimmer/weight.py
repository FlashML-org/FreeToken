from __future__ import annotations

from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.config import detect_compressed_tensors_nvfp4
from freetoken.models.loader import (
    CT_SCALE_SUFFIXES,
    ShardReader,
    ct_bf16_fuse,
    ct_nvfp4_fuse,
    iter_weight_files,
    nvfp4_parts_ct,
)
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

# The image path loads under the wrapper's vision_tower, adapter and projection included, so one prefix covers it.
_VISION_RENAMES = (
    ("model.vision_tower.", "vision_tower."),
    ("model.vision_adapter.", "vision_tower.adapter."),
    ("model.vision_projection.", "vision_tower.projection."),
)
_VISION_QKV_PARTS = (".attn.q_proj", ".attn.k_proj", ".attn.v_proj")

# Fused projections, concatenated on the output dim in this exact order to match the
# model's merged-linear splits. The attention gate rides the q/k/v fusion (it is computed
# from the same layer input); ``.self_attn.gate_proj`` and the SwiGLU ``.mlp.gate_proj``
# are disambiguated by the full suffix.
_FUSIONS: dict[str, tuple[str, ...]] = {
    ".self_attn.qkvg_proj": (
        ".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.gate_proj",
    ),
    ".mlp.gate_up_proj": (".mlp.gate_proj", ".mlp.up_proj"),
}


def _rename(raw_name: str) -> str:
    """HF text key -> FreeToken state-dict key."""
    if raw_name.startswith("model.language_model."):
        return "model." + raw_name[len("model.language_model.") :]
    if raw_name.startswith("language_model."):
        return "model." + raw_name[len("language_model.") :]
    return raw_name  # lm_head.weight


def _vision_name(raw_name: str) -> str | None:
    """FreeToken state-dict key of an image-path tensor, None for text keys."""
    for hf_prefix, prefix in _VISION_RENAMES:
        if raw_name.startswith(hf_prefix):
            return prefix + raw_name[len(hf_prefix) :]
    return None


def _vision_tensors(name: str, tensor: torch.Tensor, buf: dict) -> list[tuple[str, torch.Tensor]]:
    """What one image-path tensor yields: q/k/v weights and biases wait in buf and leave as one fused attn.qkv."""
    stem, _, leaf = name.rpartition(".")
    for idx, part in enumerate(_VISION_QKV_PARTS):
        if stem.endswith(part):
            key = f"{stem[: -len(part)]}.attn.qkv.{leaf}"
            slots = buf.setdefault(key, {})
            slots[idx] = tensor
            if len(slots) < len(_VISION_QKV_PARTS):
                return []
            del buf[key]
            return [(key, torch.cat([slots[i] for i in range(len(_VISION_QKV_PARTS))], dim=0))]
    return [(name, tensor)]


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    if not include_non_moe:
        return  # dense model: there is no experts-only pass
    if get_tp_info().size > 1:
        raise NotImplementedError("muse_glimmer weight loading currently supports TP=1 only")

    if detect_compressed_tensors_nvfp4(cached_load_hf_config(model_path)):
        yield from _iter_weights_compressed_tensors(model_path, device, include_vision)
        return

    fuse_buf: dict = {}
    vision_buf: dict = {}
    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading weights",
        disable=not get_tp_info().is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                vision = _vision_name(raw_name)
                if vision is not None:
                    if include_vision:
                        yield from _vision_tensors(vision, f.get_tensor(raw_name), vision_buf)
                    continue
                name = _rename(raw_name)
                tensor = f.get_tensor(raw_name)
                if name.endswith(".weight"):
                    emit = ct_bf16_fuse(name[: -len(".weight")], tensor, fuse_buf, _FUSIONS)
                    if emit is not None:
                        yield from emit
                        continue
                # All norms pass through raw: the decoder's centered (1+w) norms apply the
                # +1 at runtime (GemmaPlusOneRMSNorm), the final norm / qk norms need none.
                yield name, tensor

    assert not fuse_buf, f"Incomplete projection fusions: {list(fuse_buf.keys())}"
    assert not vision_buf, f"Incomplete vision qkv fusions: {list(vision_buf.keys())}"


def _iter_weights_compressed_tensors(
    model_path: str, device: torch.device, include_vision: bool
) -> Iterator[tuple[str, torch.Tensor]]:
    """Dense pass for the compressed-tensors NVFP4 checkpoint.

    Every text Linear (q/k/v/o, the attention gate, the MLP) is kept native NVFP4
    (W4A16): ``.weight`` (uint8 packed) + ``.weight_scale`` (fp8 block) + ``.weight_global``
    (fp16 per-row, the reciprocal of the stored quant-side global). q/k/v/gate fuse into
    ``qkvg_proj`` and gate/up into ``gate_up_proj`` on the output dim, each part keeping
    its own scales, so the fused FP4 weights are exact. Embeddings, norms, lm_head and the
    vision tower are bf16 (the checkpoint's ignore list). Scale lookups go through the
    shard-map reader: they can land in a different shard than their weight_packed."""
    nvfp4_buf: dict = {}
    vision_buf: dict = {}
    reader = ShardReader(model_path, device)
    try:
        for file in tqdm(
            reader.files(),
            desc="Loading compressed-tensors weights",
            disable=not get_tp_info().is_primary(),
        ):
            for raw_name in reader.names_in(file):
                if raw_name.endswith(CT_SCALE_SUFFIXES):
                    continue  # consumed with their weight_packed
                vision = _vision_name(raw_name)
                if vision is not None:
                    if include_vision:
                        yield from _vision_tensors(vision, reader.get_tensor(raw_name), vision_buf)
                    continue
                name = _rename(raw_name)
                if raw_name.endswith(".weight_packed"):
                    base = name[: -len(".weight_packed")]
                    parts = nvfp4_parts_ct(reader, raw_name[: -len(".weight_packed")])
                    emit = ct_nvfp4_fuse(base, parts, nvfp4_buf, _FUSIONS)
                    if emit is not None:
                        yield from emit
                    else:  # standalone: o_proj, down_proj
                        w, s, g, a = parts
                        yield base + ".weight", w
                        yield base + ".weight_scale", s
                        yield base + ".weight_global", g
                        if a is not None:
                            yield base + ".input_scale", a
                    continue
                yield name, reader.get_tensor(raw_name)
    finally:
        reader.close()

    assert not nvfp4_buf, f"Incomplete NVFP4 fusions: {list(nvfp4_buf.keys())}"
    assert not vision_buf, f"Incomplete vision qkv fusions: {list(vision_buf.keys())}"


__all__ = ["iter_weights"]
