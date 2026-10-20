"""GGUF tensors as state-dict entries: float tensors in torch order, quantized ones as packed ggml rows grouped into the GGUF Linear's per-type segments."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch

from freetoken.layers.quantization.ggml_types import NVFP4, is_quantized, pad_rows, scale_name, segment_param, segments

from .reader import GgufTensorInfo, gguf_tensor_index

_FLOAT_DTYPES = {0: torch.float32, 1: torch.float16, 30: torch.bfloat16}


def float_tensor(info: GgufTensorInfo) -> torch.Tensor:
    """A float tensor in torch order (ggml dims reversed)."""
    return torch.from_numpy(np.array(info.data())).view(_FLOAT_DTYPES[info.ggml_type]).view(*reversed(info.ne))


def packed_rows(info: GgufTensorInfo) -> torch.Tensor:
    """Packed ggml rows ``[rows, row_bytes]`` of a quantized tensor."""
    rows = int(np.prod(info.ne[1:]))
    return torch.from_numpy(info.data()).view(rows, info.nbytes // rows)


def matrix(info: GgufTensorInfo) -> torch.Tensor:
    return packed_rows(info) if is_quantized(info.ggml_type) else float_tensor(info)


def check_scales(model_path: str, *, nvfp4: bool = True, known: tuple[str, ...] = ()) -> None:
    """Refuse ``.scale`` / ``.input_scale`` tensors that nothing reads.

    With ``nvfp4``, an NVFP4 tensor's ``.scale`` (its weight_scale_2) is read and its ``.input_scale`` goes unused, as in llama.cpp.
    ``known`` lists tensor-name suffixes the family's loader reads itself."""
    types = {name: info.ggml_type for name, info in gguf_tensor_index(model_path).items()}
    unread = []
    for name in types:
        weight = name.removesuffix(".input_scale") if name.endswith(".input_scale") else name.removesuffix(".scale")
        if weight == name or name.endswith(known):
            continue
        if not (nvfp4 and types.get(f"{weight}.weight", types.get(weight)) == NVFP4):
            unread.append(name)
    if unread:
        what = "scale tensors outside NVFP4 weights" if nvfp4 else "tensors with per-tensor scales"
        raise NotImplementedError(f"{model_path}: GGUF {what} are not supported (e.g. {sorted(unread)[0]})")


def weight_scale(info: GgufTensorInfo) -> torch.Tensor | None:
    """The fp32 weight_scale_2 llama.cpp's converter writes next to an NVFP4 tensor as ``<name>.scale`` (one value, or one per expert); None if there is none."""
    if info.ggml_type != NVFP4:
        return None
    scale = gguf_tensor_index(info.root or info.path).get(scale_name(info.name))
    return None if scale is None else float_tensor(scale).float().reshape(-1)


def _row_scales(prefix: str, infos: list[GgufTensorInfo], parts: list[torch.Tensor], pad: int) -> torch.Tensor | None:
    scales = [weight_scale(i) for i in infos]
    if all(s is None for s in scales):
        return None
    rows = []
    for info, part, s in zip(infos, parts, scales):
        # llama.cpp reads a dense tensor's scale as one value
        if s is not None and s.numel() != 1:
            raise NotImplementedError(f"{prefix}: {info.name} has {s.numel()} scales, expected one")
        rows.append(torch.ones(part.shape[0]) if s is None else s.expand(part.shape[0]))
    return torch.cat([*rows, torch.ones(pad)])


def fused(prefix: str, infos: list[GgufTensorInfo], parts: list[torch.Tensor], k: int, pad_to: int = 0) -> Iterator[tuple[str, torch.Tensor]]:
    """One fused module: float parts concatenated into ``weight``; with any quantized part, one matrix per ggml-type run (float runs as bf16 rows, quantized runs packed).

    NVFP4 parts with a ``.scale`` add ``ggml_scale``: each part's scale on its output rows, fp32."""
    types = [i.ggml_type for i in infos]
    pad = (-sum(p.shape[0] for p in parts)) % pad_to if pad_to else 0
    if not any(is_quantized(t) for t in types):
        rows = [p.to(torch.bfloat16) for p in parts]
        if pad:
            rows.append(torch.zeros(pad, *rows[0].shape[1:], dtype=torch.bfloat16))
        yield f"{prefix}.weight", torch.cat(rows, dim=0)
        return
    segs = segments(tuple(types), tuple(p.shape[0] for p in parts))
    for i, seg in enumerate(segs):
        rows = parts[seg.first_part : seg.first_part + seg.num_parts]
        zeros = pad if i == len(segs) - 1 else 0
        if is_quantized(seg.ggml_type):
            zeros += pad_rows(seg.ggml_type, k)
            dtype = torch.uint8
        else:
            rows = [r.to(torch.bfloat16) for r in rows]
            dtype = torch.bfloat16
        tail = [torch.zeros(zeros, rows[0].shape[1], dtype=dtype)] if zeros else []
        yield f"{prefix}.{segment_param(i)}", torch.cat([*rows, *tail], dim=0)
    scales = _row_scales(prefix, infos, parts, pad)
    if scales is not None:
        yield f"{prefix}.ggml_scale", scales


__all__ = ["check_scales", "float_tensor", "fused", "matrix", "packed_rows", "weight_scale"]
