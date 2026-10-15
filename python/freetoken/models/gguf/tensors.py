"""GGUF tensors as state-dict entries: float tensors in torch order, quantized ones as packed ggml rows grouped into the GGUF Linear's per-type segments."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch

from freetoken.layers.quantization.ggml_types import is_quantized, pad_rows, segment_param, segments

from .reader import GgufTensorInfo

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


def fused(prefix: str, infos: list[GgufTensorInfo], parts: list[torch.Tensor], k: int, pad_to: int = 0) -> Iterator[tuple[str, torch.Tensor]]:
    """One fused module: float parts concatenated into ``weight``; with any quantized part, one matrix per ggml-type run (float runs as bf16 rows, quantized runs packed)."""
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
            zeros += pad_rows(k)
            dtype = torch.uint8
        else:
            rows = [r.to(torch.bfloat16) for r in rows]
            dtype = torch.bfloat16
        tail = [torch.zeros(zeros, rows[0].shape[1], dtype=dtype)] if zeros else []
        yield f"{prefix}.{segment_param(i)}", torch.cat([*rows, *tail], dim=0)


__all__ = ["float_tensor", "fused", "matrix", "packed_rows"]
