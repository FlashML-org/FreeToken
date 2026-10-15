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
    """One fused module: float parts concatenated into ``weight``, quantized parts packed into one matrix per ggml-type run."""
    types = [i.ggml_type for i in infos]
    if not any(is_quantized(t) for t in types):
        rows = [p.to(torch.bfloat16) for p in parts]
        pad = (-sum(r.shape[0] for r in rows)) % pad_to if pad_to else 0
        if pad:
            rows.append(torch.zeros(pad, *rows[0].shape[1:], dtype=torch.bfloat16))
        yield f"{prefix}.weight", torch.cat(rows, dim=0)
        return
    for i, seg in enumerate(segments(tuple(types), tuple(p.shape[0] for p in parts))):
        rows = parts[seg.first_part : seg.first_part + seg.num_parts]
        tail = [torch.zeros(pad_rows(k), rows[0].shape[1], dtype=torch.uint8)] if pad_rows(k) else []
        yield f"{prefix}.{segment_param(i)}", torch.cat([*rows, *tail], dim=0)


__all__ = ["float_tensor", "fused", "matrix", "packed_rows"]
