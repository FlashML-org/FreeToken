"""ggml type facts shared by the GGUF dialect, its methods and the GGUF weight readers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from freetoken.third_party import gguf

# ggml reads up to this many values past the end of a packed matrix (llama.cpp MATRIX_ROW_PADDING)
MATRIX_ROW_PADDING = 512

UNQUANTIZED = frozenset(
    int(t) for t in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16)
)


_FLOAT_DTYPES = {
    int(gguf.GGMLQuantizationType.F32): torch.float32,
    int(gguf.GGMLQuantizationType.F16): torch.float16,
    int(gguf.GGMLQuantizationType.BF16): torch.bfloat16,
}


def float_dtype(ggml_type: int) -> torch.dtype | None:
    """The torch dtype of an unquantized ggml type, None for a quantized one."""
    return _FLOAT_DTYPES.get(int(ggml_type))


def type_name(ggml_type: int) -> str:
    return gguf.GGMLQuantizationType(ggml_type).name


def is_quantized(ggml_type: int) -> bool:
    return int(ggml_type) not in UNQUANTIZED


def row_bytes(ggml_type: int, k: int) -> int:
    block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(ggml_type)]
    if k % block:
        raise ValueError(f"{type_name(ggml_type)} rows need a multiple of {block} values, got {k}")
    return k // block * size


def pad_rows(k: int) -> int:
    """Zero rows to append after a packed [rows, K] matrix so ggml's tail reads stay inside it."""
    return 0 if k % MATRIX_ROW_PADDING == 0 else 1


def mmq_tail_bytes(ggml_type: int, k: int) -> int:
    """Bytes past the last row of a packed [rows, K] matrix that MMQ reads and that must read as zeros."""
    rem = k % MATRIX_ROW_PADDING
    return 0 if rem == 0 or not is_quantized(ggml_type) else row_bytes(ggml_type, MATRIX_ROW_PADDING - rem)


@dataclass(frozen=True)
class Segment:
    """A run of consecutive fused parts that share one ggml type: one packed matrix and one GEMM."""

    ggml_type: int
    first_part: int
    num_parts: int
    out_start: int
    rows: int


def segments(types: tuple[int, ...], output_sizes: tuple[int, ...]) -> list[Segment]:
    if len(types) != len(output_sizes):
        raise ValueError(f"{len(types)} ggml types for {len(output_sizes)} fused parts")
    out: list[Segment] = []
    start = 0
    for i, (t, n) in enumerate(zip(types, output_sizes)):
        if out and out[-1].ggml_type == t:
            last = out[-1]
            out[-1] = Segment(t, last.first_part, last.num_parts + 1, last.out_start, last.rows + n)
        else:
            out.append(Segment(int(t), i, 1, start, n))
        start += n
    return out


def segment_param(index: int) -> str:
    """State-dict leaf of segment ``index``: ``weight`` for the first, then ``weight_1``, ``weight_2``."""
    return "weight" if index == 0 else f"weight_{index}"


__all__ = ["MATRIX_ROW_PADDING", "Segment", "float_dtype", "is_quantized", "mmq_tail_bytes", "pad_rows", "row_bytes", "segment_param", "segments", "type_name"]
