from __future__ import annotations

from dataclasses import dataclass

import torch

ROW = -1  # one scale per row


@dataclass(frozen=True)
class KVFormat:
    """How one K or V row is stored in the KV pool; the encoding lives in the ``kv_rows.py`` arms of the same name."""

    name: str  # --kv-cache-dtype value and the kernels' KV_FMT
    elem_dtype: torch.dtype | None  # logical element type; None follows the compute dtype
    elem_store_dtype: torch.dtype | None  # dtype of the element tensor; None follows the compute dtype
    scale_dtype: torch.dtype | None = None  # logical scale type; None means no scale tensor
    scale_store_dtype: torch.dtype | None = None  # dtype of the scale tensor
    scale_group: int = ROW  # channels per scale

    def __post_init__(self) -> None:
        if (self.elem_dtype is None) != (self.elem_store_dtype is None):
            raise ValueError(f"{self.name}: elem_dtype and elem_store_dtype must be set together")
        if (self.scale_dtype is None) != (self.scale_store_dtype is None):
            raise ValueError(f"{self.name}: scale_dtype and scale_store_dtype must be set together")
        if self.scale_dtype is not None and self.scale_dtype.itemsize != self.scale_store_dtype.itemsize:
            raise ValueError(f"{self.name}: scale_store_dtype must have the size of scale_dtype")

    @property
    def quantized(self) -> bool:
        return self.elem_dtype is not None

    def elem_tensor_dtype(self, compute_dtype: torch.dtype) -> torch.dtype:
        return compute_dtype if self.elem_store_dtype is None else self.elem_store_dtype

    def elem_dim(self, head_dim: int) -> int:
        """Elements of the element tensor per row."""
        if self.elem_dtype is None:
            return head_dim
        row_bytes = head_dim * self.elem_dtype.itemsize
        if row_bytes % self.elem_store_dtype.itemsize:
            raise ValueError(f"{self.name} cannot store a {head_dim}-wide row")
        return row_bytes // self.elem_store_dtype.itemsize

    def scale_dim(self, head_dim: int) -> int:
        """Scales per row."""
        if self.scale_dtype is None:
            return 0
        if self.scale_group == ROW:
            return 1
        if head_dim % self.scale_group:
            raise ValueError(f"{self.name} needs head_dim divisible by {self.scale_group}, got {head_dim}")
        return head_dim // self.scale_group

    def row_bytes(self, head_dim: int, compute_dtype: torch.dtype, segments: int = 1) -> int:
        """Bytes of one K or V row split into ``segments`` separately scaled parts, its scales included."""
        elem_bytes = self.elem_dim(head_dim) * self.elem_tensor_dtype(compute_dtype).itemsize
        if self.scale_dtype is None:
            return elem_bytes
        return elem_bytes + segments * self.scale_dim(head_dim) * self.scale_store_dtype.itemsize


NONE = KVFormat("none", elem_dtype=None, elem_store_dtype=None)
FP8_PER_TOKEN_HEAD = KVFormat(
    "fp8_per_token_head",
    elem_dtype=torch.float8_e4m3fn,
    elem_store_dtype=torch.uint8,
    scale_dtype=torch.float32,
    scale_store_dtype=torch.float32,
)
INT8_PER_TOKEN_HEAD = KVFormat(
    "int8_per_token_head",
    elem_dtype=torch.int8,
    elem_store_dtype=torch.int8,
    scale_dtype=torch.float32,
    scale_store_dtype=torch.float32,
)

KV_FORMATS: dict[str, KVFormat] = {fmt.name: fmt for fmt in (NONE, FP8_PER_TOKEN_HEAD, INT8_PER_TOKEN_HEAD)}
# what a pool with per-row scale tensors and a reader built on kv_rows.py can declare
ROW_SCALE_FORMATS = frozenset({NONE.name, FP8_PER_TOKEN_HEAD.name, INT8_PER_TOKEN_HEAD.name})


def resolve_kv_format(kv_cache_dtype: str) -> KVFormat:
    """The format of a --kv-cache-dtype value; "auto" never quantizes."""
    if kv_cache_dtype == "auto":
        return NONE
    try:
        return KV_FORMATS[kv_cache_dtype]
    except KeyError:
        raise ValueError(f"unknown --kv-cache-dtype {kv_cache_dtype!r}; one of auto, {', '.join(KV_FORMATS)}") from None
