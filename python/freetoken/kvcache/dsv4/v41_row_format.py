"""RowFormat: how one DeepSeek-V4.1 KV row (a head_dim-wide vector) is stored in a packed byte pool.

DeepSeek-V4.1 keeps three cache tiers in three storage formats (tech report sec. 2.4.4 and
``inference/model.py``):

* SWA KV      -- ``fp8_e8m0_b32``: e4m3 values, one ue8m0 (power-of-two) scale per 32 channels
* main KV     -- ``fp4_e4m3_b16``: e2m1 values, one e4m3 scale per 16 channels (NVFP4 without the global scale)
* indexer K   -- ``fp4_e8m0_b32``: e2m1 values, one ue8m0 scale per 32 channels (OCP MXFP4)

A row is laid out as ``[values | scales]``: ``value_bytes(dim)`` bytes of (packed) values followed by
``scale_bytes(dim)`` scale bytes, so one gather fetches both. e2m1 pairs pack the even channel into the low
nibble and the odd channel into the high nibble (``torch.float4_e2m1fn_x2`` / ``convert.py`` order).

The Triton side (``kernel/triton/dsv41/row_format.py``) selects the dequant path from ``RowFormat.code``
(a constexpr); the pack / unpack wrappers in ``kernel/triton/dsv41/pack.py`` reproduce the reference
``act_quant`` / ``fp4_act_quant`` numerics exactly, including the bf16 rounding of the dequantized value
the reference bakes into its cache.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RowFormat:
    name: str
    elem_bits: int  # 16 (bf16), 8 (e4m3) or 4 (e2m1)
    group: int  # channels per scale; 0 = unscaled
    scale: str | None  # None | "e8m0" | "e4m3"
    code: int  # Triton constexpr selector

    def value_bytes(self, dim: int) -> int:
        assert (dim * self.elem_bits) % 8 == 0, (dim, self.elem_bits)
        return dim * self.elem_bits // 8

    def scale_bytes(self, dim: int) -> int:
        if not self.group:
            return 0
        assert dim % self.group == 0, (dim, self.group)
        return dim // self.group

    def row_bytes(self, dim: int) -> int:
        return self.value_bytes(dim) + self.scale_bytes(dim)

    def num_groups(self, dim: int) -> int:
        return dim // self.group if self.group else 0

    def validate_dim(self, dim: int) -> None:
        if dim <= 0 or (dim * self.elem_bits) % 8 or (self.group and dim % self.group):
            raise ValueError(f"row format {self.name} cannot hold a {dim}-wide row")


BF16 = RowFormat("bf16", elem_bits=16, group=0, scale=None, code=0)
FP8_E8M0_B32 = RowFormat("fp8_e8m0_b32", elem_bits=8, group=32, scale="e8m0", code=1)
FP4_E4M3_B16 = RowFormat("fp4_e4m3_b16", elem_bits=4, group=16, scale="e4m3", code=2)
FP4_E8M0_B32 = RowFormat("fp4_e8m0_b32", elem_bits=4, group=32, scale="e8m0", code=3)

ROW_FORMATS: dict[str, RowFormat] = {f.name: f for f in (BF16, FP8_E8M0_B32, FP4_E4M3_B16, FP4_E8M0_B32)}


def row_format(name: str) -> RowFormat:
    try:
        return ROW_FORMATS[name]
    except KeyError:
        raise ValueError(f"unknown row format {name!r}; one of {sorted(ROW_FORMATS)}") from None


__all__ = ["RowFormat", "BF16", "FP8_E8M0_B32", "FP4_E4M3_B16", "FP4_E8M0_B32", "ROW_FORMATS", "row_format"]
