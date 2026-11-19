"""RowFormat: how one DeepSeek-V4 / V4.1 KV row (a head_dim-wide vector) is stored in a packed byte pool.

DeepSeek-V4.1 keeps three cache tiers in three storage formats (tech report sec. 2.4.4 and
``inference/model.py``):

* SWA KV      -- ``fp8_e8m0_b32``: e4m3 values, one ue8m0 (power-of-two) scale per 32 channels
* main KV     -- ``fp4_e4m3_b16``: e2m1 values, one e4m3 scale per 16 channels (NVFP4 without the global scale)
* indexer K   -- ``fp4_e8m0_b32``: e2m1 values, one ue8m0 scale per 32 channels (OCP MXFP4)

DeepSeek-V4 stores its window and compressed KV as ``fp8_e8m0_b64_rope`` (e4m3 values, one ue8m0 scale per
64 channels, the last 64 (rope) channels kept as bf16) and its indexer K as ``fp4_e8m0_b32``.

A row is laid out as ``[values | rope | scales | pad]``: ``value_bytes(dim)`` bytes of (packed) values,
``2 * rope`` bytes of raw bf16 rope channels, ``scale_bytes(dim)`` scale bytes, then padding up to
``align``, so one gather fetches all of it. e2m1 pairs pack the even channel into the low nibble and the
odd channel into the high nibble (``torch.float4_e2m1fn_x2`` / ``convert.py`` order). A 512-wide
``fp8_e8m0_b64_rope`` row has the fields of vLLM's DeepSeek-V4 ``fp8_ds_mla`` record: 448 + 128 + 7 + 1 = 584 bytes.
vLLM puts a page's scale bytes after all its data rows; here each row keeps its own.

The Triton side (``kernel/triton/dsv4/row_format.py``) selects the dequant path from ``RowFormat.code``
(a constexpr); the pack / unpack wrappers in ``kernel/triton/dsv4/pack.py`` reproduce the reference
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
    rope: int = 0  # trailing channels stored unquantized as bf16
    align: int = 1  # row stride multiple

    def value_bytes(self, dim: int) -> int:
        assert ((dim - self.rope) * self.elem_bits) % 8 == 0, (dim, self.elem_bits)
        return (dim - self.rope) * self.elem_bits // 8

    def scale_bytes(self, dim: int) -> int:
        if not self.group:
            return 0
        assert (dim - self.rope) % self.group == 0, (dim, self.group)
        return (dim - self.rope) // self.group

    def row_bytes(self, dim: int) -> int:
        n = self.value_bytes(dim) + 2 * self.rope + self.scale_bytes(dim)
        return -(-n // self.align) * self.align

    def num_groups(self, dim: int) -> int:
        return (dim - self.rope) // self.group if self.group else 0

    def validate_dim(self, dim: int) -> None:
        nope = dim - self.rope
        if nope <= 0 or (nope * self.elem_bits) % 8 or (self.group and nope % self.group):
            raise ValueError(f"row format {self.name} cannot hold a {dim}-wide row")


BF16 = RowFormat("bf16", elem_bits=16, group=0, scale=None, code=0)
FP8_E8M0_B32 = RowFormat("fp8_e8m0_b32", elem_bits=8, group=32, scale="e8m0", code=1)
FP4_E4M3_B16 = RowFormat("fp4_e4m3_b16", elem_bits=4, group=16, scale="e4m3", code=2)
FP4_E8M0_B32 = RowFormat("fp4_e8m0_b32", elem_bits=4, group=32, scale="e8m0", code=3)
FP8_E8M0_B64_ROPE = RowFormat("fp8_e8m0_b64_rope", elem_bits=8, group=64, scale="e8m0", code=4, rope=64, align=8)

ROW_FORMATS: dict[str, RowFormat] = {f.name: f for f in (BF16, FP8_E8M0_B32, FP4_E4M3_B16, FP4_E8M0_B32, FP8_E8M0_B64_ROPE)}


def row_format(name: str) -> RowFormat:
    try:
        return ROW_FORMATS[name]
    except KeyError:
        raise ValueError(f"unknown row format {name!r}; one of {sorted(ROW_FORMATS)}") from None


__all__ = ["RowFormat", "BF16", "FP8_E8M0_B32", "FP4_E4M3_B16", "FP4_E8M0_B32", "FP8_E8M0_B64_ROPE", "ROW_FORMATS", "row_format"]
