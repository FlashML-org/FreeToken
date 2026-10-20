from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view, e4m3_native_cx, e4m3_u8_to_f32

# E2M1 (NVFP4) value table indexed by the 4-bit code.
_E2M1_VALUES = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


@functools.lru_cache(maxsize=None)
def _e2m1_lut(device_index: int) -> torch.Tensor:
    return torch.tensor(
        _E2M1_VALUES, dtype=torch.float32, device=torch.device("cuda", device_index)
    )


@triton.jit
def _dequant_nvfp4_kernel(
    packed_ptr,      # [S, OUT, IN // 2] uint8
    scale_ptr,       # [S, OUT, IN // 16] fp8-e4m3 per-block scale (checkpoint native)
    global_ptr,      # [S, OUT] fp32 per-output-row global scale (weight_scale_2)
    slots_ptr,       # [N] int32 -> cache slot for each output expert
    out_ptr,         # [N, OUT, IN] compute dtype
    lut_ptr,         # [16] float32 E2M1 values
    OUT: tl.constexpr,
    IN: tl.constexpr,
    IN_PACKED: tl.constexpr,   # IN // 2
    NUM_BLOCKS: tl.constexpr,  # IN // 16
    BLOCK_BYTES: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_b = tl.program_id(1)

    n = pid_row // OUT
    out_idx = pid_row % OUT
    # Cache slots and per-row offsets are computed in int64: with a full cache (slots up to
    # a few thousand) or a large active set (prefill), ``slot * OUT * IN`` and ``pid_row * IN``
    # exceed int32 and would silently wrap to a negative address (illegal memory access).
    slot = tl.load(slots_ptr + n).to(tl.int64)
    row = slot * OUT + out_idx.to(tl.int64)

    # Per-tensor global scale is kept separate (folding it into the fp8 block scale would
    # underflow), and applied here in fp32: weight = fp4 * block_scale * global_scale.
    g = tl.load(global_ptr + row).to(tl.float32)

    byte_off = pid_b * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    byte_mask = byte_off < IN_PACKED
    byte_off = byte_off.to(tl.int64)

    packed_base = row * IN_PACKED
    bytes_ = tl.load(packed_ptr + packed_base + byte_off, mask=byte_mask, other=0).to(tl.int32)

    lo = bytes_ & 0xF
    hi = (bytes_ >> 4) & 0xF
    val_lo = tl.load(lut_ptr + lo)
    val_hi = tl.load(lut_ptr + hi)

    # The two nibbles in a byte (elements 2b, 2b+1) always share one 16-wide block.
    scale_idx = byte_off // 8
    scale_base = row * NUM_BLOCKS
    if e4m3_native_cx():
        scale = tl.load(scale_ptr + scale_base + scale_idx, mask=byte_mask, other=0.0).to(tl.float32)
    else:
        scale = e4m3_u8_to_f32(tl.load(scale_ptr + scale_base + scale_idx, mask=byte_mask, other=0))
    scale = scale * g

    val_lo = val_lo * scale
    val_hi = val_hi * scale

    out_base = pid_row.to(tl.int64) * IN
    tl.store(out_ptr + out_base + 2 * byte_off, val_lo.to(out_ptr.dtype.element_ty), mask=byte_mask)
    tl.store(out_ptr + out_base + 2 * byte_off + 1, val_hi.to(out_ptr.dtype.element_ty), mask=byte_mask)


def dequant_nvfp4(
    packed_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    global_cache: torch.Tensor,
    slots: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize the cache rows selected by ``slots`` into a compact dense tensor.

    ``packed_cache`` is ``[S, OUT, IN//2]`` uint8 (two FP4 codes per byte, low nibble
    first), ``scale_cache`` is ``[S, OUT, IN//16]`` fp8-e4m3 per-block scales (the
    checkpoint's native format), and ``global_cache`` is ``[S, OUT]`` per-output-row
    global scale. The dequant is ``fp4 * block_scale * global_scale``. Returns ``[N, OUT, IN]``.
    """
    assert packed_cache.dtype == torch.uint8
    assert packed_cache.is_contiguous() and scale_cache.is_contiguous()
    assert global_cache.is_contiguous()
    S, OUT, IN_PACKED = packed_cache.shape
    IN = IN_PACKED * 2
    NUM_BLOCKS = scale_cache.shape[2]
    assert scale_cache.shape[:2] == (S, OUT)
    assert global_cache.shape == (S, OUT)
    assert NUM_BLOCKS == IN // 16
    N = slots.shape[0]
    if out is None:
        out = torch.empty((N, OUT, IN), dtype=dtype, device=packed_cache.device)
    else:
        assert out.shape == (N, OUT, IN)

    BLOCK_BYTES = 256
    grid = (N * OUT, triton.cdiv(IN_PACKED, BLOCK_BYTES))
    scale_cache = e4m3_kernel_view(scale_cache)
    _dequant_nvfp4_kernel[grid](
        packed_cache,
        scale_cache,
        global_cache,
        slots,
        out,
        _e2m1_lut(packed_cache.device.index),
        OUT=OUT,
        IN=IN,
        IN_PACKED=IN_PACKED,
        NUM_BLOCKS=NUM_BLOCKS,
        BLOCK_BYTES=BLOCK_BYTES,
    )
    return out


__all__ = ["dequant_nvfp4"]
