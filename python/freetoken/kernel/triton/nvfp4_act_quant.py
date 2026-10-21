"""NVFP4 activation quantization under a static global scale: x ~= e2m1 * e4m3 block scale (per 16) * g."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _e2m1x2(hi, lo):
    # sm_100+ / sm_120a; cvt packs its first source into the high nibble
    return tl.inline_asm_elementwise(
        "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $1, $2; cvt.u16.u8 $0, t; }",
        "=h,r,r", [hi, lo], dtype=tl.uint16, is_pure=True, pack=1)


@triton.jit
def _quant_kernel(x_ptr, q_ptr, s_ptr, g_ptr, R, K, stride_x, k_tiles,
                  SWIZZLED: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    # with a static global every 16-wide block quantizes on its own, so a row splits over programs
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    row_ok = rows < R
    r64 = rows.to(tl.int64)
    x = tl.load(x_ptr + r64[:, None] * stride_x + cols[None, :], mask=row_ok[:, None] & (cols < K)[None, :], other=0.0)
    g = tl.load(g_ptr)
    xb = tl.reshape(x.to(tl.float32), [BLOCK_M, BLOCK_K // 16, 16])
    s = (tl.max(tl.abs(xb), 2) / (6.0 * g)).to(tl.float8e4nv)
    sf = s.to(tl.float32) * g
    inv = tl.where(sf > 0, 1.0 / sf, 0.0)
    lo, hi = tl.split(tl.reshape(xb * inv[:, :, None], [BLOCK_M, BLOCK_K // 2, 2]))
    oq = tl.program_id(1) * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
    tl.store(q_ptr + r64[:, None] * (K // 2) + oq[None, :], _e2m1x2(hi, lo).to(tl.uint8),
             mask=row_ok[:, None] & (oq < K // 2)[None, :])
    sc = tl.program_id(1) * (BLOCK_K // 16) + tl.arange(0, BLOCK_K // 16)
    if SWIZZLED:
        # cuBLAS / CUTLASS 128x4 block-scale atoms: [row tile, k tile, row % 32, (row % 128) // 32, col % 4]
        s_off = ((r64 // 128)[:, None] * k_tiles + (sc // 4)[None, :]) * 512 \
            + ((rows % 32) * 16 + (rows % 128) // 32 * 4)[:, None] + (sc % 4)[None, :]
    else:
        s_off = r64[:, None] * (K // 16) + sc[None, :]
    tl.store(s_ptr + s_off, s.to(tl.uint8, bitcast=True), mask=row_ok[:, None] & (sc < K // 16)[None, :])


def quantize_nvfp4(x: torch.Tensor, g: torch.Tensor, swizzled: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """x [R, K] -> (codes [R, K / 2] uint8, low nibble first; e4m3 block scales) under the fp32 global g [1]; K % 16 == 0.

    The scales are [R, K / 16], or with ``swizzled`` the flat cuBLAS / CUTLASS 128x4 layout that ``torch.nn.functional.scaled_mm`` takes,
    rows padded to 128 and K / 16 to 4 with zeros.
    """
    R, K = x.shape
    assert K % 16 == 0 and x.stride(1) == 1
    q = torch.empty(R, K // 2, dtype=torch.uint8, device=x.device)
    k_tiles = triton.cdiv(K // 16, 4)
    if swizzled:
        n = triton.cdiv(R, 128) * 128 * k_tiles * 4
        exact = R % 128 == 0 and (K // 16) % 4 == 0
        s = (torch.empty if exact else torch.zeros)(n, dtype=torch.uint8, device=x.device)
    else:
        s = torch.empty(R, K // 16, dtype=torch.uint8, device=x.device)
    if R:
        block_k = min(triton.next_power_of_2(K), 1024)
        block_m = max(1, 1024 // block_k)
        grid = (triton.cdiv(R, block_m), triton.cdiv(K, block_k))
        _quant_kernel[grid](x, q, s, g, R, K, x.stride(0), k_tiles, SWIZZLED=swizzled, BLOCK_M=block_m, BLOCK_K=block_k,
                            num_warps=4)
    return q, s.view(torch.float8_e4m3fn)
