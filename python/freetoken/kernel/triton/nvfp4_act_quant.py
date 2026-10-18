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
def _quant_kernel(x_ptr, q_ptr, s_ptr, g_ptr, K, stride_x, BLOCK_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_K)
    x = tl.load(x_ptr + row * stride_x + offs, mask=offs < K, other=0.0).to(tl.float32)
    g = tl.load(g_ptr)
    xb = tl.reshape(x, [BLOCK_K // 16, 16])
    s = (tl.max(tl.abs(xb), 1) / (6.0 * g)).to(tl.float8e4nv)
    sf = s.to(tl.float32) * g
    inv = tl.where(sf > 0, 1.0 / sf, 0.0)
    lo, hi = tl.split(tl.reshape(xb * inv[:, None], [BLOCK_K // 2, 2]))
    oq = tl.arange(0, BLOCK_K // 2)
    tl.store(q_ptr + row * (K // 2) + oq, _e2m1x2(hi, lo).to(tl.uint8), mask=oq < K // 2)
    os_ = tl.arange(0, BLOCK_K // 16)
    tl.store(s_ptr + row * (K // 16) + os_, s.to(tl.uint8, bitcast=True), mask=os_ < K // 16)


def quantize_nvfp4(x: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x [R, K] -> (codes [R, K / 2] uint8, low nibble first; block scales [R, K / 16] e4m3) under the fp32 global g [1]; K % 16 == 0."""
    R, K = x.shape
    q = torch.empty(R, K // 2, dtype=torch.uint8, device=x.device)
    s = torch.empty(R, K // 16, dtype=torch.uint8, device=x.device)
    if R:
        _quant_kernel[(R,)](x, q, s, g, K, x.stride(0), BLOCK_K=triton.next_power_of_2(K), num_warps=4)
    return q, s.view(torch.float8_e4m3fn)
