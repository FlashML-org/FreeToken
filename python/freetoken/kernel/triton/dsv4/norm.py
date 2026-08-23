"""Fused RMSNorm for DeepSeek-V4 decode.

The eager ``RMSNorm`` (``x.float(); x*rsqrt(mean(x^2)+eps); (w*x).to(dtype)``) is a
chain of ~5 elementwise/reduction launches, run ~5x per layer (q/kv/attn/ffn/
compressor norms). At bs=1 these are latency-bound; collapsing each into one kernel
(row in registers, single reduction) removes most of the decode "tail" of tiny ops.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_TL = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


@triton.jit
def _rmsnorm_kernel(
    x_ptr, w_ptr, out_ptr, M, D, eps,
    stride_xm, stride_om,
    BLOCK_D: tl.constexpr, HAS_W: tl.constexpr, compute_type: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / D
    y = x * tl.rsqrt(var + eps)
    if HAS_W:
        y = y * tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * stride_om + offs, y.to(compute_type), mask=mask)


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """``y = x * rsqrt(mean(x^2, -1) + eps) * weight`` (weight optional), fused."""
    D = x.shape[-1]
    x2d = x.reshape(-1, D)
    M = x2d.shape[0]
    out_dtype = x.dtype if x.dtype in _TL else torch.bfloat16
    out = torch.empty_like(x2d, dtype=out_dtype)
    BLOCK_D = triton.next_power_of_2(D)
    num_warps = 4 if BLOCK_D <= 1024 else (8 if BLOCK_D <= 4096 else 16)
    _rmsnorm_kernel[(M,)](
        x2d, weight, out, M, D, eps,
        x2d.stride(0), out.stride(0),
        BLOCK_D=BLOCK_D, HAS_W=weight is not None,
        compute_type=_TL[out_dtype], num_warps=num_warps,
    )
    return out.reshape(x.shape)


@triton.jit
def _inv_rms_kernel(x_ptr, o_ptr, D, eps, stride_xm, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for off in range(0, D, BLOCK_D):
        offs = off + tl.arange(0, BLOCK_D)
        v = tl.load(x_ptr + row * stride_xm + offs, mask=offs < D, other=0.0).to(tl.float32)
        acc += v * v
    tl.store(o_ptr + row, tl.rsqrt(tl.sum(acc, axis=0) / D + eps))


def inv_rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    """``rsqrt(mean(x^2, -1) + eps)`` -> ``[..., 1]`` fp32, one pass, no fp32 temp.

    :func:`rms_norm` applies the scale and writes a full-size output; callers that need
    the *scalar* (DSV4's hyper-connection pre-norm multiplies it into a [.., mix_hc]
    tensor, not into x) were spelling it ``x.float().square().mean(-1)``, which
    materialises a second full fp32 copy of the hidden state purely to reduce it away.
    At hc_dim 16384, T=8192 that is a 537 MB write plus a 537 MB read for a [T, 1]
    result. Reading the bf16 source instead of its fp32 upcast halves the bytes again;
    the upcast is exact, so the sum of squares is unchanged.

    RTX 6000 Ada, hc_dim 16384: 1.944 ms -> 0.304 ms at T=8192 (6.4x), 0.020 -> 0.013 ms
    at T=64. The large end lands at ~880 GB/s against a 960 GB/s peak -- the memory roof.

    NOT bit-identical to ``square().mean()``: the reduction order differs, and ATen's
    order is not reproducible anyway (it varies with M -- a tree/2 fold matches at M=4
    and nothing matches at M=64, because the block/grid split is chosen per shape). The
    accuracy is equivalent, not merely close: against an fp64 reference the mean relative
    error is 3.65e-08 here vs 3.60e-08 for ATen at M=8192, and 3.60e-08 vs 3.63e-08 at
    M=4096 -- i.e. it wins at some shapes and loses at others, within 8% of ATen's own
    distance from the truth. (``linalg.vector_norm`` is the same speed but consistently
    ~23% worse, at 4.47e-08, because sqrt-then-square rounds twice.)
    """
    D = x.shape[-1]
    x2d = x.reshape(-1, D)
    out = torch.empty(x2d.shape[0], dtype=torch.float32, device=x.device)
    _inv_rms_kernel[(x2d.shape[0],)](
        x2d, out, D, eps, x2d.stride(0),
        BLOCK_D=min(2048, triton.next_power_of_2(D)), num_warps=8,
    )
    return out.view(*x.shape[:-1], 1)


__all__ = ["rms_norm", "inv_rms"]
