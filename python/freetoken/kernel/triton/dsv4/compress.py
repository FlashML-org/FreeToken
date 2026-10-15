"""Fused gated-pool for the DeepSeek-V4 KV compressor (CSA/HCA).

The compressor collapses ``ratio`` (or ``2*ratio`` with overlap) cached KV rows into
one via a softmax over the row axis: ``out[d] = sum_r kv[r,d] * softmax(score[:,d])[r]``.
In eager torch this is ``cunn_SpatialSoftMaxForward`` (softmax over the *middle* dim,
which is occupancy-starved at bs=1) + a broadcast mul + a sum = 3 launches per call,
and there are 62 such calls/token (21 CSA x2 + 20 HCA). This kernel does it in one
pass with an online softmax in fp32, parallelized over the D (column) axis.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_TL = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


@triton.jit
def _gated_pool_kernel(
    kv_ptr, score_ptr, out_ptr, R, D,
    stride_kb, stride_kr, stride_kd,
    stride_sb, stride_sr, stride_sd,
    stride_ob, stride_od,
    BLOCK_D: tl.constexpr, OUT: tl.constexpr,
):
    b = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D
    kv_base = kv_ptr + b * stride_kb + offs * stride_kd
    sc_base = score_ptr + b * stride_sb + offs * stride_sd
    maxv = tl.full((BLOCK_D,), float("-inf"), tl.float32)
    # Explicit unroll: ptxas does not unroll the plain loop Triton 3.8 emits, which serializes these latency-bound loads.
    for r in tl.range(R, loop_unroll_factor=2):
        s = tl.load(sc_base + r * stride_sr, mask=mask, other=float("-inf")).to(tl.float32)
        maxv = tl.maximum(maxv, s)
    denom = tl.zeros((BLOCK_D,), tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)
    for r in tl.range(R, loop_unroll_factor=2):
        s = tl.load(sc_base + r * stride_sr, mask=mask, other=float("-inf")).to(tl.float32)
        w = tl.exp(s - maxv)
        denom += w
        k = tl.load(kv_base + r * stride_kr, mask=mask, other=0.0).to(tl.float32)
        acc += w * k
    tl.store(out_ptr + b * stride_ob + offs * stride_od, (acc / denom).to(OUT), mask=mask)


def gated_pool(kv: torch.Tensor, score: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """``out[b,d] = sum_r kv[b,r,d] * softmax(score[b,:,d], dim=row)[r]``.

    ``kv``/``score``: ``[B, R, D]``. Returns ``[B, 1, D]`` (keepdim, matching the
    reference ``(kv * score.softmax(dim=1)).sum(dim=1, keepdim=True)``)."""
    B, R, D = kv.shape
    out = torch.empty((B, D), dtype=out_dtype, device=kv.device)
    BLOCK_D = 256
    grid = (B, triton.cdiv(D, BLOCK_D))
    _gated_pool_kernel[grid](
        kv, score, out, R, D,
        kv.stride(0), kv.stride(1), kv.stride(2),
        score.stride(0), score.stride(1), score.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_D=BLOCK_D, OUT=_TL[out_dtype], num_warps=4,
    )
    return out.unsqueeze(1)


__all__ = ["gated_pool"]
