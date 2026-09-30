"""One activation row times a bf16 weight: the GEMV a batch-1 decode step runs for every projection.

The weight streams from HBM once, upcast to fp32 in registers and accumulated in fp32, so the result
is F.linear's up to the fp32 summation order. At one row the projection is bound by how fast the
weight streams in, and cuBLAS runs it on tensor-core GEMM tiles that leave part of the bandwidth
unused; this kernel reads nothing but the weight and the row.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_gemv_kernel(
    x_ptr, w_ptr, out_ptr, N, K,
    stride_wn, stride_wk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    w_row = w_ptr + offs_n[:, None] * stride_wn
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        w = tl.load(w_row + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        xk = tl.load(x_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        acc += tl.sum(w * xk[None, :], axis=1)
    tl.store(out_ptr + offs_n, acc.to(out_ptr.dtype.element_ty), mask=n_mask)


def bf16_gemv(x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """``x @ weight.T`` for a single row: ``x`` ``[..., K]`` holding one row, ``weight`` ``[N, K]``
    bf16; returns ``[..., N]`` in ``out_dtype``."""
    *lead, K = x.shape
    N = weight.shape[0]
    x1 = x.reshape(K).contiguous()
    out = torch.empty(N, dtype=out_dtype, device=x.device)
    # two rows a program spreads even a small GEMV over every SM, and a K tile up to 4096 wide
    # keeps most rows to one pass
    BLOCK_N = 2
    BLOCK_K = min(triton.next_power_of_2(K), 4096)
    _bf16_gemv_kernel[(triton.cdiv(N, BLOCK_N),)](
        x1, weight, out, N, K,
        weight.stride(0), weight.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4,
    )
    return out.reshape(*lead, N)


__all__ = ["bf16_gemv"]
