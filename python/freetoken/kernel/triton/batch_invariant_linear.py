"""BF16/FP32 linear projections whose output rows do not depend on batch size or row order.

With weights, dtypes, device and launch configuration unchanged, splitting or reordering the
input rows preserves their results. Fixed tiles and a fixed K reduction order provide this
property; shape-selected library GEMMs need not. This is useful when prefill caches must be
reusable across batching and chunking decisions. It is not cross-backend or cross-device determinism.

BF16 operands use tensor cores; any FP32 operand selects IEEE FP32 dots (no TF32). Both paths
accumulate in FP32. Callers choose when to require batch invariance; this module has no model
or prefill/decode policy. FP16 operands are rejected instead of silently rounded to BF16.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 64


@triton.jit
def _batch_invariant_linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xg, stride_xk, stride_wn, stride_wk, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    IEEE: tl.constexpr, HAS_BIAS: tl.constexpr, OUT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    g = tl.program_id(2).to(tl.int64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N
    x_base = x_ptr + g * stride_xg + offs_m[:, None] * stride_xm
    w_base = w_ptr + (g * N + offs_n)[:, None] * stride_wn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < K
        a = tl.load(x_base + k[None, :] * stride_xk, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(w_base + k[None, :] * stride_wk, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        if IEEE:
            acc += tl.dot(a.to(tl.float32), tl.trans(b.to(tl.float32)), input_precision="ieee")
        else:
            acc += tl.dot(a.to(tl.bfloat16), tl.trans(b.to(tl.bfloat16)))
    if HAS_BIAS:
        acc += tl.load(b_ptr + g * N + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + (g * N + offs_n)[None, :] * stride_on
    tl.store(out_ptrs, acc.to(OUT), mask=m_mask[:, None] & n_mask[None, :])


_TL = {torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}


def _check_operands(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """bf16 / fp32 operands only; returns whether the dots run in IEEE fp32 (any fp32 operand)."""
    for t, what in ((x, "input"), (weight, "weight")):
        if t.dtype not in _TL:
            raise TypeError(f"batch_invariant_linear takes bf16 or fp32 operands, got {what} {t.dtype}")
    return x.dtype == torch.float32 or weight.dtype == torch.float32


def batch_invariant_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, *, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """``x [..., K] @ weight [N, K]^T (+ bias)`` with a fixed K walk: bf16 operands use bf16 tensor-core
    dots, fp32 operands IEEE fp32 dots (either operand fp32 promotes both, like ``F.linear`` on an
    fp32 stream); fp32 accumulation; ``out_dtype`` (bf16 / fp32) defaults to the operand dtype. Row
    ``m`` of the result depends on ``x[m]`` alone -- not on M or on which rows share the batch."""
    *lead, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K, (weight.shape, K)
    x2 = x.reshape(-1, K)
    ieee = _check_operands(x, weight)
    out_dtype = out_dtype or (torch.float32 if ieee else x.dtype)
    if out_dtype not in _TL:
        raise TypeError(f"batch_invariant_linear writes bf16 or fp32, got out_dtype {out_dtype}")
    out = torch.empty((x2.shape[0], N), dtype=out_dtype, device=x.device)
    M = x2.shape[0]
    if M and N:
        _batch_invariant_linear_kernel[(triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M), 1)](
            x2, weight, bias if bias is not None else out, out, M, N, K,
            x2.stride(0), 0, x2.stride(1), weight.stride(0), weight.stride(1), out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, IEEE=ieee, HAS_BIAS=bias is not None, OUT=_TL[out_dtype], num_warps=4,
        )
    return out.reshape(*lead, N)


def batch_invariant_grouped_linear(x: torch.Tensor, weight: torch.Tensor, *, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Block-diagonal ``x [T, G, K]`` x ``weight [G * N, K]`` -> ``[T, G * N]`` (the reference's grouped
    einsum ``tgd,grd->tgr``) with the same fixed K walk per row."""
    T, G, K = x.shape
    GN = weight.shape[0]
    N = GN // G
    assert weight.shape[1] == K and GN % G == 0, (weight.shape, G, K)
    ieee = _check_operands(x, weight)
    out_dtype = out_dtype or (torch.float32 if ieee else x.dtype)
    if out_dtype not in _TL:
        raise TypeError(f"batch_invariant_grouped_linear writes bf16 or fp32, got out_dtype {out_dtype}")
    out = torch.empty((T, GN), dtype=out_dtype, device=x.device)
    if T:
        _batch_invariant_linear_kernel[(triton.cdiv(N, BLOCK_N), triton.cdiv(T, BLOCK_M), G)](
            x, weight, out, out, T, N, K,
            x.stride(0), x.stride(1), x.stride(2), weight.stride(0), weight.stride(1), out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, IEEE=ieee, HAS_BIAS=False, OUT=_TL[out_dtype], num_warps=4,
        )
    return out


__all__ = ["batch_invariant_linear", "batch_invariant_grouped_linear"]
