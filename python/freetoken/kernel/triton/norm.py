"""Single-pass Triton RMSNorm / fused_add_rmsnorm (provenance: norm<-sglang).

Adapted from sglang's srt/layers/elementwise.py (fused_rmsnorm_kernel /
fused_dual_residual_rmsnorm_kernel). Key idea from sglang: cover the ENTIRE
hidden row with one block (BLOCK = next_power_of_2(H)) so x/residual are read
exactly once (single pass), vs a two-pass loop that re-reads x.

Three sglang traits kept verbatim because they measurably matter (H100 A/B vs
their kernel): H and eps are constexpr (BLOCK == H lets the compiler drop the
column mask -> unmasked vectorized loads; one specialization per (H, eps) pair,
both per-model constants), and the hot 2D-contiguous case takes a CONTIG path
addressed as pid*H with no stride math. Unlike sglang we also keep a strided
path (CONTIG=False) -- per-head qk-norm on a fused-QKV slice (q.view(N, heads,
head_dim), row stride = full qkv width) works in place, only the norm axis must
be unit-stride -- and num_warps drops to 4 for prefill row counts (sglang's
fixed w8 loses ~12% there).

Launch params are fixed (no autotune); PDL is used on sm_90+ like the
activation kernels to hide launch latency in decode graph replays.

Optional pure-triton drop-in for freetoken.kernel.norm.{rmsnorm,fused_add_rmsnorm}:
  rmsnorm(x, w, eps)          -> y = (x * rsqrt(mean(x^2) + eps)) * w
  fused_add_rmsnorm(x, r, w)  -> r <- x + r ; x <- (r * rsqrt(mean(r^2)+eps)) * w
Reductions accumulate in fp32; output/residual cast back to the input dtype.
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

from freetoken.utils.arch import is_rocm, is_sm90_supported

_HEUR = {"BLOCK": lambda a: triton.next_power_of_2(a["H"])}


def _num_warps(M: int) -> int:
    return 4 if M >= 1024 else 8


@triton.heuristics(_HEUR)
@triton.jit
def _rmsnorm_kernel(
    OUT, X, W,
    eps: tl.constexpr, H: tl.constexpr,
    sxa, sxb, soa, sob,
    CONTIG: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    BLOCK: tl.constexpr,
    GEMMA: tl.constexpr = False,
):
    a = tl.program_id(0)
    b = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    if CONTIG:
        xoff = a.to(tl.int64) * H  # int64: row offsets can pass 2^31 (scalar, free)
        ooff = xoff
    else:
        xoff = a.to(tl.int64) * sxa + b * sxb
        ooff = a.to(tl.int64) * soa + b * sob
    if ENABLE_PDL:
        gdc_wait()
    x = tl.load(X + xoff + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    if ENABLE_PDL:
        gdc_launch_dependents()
    if GEMMA:
        # Gemma semantics: checkpoint stores (scale - 1); apply (1 + w) in fp32
        # (matches flashinfer's gemma_rmsnorm, which never rounds 1+w to bf16).
        w = w + 1.0
    inv = tl.math.rsqrt(tl.sum(x * x, axis=0) / H + eps)
    y = x * inv * w
    tl.store(OUT + ooff + cols, y.to(OUT.dtype.element_ty), mask=mask)


@triton.heuristics(_HEUR)
@triton.jit
def _fused_add_rmsnorm_kernel(
    X, R, W,
    eps: tl.constexpr, H: tl.constexpr,
    sxa, sxb, sra, srb,
    CONTIG: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    BLOCK: tl.constexpr,
    GEMMA: tl.constexpr = False,
):
    a = tl.program_id(0)
    b = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    if CONTIG:
        xoff = a.to(tl.int64) * H  # int64: row offsets can pass 2^31 (scalar, free)
        roff = xoff
    else:
        xoff = a.to(tl.int64) * sxa + b * sxb
        roff = a.to(tl.int64) * sra + b * srb
    if ENABLE_PDL:
        gdc_wait()
    x = tl.load(X + xoff + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R + roff + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    if ENABLE_PDL:
        gdc_launch_dependents()
    if GEMMA:
        w = w + 1.0
    z = x + r
    tl.store(R + roff + cols, z.to(R.dtype.element_ty), mask=mask)
    inv = tl.math.rsqrt(tl.sum(z * z, axis=0) / H + eps)
    y = z * inv * w
    tl.store(X + xoff + cols, y.to(X.dtype.element_ty), mask=mask)


def _leading(t):
    # View t as (A, B, H) for the 2D program grid; last dim (H) is the norm axis.
    # 2D -> (M, 1); 3D -> (rows, heads). Returns (A, B, strideA, strideB).
    if t.ndim == 2:
        return t.shape[0], 1, t.stride(0), 0
    return t.shape[0], t.shape[1], t.stride(0), t.stride(1)


def _rmsnorm(input, weight, eps, out, gemma: bool):
    import torch

    assert weight.ndim == 1 and input.shape[-1] == weight.shape[0]
    H = input.shape[-1]
    if input.ndim < 2 or input.ndim > 3:
        # Collapse leading dims (rare: never hit by the shipped models, which pass
        # 2D hidden norms or 3D per-head qk-norms). reshape may copy for a strided
        # input, so this path is out-of-place only.
        assert out is None, "rmsnorm on <2D/>3D input must be out-of-place"
        return _rmsnorm(input.reshape(-1, H), weight, eps, None, gemma).reshape(input.shape)
    assert input.stride(-1) == 1  # norm axis must be unit-stride (as flashinfer requires)
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape and out.stride(-1) == 1
    A, B, sxa, sxb = _leading(input)
    _, _, soa, sob = _leading(out)
    contig = input.ndim == 2 and input.is_contiguous() and out.is_contiguous()
    # PDL only on the contiguous (decode-replay) path: on the strided qk-norm's
    # 32k-CTA prefill grids the per-CTA gdc_wait poll costs more than it hides.
    pdl = contig and is_sm90_supported()
    _rmsnorm_kernel[(A, B)](
        out, input, weight, eps, H, sxa, sxb, soa, sob,
        CONTIG=contig, ENABLE_PDL=pdl, GEMMA=gemma,
        **({} if is_rocm() else {"launch_pdl": pdl}),
        num_warps=_num_warps(A * B), num_stages=1,
    )
    return out


def rmsnorm(input, weight, eps: float = 1e-6, out=None, enable_pdl: bool = False):
    return _rmsnorm(input, weight, eps, out, gemma=False)


def gemma_rmsnorm(input, weight, eps: float = 1e-6, out=None, enable_pdl: bool = False):
    """(1 + w)-scaled RMSNorm (Gemma/MiniMax-M3 semantics: the checkpoint stores
    ``scale - 1``); drop-in for flashinfer's ``norm.gemma_rmsnorm``."""
    return _rmsnorm(input, weight, eps, out, gemma=True)


def _fused_add_rmsnorm(input, residual, weight, eps, gemma: bool):
    assert weight.ndim == 1 and input.shape[-1] == weight.shape[0]
    assert input.shape == residual.shape
    H = input.shape[-1]
    assert 2 <= input.ndim <= 3
    assert input.stride(-1) == 1 and residual.stride(-1) == 1
    A, B, sxa, sxb = _leading(input)
    _, _, sra, srb = _leading(residual)
    contig = input.ndim == 2 and input.is_contiguous() and residual.is_contiguous()
    pdl = contig and is_sm90_supported()
    _fused_add_rmsnorm_kernel[(A, B)](
        input, residual, weight, eps, H, sxa, sxb, sra, srb,
        CONTIG=contig, ENABLE_PDL=pdl, GEMMA=gemma,
        **({} if is_rocm() else {"launch_pdl": pdl}),
        num_warps=_num_warps(A * B), num_stages=1,
    )


def fused_add_rmsnorm(input, residual, weight, eps: float = 1e-6, enable_pdl: bool = False):
    _fused_add_rmsnorm(input, residual, weight, eps, gemma=False)


def gemma_fused_add_rmsnorm(input, residual, weight, eps: float = 1e-6, enable_pdl: bool = False):
    """In-place ``residual += input; input = gemma_rmsnorm(residual)`` (Gemma (1+w)
    scaling); drop-in for flashinfer's ``norm.gemma_fused_add_rmsnorm``."""
    _fused_add_rmsnorm(input, residual, weight, eps, gemma=True)


__all__ = ["rmsnorm", "fused_add_rmsnorm", "gemma_rmsnorm", "gemma_fused_add_rmsnorm"]
