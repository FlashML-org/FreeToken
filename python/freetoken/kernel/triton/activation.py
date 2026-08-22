"""Triton *_and_mul activation kernels (PDL usage modeled on flashinfer).

Optional pure-triton drop-in for
freetoken.kernel.activation.{silu,gelu,gelu_tanh}_and_mul:

    y = act(x[..., :d]) * x[..., d:],   d = x.shape[-1] // 2

Elementwise / memory-bound. Single kernel; the activation kind is a
compile-time constexpr so each variant specializes cleanly. Uses Hopper PDL
(griddepcontrol) like flashinfer's enable_pdl path to close the fixed per-launch
gap that dominates tiny decode graph replays.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.language.extra.cuda import gdc_wait, gdc_launch_dependents

from triton.language import target_info
from triton.runtime.jit import constexpr_function

from freetoken.utils.arch import is_sm90_supported

SILU = 0
GELU = 1
GELU_TANH = 2
# SwiGLU-OAI (gpt-oss / MiniMax-M3 "swigluoai"), UNINTERLEAVED halves (gate [:d],
# up [d:], matching this file's *_and_mul layout -- the interleaved gpt-oss bank
# variant lives in triton/mxfp4_moe.py):
#   y = clamp(gate, max=limit) * sigmoid(alpha * gate) * (clamp(up, +-limit) + 1)
SWIGLUOAI = 3

_SQRT_2_OVER_PI = 0.7978845608028654  # sqrt(2/pi)
_GELU_C = 0.044715

_LOG2E = tl.constexpr(1.4426950408889634)


@functools.cache
def _pdl_supported() -> bool:
    # PDL / griddepcontrol is an sm_90+ (Hopper/Blackwell) feature; pre-Hopper
    # ptxas rejects the intrinsic. The fallback path is exactly where a non-Hopper
    # card (3090 sm_86, 4090/4060 sm_89, A100 sm_80) is most likely, so gate it off.
    return is_sm90_supported()


@constexpr_function
def _fast_tanh_cx():
    """Compile-time: can the target issue ``tanh.approx.f32``? It is an sm_75+
    instruction and pre-Turing ptxas rejects the whole module ("Feature 'tanh' requires
    .target sm_75 or higher"), so older cards take the libdevice fallback below. Resolved
    from the compilation target like :mod:`freetoken.kernel.triton.e4m3_compat` does."""
    return target_info.cuda_capability_geq(7, 5)


@triton.jit
def _fast_tanh(x):
    # PTX tanh.approx.f32 — single HW op, matches flashinfer math::tanh.
    return tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;", "=f,f", [x],
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _fast_ex2(x):
    # PTX ex2.approx.f32 — matches __expf fast path used by flashinfer silu.
    return tl.inline_asm_elementwise(
        "ex2.approx.f32 $0, $1;", "=f,f", [x],
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _act_and_mul_kernel(
    out_ptr,
    x_ptr,
    d,
    alpha,
    limit,
    ACT: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # One program handles a contiguous BLOCK_D chunk of one output row.
    row = tl.program_id(0).to(tl.int64)
    col_blk = tl.program_id(1)
    cols = col_blk * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = cols < d

    # Input row stride is 2*d; gate is [:d], up is [d:].
    x_row = x_ptr + row * (2 * d)
    # PDL (Hopper griddepcontrol): wait for the producer kernel's writes to be
    # visible before loading, then signal the next launch may begin its prologue
    # once all reads are issued.
    if ENABLE_PDL:
        gdc_wait()
    gate = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_row + d + cols, mask=mask, other=0.0).to(tl.float32)
    if ENABLE_PDL:
        gdc_launch_dependents()

    if ACT == 0:  # SILU: x / (1 + exp(-x)) via ex2.approx
        act = gate / (1.0 + _fast_ex2(-gate * _LOG2E))
        y = act * up
    elif ACT == 2:  # GELU_TANH via tanh.approx (sm_75+), else libdevice
        inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
        if _fast_tanh_cx():
            tanh_inner = _fast_tanh(inner)
        else:
            tanh_inner = libdevice.tanh(inner)
        act = 0.5 * gate * (1.0 + tanh_inner)
        y = act * up
    elif ACT == 3:  # SWIGLUOAI: clamped gate/up, sigmoid(alpha*gate), (up + 1) bias
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
        act = gate / (1.0 + _fast_ex2(-gate * alpha * _LOG2E))
        y = act * (up + 1.0)
    else:  # GELU (erf)
        act = 0.5 * gate * (1.0 + libdevice.erf(gate * 0.7071067811865476))
        y = act * up
    out_row = out_ptr + row * d
    tl.store(out_row + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def _act_and_mul(
    kind: int,
    x: torch.Tensor,
    out: torch.Tensor | None,
    alpha: float = 0.0,
    limit: float = 0.0,
):
    assert x.is_cuda and x.is_contiguous()
    d = x.shape[-1] // 2
    out_shape = x.shape[:-1] + (d,)
    if out is None:
        out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    x2 = x.reshape(-1, x.shape[-1])
    o2 = out.reshape(-1, d)
    M = x2.shape[0]
    grid = lambda meta: (M, triton.cdiv(d, meta["BLOCK_D"]))
    pdl = _pdl_supported()
    # Fixed via H100 sweep (72-config grid; 512/w4/s3 within 11% everywhere,
    # 1024/w4/s2 best at rows>=4096).
    block_d = min(triton.next_power_of_2(d), 1024 if M >= 4096 else 512)
    num_stages = 2 if block_d == 1024 else 3
    _act_and_mul_kernel[grid](
        o2, x2, d, alpha, limit, ACT=kind, ENABLE_PDL=pdl, launch_pdl=pdl,
        BLOCK_D=block_d, num_warps=4, num_stages=num_stages,
    )
    return out


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    return _act_and_mul(SILU, x, out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    return _act_and_mul(GELU, x, out)


def gelu_tanh_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    return _act_and_mul(GELU_TANH, x, out)


def swigluoai_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    """SwiGLU-OAI over UNINTERLEAVED halves (gate ``x[..., :d]``, up ``x[..., d:]``):
    ``clamp(gate, max=limit) * sigmoid(alpha * gate) * (clamp(up, +-limit) + 1)``.
    Same math as gpt-oss's interleaved ``gpt_oss_swiglu_kernel``; this layout matches
    the [gate; up] halves the NVFP4 expert banks and merged gate_up projections use."""
    return _act_and_mul(SWIGLUOAI, x, out, alpha=alpha, limit=limit)


__all__ = ["silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul", "swigluoai_and_mul"]
