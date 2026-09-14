# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# Adapted from SGLang (kernels/ops/elementwise.py, ``_fused_gate_sigmoid_mul_add``)
"""Gated shared-expert epilogue: the gate reduction and the sigmoid-mul-add.

The routed experts may write into ``hidden_states`` in place, so the gate reduction runs
before them and the mul-add after.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from freetoken.utils.arch import is_sm90_supported


@functools.cache
def _pdl_supported() -> bool:
    return is_sm90_supported()


def _reduction_warps(hidden_dim: int, num_tokens: int) -> int:
    warps = max(min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), 32), 4)
    return min(warps, 8) if num_tokens >= 1024 else warps


@triton.jit
def _gate_sigmoid_kernel(
    hidden_ptr,
    weight_ptr,
    gate_ptr,
    stride_h,
    HIDDEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    # row * stride can overflow int32 for large token counts.
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < HIDDEN

    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    h = tl.load(hidden_ptr + row * stride_h + offs, mask=mask, other=0.0).to(tl.float32)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    tl.store(gate_ptr + row, tl.sigmoid(tl.sum(h * w, axis=0)))


def shared_gate_sigmoid(hidden_states: torch.Tensor, gate_weight: torch.Tensor) -> torch.Tensor:
    """Per-token ``sigmoid(hidden_states @ gate_weight)`` as fp32 [num_tokens]."""
    num_tokens, hidden_dim = hidden_states.shape
    assert hidden_states.stride(1) == 1, "shared gate requires unit inner stride"
    assert gate_weight.shape == (hidden_dim,) and gate_weight.is_contiguous()

    gate = torch.empty(num_tokens, dtype=torch.float32, device=hidden_states.device)
    _gate_sigmoid_kernel[(num_tokens,)](
        hidden_states,
        gate_weight,
        gate,
        hidden_states.stride(0),
        HIDDEN=hidden_dim,
        BLOCK_SIZE=triton.next_power_of_2(hidden_dim),
        launch_pdl=_pdl_supported(),
        num_warps=_reduction_warps(hidden_dim, num_tokens),
    )
    return gate


@triton.jit
def _gate_mul_add_kernel(
    routed_ptr,
    shared_ptr,
    gate_ptr,
    out_ptr,
    stride_r,
    stride_s,
    stride_o,
    HIDDEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    offs = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < HIDDEN

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    gate = tl.load(gate_ptr + row)
    routed = tl.load(routed_ptr + row * stride_r + offs, mask=mask, other=0.0).to(tl.float32)
    shared = tl.load(shared_ptr + row * stride_s + offs, mask=mask, other=0.0).to(tl.float32)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    tl.store(out_ptr + row * stride_o + offs, routed + gate * shared, mask=mask)


def shared_gate_mul_add(
    routed: torch.Tensor, shared: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """``routed + gate[:, None] * shared`` into a fresh tensor."""
    num_tokens, hidden_dim = routed.shape
    assert shared.shape == routed.shape
    assert routed.stride(1) == 1 and shared.stride(1) == 1
    assert gate.shape == (num_tokens,)

    out = torch.empty_like(routed)
    block_size = min(triton.next_power_of_2(hidden_dim), 2048)
    _gate_mul_add_kernel[(num_tokens, triton.cdiv(hidden_dim, block_size))](
        routed,
        shared,
        gate,
        out,
        routed.stride(0),
        shared.stride(0),
        out.stride(0),
        HIDDEN=hidden_dim,
        BLOCK_SIZE=block_size,
        launch_pdl=_pdl_supported(),
        num_warps=4,
    )
    return out


__all__ = ["shared_gate_mul_add", "shared_gate_sigmoid"]
