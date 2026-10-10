# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM (vllm/models/qwen4_exp/nvidia/ops/hc.py)
"""NVIDIA HyperConnection kernels for Qwen4Exp."""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from freetoken.utils.arch import is_sm90_supported


@functools.cache
def _pdl_supported() -> bool:
    return is_sm90_supported()


@triton.jit
def _grouped_gemma_rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    stride_x,
    stride_y,
    DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    GROUP_DIM: tl.constexpr = DIM // NUM_GROUPS
    BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(GROUP_DIM)

    pid = tl.program_id(0)
    group_id = pid % NUM_GROUPS
    # row * stride can overflow int32 for large token counts.
    row = (pid // NUM_GROUPS).to(tl.int64)

    offs_g = tl.arange(0, BLOCK_SIZE)
    offsets = group_id * GROUP_DIM + offs_g
    mask = offs_g < GROUP_DIM
    # A [GROUP_DIM] affine is shared; a [DIM] affine follows the grouped
    # checkpoint layout.
    w_offs = offs_g if W_SHARED else offsets

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    x = tl.load(x_ptr + row * stride_x + offsets, mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + w_offs, mask, other=0.0)

    rrms = tl.rsqrt(tl.sum(x * x) / GROUP_DIM + EPS)
    # Gemma's (1 + w) affine is written this way to lower to an FMA.
    y = x * rrms
    y += y * w.to(tl.float32)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(y_ptr + row * stride_y + offsets, y, mask)


def grouped_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int
) -> torch.Tensor:
    N, DIM = x.shape
    assert x.stride(1) == 1, "grouped Gemma RMSNorm requires unit inner stride"
    assert weight.is_contiguous(), "grouped Gemma RMSNorm weight must be contiguous"
    assert DIM % num_groups == 0
    group_dim = DIM // num_groups
    assert weight.numel() in (group_dim, DIM)

    y = x.new_empty(x.shape)
    _grouped_gemma_rmsnorm_kernel[(N * num_groups,)](
        x,
        weight,
        y,
        x.stride(0),
        y.stride(0),
        DIM,
        num_groups,
        W_SHARED=weight.numel() == group_dim,
        EPS=eps,
        launch_pdl=_pdl_supported(),
    )
    return y


@triton.jit
def _hc_silu_kernel(
    x_ptr,
    y_ptr,
    stride_x,
    stride_y,
    DIM: tl.constexpr,
    HC: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(DIM)

    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < DIM

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    x = tl.load(x_ptr + row * stride_x + offs, mask).to(tl.float32) / HC
    y = x * tl.sigmoid(x)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(y_ptr + row * stride_y + offs, y, mask)


def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    num_tokens, DIM = x.shape
    assert x.stride(1) == 1

    output = x.new_empty(x.shape)
    _hc_silu_kernel[(num_tokens,)](
        x,
        output,
        x.stride(0),
        output.stride(0),
        DIM=DIM,
        HC=hc_count,
        launch_pdl=_pdl_supported(),
    )
    return output


@triton.jit
def _hc_gate_mix_kernel(
    x_ptr,
    g_ptr,
    y_ptr,
    stride_x,
    stride_g,
    stride_y,
    DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    HC_DIM: tl.constexpr = DIM // HC

    row = tl.program_id(0).to(tl.int64)
    tile_id = tl.program_id(1)
    offs_inner = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_inner < HC_DIM

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    # The constexpr loop is unrolled and keeps one stream live at a time.
    # Materializing [HC, BLOCK_SIZE] more than doubles latency at large M.
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for stream in tl.static_range(HC):
        offsets = stream * HC_DIM + offs_inner
        g = tl.load(g_ptr + row * stride_g + offsets, mask, other=0.0)
        x = tl.load(x_ptr + row * stride_x + offsets, mask, other=0.0)
        acc += tl.sigmoid(g.to(tl.float32)) * x.to(tl.float32)
    acc /= HC

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(y_ptr + row * stride_y + offs_inner, acc, mask)


def hc_gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    N, DIM = gate.shape
    assert x.shape == gate.shape
    assert DIM % hc_count == 0
    assert x.stride(1) == 1
    assert gate.stride(1) == 1

    HC_DIM = DIM // hc_count
    out = x.new_empty(N, HC_DIM)
    BLOCK_SIZE = 512
    _hc_gate_mix_kernel[(N, triton.cdiv(HC_DIM, BLOCK_SIZE))](
        x,
        gate,
        out,
        x.stride(0),
        gate.stride(0),
        out.stride(0),
        DIM,
        hc_count,
        BLOCK_SIZE,
        launch_pdl=_pdl_supported(),
    )
    return out


@triton.jit
def _hc_combine_kernel(
    block_ptr,
    res_ptr,
    inj_ptr,
    out_ptr,
    stride_block,
    stride_res,
    stride_inj,
    stride_out,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    HC_PAD: tl.constexpr = triton.next_power_of_2(HC)

    row = tl.program_id(0).to(tl.int64)
    tile_id = tl.program_id(1)

    offs_inner = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_inner = offs_inner < HC_DIM
    offs_hc = tl.arange(0, HC_PAD)
    mask_hc = offs_hc < HC
    offs = offs_hc[:, None] * HC_DIM + offs_inner[None, :]
    mask = mask_hc[:, None] & mask_inner[None, :]

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    inj = tl.load(inj_ptr + row * stride_inj + offs_hc, mask_hc, other=0.0)
    block = tl.load(block_ptr + row * stride_block + offs_inner, mask_inner, other=0.0)
    res = tl.load(res_ptr + row * stride_res + offs, mask, other=0.0)

    # Keeping HC as a broadcast dimension is faster here than four separate
    # residual load/store sequences.
    inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)
    out = res.to(tl.float32) + block.to(tl.float32)[None, :] * inj[:, None]

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(out_ptr + row * stride_out + offs, out, mask=mask)


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    N, DIM = residual.shape
    assert DIM % hc_count == 0
    hc_dim = DIM // hc_count
    assert block_output.shape == (N, hc_dim)
    assert injection_logits.shape == (N, hc_count)
    assert residual.stride(1) == 1
    assert block_output.stride(1) == 1
    assert injection_logits.stride(1) == 1

    out = residual.new_empty(residual.shape)
    BLOCK_SIZE = 512
    _hc_combine_kernel[(N, triton.cdiv(hc_dim, BLOCK_SIZE))](
        block_output,
        residual,
        injection_logits,
        out,
        block_output.stride(0),
        residual.stride(0),
        injection_logits.stride(0),
        out.stride(0),
        hc_dim,
        hc_count,
        BLOCK_SIZE,
        launch_pdl=_pdl_supported(),
    )
    return out


@triton.jit
def _hc_combine_norm_kernel(
    block_ptr,
    res_ptr,
    inj_ptr,
    w_ptr,
    out_ptr,
    y_ptr,
    shared_ptr,
    gate_ptr,
    stride_block,
    stride_res,
    stride_inj,
    stride_out,
    stride_y,
    stride_shared,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GATED: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    HC_PAD: tl.constexpr = triton.next_power_of_2(HC)
    NUM_TILES: tl.constexpr = triton.cdiv(HC_DIM, BLOCK_SIZE)
    NUM_TILES_PAD: tl.constexpr = triton.next_power_of_2(NUM_TILES)

    row = tl.program_id(0).to(tl.int64)
    stream = tl.program_id(1)
    offs_hc = tl.arange(0, HC_PAD)
    mask_hc = offs_hc < HC
    tile_ids = tl.arange(0, NUM_TILES_PAD)
    offs_inner = tile_ids[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    mask_inner = offs_inner < HC_DIM
    offs = stream * HC_DIM + offs_inner
    # Shared norm weights repeat across streams; per-branch weights use the
    # same flattened HC layout as the residual.
    w_offs = offs_inner if W_SHARED else offs

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    # Start the uncached residual load first, then issue the other combine
    # loads before consuming any of them.
    res = tl.load(res_ptr + row * stride_res + offs, mask_inner, other=0.0)
    inj = tl.load(inj_ptr + row * stride_inj + offs_hc, mask_hc, other=0.0)
    block = tl.load(block_ptr + row * stride_block + offs_inner, mask_inner, other=0.0)
    if GATED:
        # Fold the shared-expert gate epilogue in: same fp32 ops and the
        # same bf16 store boundary as the standalone gate kernel, so the
        # combine below sees the identical block row.
        g = tl.load(gate_ptr + row)
        sh = tl.load(shared_ptr + row * stride_shared + offs_inner, mask_inner, other=0.0)
        block = (block.to(tl.float32) + g * sh.to(tl.float32)).to(block_ptr.dtype.element_ty)
    inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)
    inj = tl.sum(tl.where(offs_hc == stream, inj, 0.0))
    # Round the materialized combine result before normalization. This matches
    # the unfused combine -> RMSNorm boundary.
    out = (res.to(tl.float32) + block.to(tl.float32) * inj).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * stride_out + offs, out, mask=mask_inner)

    out = out.to(tl.float32)
    # The standalone grouped RMSNorm reduces one padded [next_pow2(HC_DIM)]
    # vector; reshape keeps the identical tree (same element
    # order, same masked zeros), so the fused statistic is bit-exact vs the split.
    flat = tl.reshape(out * out, [NUM_TILES_PAD * BLOCK_SIZE])
    sum_sq = tl.sum(flat)
    rrms = tl.rsqrt(sum_sq / HC_DIM + EPS)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    # Loading the weight earlier helps decode but keeps the tile live across
    # the reduction and regresses larger batches, so defer it to the norm.
    w = tl.load(w_ptr + w_offs, mask_inner, other=0.0)
    y = out * rrms
    y += y * w.to(tl.float32)
    tl.store(y_ptr + row * stride_y + offs, y, mask_inner)


def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
    shared: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """combine + grouped RMSNorm; with ``shared``/``gate`` the shared-expert gate
    epilogue (``block_output + gate * shared``) is folded in bit-exactly."""
    N, DIM = residual.shape
    assert DIM % hc_count == 0
    hc_dim = DIM // hc_count
    assert block_output.shape == (N, hc_dim)
    assert injection_logits.shape == (N, hc_count)
    assert residual.stride(1) == 1
    assert block_output.stride(1) == 1
    assert injection_logits.stride(1) == 1
    assert norm_weight.is_contiguous()
    assert norm_weight.numel() in (hc_dim, DIM)
    if shared is not None:
        assert shared.shape == block_output.shape and shared.stride(1) == 1
        assert gate is not None and gate.shape == (N,) and gate.dtype == torch.float32

    out = residual.new_empty(residual.shape)
    y = residual.new_empty(residual.shape)
    BLOCK_SIZE = 512
    _hc_combine_norm_kernel[(N, hc_count)](
        block_output,
        residual,
        injection_logits,
        norm_weight,
        out,
        y,
        block_output if shared is None else shared,
        block_output if gate is None else gate,
        block_output.stride(0),
        residual.stride(0),
        injection_logits.stride(0),
        out.stride(0),
        y.stride(0),
        0 if shared is None else shared.stride(0),
        hc_dim,
        hc_count,
        W_SHARED=norm_weight.numel() == hc_dim,
        EPS=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        GATED=shared is not None,
        launch_pdl=_pdl_supported(),
    )
    return out, y


__all__ = [
    "grouped_gemma_rmsnorm",
    "hc_combine",
    "hc_combine_norm",
    "hc_gate_mix",
    "hc_silu",
]
