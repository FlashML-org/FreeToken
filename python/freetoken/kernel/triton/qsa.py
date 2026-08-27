"""Exact gathered-row GQA attention for Qwen compressed sparse attention."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_H = 16
BLOCK_T = 32


@triton.jit
def _compact_qsa_blocks_kernel(
    blocks_ptr,
    positions_ptr,
    output_ptr,
    counts_ptr,
    stride_bn,
    stride_bt,
    stride_on,
    stride_ot,
    BLOCK_COUNT: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOKEN_BUDGET: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    BLOCK_BLOCKS: tl.constexpr,
    BLOCK_OUTPUT: tl.constexpr,
):
    row = tl.program_id(0)
    block_offsets = tl.arange(0, BLOCK_BLOCKS)
    chosen = tl.load(
        blocks_ptr + row * stride_bn + block_offsets * stride_bt,
        mask=block_offsets < BLOCK_COUNT,
        other=-1,
    )
    live_blocks = tl.sum((chosen >= 0).to(tl.int32), axis=0)
    position = tl.load(positions_ptr + row).to(tl.int64)
    visible = position + 1
    tail_start = (visible // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = visible - tail_start
    live_tokens = live_blocks * COMPRESS_RATIO

    columns = tl.arange(0, BLOCK_OUTPUT)
    block_slot = columns // COMPRESS_RATIO
    block_value = tl.load(
        blocks_ptr + row * stride_bn + block_slot * stride_bt,
        mask=(columns < live_tokens) & (block_slot < BLOCK_COUNT),
        other=-1,
    ).to(tl.int64)
    expanded = block_value * COMPRESS_RATIO + columns % COMPRESS_RATIO
    tail_offset = columns - live_tokens
    value = tl.where(
        columns < live_tokens,
        expanded,
        tl.where(tail_offset < tail_count, tail_start + tail_offset, -1),
    )
    tl.store(
        output_ptr + row * stride_on + columns * stride_ot,
        value.to(tl.int32),
        mask=columns < OUTPUT_WIDTH,
    )
    tl.store(counts_ptr + row, (live_tokens + tail_count).to(tl.int32))


def compact_qsa_blocks(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    compress_ratio: int,
    token_budget: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused CUDA expansion, tail insertion, and live-count calculation."""
    if not block_indices.is_cuda or not query_positions.is_cuda:
        raise ValueError("compact_qsa_blocks is a CUDA kernel")
    rows, blocks = block_indices.shape
    width = token_budget + compress_ratio - 1
    output = torch.empty((rows, width), dtype=torch.int32, device=block_indices.device)
    counts = torch.empty(rows, dtype=torch.int32, device=block_indices.device)
    if not rows:
        return output, counts
    blocks_i32 = block_indices.to(torch.int32).contiguous()
    positions = query_positions.to(torch.int64).contiguous()
    _compact_qsa_blocks_kernel[(rows,)](
        blocks_i32,
        positions,
        output,
        counts,
        blocks_i32.stride(0),
        blocks_i32.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_COUNT=blocks,
        COMPRESS_RATIO=compress_ratio,
        TOKEN_BUDGET=token_budget,
        OUTPUT_WIDTH=width,
        BLOCK_BLOCKS=triton.next_power_of_2(blocks),
        BLOCK_OUTPUT=triton.next_power_of_2(width),
        num_warps=8,
        num_stages=1,
    )
    return output, counts


@triton.jit
def _qsa_sparse_gqa_kernel(
    q_ptr, k_ptr, v_ptr, rows_ptr, counts_ptr, out_ptr,
    scale,
    H, KVH, D, TOPK, GQA,
    stride_qn, stride_qh, stride_qd,
    stride_kr, stride_kh, stride_kd,
    stride_vr, stride_vh, stride_vd,
    stride_rn, stride_rt,
    stride_on, stride_oh, stride_od,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)

    head_offsets = tl.arange(0, BLOCK_H)
    heads = kv_head * GQA + head_offsets
    head_mask = (head_offsets < GQA) & (heads < H)
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < D

    q = tl.load(
        q_ptr + row * stride_qn + heads[:, None] * stride_qh + dims[None, :] * stride_qd,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    running_max = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)
    active = tl.load(counts_ptr + row)
    row_base = rows_ptr + row * stride_rn

    for tile in range(0, tl.cdiv(active, BLOCK_T)):
        token_offsets = tile * BLOCK_T + tl.arange(0, BLOCK_T)
        token_mask = token_offsets < active
        physical = tl.load(
            row_base + token_offsets * stride_rt, mask=token_mask, other=-1
        )
        valid = token_mask & (physical >= 0)
        physical = tl.maximum(physical, 0)
        k = tl.load(
            k_ptr
            + physical[None, :] * stride_kr
            + kv_head * stride_kh
            + dims[:, None] * stride_kd,
            mask=dim_mask[:, None] & valid[None, :],
            other=0.0,
        )
        scores = tl.dot(q, k) * scale
        scores = tl.where(valid[None, :], scores, -float("inf"))
        new_max = tl.maximum(running_max, tl.max(scores, axis=1))
        alpha = tl.where(new_max == -float("inf"), 1.0, tl.exp(running_max - new_max))
        probs = tl.where(valid[None, :], tl.exp(scores - new_max[:, None]), 0.0)
        running_sum = running_sum * alpha + tl.sum(probs, axis=1)
        acc *= alpha[:, None]
        v = tl.load(
            v_ptr
            + physical[:, None] * stride_vr
            + kv_head * stride_vh
            + dims[None, :] * stride_vd,
            mask=valid[:, None] & dim_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(probs.to(v.dtype), v)
        running_max = new_max

    output = tl.where(running_sum[:, None] > 0, acc / running_sum[:, None], 0.0)
    tl.store(
        out_ptr + row * stride_on + heads[:, None] * stride_oh + dims[None, :] * stride_od,
        output.to(out_ptr.dtype.element_ty),
        mask=head_mask[:, None] & dim_mask[None, :],
    )


def _torch_qsa_sparse_gqa(
    q: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    selected_rows: torch.Tensor,
    counts: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Small CPU/reference implementation used by unit tests."""
    output = torch.zeros_like(q)
    gqa = q.shape[1] // k_rows.shape[1]
    for row in range(q.shape[0]):
        count = int(counts[row])
        if count == 0:
            continue
        indices = selected_rows[row, :count].long()
        keys = k_rows.index_select(0, indices)
        values = v_rows.index_select(0, indices)
        for kv_head in range(k_rows.shape[1]):
            heads = slice(kv_head * gqa, (kv_head + 1) * gqa)
            scores = torch.einsum(
                "hd,td->ht", q[row, heads].float(), keys[:, kv_head].float()
            ) * sm_scale
            probs = torch.softmax(scores, dim=-1).to(values.dtype)
            output[row, heads] = torch.einsum(
                "ht,td->hd", probs, values[:, kv_head]
            ).to(output.dtype)
    return output


@torch.no_grad()
def qsa_sparse_gqa(
    q: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    selected_rows: torch.Tensor,
    counts: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Attend to an arbitrary physical-row list for each query.

    Shapes are ``q[N,H,D]``, ``k/v[R,KVH,D]``, and
    ``selected_rows[N,K]``.  Valid entries are compact and ``counts[N]`` gives
    the live width of each row.
    """
    if q.ndim != 3 or k_rows.ndim != 3 or v_rows.shape != k_rows.shape:
        raise ValueError("QSA expects q [N,H,D] and matching k/v [R,KVH,D]")
    if q.shape[-1] != k_rows.shape[-1] or q.shape[1] % k_rows.shape[1]:
        raise ValueError("QSA requires matching head dimensions and integral GQA groups")
    if selected_rows.shape[0] != q.shape[0] or counts.numel() != q.shape[0]:
        raise ValueError("QSA selection rows must match query rows")
    if not q.is_cuda:
        return _torch_qsa_sparse_gqa(q, k_rows, v_rows, selected_rows, counts, sm_scale)

    q = q.contiguous()
    k_rows = k_rows.contiguous()
    v_rows = v_rows.contiguous()
    selected_rows = selected_rows.to(torch.int32).contiguous()
    counts = counts.to(torch.int32).contiguous()
    output = torch.empty_like(q)
    n, heads, dim = q.shape
    kv_heads = k_rows.shape[1]
    gqa = heads // kv_heads
    if gqa > BLOCK_H:
        raise ValueError(f"QSA Triton kernel supports a GQA group up to {BLOCK_H}, got {gqa}")
    _qsa_sparse_gqa_kernel[(n, kv_heads)](
        q, k_rows, v_rows, selected_rows, counts, output,
        float(sm_scale),
        heads, kv_heads, dim, selected_rows.shape[1], gqa,
        q.stride(0), q.stride(1), q.stride(2),
        k_rows.stride(0), k_rows.stride(1), k_rows.stride(2),
        v_rows.stride(0), v_rows.stride(1), v_rows.stride(2),
        selected_rows.stride(0), selected_rows.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_D=triton.next_power_of_2(dim),
        BLOCK_H=BLOCK_H,
        BLOCK_T=BLOCK_T,
        num_warps=8,
        num_stages=2,
    )
    return output


__all__ = ["compact_qsa_blocks", "qsa_sparse_gqa"]
