# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM (vllm/models/qwen4_exp/nvidia/ops/qsa.py and ops/qsa_pre_indexer.py)
"""QSA index-key compression, indexer norm+rope, and fixed-width row stores.

The pending ring is one row per (request slot, ring position) keyed by ``Req.table_idx``,
the caches are row-flat, and the fused (1+w) RMSNorm + partial NeoX rope takes the rotary
width as a parameter.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _compress_qsa_groups_kernel(
    raw_keys_ptr,
    ring_ptr,
    ring_slots_ptr,
    token_to_req_ptr,
    query_start_loc_ptr,
    logical_positions_ptr,
    pooled_ptr,
    first_positions_ptr,
    stride_raw_row,
    stride_ring_slot,
    stride_ring_row,
    stride_pooled_row,
    num_rows,
    num_ring_slots,
    num_requests,
    RING_CAPACITY: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    valid_request = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_row_start = tl.load(
        query_start_loc_ptr + safe_request, mask=valid_request, other=0
    )
    query_row_end = tl.load(
        query_start_loc_ptr + safe_request + 1, mask=valid_request, other=0
    )
    chunk_start_position = end_position - (row - query_row_start)
    ring_slot = tl.load(ring_slots_ptr + safe_request, mask=valid_request, other=-1)
    valid_ring_slot = (ring_slot >= 0) & (ring_slot < num_ring_slots)
    valid_row = (
        (row < num_rows)
        & valid_request
        & (row >= query_row_start)
        & (row < query_row_end)
        & (end_position >= COMPRESS_RATIO - 1)
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # A group can span the pending ring (older members) and this step's raw rows
    # (members at positions >= chunk_start_position).
    for group_offset in tl.range(0, COMPRESS_RATIO):
        position = end_position - (COMPRESS_RATIO - 1 - group_offset)
        use_raw = position >= chunk_start_position
        raw_row = query_row_start + position - chunk_start_position
        raw_values = tl.load(
            raw_keys_ptr + raw_row * stride_raw_row + dims,
            mask=valid_row
            & use_raw
            & (raw_row >= query_row_start)
            & (raw_row < query_row_end)
            & (raw_row < num_rows)
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        ring_values = tl.load(
            ring_ptr
            + tl.maximum(ring_slot, 0).to(tl.int64) * stride_ring_slot
            + (position % RING_CAPACITY) * stride_ring_row
            + dims,
            mask=valid_row
            & ~use_raw
            & valid_ring_slot
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.where(use_raw, raw_values, ring_values)

    tl.store(
        pooled_ptr + row * stride_pooled_row + dims,
        accumulator / COMPRESS_RATIO,
        mask=(row < num_rows) & (dims < HEAD_DIM),
    )
    first_position = end_position - COMPRESS_RATIO + 1
    tl.store(
        first_positions_ptr + row,
        tl.where(valid_row, first_position, 0),
        mask=row < num_rows,
    )


@triton.jit
def _index_norm_rope_kernel(
    x_ptr,
    positions_ptr,
    cos_sin_ptr,
    weight_ptr,
    out_ptr,
    dest_rows_ptr,
    stride_x_row,
    stride_out_row,
    stride_cos_sin_row,
    num_rows,
    eps,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_HALF: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_DEST_ROWS: tl.constexpr,
) -> None:
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    live = rows < num_rows
    dims = tl.arange(0, BLOCK_D)
    in_dim = dims < HEAD_DIM
    in_rotary = dims < 2 * ROTARY_HALF
    # NeoX pairs dim d with d + rotary_dim/2; both halves are read so the rotation needs
    # no cross-lane shuffle.
    pair = dims % ROTARY_HALF
    partner = tl.where(dims < ROTARY_HALF, dims + ROTARY_HALF, dims - ROTARY_HALF)
    partner = tl.where(in_rotary, partner, dims)

    base = x_ptr + rows[:, None].to(tl.int64) * stride_x_row
    mask = live[:, None] & in_dim[None, :]
    x = tl.load(base + dims[None, :], mask=mask, other=0.0).to(tl.float32)
    x_partner = tl.load(base + partner[None, :], mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + dims, mask=in_dim, other=0.0).to(tl.float32) + 1.0
    weight_partner = (
        tl.load(weight_ptr + partner, mask=in_dim, other=0.0).to(tl.float32) + 1.0
    )
    rrms = tl.rsqrt(tl.sum(x * x, axis=1) / HEAD_DIM + eps)
    y = x * rrms[:, None] * weight[None, :]
    y_partner = x_partner * rrms[:, None] * weight_partner[None, :]

    position = tl.load(positions_ptr + rows // HEADS, mask=live, other=0).to(tl.int64)
    cos_base = cos_sin_ptr + position[:, None] * stride_cos_sin_row
    rotary_mask = live[:, None] & in_rotary[None, :]
    cos = tl.load(cos_base + pair[None, :], mask=rotary_mask, other=1.0)
    sin = tl.load(cos_base + ROTARY_HALF + pair[None, :], mask=rotary_mask, other=0.0)
    sign = tl.where(dims < ROTARY_HALF, -1.0, 1.0)
    result = tl.where(in_rotary[None, :], y * cos + sign[None, :] * y_partner * sin, y)

    if HAS_DEST_ROWS:
        dest = tl.load(dest_rows_ptr + rows, mask=live, other=-1)
        live = live & (dest >= 0)
        dest_row = tl.maximum(dest, 0).to(tl.int64)
    else:
        dest_row = rows.to(tl.int64)
    tl.store(
        out_ptr + dest_row[:, None] * stride_out_row + dims[None, :],
        result.to(out_ptr.dtype.element_ty),
        mask=live[:, None] & in_dim[None, :],
    )


@triton.jit
def _store_qsa_rows_kernel(
    cache_ptr,
    slots_ptr,
    rows_ptr,
    stride_cache_block,
    stride_cache_token,
    stride_rows_row,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_blocks * PAGE_SIZE)
    block = tl.maximum(slot, 0) // PAGE_SIZE
    token = tl.maximum(slot, 0) % PAGE_SIZE
    values = tl.load(
        rows_ptr + row * stride_rows_row + dims,
        mask=valid & (dims < WIDTH),
        other=0,
    )
    tl.store(
        cache_ptr
        + block.to(tl.int64) * stride_cache_block
        + token * stride_cache_token
        + dims,
        values,
        mask=valid & (dims < WIDTH),
    )


def qsa_compress_groups(
    raw_keys: torch.Tensor,
    ring: torch.Tensor,
    ring_slots: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compress_ratio: int,
    pooled: torch.Tensor,
    first_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool each row's closing group from the pending ring and this step's raw rows."""

    rows = raw_keys.shape[0]
    head_dim = raw_keys.shape[1]
    if ring.ndim != 3 or ring.shape[2] != head_dim:
        raise ValueError("QSA pending ring must be [slots, capacity, head_dim]")
    if ring.shape[1] < compress_ratio:
        raise ValueError("QSA ring capacity must cover a whole group")
    if raw_keys.stride(1) != 1 or ring.stride(2) != 1 or pooled.stride(1) != 1:
        raise ValueError("QSA compression needs unit-stride key rows")
    if not rows:
        return pooled, first_positions
    _compress_qsa_groups_kernel[(rows,)](
        raw_keys,
        ring,
        ring_slots,
        token_to_req,
        query_start_loc,
        logical_positions,
        pooled,
        first_positions,
        raw_keys.stride(0),
        ring.stride(0),
        ring.stride(1),
        pooled.stride(0),
        rows,
        ring.shape[0],
        query_start_loc.shape[0] - 1,
        RING_CAPACITY=ring.shape[1],
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return pooled, first_positions


def qsa_index_norm_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    heads: int = 1,
    dest_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """Zero-centered RMSNorm then partial NeoX rope on [rows, head_dim] indexer rows."""

    rows, head_dim = x.shape
    rotary_dim = cos_sin_cache.shape[1]
    if rotary_dim % 2 or rotary_dim > head_dim:
        raise ValueError("QSA indexer rope needs an even rotary_dim <= head_dim")
    if x.stride(1) != 1 or out.stride(1) != 1 or not cos_sin_cache.is_contiguous():
        raise ValueError("QSA indexer norm+rope needs unit-stride rows")
    if rows % heads:
        raise ValueError("QSA indexer rows must be a whole number of head groups")
    if not rows:
        return out
    block_r = 8 if head_dim >= 128 else 16
    _index_norm_rope_kernel[(triton.cdiv(rows, block_r),)](
        x,
        positions,
        cos_sin_cache,
        norm_weight,
        out,
        dest_rows,
        x.stride(0),
        out.stride(0),
        cos_sin_cache.stride(0),
        rows,
        eps,
        HEADS=heads,
        HEAD_DIM=head_dim,
        ROTARY_HALF=rotary_dim // 2,
        BLOCK_R=block_r,
        BLOCK_D=triton.next_power_of_2(head_dim),
        HAS_DEST_ROWS=dest_rows is not None,
        num_warps=4,
    )
    return out


def qsa_store_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Scatter rows into a ``[blocks, block_size, width]`` cache at ``block * block_size +
    offset``; negative slots are dropped. The cache may be a strided per-layer view."""

    if cache.ndim != 3 or rows.ndim != 2 or rows.shape[1] != cache.shape[2]:
        raise ValueError("QSA row store needs a [blocks, block_size, width] cache")
    if cache.stride(2) != 1 or rows.stride(1) != 1:
        raise ValueError("QSA row store needs unit-stride rows")
    if rows.shape[0] != slot_mapping.numel():
        raise ValueError("QSA row store slots and rows disagree")
    if not rows.shape[0]:
        return
    _store_qsa_rows_kernel[(rows.shape[0],)](
        cache,
        slot_mapping,
        rows,
        cache.stride(0),
        cache.stride(1),
        rows.stride(0),
        rows.shape[0],
        cache.shape[0],
        PAGE_SIZE=cache.shape[1],
        WIDTH=cache.shape[2],
        BLOCK_D=triton.next_power_of_2(cache.shape[2]),
        num_warps=4,
    )


__all__ = ["qsa_compress_groups", "qsa_index_norm_rope", "qsa_store_rows"]
