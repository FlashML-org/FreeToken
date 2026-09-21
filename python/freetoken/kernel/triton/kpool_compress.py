"""Pool one token row's group into ``slab[cmp_rows[row]]``: members inside this
forward read the raw K/gate rows, older members read the per-request tail ring.
Adapted from ``qsa/compress.py`` with the mean replaced by a per-channel
softmax(gate + APE) weighted sum (hence the gate stream/ring and the APE input).
Non-closing rows land on a scratch row that scoring never reads."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _compress_kpool_groups_kernel(
    raw_k_ptr,
    raw_g_ptr,
    ring_k_ptr,
    ring_g_ptr,
    ape_ptr,
    ring_slots_ptr,
    token_to_req_ptr,
    query_start_loc_ptr,
    positions_ptr,
    slab_ptr,
    cmp_rows_ptr,
    stride_raw_k_row,
    stride_raw_g_row,
    stride_ring_row,
    stride_slab_row,
    num_rows,
    num_ring_rows,
    num_requests,
    RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    in_dim = dims < HEAD_DIM

    request = tl.load(token_to_req_ptr + row, mask=row < num_rows, other=-1)
    end_position = tl.load(positions_ptr + row, mask=row < num_rows, other=0).to(tl.int64)
    valid_request = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_row_start = tl.load(
        query_start_loc_ptr + safe_request, mask=valid_request, other=0
    ).to(tl.int64)
    chunk_start_position = end_position - (row - query_row_start)
    ring_slot = tl.load(ring_slots_ptr + safe_request, mask=valid_request, other=0).to(
        tl.int64
    )
    # A row whose group has members before position 0 (end_position < RATIO - 1)
    # can never close; keep its loads masked off entirely -- member positions would
    # be negative and C-style % would produce NEGATIVE ring rows (illegal address).
    valid_row = (row < num_rows) & valid_request & (end_position >= RATIO - 1)

    # Two-pass per-channel softmax over the RATIO group members (static loop; the
    # doubled loads are 4 x 512B and free next to the fp32 math).
    m = tl.full((BLOCK_D,), float("-inf"), tl.float32)
    for off in tl.static_range(RATIO):
        position = end_position - (RATIO - 1 - off)
        use_raw = position >= chunk_start_position
        raw_row = query_row_start + position - chunk_start_position
        ring_row = ring_slot * RATIO + position % RATIO
        g_raw = tl.load(
            raw_g_ptr + raw_row * stride_raw_g_row + dims,
            mask=valid_row & use_raw & (raw_row >= 0) & (raw_row < num_rows) & in_dim,
            other=0.0,
        ).to(tl.float32)
        g_ring = tl.load(
            ring_g_ptr + tl.maximum(ring_row, 0) * stride_ring_row + dims,
            mask=valid_row & (~use_raw) & (ring_row >= 0) & (ring_row < num_ring_rows) & in_dim,
            other=0.0,
        ).to(tl.float32)
        ape = tl.load(ape_ptr + off * HEAD_DIM + dims, mask=in_dim, other=0.0)
        m = tl.maximum(m, tl.where(use_raw, g_raw, g_ring) + ape)

    s = tl.zeros((BLOCK_D,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for off in tl.static_range(RATIO):
        position = end_position - (RATIO - 1 - off)
        use_raw = position >= chunk_start_position
        raw_row = query_row_start + position - chunk_start_position
        ring_row = ring_slot * RATIO + position % RATIO
        raw_mask = valid_row & use_raw & (raw_row >= 0) & (raw_row < num_rows) & in_dim
        ring_mask = (
            valid_row & (~use_raw) & (ring_row >= 0) & (ring_row < num_ring_rows) & in_dim
        )
        g_raw = tl.load(
            raw_g_ptr + raw_row * stride_raw_g_row + dims, mask=raw_mask, other=0.0
        ).to(tl.float32)
        g_ring = tl.load(
            ring_g_ptr + tl.maximum(ring_row, 0) * stride_ring_row + dims,
            mask=ring_mask, other=0.0,
        ).to(tl.float32)
        k_raw = tl.load(
            raw_k_ptr + raw_row * stride_raw_k_row + dims, mask=raw_mask, other=0.0
        ).to(tl.float32)
        k_ring = tl.load(
            ring_k_ptr + tl.maximum(ring_row, 0) * stride_ring_row + dims,
            mask=ring_mask, other=0.0,
        ).to(tl.float32)
        ape = tl.load(ape_ptr + off * HEAD_DIM + dims, mask=in_dim, other=0.0)
        e = tl.exp(tl.where(use_raw, g_raw, g_ring) + ape - m)
        s += e
        acc += e * tl.where(use_raw, k_raw, k_ring)

    pooled = acc / s
    dest = tl.load(cmp_rows_ptr + row, mask=row < num_rows, other=0).to(tl.int64)
    tl.store(
        slab_ptr + dest * stride_slab_row + dims,
        pooled.to(slab_ptr.dtype.element_ty),
        mask=(row < num_rows) & in_dim,
    )


def kpool_compress_store(
    k: torch.Tensor,  # [T, D] raw index keys (this forward)
    gate: torch.Tensor,  # [T, D] raw gate scores
    ring_k: torch.Tensor,  # [slots * ratio, D] flat tail ring (keys)
    ring_g: torch.Tensor,  # [slots * ratio, D] flat tail ring (gates)
    ape: torch.Tensor,  # [ratio, D] fp32 model parameter
    ring_slots: torch.Tensor,  # [n_req] Req.table_idx
    token_to_req: torch.Tensor,  # [T]
    cu_seqlens: torch.Tensor,  # [n_req + 1]
    positions: torch.Tensor,  # [T]
    slab: torch.Tensor,  # [shadow_rows + scratch, D]
    cmp_rows: torch.Tensor,  # [T] shadow row (closing) or scratch row
    ratio: int,
) -> None:
    t, d = k.shape
    if t == 0:
        return
    assert ape.dtype == torch.float32 and ape.shape == (ratio, d)
    _compress_kpool_groups_kernel[(t,)](
        k, gate, ring_k, ring_g, ape,
        ring_slots, token_to_req, cu_seqlens, positions,
        slab, cmp_rows,
        k.stride(0), gate.stride(0), ring_k.stride(0), slab.stride(0),
        t, ring_k.shape[0], ring_slots.numel(),
        RATIO=ratio, HEAD_DIM=d, BLOCK_D=triton.next_power_of_2(d),
    )


__all__ = ["kpool_compress_store"]
