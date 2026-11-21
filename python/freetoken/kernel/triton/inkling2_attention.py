"""Paged attention with Inkling 2's data-dependent relative position bias and per-token sink logit.

One kernel serves prefill (any number of query rows per request, keys = the cached prefix plus this
chunk, all read back from the pool after the store) and decode / verify (a few rows per request).
The score of query row ``i`` (position ``p_i``) against key ``j`` (position ``j``, its index in the
request) is ``scale * q_i.k_j + bias_i(p_i - j)`` with ``bias_i(d) = sum_r rel_i[r] * table[r, d]``
for ``0 <= d < extent`` and 0 beyond; each row also sees one extra logit ``sink_i`` with a zero value.
A sliding layer keeps keys with ``p_i - j < window``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _inkling2_attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, rel_ptr, table_ptr, sink_ptr,
    qo_indptr_ptr, kv_indptr_ptr, kv_indices_ptr, q_pos_ptr,
    sm_scale,
    stride_qt, stride_qh,
    stride_kt, stride_kh,
    stride_vt, stride_vh,
    stride_ot, stride_oh,
    stride_rt, stride_rh,
    stride_st, stride_sh,
    GROUP: tl.constexpr, HEAD_DIM: tl.constexpr, REL_DIM: tl.constexpr, EXTENT: tl.constexpr,
    WINDOW: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    req = tl.program_id(0)
    m_block = tl.program_id(1)
    head = tl.program_id(2)
    kv_head = head // GROUP

    q_start = tl.load(qo_indptr_ptr + req)
    q_end = tl.load(qo_indptr_ptr + req + 1)
    m0 = q_start + m_block * BLOCK_M
    if m0 >= q_end:
        return
    rows = m0 + tl.arange(0, BLOCK_M)
    row_mask = rows < q_end
    rows64 = rows.to(tl.int64)
    q_pos = tl.load(q_pos_ptr + rows64, mask=row_mask, other=0).to(tl.int32)

    kv_start = tl.load(kv_indptr_ptr + req)
    kv_len = tl.load(kv_indptr_ptr + req + 1) - kv_start

    d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + rows64[:, None] * stride_qt + head * stride_qh + d[None, :], mask=row_mask[:, None], other=0.0)
    sink = tl.load(sink_ptr + rows64 * stride_st + head * stride_sh, mask=row_mask, other=0.0).to(tl.float32)

    # the sink is a key with logit `sink` and value 0: it seeds the running max and denominator
    m_i = sink
    l_i = tl.full((BLOCK_M,), 1.0, tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    pos_max = tl.max(tl.where(row_mask, q_pos, 0), axis=0)
    hi = tl.minimum(pos_max + 1, kv_len)
    lo = 0
    if WINDOW > 0:
        pos_min = tl.min(tl.where(row_mask, q_pos, pos_max), axis=0)
        lo = tl.maximum(pos_min - WINDOW + 1, 0)

    for n0 in range(lo, hi, BLOCK_N):
        cols = n0 + tl.arange(0, BLOCK_N)
        col_mask = cols < hi
        slots = tl.load(kv_indices_ptr + kv_start + cols, mask=col_mask, other=0).to(tl.int64)
        k = tl.load(k_ptr + slots[:, None] * stride_kt + kv_head * stride_kh + d[None, :], mask=col_mask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * sm_scale

        dist = q_pos[:, None] - cols[None, :]
        in_extent = (dist >= 0) & (dist < EXTENT)
        idx = tl.minimum(tl.maximum(dist, 0), EXTENT - 1)
        bias = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for r in tl.static_range(REL_DIM):
            rel_r = tl.load(rel_ptr + rows64 * stride_rt + head * stride_rh + r, mask=row_mask, other=0.0).to(tl.float32)
            prof = tl.load(table_ptr + r * EXTENT + idx).to(tl.float32)
            bias += rel_r[:, None] * prof
        s += tl.where(in_extent, bias, 0.0)

        valid = (dist >= 0) & col_mask[None, :] & row_mask[:, None]
        if WINDOW > 0:
            valid = valid & (dist < WINDOW)
        s = tl.where(valid, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + slots[:, None] * stride_vt + kv_head * stride_vh + d[None, :], mask=col_mask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(o_ptr + rows64[:, None] * stride_ot + head * stride_oh + d[None, :], out.to(o_ptr.dtype.element_ty), mask=row_mask[:, None])


def inkling2_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    q_positions: torch.Tensor,
    max_q_len: int,
    rel: torch.Tensor,
    table: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float,
    sliding_window: int | None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``q`` [T, H, D]; ``k_cache`` / ``v_cache`` [slots, KVH, D]; request ``b`` owns query rows
    ``qo_indptr[b]:qo_indptr[b+1]`` and keys ``kv_indices[kv_indptr[b]:kv_indptr[b+1]]`` (key ``j`` at
    position ``j``); ``rel`` [T, H, R], ``table`` [R, E], ``sinks`` [T, H]. Returns [T, H, D] in q's dtype."""
    T, H, D = q.shape
    kvh = k_cache.shape[1]
    assert H % kvh == 0 and k_cache.shape[-1] == D and q.stride(-1) == 1 and k_cache.stride(-1) == 1
    R, E = table.shape
    assert rel.shape == (T, H, R) and rel.stride(-1) == 1 and sinks.shape == (T, H)
    table = table.contiguous()
    if out is None:
        out = torch.empty_like(q)
    bs = qo_indptr.numel() - 1
    if T == 0 or bs == 0:
        return out
    block_m = 16 if max_q_len <= 16 else 64
    block_n = 64
    grid = (bs, triton.cdiv(max_q_len, block_m), H)
    _inkling2_attn_kernel[grid](
        q, k_cache, v_cache, out, rel, table, sinks,
        qo_indptr, kv_indptr, kv_indices, q_positions,
        float(sm_scale),
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1),
        v_cache.stride(0), v_cache.stride(1),
        out.stride(0), out.stride(1),
        rel.stride(0), rel.stride(1),
        sinks.stride(0), sinks.stride(1),
        GROUP=H // kvh, HEAD_DIM=D, REL_DIM=R, EXTENT=E,
        WINDOW=int(sliding_window or 0), BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=4, num_stages=2,
    )
    return out


__all__ = ["inkling2_paged_attention"]
