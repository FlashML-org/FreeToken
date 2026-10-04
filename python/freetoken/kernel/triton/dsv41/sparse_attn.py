"""Sparse gathered-KV flash attention over PACKED pools (DeepSeek-V4.1).

Same contract as ``kernel/triton/dsv4/sparse_attn.py`` -- each query attends to a per-query list of
GLOBAL slots laid out ``[window part | compressed part]`` plus a per-head attention sink, K == V is one
shared latent head -- but the two pools are byte pools in (possibly different) ``RowFormat``s: the
window ring holds fp8 rows, the compressed KV holds fp4 rows, and every gathered row is dequantized
in-kernel (``row_format.load_rows``) to the bf16 value the reference cache would have held.

Because the two pools dequantize differently, a tile never mixes them: the window part is
``n_window`` columns with ``n_window % BLOCK_T == 0`` (the caller pads it with ``-1``), so the kernel
walks the window tiles and then the compressed tiles in two loops, each with a single load site.
Both dots take bf16 operands with fp32 accumulation, like the reference ``sparse_attn`` kernel.

Two implementations, picked from the launch shape (no knob):
  * prefill (``m`` > 1): one program per (query, request, head block)
  * decode (``m`` == 1): flash-decoding split over the candidate axis, merged through log-sum-exp
    (the sink joins in the merge). Not bit-identical to the single-program path.

``cmp_counts`` (device ``[b, m]`` int32) bounds the compressed columns a CUDA-graph replay visits;
without it the full ``topk_idxs`` width is walked.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kvcache.dsv4.v41_row_format import RowFormat

from .row_format import load_rows

BLOCK_H = 16
BLOCK_T = 32
MAX_SPLITS = 32
# Decode splits at tile boundaries, bounded by the device's SM count below.
MIN_TILES_PER_SPLIT = 1


@triton.jit
def _attend_tile(q, kv, valid, scale, m_i, l_i, acc):
    """One online-softmax step over a ``[BLOCK_T, D]`` fp32 KV tile (bf16-exact values)."""
    kv16 = kv.to(tl.bfloat16)
    scores = tl.dot(q, tl.trans(kv16)) * scale  # [BLOCK_H, BLOCK_T] fp32
    scores = tl.where(valid[None, :], scores, -float("inf"))
    m_new = tl.maximum(m_i, tl.max(scores, axis=1))
    # an all-masked tile keeps m_new at -inf; short-circuit so -inf - -inf never forms
    alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))
    p = tl.where(valid[None, :], tl.exp(scores - m_new[:, None]), 0.0)
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv16)
    return m_new, l_i, acc


@triton.jit
def _sparse_attn_packed_kernel(
    q_ptr, win_ptr, cmp_ptr, o_ptr, sink_ptr, idx_ptr, cnt_ptr,
    scale,
    H, TOPK, N_WINDOW,
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_ob, stride_om, stride_oh, stride_od,
    stride_ib, stride_im, stride_it,
    stride_nb, stride_nm,
    D: tl.constexpr,
    WIN_FMT: tl.constexpr, WIN_ROW_BYTES: tl.constexpr,
    CMP_FMT: tl.constexpr, CMP_ROW_BYTES: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_COUNTS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs_h < H
    offs_d = tl.arange(0, D)

    q_ptrs = q_ptr + pid_b * stride_qb + pid_m * stride_qm + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0).to(tl.bfloat16)  # [BLOCK_H, D]

    m_i = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, D), dtype=tl.float32)

    n_active = TOPK
    if HAS_COUNTS:
        n_active = N_WINDOW + tl.load(cnt_ptr + pid_b * stride_nb + pid_m * stride_nm)

    idx_base = idx_ptr + pid_b * stride_ib + pid_m * stride_im
    for start in range(0, N_WINDOW, BLOCK_T):
        offs_t = start + tl.arange(0, BLOCK_T)
        idxs = tl.load(idx_base + offs_t * stride_it)
        valid = idxs >= 0
        kv = load_rows(win_ptr, idxs, valid, offs_d, D, WIN_ROW_BYTES, WIN_FMT)
        m_i, l_i, acc = _attend_tile(q, kv, valid, scale, m_i, l_i, acc)
    for start in range(N_WINDOW, n_active, BLOCK_T):
        offs_t = start + tl.arange(0, BLOCK_T)
        t_mask = offs_t < n_active
        idxs = tl.load(idx_base + offs_t * stride_it, mask=t_mask, other=-1)
        valid = idxs >= 0
        kv = load_rows(cmp_ptr, idxs, valid, offs_d, D, CMP_ROW_BYTES, CMP_FMT)
        m_i, l_i, acc = _attend_tile(q, kv, valid, scale, m_i, l_i, acc)

    sink = tl.load(sink_ptr + offs_h, mask=h_mask, other=0.0).to(tl.float32)
    l_i = l_i + tl.exp(sink - m_i)
    o = acc / l_i[:, None]

    o_ptrs = o_ptr + pid_b * stride_ob + pid_m * stride_om + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o.to(o_ptr.dtype.element_ty), mask=h_mask[:, None])


@triton.jit
def _sparse_attn_packed_splitk_kernel(
    q_ptr, win_ptr, cmp_ptr, mid_o_ptr, mid_lse_ptr, idx_ptr, cnt_ptr,
    scale,
    H, TOPK, N_WINDOW,
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_mb, stride_mm, stride_mh, stride_ms, stride_md,
    stride_lb, stride_lm, stride_lh, stride_ls,
    stride_ib, stride_im, stride_it,
    stride_nb, stride_nm,
    D: tl.constexpr,
    WIN_FMT: tl.constexpr, WIN_ROW_BYTES: tl.constexpr,
    CMP_FMT: tl.constexpr, CMP_ROW_BYTES: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_COUNTS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """Stage 1: each program reduces one BLOCK_T-aligned slice of the candidate list and writes
    its normalized partial output + log-sum-exp. A slice may straddle the window / compressed
    boundary; it is walked as (window tiles, compressed tiles) since N_WINDOW is tile-aligned."""
    pid_ms = tl.program_id(0)
    pid_m = pid_ms // NUM_SPLITS
    split_id = pid_ms % NUM_SPLITS
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs_h < H
    offs_d = tl.arange(0, D)

    n_active = TOPK
    if HAS_COUNTS:
        n_active = N_WINDOW + tl.load(cnt_ptr + pid_b * stride_nb + pid_m * stride_nm)

    per_split = tl.cdiv(tl.cdiv(n_active, NUM_SPLITS), BLOCK_T) * BLOCK_T
    split_start = per_split * split_id
    split_end = tl.minimum(split_start + per_split, n_active)

    m_i = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, D), dtype=tl.float32)

    if split_end > split_start:
        q_ptrs = (
            q_ptr + pid_b * stride_qb + pid_m * stride_qm
            + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
        )
        q = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0).to(tl.bfloat16)
        idx_base = idx_ptr + pid_b * stride_ib + pid_m * stride_im

        win_end = tl.minimum(split_end, N_WINDOW)
        for start in range(split_start, win_end, BLOCK_T):
            offs_t = start + tl.arange(0, BLOCK_T)
            idxs = tl.load(idx_base + offs_t * stride_it)
            valid = idxs >= 0
            kv = load_rows(win_ptr, idxs, valid, offs_d, D, WIN_ROW_BYTES, WIN_FMT)
            m_i, l_i, acc = _attend_tile(q, kv, valid, scale, m_i, l_i, acc)
        cmp_start = tl.maximum(split_start, N_WINDOW)
        for start in range(cmp_start, split_end, BLOCK_T):
            offs_t = start + tl.arange(0, BLOCK_T)
            t_mask = offs_t < split_end
            idxs = tl.load(idx_base + offs_t * stride_it, mask=t_mask, other=-1)
            valid = idxs >= 0
            kv = load_rows(cmp_ptr, idxs, valid, offs_d, D, CMP_ROW_BYTES, CMP_FMT)
            m_i, l_i, acc = _attend_tile(q, kv, valid, scale, m_i, l_i, acc)

    out = tl.where(l_i[:, None] == 0.0, 0.0, acc / l_i[:, None])
    lse = tl.where(l_i == 0.0, -float("inf"), m_i + tl.log(l_i))

    mid_base = (
        mid_o_ptr + pid_b * stride_mb + pid_m * stride_mm
        + offs_h[:, None] * stride_mh + split_id * stride_ms + offs_d[None, :] * stride_md
    )
    tl.store(mid_base, out, mask=h_mask[:, None])
    lse_base = (
        mid_lse_ptr + pid_b * stride_lb + pid_m * stride_lm
        + offs_h * stride_lh + split_id * stride_ls
    )
    tl.store(lse_base, lse, mask=h_mask)


@triton.jit
def _sparse_attn_splitk_merge_kernel(
    mid_o_ptr, mid_lse_ptr, o_ptr, sink_ptr,
    stride_mb, stride_mm, stride_mh, stride_ms, stride_md,
    stride_lb, stride_lm, stride_lh, stride_ls,
    stride_ob, stride_om, stride_oh, stride_od,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """Stage 2: log-sum-exp merge over the splits; the attention sink joins here once, as a null
    key with logit ``attn_sink[h]`` and zero value. One program per (query, request, head)."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    offs_d = tl.arange(0, D)

    m_i = tl.load(sink_ptr + pid_h).to(tl.float32)
    l_i = 1.0
    acc = tl.zeros((D,), dtype=tl.float32)

    mid_base = mid_o_ptr + pid_b * stride_mb + pid_m * stride_mm + pid_h * stride_mh + offs_d * stride_md
    lse_base = mid_lse_ptr + pid_b * stride_lb + pid_m * stride_lm + pid_h * stride_lh

    for split_id in tl.range(0, NUM_SPLITS, num_stages=2):
        partial = tl.load(mid_base + split_id * stride_ms)
        lse = tl.load(lse_base + split_id * stride_ls)
        m_new = tl.maximum(m_i, lse)
        alpha = tl.exp(m_i - m_new)
        beta = tl.where(lse == -float("inf"), 0.0, tl.exp(lse - m_new))
        acc = acc * alpha + partial * beta
        l_i = l_i * alpha + beta
        m_i = m_new

    o = acc / l_i
    o_ptrs = o_ptr + pid_b * stride_ob + pid_m * stride_om + pid_h * stride_oh + offs_d * stride_od
    tl.store(o_ptrs, o.to(o_ptr.dtype.element_ty))


def split_count(b: int, m: int, h: int, topk: int, device) -> int:
    """How many ways to split the candidate axis; 0 means run the single-program kernel. Decode
    splits until every SM has a program or every split is down to one tile."""
    if m != 1:
        return 0
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    n_splits = min(
        MAX_SPLITS,
        triton.cdiv(topk, MIN_TILES_PER_SPLIT * BLOCK_T),
        max(1, sm_count // (b * triton.cdiv(h, BLOCK_H))),
    )
    return n_splits if n_splits > 1 else 0


def sparse_attn_packed(
    q: torch.Tensor,             # [b, m, h, d] bf16
    window_pool: torch.Tensor,   # [n_win_slots, win_fmt.row_bytes(d)] uint8
    win_fmt: RowFormat,
    cmp_pool: torch.Tensor,      # [n_cmp, cmp_fmt.row_bytes(d)] uint8
    cmp_fmt: RowFormat,
    attn_sink: torch.Tensor,     # [h]
    topk_idxs: torch.Tensor,     # [b, m, topk] int32, GLOBAL rows, layout [window | compressed]
    n_window: int,               # window columns (a multiple of BLOCK_T; pad with -1)
    softmax_scale: float,
    cmp_counts: torch.Tensor | None = None,  # [b, m] int32 live compressed columns per query
) -> torch.Tensor:
    """Paged sparse MLA attention over packed KV pools; see the module docstring for the contract."""
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]
    assert n_window % BLOCK_T == 0 and 0 <= n_window <= topk, (n_window, topk)
    assert window_pool.dtype == torch.uint8 and window_pool.shape[1] == win_fmt.row_bytes(d), (window_pool.shape, win_fmt)
    assert cmp_pool.dtype == torch.uint8 and cmp_pool.shape[1] == cmp_fmt.row_bytes(d), (cmp_pool.shape, cmp_fmt)
    assert window_pool.is_contiguous() and cmp_pool.is_contiguous()
    q = q.contiguous()
    idx = topk_idxs.contiguous().to(torch.int32)
    sink = attn_sink.contiguous().to(torch.float32)
    o = torch.empty_like(q)

    has_counts = cmp_counts is not None
    if has_counts:
        cnt = cmp_counts.contiguous().to(torch.int32).view(b, m)
        stride_nb, stride_nm = cnt.stride()
    else:
        cnt, stride_nb, stride_nm = idx, 0, 0

    fmt_args = dict(
        WIN_FMT=win_fmt.code, WIN_ROW_BYTES=win_fmt.row_bytes(d),
        CMP_FMT=cmp_fmt.code, CMP_ROW_BYTES=cmp_fmt.row_bytes(d),
    )
    n_splits = split_count(b, m, h, topk, q.device)
    if n_splits:
        head_blocks = triton.cdiv(h, BLOCK_H)
        mid_o = torch.empty((b, m, h, n_splits, d), dtype=torch.float32, device=q.device)
        mid_lse = torch.empty((b, m, h, n_splits), dtype=torch.float32, device=q.device)
        _sparse_attn_packed_splitk_kernel[(m * n_splits, b, head_blocks)](
            q, window_pool, cmp_pool, mid_o, mid_lse, idx, cnt,
            float(softmax_scale),
            h, topk, int(n_window),
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3), mid_o.stride(4),
            mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
            idx.stride(0), idx.stride(1), idx.stride(2),
            stride_nb, stride_nm,
            D=d, BLOCK_H=BLOCK_H, BLOCK_T=BLOCK_T, HAS_COUNTS=has_counts, NUM_SPLITS=n_splits,
            **fmt_args, num_warps=8, num_stages=2,
        )
        _sparse_attn_splitk_merge_kernel[(m, b, h)](
            mid_o, mid_lse, o, sink,
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3), mid_o.stride(4),
            mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            D=d, NUM_SPLITS=n_splits, num_warps=4,
        )
        return o

    grid = (m, b, triton.cdiv(h, BLOCK_H))
    _sparse_attn_packed_kernel[grid](
        q, window_pool, cmp_pool, o, sink, idx, cnt,
        float(softmax_scale),
        h, topk, int(n_window),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        idx.stride(0), idx.stride(1), idx.stride(2),
        stride_nb, stride_nm,
        D=d, BLOCK_H=BLOCK_H, BLOCK_T=BLOCK_T, HAS_COUNTS=has_counts,
        **fmt_args, num_warps=8, num_stages=2,
    )
    return o


__all__ = ["sparse_attn_packed", "split_count", "BLOCK_T"]
