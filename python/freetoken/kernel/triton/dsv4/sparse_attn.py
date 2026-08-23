"""Sparse gathered-KV flash attention for DeepSeek-V4 MLA.

Each query attends to a per-query *gathered* set of KV positions plus a per-head
attention sink (a null key with logit ``attn_sink[h]`` and zero value, contributing
only to the softmax denominator). KV is a single shared latent head of width
``head_dim`` (MLA: K == V), broadcast across all query heads.

The gather is PAGED: the candidate KV lives in two GLOBAL pools -- the window-ring
pool ``window_pool[n_win_slots, d]`` and the compressed pool ``cmp_pool[n_cmp, d]``
(both batch-less, addressed by physical slot). Per (request, query) the top-k is a
list of GLOBAL slots laid out ``[window part | compressed part]``: entry ``topk[j]``
loads from ``window_pool`` when ``j < n_window`` else from ``cmp_pool`` (``-1`` is
masked). This mirrors sglang's FlashMLA backend, which takes global page indices
directly -- no per-forward ``index_select`` staging slab.

``topk_idxs``' width is a STATIC buffer width (a CUDA-graph capture bakes it), but the
number of columns a replay must actually visit is not: pass ``cmp_counts`` (a device
``[b, m]`` int32 tensor of per-query VALID compressed columns) and the kernel reads its
loop bound from memory instead of walking the whole buffer. This is the same contract as
sglang's ``flash_mla_with_kvcache(indices=..., topk_length=...)``. Without it the loop
covers the full width, which is correct but pays for every ``-1`` column.

Two implementations, picked from the launch shape -- there is no knob:
  * prefill (``m`` > 1) -- one program per (query, request, head block). The query axis alone
    already fills the device, and a split would only add a merge pass.
  * decode (``m`` == 1) -- flash-decoding: the candidate axis is split across programs and
    merged through log-sum-exp. Decode otherwise launches ``b * cdiv(H, BLOCK_H)`` programs
    (4 at bs=1), leaving nearly every SM idle. Falls back to the prefill kernel when the
    candidate list is too short to slice. NOT bit-identical (the reduction order differs).

Shapes: ``q[b, m, h, d]``, ``window_pool[n_win, d]``, ``cmp_pool[n_cmp, d]``,
``topk_idxs[b, m, topk]`` int32 (global slots, window-first), ``attn_sink[h]`` fp32
-> ``o[b, m, h, d]``. ``n_window`` (int) splits the top-k into its two pool halves.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_H = 16
# The gather has exactly ONE tl.load site (the pool base is selected per column), so it stages
# a single [BLOCK_T, D] KV tile -- 67968 B at BLOCK_T=32, num_stages=2, which fits the ~99KB
# consumer-Blackwell (sm_120, e.g. RTX 5090) budget. (BLOCK_T=64 would need ~103KB.)
BLOCK_T = 32
MAX_SPLITS = 32
MIN_TILES_PER_SPLIT = 4


@triton.jit
def _sparse_attn_paged_kernel(
    q_ptr, win_ptr, cmp_ptr, win_s_ptr, cmp_s_ptr, o_ptr, sink_ptr, idx_ptr, cnt_ptr,
    scale,
    H, TOPK, N_WINDOW,
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_wn, stride_wd,
    stride_cn, stride_cd,
    stride_sn, stride_sd,
    stride_ob, stride_om, stride_oh, stride_od,
    stride_ib, stride_im, stride_it,
    stride_nb, stride_nm,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_COUNTS: tl.constexpr,
    QUANT: tl.constexpr,
    QBLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs_h < H
    offs_d = tl.arange(0, D)

    q_ptrs = q_ptr + pid_b * stride_qb + pid_m * stride_qm + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0).to(tl.float32)  # [BLOCK_H, D]

    m_i = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, D), dtype=tl.float32)

    # Columns this query must visit. The window half is always walked in full (N_WINDOW is a
    # small constant); only the compressed tail is bounded by the live count.
    n_active = TOPK
    if HAS_COUNTS:
        n_active = N_WINDOW + tl.load(cnt_ptr + pid_b * stride_nb + pid_m * stride_nm)

    idx_base = idx_ptr + pid_b * stride_ib + pid_m * stride_im
    for t in range(0, tl.cdiv(n_active, BLOCK_T)):
        offs_t = t * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = offs_t < n_active
        idxs = tl.load(idx_base + offs_t * stride_it, mask=t_mask, other=-1)
        valid = idxs >= 0
        # Window-first partition: top-k column j < N_WINDOW reads window_pool, else cmp_pool.
        # Both pools are contiguous [*, D] with identical strides (asserted in the wrapper), so we
        # select the per-column pool BASE pointer and issue a SINGLE gather load. Keeping exactly
        # one ``tl.load`` site (rather than a two-pool ``tl.where`` over two loaded tiles, or an
        # if/elif/else with several load sites) stops Triton's software pipeliner from staging
        # multiple KV tiles in shared memory. Result is bit-identical to the two-pool form -- each
        # column still reads from the same pool/slot.
        is_win = offs_t < N_WINDOW
        base = tl.where(is_win, win_ptr, cmp_ptr)  # [BLOCK_T] per-column pool base pointer
        kv_ptrs = base[:, None] + idxs[:, None] * stride_wn + offs_d[None, :] * stride_wd
        kv = tl.load(kv_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)  # [BLOCK_T, D]

        if QUANT:
            # 8-bit pool: one fp16 scale per QBLOCK dims along head_dim. The scale varies
            # along the reduction axis of q @ kv, so it cannot be folded in after the dot --
            # the tile is dequantized here, before it is used. Indexing the scale per-d
            # (rather than loading [BLOCK_T, D//QBLOCK] and broadcasting) keeps this to one
            # extra load site: the D/QBLOCK distinct values per row stay hot in L1.
            s_base = tl.where(is_win, win_s_ptr, cmp_s_ptr)
            s_ptrs = (s_base[:, None] + idxs[:, None] * stride_sn
                      + (offs_d[None, :] // QBLOCK) * stride_sd)
            kv = kv * tl.load(s_ptrs, mask=valid[:, None], other=1.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(kv)) * scale  # [BLOCK_H, BLOCK_T]
        scores = tl.where(valid[None, :], scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        # A tile with no valid column leaves m_new at -inf; -inf - -inf is NaN, so short-circuit
        # both terms. Identical results whenever m_new is finite (alpha -> exp(m_i - m_new),
        # masked-off p -> exp(-inf - finite) == 0.0), which is every non-degenerate query.
        alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))
        p = tl.where(valid[None, :], tl.exp(scores - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new

    sink = tl.load(sink_ptr + offs_h, mask=h_mask, other=0.0).to(tl.float32)
    l_i = l_i + tl.exp(sink - m_i)
    o = acc / l_i[:, None]

    o_ptrs = o_ptr + pid_b * stride_ob + pid_m * stride_om + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o.to(o_ptr.dtype.element_ty), mask=h_mask[:, None])


@triton.jit
def _sparse_attn_paged_splitk_kernel(
    q_ptr, win_ptr, cmp_ptr, win_s_ptr, cmp_s_ptr, mid_o_ptr, mid_lse_ptr, idx_ptr, cnt_ptr,
    scale,
    H, TOPK, N_WINDOW,
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_wn, stride_wd,
    stride_cn, stride_cd,
    stride_sn, stride_sd,
    stride_mb, stride_mm, stride_mh, stride_ms, stride_md,
    stride_lb, stride_lm, stride_lh, stride_ls,
    stride_ib, stride_im, stride_it,
    stride_nb, stride_nm,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_COUNTS: tl.constexpr,
    QUANT: tl.constexpr,
    QBLOCK: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """Stage 1: each program reduces one BLOCK_T-aligned slice of the candidate list and writes
    its normalized partial output + log-sum-exp. The split axis is folded into program_id(0)."""
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

    # BLOCK_T-aligned slices so no program straddles a tile; trailing splits fall off the end
    # and exit, which is how a short candidate list stops paying for the full grid.
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
        q = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0).to(tl.float32)
        idx_base = idx_ptr + pid_b * stride_ib + pid_m * stride_im

        for start in range(split_start, split_end, BLOCK_T):
            offs_t = start + tl.arange(0, BLOCK_T)
            t_mask = offs_t < split_end
            idxs = tl.load(idx_base + offs_t * stride_it, mask=t_mask, other=-1)
            valid = idxs >= 0
            is_win = offs_t < N_WINDOW
            base = tl.where(is_win, win_ptr, cmp_ptr)
            kv_ptrs = base[:, None] + idxs[:, None] * stride_wn + offs_d[None, :] * stride_wd
            kv = tl.load(kv_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)

            if QUANT:
                # 8-bit pool: one fp16 scale per QBLOCK dims along head_dim. The scale varies
                # along the reduction axis of q @ kv, so it cannot be folded in after the dot --
                # the tile is dequantized here, before it is used. Indexing the scale per-d
                # (rather than loading [BLOCK_T, D//QBLOCK] and broadcasting) keeps this to one
                # extra load site: the D/QBLOCK distinct values per row stay hot in L1.
                s_base = tl.where(is_win, win_s_ptr, cmp_s_ptr)
                s_ptrs = (s_base[:, None] + idxs[:, None] * stride_sn
                          + (offs_d[None, :] // QBLOCK) * stride_sd)
                kv = kv * tl.load(s_ptrs, mask=valid[:, None], other=1.0).to(tl.float32)

            scores = tl.dot(q, tl.trans(kv)) * scale
            scores = tl.where(valid[None, :], scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))
            p = tl.where(valid[None, :], tl.exp(scores - m_new[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
            m_i = m_new

    # Normalized partial + its lse; an empty (or all-masked) split reports lse = -inf so the
    # merge's exp(lse - m) weight is exactly 0.
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
    """Stage 2: log-sum-exp merge over the splits. The attention sink joins here (once, as a
    null key with logit ``attn_sink[h]`` and zero value) exactly as v1 applies it at the end.

    One program per (query, request, HEAD) so the accumulator is a [D] vector: a [BLOCK_H, D]
    tile would be 32 KB of registers per program and spill.
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    offs_d = tl.arange(0, D)

    m_i = tl.load(sink_ptr + pid_h).to(tl.float32)
    l_i = 1.0  # the sink's own exp(sink - m_i) == 1
    acc = tl.zeros((D,), dtype=tl.float32)

    mid_base = (
        mid_o_ptr + pid_b * stride_mb + pid_m * stride_mm + pid_h * stride_mh
        + offs_d * stride_md
    )
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
    o_ptrs = (
        o_ptr + pid_b * stride_ob + pid_m * stride_om + pid_h * stride_oh
        + offs_d * stride_od
    )
    tl.store(o_ptrs, o.to(o_ptr.dtype.element_ty))


def split_count(b: int, m: int, h: int, topk: int, device) -> int:
    """How many ways to split the candidate axis; 0 means run the single-program kernel.

    Prefill (m > 1) never splits. Decode splits until either every SM has a program or the
    slices drop below MIN_TILES_PER_SPLIT tiles of real work -- every split, including the ones
    a small live count leaves empty, still stores a [D] partial the merge reads back, so
    over-splitting buys occupancy in stage 1 and pays for it twice in stage 2.
    """
    if m != 1:
        return 0
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    n_splits = min(
        MAX_SPLITS,
        triton.cdiv(topk, MIN_TILES_PER_SPLIT * BLOCK_T),
        max(1, sm_count // (b * triton.cdiv(h, BLOCK_H))),
    )
    return n_splits if n_splits > 1 else 0


def sparse_attn_paged(
    q: torch.Tensor,            # [b, m, h, d]
    window_pool: torch.Tensor,  # [n_win_slots, d]  GLOBAL window-ring pool (this layer)
    cmp_pool: torch.Tensor,     # [n_cmp, d]        GLOBAL compressed pool (this layer)
    attn_sink: torch.Tensor,    # [h]
    topk_idxs: torch.Tensor,    # [b, m, topk] int32, GLOBAL slots, layout [window | compressed]
    n_window: int,              # # of window entries (the first n_window cols of topk)
    softmax_scale: float,
    cmp_counts: torch.Tensor | None = None,  # [b, m] int32, live compressed columns per query
    window_scale: torch.Tensor | None = None,  # [n_win_slots, d//QBLOCK] fp16, 8-bit pools only
    cmp_scale: torch.Tensor | None = None,     # [n_cmp, d//QBLOCK] fp16, 8-bit pools only
) -> torch.Tensor:
    """Paged sparse MLA attention: gather KV from the two global pools inside the kernel.

    ``topk_idxs[..., j]`` is a GLOBAL physical slot: ``j < n_window`` -> ``window_pool``,
    else ``cmp_pool`` (``-1`` masked). No per-forward staging slab.

    ``cmp_counts`` bounds the compressed half per query, read from DEVICE memory so a captured
    graph's work tracks the live position instead of the buffer width. Omit it (prefill/extend,
    where the width is already the real length) to walk the full ``topk_idxs``.

    The kernel variant follows the launch shape (see the module docstring); nothing selects it.
    """
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]
    assert window_pool.shape[1] == d and cmp_pool.shape[1] == d, (window_pool.shape, cmp_pool.shape, d)
    assert 0 <= n_window <= topk, (n_window, topk)
    q = q.contiguous()
    window_pool = window_pool.contiguous()
    cmp_pool = cmp_pool.contiguous()
    idx = topk_idxs.contiguous().to(torch.int32)
    sink = attn_sink.contiguous().to(torch.float32)
    o = torch.empty_like(q)
    # The kernel selects one pool base per column and gathers with a single stride pair, so both
    # pools must share strides. Guaranteed here by .contiguous() on their [*, D] shape.
    assert window_pool.stride() == cmp_pool.stride(), (window_pool.stride(), cmp_pool.stride())

    # 8-bit pools carry a parallel scale array. Same base-pointer selection as the pools, so
    # the same stride-sharing rule applies.
    quant = window_scale is not None
    assert quant == (cmp_scale is not None), "both pools are quantized or neither"
    if quant:
        window_scale = window_scale.contiguous()
        cmp_scale = cmp_scale.contiguous()
        assert window_scale.stride() == cmp_scale.stride(), (
            window_scale.stride(), cmp_scale.stride())
        assert d % window_scale.shape[1] == 0, (d, window_scale.shape)
        qblock = d // window_scale.shape[1]
        stride_sn, stride_sd = window_scale.stride()
    else:
        window_scale = cmp_scale = window_pool  # unused; keeps the arg a valid pointer
        qblock, stride_sn, stride_sd = 1, 0, 0

    has_counts = cmp_counts is not None
    if has_counts:
        cnt = cmp_counts.contiguous().to(torch.int32).view(b, m)
        stride_nb, stride_nm = cnt.stride()
    else:
        cnt, stride_nb, stride_nm = idx, 0, 0

    n_splits = split_count(b, m, h, topk, q.device)
    if n_splits:
        return _sparse_attn_paged_splitk(
            q, window_pool, cmp_pool, window_scale, cmp_scale, sink, idx, cnt, o,
            b, m, h, d, topk, n_window, softmax_scale, has_counts, stride_nb, stride_nm,
            n_splits, quant, qblock, stride_sn, stride_sd,
        )

    grid = (m, b, triton.cdiv(h, BLOCK_H))
    _sparse_attn_paged_kernel[grid](
        q, window_pool, cmp_pool, window_scale, cmp_scale, o, sink, idx, cnt,
        float(softmax_scale),
        h, topk, int(n_window),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        window_pool.stride(0), window_pool.stride(1),
        cmp_pool.stride(0), cmp_pool.stride(1),
        stride_sn, stride_sd,
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        idx.stride(0), idx.stride(1), idx.stride(2),
        stride_nb, stride_nm,
        D=d,
        BLOCK_H=BLOCK_H,
        BLOCK_T=BLOCK_T,
        HAS_COUNTS=has_counts,
        QUANT=quant,
        QBLOCK=qblock,
        num_warps=8,
        num_stages=2,
    )
    return o


def _sparse_attn_paged_splitk(
    q, window_pool, cmp_pool, window_scale, cmp_scale, sink, idx, cnt, o,
    b, m, h, d, topk, n_window, softmax_scale, has_counts, stride_nb, stride_nm, n_splits,
    quant, qblock, stride_sn, stride_sd,
):
    head_blocks = triton.cdiv(h, BLOCK_H)
    mid_o = torch.empty((b, m, h, n_splits, d), dtype=torch.float32, device=q.device)
    mid_lse = torch.empty((b, m, h, n_splits), dtype=torch.float32, device=q.device)

    _sparse_attn_paged_splitk_kernel[(m * n_splits, b, head_blocks)](
        q, window_pool, cmp_pool, window_scale, cmp_scale, mid_o, mid_lse, idx, cnt,
        float(softmax_scale),
        h, topk, int(n_window),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        window_pool.stride(0), window_pool.stride(1),
        cmp_pool.stride(0), cmp_pool.stride(1),
        stride_sn, stride_sd,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3), mid_o.stride(4),
        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
        idx.stride(0), idx.stride(1), idx.stride(2),
        stride_nb, stride_nm,
        D=d,
        BLOCK_H=BLOCK_H,
        BLOCK_T=BLOCK_T,
        HAS_COUNTS=has_counts,
        QUANT=quant,
        QBLOCK=qblock,
        NUM_SPLITS=n_splits,
        num_warps=8,
        num_stages=2,
    )
    _sparse_attn_splitk_merge_kernel[(m, b, h)](
        mid_o, mid_lse, o, sink,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3), mid_o.stride(4),
        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        D=d,
        NUM_SPLITS=n_splits,
        num_warps=4,
    )
    return o


__all__ = ["sparse_attn_paged", "split_count"]
