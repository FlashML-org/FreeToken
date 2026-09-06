"""Sparse gathered-KV MLA attention for GLM-5.2 DSA.

Each query attends a per-query top-k set of latent rows gathered from the paged MLA
pool. One global pool ``pool[rows, 576]`` holds ``ckv (512) | kpe (64)`` per token
(page_size=1, so a "page index" IS a token row). Keys are the full 576-dim row;
values are the first 512 (ckv) -- MLA weight absorption, so the output is the
512-wide latent context that the model applies ``W_uv`` to.

Structure follows ``dsv4/sparse_attn.py``'s prefill kernel, with three deltas:

* the row width 576 is not a power of two, so the gather loads the 512-dim value
  part and the 64-dim rope tail separately (``tl.arange`` needs pow-2 spans) and the
  score is the sum of the two partial dots. The accumulator only carries the 512
  value dims -- the rope tail contributes to scores, never to the output.
* one pool, no window half (``n_window == 0`` always), no attention sink.
* ``counts`` (device ``[b, m]`` int32) bounds the columns a CUDA-graph replay
  visits, exactly like DSV4's ``cmp_counts``; ``-1`` indices are masked, so the
  kernel is exact for ``kv_len < topk`` (short contexts select every live row and
  the result equals dense attention).

Shapes: ``q[b, m, h, 576]``, ``pool[rows, 576]``, ``topk_idxs[b, m, topk]`` int32
global rows -> ``o[b, m, h, 512]``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_H = 16
BLOCK_T = 32
MAX_SPLITS = 32
MIN_TILES_PER_SPLIT = 4


@triton.jit
def _glm_dsa_sparse_kernel(
    q_ptr, pool_ptr, o_ptr, idx_ptr, cnt_ptr,
    scale,
    H, TOPK,
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_pn, stride_pd,
    stride_ob, stride_om, stride_oh, stride_od,
    stride_ib, stride_im, stride_it,
    stride_nb, stride_nm,
    D_V: tl.constexpr,
    D_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_COUNTS: tl.constexpr,
    HAS_ROPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs_h < H
    offs_v = tl.arange(0, D_V)

    q_base = q_ptr + pid_b * stride_qb + pid_m * stride_qm + offs_h[:, None] * stride_qh
    q_v = tl.load(q_base + offs_v[None, :] * stride_qd, mask=h_mask[:, None], other=0.0).to(tl.float32)
    if HAS_ROPE:
        # NoPE checkpoints (glm5_next) have D_R == 0: tl.arange needs a non-empty
        # span, so the whole rope half is compiled out on the constexpr.
        offs_r = tl.arange(0, D_R)
        q_r = tl.load(q_base + (D_V + offs_r[None, :]) * stride_qd, mask=h_mask[:, None], other=0.0).to(tl.float32)

    m_i = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, D_V), dtype=tl.float32)

    n_active = TOPK
    if HAS_COUNTS:
        n_active = tl.load(cnt_ptr + pid_b * stride_nb + pid_m * stride_nm)

    idx_base = idx_ptr + pid_b * stride_ib + pid_m * stride_im
    for t in range(0, tl.cdiv(n_active, BLOCK_T)):
        offs_t = t * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = offs_t < n_active
        idxs = tl.load(idx_base + offs_t * stride_it, mask=t_mask, other=-1)
        valid = idxs >= 0
        kv_base = pool_ptr + idxs[:, None] * stride_pn
        kv_v = tl.load(kv_base + offs_v[None, :] * stride_pd, mask=valid[:, None], other=0.0).to(tl.float32)

        scores = tl.dot(q_v, tl.trans(kv_v))
        if HAS_ROPE:
            kv_r = tl.load(kv_base + (D_V + offs_r[None, :]) * stride_pd, mask=valid[:, None], other=0.0).to(tl.float32)
            scores += tl.dot(q_r, tl.trans(kv_r))
        scores = scores * scale
        scores = tl.where(valid[None, :], scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        # All-masked tile: keep the -inf running max NaN-free (see dsv4/sparse_attn.py).
        alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))
        p = tl.where(valid[None, :], tl.exp(scores - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv_v.dtype), kv_v)
        m_i = m_new

    o = acc / l_i[:, None]
    o_ptrs = o_ptr + pid_b * stride_ob + pid_m * stride_om + offs_h[:, None] * stride_oh + offs_v[None, :] * stride_od
    tl.store(o_ptrs, o.to(o_ptr.dtype.element_ty), mask=h_mask[:, None])


@triton.jit
def _glm_dsa_decode_logits_kernel(
    q_ptr, w_ptr, pool_ptr, rows_ptr, valid_ptr, out_ptr,
    N_STAGE,
    stride_qb, stride_qh, stride_qd,
    stride_wb, stride_wh,
    stride_pr, stride_pd,
    stride_rb, stride_rw,
    stride_ob, stride_ot,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """One decode query per request, gathering its own index keys (GLM variant of
    dsv4's ``_indexer_decode_logits_kernel``: token-level rows straight off the row
    snapshot, no compressed-block RATIO math).

    Head-reduced score ``logits[t] = sum_h w_h * relu(q_h . k_t)`` (the caller folds
    ``index_head_dim**-0.5`` into ``w``). Keys are gathered from the paged index-key
    pool inside the kernel and the column count is read from DEVICE memory, so a
    captured graph's work tracks the live position instead of the staged width; tiles
    past the live count store ``-inf`` and return, which also blanks the garbage the
    page table carries beyond each request's length.
    """
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    out_ptrs = out_ptr + pid_b * stride_ob + offs_t * stride_ot
    store_mask = offs_t < N_STAGE

    n_valid = tl.load(valid_ptr + pid_b)
    if pid_t * BLOCK_T >= n_valid:
        tl.store(out_ptrs, tl.full((BLOCK_T,), float("-inf"), tl.float32), mask=store_mask)
        return

    t_mask = offs_t < n_valid
    offs_d = tl.arange(0, D)
    offs_h = tl.arange(0, BLOCK_H)
    h_mask = offs_h < H

    rows = tl.load(rows_ptr + pid_b * stride_rb + offs_t * stride_rw, mask=t_mask, other=0)
    rows = tl.maximum(rows, 0)
    k = tl.load(pool_ptr + rows[:, None] * stride_pr + offs_d[None, :] * stride_pd,
                mask=t_mask[:, None], other=0.0)
    q = tl.load(q_ptr + pid_b * stride_qb + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd,
                mask=h_mask[:, None], other=0.0)
    w = tl.load(w_ptr + pid_b * stride_wb + offs_h * stride_wh, mask=h_mask, other=0.0).to(tl.float32)

    score = tl.dot(q, tl.trans(k))  # [BLOCK_H, BLOCK_T] fp32
    score = tl.maximum(score, 0.0) * w[:, None]
    logits = tl.sum(score, axis=0)  # [BLOCK_T]

    tl.store(out_ptrs, tl.where(t_mask, logits, float("-inf")), mask=store_mask)


def glm_dsa_decode_logits(
    q: torch.Tensor,       # [B, H, D] bf16, one index query per request
    weights: torch.Tensor, # [B, H] fp32 (head gate; softmax scale folded in by caller)
    idx_pool: torch.Tensor,# [R, D] bf16, this slot's paged index keys
    rows: torch.Tensor,    # [B, W] int, physical-row snapshot in position order
    valid: torch.Tensor,   # [B] int32, live length per request (device-read)
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Head-reduced indexer logits ``[B, W]`` fp32 for a decode step.

    Equivalent to gathering ``idx_pool[rows]``, ``einsum('bhd,btd->bht').relu() * w``
    and a head sum with ``-inf`` past ``valid`` -- without the [B, W, D] key transient
    or the [B, H, W] score transient, and doing real work only over live columns.
    """
    b, h, d = q.shape
    w_stage = rows.shape[1]
    assert d == triton.next_power_of_2(d), f"index_head_dim must be pow2, got {d}"
    assert h >= 16, f"tl.dot needs >= 16 index heads (BLOCK_H), got {h}"  # GLM-5.2: 32
    weights = weights.to(torch.float32).contiguous()
    if out is None:
        out = torch.empty(b, w_stage, dtype=torch.float32, device=q.device)
    BLOCK_T = 64
    _glm_dsa_decode_logits_kernel[(b, triton.cdiv(w_stage, BLOCK_T))](
        q, weights, idx_pool, rows, valid, out,
        w_stage,
        q.stride(0), q.stride(1), q.stride(2),
        weights.stride(0), weights.stride(1),
        idx_pool.stride(0), idx_pool.stride(1),
        rows.stride(0), rows.stride(1),
        out.stride(0), out.stride(1),
        H=h, D=d,
        BLOCK_H=triton.next_power_of_2(h), BLOCK_T=BLOCK_T,
        num_warps=4, num_stages=2,
    )
    return out


@triton.jit
def _glm_dsa_splitk_kernel(
    q_ptr, pool_ptr, mid_o_ptr, mid_lse_ptr, idx_ptr, cnt_ptr,
    scale,
    H, TOPK,
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_pn, stride_pd,
    stride_mb, stride_mm, stride_mh, stride_ms, stride_md,
    stride_lb, stride_lm, stride_lh, stride_ls,
    stride_ib, stride_im, stride_it,
    stride_nb, stride_nm,
    D_V: tl.constexpr,
    D_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_COUNTS: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """Stage 1 (decode flash-decoding): each program reduces one BLOCK_T-aligned slice of
    the candidate list and writes a normalized [D_V] partial + log-sum-exp. Same structure
    as dsv4's split-k stage, with the two-part 512|64 gather and no window half."""
    pid_ms = tl.program_id(0)
    pid_m = pid_ms // NUM_SPLITS
    split_id = pid_ms % NUM_SPLITS
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs_h < H
    offs_v = tl.arange(0, D_V)

    n_active = TOPK
    if HAS_COUNTS:
        n_active = tl.load(cnt_ptr + pid_b * stride_nb + pid_m * stride_nm)

    per_split = tl.cdiv(tl.cdiv(n_active, NUM_SPLITS), BLOCK_T) * BLOCK_T
    split_start = per_split * split_id
    split_end = tl.minimum(split_start + per_split, n_active)

    m_i = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, D_V), dtype=tl.float32)

    if split_end > split_start:
        q_base = q_ptr + pid_b * stride_qb + pid_m * stride_qm + offs_h[:, None] * stride_qh
        q_v = tl.load(q_base + offs_v[None, :] * stride_qd, mask=h_mask[:, None], other=0.0).to(tl.float32)
        if HAS_ROPE:
            offs_r = tl.arange(0, D_R)
            q_r = tl.load(q_base + (D_V + offs_r[None, :]) * stride_qd, mask=h_mask[:, None], other=0.0).to(tl.float32)
        idx_base = idx_ptr + pid_b * stride_ib + pid_m * stride_im

        for start in range(split_start, split_end, BLOCK_T):
            offs_t = start + tl.arange(0, BLOCK_T)
            t_mask = offs_t < split_end
            idxs = tl.load(idx_base + offs_t * stride_it, mask=t_mask, other=-1)
            valid = idxs >= 0
            kv_base = pool_ptr + idxs[:, None] * stride_pn
            kv_v = tl.load(kv_base + offs_v[None, :] * stride_pd, mask=valid[:, None], other=0.0).to(tl.float32)

            scores = tl.dot(q_v, tl.trans(kv_v))
            if HAS_ROPE:
                kv_r = tl.load(kv_base + (D_V + offs_r[None, :]) * stride_pd, mask=valid[:, None], other=0.0).to(tl.float32)
                scores += tl.dot(q_r, tl.trans(kv_r))
            scores = scores * scale
            scores = tl.where(valid[None, :], scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))
            p = tl.where(valid[None, :], tl.exp(scores - m_new[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(kv_v.dtype), kv_v)
            m_i = m_new

    out = tl.where(l_i[:, None] == 0.0, 0.0, acc / l_i[:, None])
    lse = tl.where(l_i == 0.0, -float("inf"), m_i + tl.log(l_i))

    mid_base = (
        mid_o_ptr + pid_b * stride_mb + pid_m * stride_mm
        + offs_h[:, None] * stride_mh + split_id * stride_ms + offs_v[None, :] * stride_md
    )
    tl.store(mid_base, out, mask=h_mask[:, None])
    lse_base = (
        mid_lse_ptr + pid_b * stride_lb + pid_m * stride_lm
        + offs_h * stride_lh + split_id * stride_ls
    )
    tl.store(lse_base, lse, mask=h_mask)


@triton.jit
def _glm_dsa_merge_kernel(
    mid_o_ptr, mid_lse_ptr, o_ptr,
    stride_mb, stride_mm, stride_mh, stride_ms, stride_md,
    stride_lb, stride_lm, stride_lh, stride_ls,
    stride_ob, stride_om, stride_oh, stride_od,
    D_V: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """Stage 2: log-sum-exp merge over the splits (no attention sink -- GLM DSA has none,
    so the running max/denominator start empty instead of seeded by a null key)."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    offs_v = tl.arange(0, D_V)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((D_V,), dtype=tl.float32)

    mid_base = (
        mid_o_ptr + pid_b * stride_mb + pid_m * stride_mm + pid_h * stride_mh
        + offs_v * stride_md
    )
    lse_base = mid_lse_ptr + pid_b * stride_lb + pid_m * stride_lm + pid_h * stride_lh

    for split_id in tl.range(0, NUM_SPLITS, num_stages=2):
        partial = tl.load(mid_base + split_id * stride_ms)
        lse = tl.load(lse_base + split_id * stride_ls)
        m_new = tl.maximum(m_i, lse)
        alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))
        beta = tl.where(lse == -float("inf"), 0.0, tl.exp(lse - m_new))
        acc = acc * alpha + partial * beta
        l_i = l_i * alpha + beta
        m_i = m_new

    o = acc / l_i
    o_ptrs = (
        o_ptr + pid_b * stride_ob + pid_m * stride_om + pid_h * stride_oh
        + offs_v * stride_od
    )
    tl.store(o_ptrs, o.to(o_ptr.dtype.element_ty))


def _split_count(b: int, m: int, h: int, topk: int, device) -> int:
    """0 = single-program kernel; decode splits until SMs are covered (dsv4 policy)."""
    if m != 1:
        return 0
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    n_splits = min(
        MAX_SPLITS,
        triton.cdiv(topk, MIN_TILES_PER_SPLIT * BLOCK_T),
        max(1, sm_count // (b * triton.cdiv(h, BLOCK_H))),
    )
    return n_splits if n_splits > 1 else 0


def glm_dsa_sparse_attn(
    q: torch.Tensor,          # [b, m, h, d_v + d_r]  (ckv-absorbed | rope)
    pool: torch.Tensor,       # [rows, d_v + d_r]     GLOBAL latent pool slab (this layer)
    topk_idxs: torch.Tensor,  # [b, m|1, topk] int32 global rows, -1 masked
    softmax_scale: float,
    counts: torch.Tensor | None = None,  # [b, m] int32 live columns per query (device-read)
    d_v: int = 512,
    force_splits: int | None = None,     # tests only: 0 = single-program, N = split-k N
) -> torch.Tensor:
    """Sparse MLA attention over gathered latent rows; returns ``[b, m, h, d_v]``.

    ``topk_idxs`` with a singleton query dim while ``m > 1`` broadcasts one shared row
    list across all queries with STRIDE 0 -- the identity-selection dense path: every
    query reads the same position-ordered row list, causally bounded by its own
    ``counts[q] = position + 1``, with zero per-query index materialization.
    """
    b, m, h, d = q.shape
    d_r = d - d_v
    topk = topk_idxs.shape[-1]
    assert pool.shape[-1] == d, (pool.shape, d)
    q = q.contiguous()
    pool_2d = pool.reshape(-1, d)
    assert pool_2d.stride(-1) == 1
    idx = topk_idxs.contiguous().to(torch.int32)
    broadcast_m = idx.shape[1] == 1 and m > 1
    assert broadcast_m or idx.shape[1] == m, (idx.shape, m)
    o = q.new_empty(b, m, h, d_v)

    has_counts = counts is not None
    if has_counts:
        cnt = counts.contiguous().to(torch.int32).view(b, m)
        stride_nb, stride_nm = cnt.stride()
    else:
        cnt, stride_nb, stride_nm = idx, 0, 0

    n_splits = _split_count(b, m, h, topk, q.device) if force_splits is None else force_splits
    if n_splits:
        mid_o = q.new_empty(b, m, h, n_splits, d_v, dtype=torch.float32)
        mid_lse = q.new_empty(b, m, h, n_splits, dtype=torch.float32)
        grid1 = (m * n_splits, b, triton.cdiv(h, BLOCK_H))
        _glm_dsa_splitk_kernel[grid1](
            q, pool_2d, mid_o, mid_lse, idx, cnt,
            float(softmax_scale),
            h, topk,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            pool_2d.stride(0), pool_2d.stride(1),
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3), mid_o.stride(4),
            mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
            idx.stride(0), 0 if broadcast_m else idx.stride(1), idx.stride(2),
            stride_nb, stride_nm,
            D_V=d_v, D_R=d_r,
            BLOCK_H=BLOCK_H, BLOCK_T=BLOCK_T,
            HAS_COUNTS=has_counts, HAS_ROPE=d_r > 0, NUM_SPLITS=n_splits,
            num_warps=4, num_stages=2,
        )
        grid2 = (m, b, h)
        _glm_dsa_merge_kernel[grid2](
            mid_o, mid_lse, o,
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3), mid_o.stride(4),
            mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), mid_lse.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            D_V=d_v, NUM_SPLITS=n_splits,
            num_warps=4,
        )
        return o

    grid = (m, b, triton.cdiv(h, BLOCK_H))
    _glm_dsa_sparse_kernel[grid](
        q, pool_2d, o, idx, cnt,
        float(softmax_scale),
        h, topk,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        pool_2d.stride(0), pool_2d.stride(1),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        idx.stride(0), 0 if broadcast_m else idx.stride(1), idx.stride(2),
        stride_nb, stride_nm,
        D_V=d_v, D_R=d_r,
        BLOCK_H=BLOCK_H, BLOCK_T=BLOCK_T,
        HAS_COUNTS=has_counts, HAS_ROPE=d_r > 0,
        num_warps=4, num_stages=2,
    )
    return o


__all__ = ["glm_dsa_sparse_attn", "glm_dsa_decode_logits"]
