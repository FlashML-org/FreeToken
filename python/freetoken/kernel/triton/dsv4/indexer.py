"""Lightning Indexer logits for DeepSeek-V4 (fused, memory-frugal).

Computes the head-reduced index score

    ``logits[b, s, t] = sum_h relu(q[b, s, h, :] . k[b, t, :]) * weights[b, s, h]``

in a single Triton pass. This is the ``mqa_attn_return_logits`` of DeepSeek's
Lightning Indexer: ``q`` carries ``n_heads`` index heads, ``k`` is a single shared
(MQA) compressed-KV head broadcast across them, and ``weights`` already folds in
``softmax_scale`` (so the kernel applies no extra scaling).

The reference path materialises the per-head score tensor
``index_score = einsum("bshd,btd->bsht")`` of shape ``[b, s, n_heads, t]`` before
reducing over heads -- an O(s * n_heads * t) transient that OOMs long-context
prefill on a 32 GB card. This kernel reduces over heads *inside* the kernel, so
only the O(s * t) ``logits`` ever hit HBM. Masking + top-k stay in PyTorch (they
operate on the small ``[b, s, t]`` tensor).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _indexer_logits_kernel(
    q_ptr, k_ptr, w_ptr, out_ptr,
    S, T,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kd,
    stride_wb, stride_ws, stride_wh,
    stride_ob, stride_os, stride_ot,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_t = tl.program_id(2)

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, D)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    h_mask = offs_h < H
    t_mask = offs_t < T

    # Q for this query row: [BLOCK_H, D]. Padded heads load 0 (-> 0 score -> 0 weight).
    q_ptrs = (
        q_ptr + pid_b * stride_qb + pid_s * stride_qs
        + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)

    # K for this KV tile: [BLOCK_T, D].
    k_ptrs = (
        k_ptr + pid_b * stride_kb
        + offs_t[:, None] * stride_kt + offs_d[None, :] * stride_kd
    )
    k = tl.load(k_ptrs, mask=t_mask[:, None], other=0.0)

    w = tl.load(
        w_ptr + pid_b * stride_wb + pid_s * stride_ws + offs_h * stride_wh,
        mask=h_mask, other=0.0,
    ).to(tl.float32)

    # score[h, t] = relu(q_h . k_t); accumulate in fp32 (more precise than the bf16 ref).
    score = tl.dot(q, tl.trans(k))            # [BLOCK_H, BLOCK_T] fp32
    score = tl.maximum(score, 0.0)
    score = score * w[:, None]
    logits = tl.sum(score, axis=0)            # [BLOCK_T]

    out_ptrs = out_ptr + pid_b * stride_ob + pid_s * stride_os + offs_t * stride_ot
    tl.store(out_ptrs, logits, mask=t_mask)


def indexer_logits(
    q: torch.Tensor,        # [B, S, H, D] bf16  (index queries, fp4 round-tripped)
    k: torch.Tensor,        # [B, T, D]    bf16  (compressed KV, shared across heads)
    weights: torch.Tensor,  # [B, S, H]          (per-head gate, softmax_scale folded in)
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Head-reduced Lightning-Indexer logits ``[B, S, T]`` (fp32).

    Equivalent to ``(einsum("bshd,btd->bsht", q, k).relu_() * weights[..., None]).sum(2)``
    but without the ``[B, S, H, T]`` transient.
    """
    B, S, H, D = q.shape
    T = k.shape[1]
    assert k.shape[0] == B and k.shape[2] == D, (q.shape, k.shape)
    assert weights.shape == (B, S, H), (weights.shape, (B, S, H))
    assert D == triton.next_power_of_2(D), f"index_head_dim must be pow2, got {D}"

    if out is None:
        out = torch.empty((B, S, T), dtype=torch.float32, device=q.device)
    if T == 0:
        return out

    BLOCK_H = triton.next_power_of_2(H)
    BLOCK_T = 128
    grid = (S, B, triton.cdiv(T, BLOCK_T))
    _indexer_logits_kernel[grid](
        q, k, weights, out,
        S, T,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2),
        weights.stride(0), weights.stride(1), weights.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        H=H, D=D,
        BLOCK_H=BLOCK_H, BLOCK_T=BLOCK_T,
        num_warps=4, num_stages=2,
    )
    return out


@triton.jit
def _indexer_decode_logits_kernel(
    q_ptr, w_ptr, pool_ptr, snap_ptr, rows_ptr, valid_ptr, out_ptr,
    N_STAGE, RATIO,
    stride_qb, stride_qh, stride_qd,
    stride_wb, stride_wh,
    stride_pr, stride_pd,
    stride_sb, stride_sw,
    stride_ob, stride_ot,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """One decode query per row, gathering its own compressed keys.

    Same head-reduced score as ``_indexer_logits_kernel``, but the keys are gathered from the
    paged ``idx_pool`` inside the kernel (block b lives at row ``full_snap[row, b*RATIO] //
    RATIO``), and the number of blocks to score is read from DEVICE memory (``valid_ptr``), so
    a captured graph's work tracks the live position instead of the staged width. Tiles past
    the live count store ``-inf`` and return -- the top-k downstream still sees a full row.
    """
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    out_ptrs = out_ptr + pid_b * stride_ob + offs_t * stride_ot
    store_mask = offs_t < N_STAGE

    n_valid = tl.load(valid_ptr + pid_b)
    if pid_t * BLOCK_T >= n_valid:
        tl.store(out_ptrs, tl.full((BLOCK_T,), float("-inf"), tl.float32).to(
            out_ptr.dtype.element_ty), mask=store_mask)
        return

    t_mask = offs_t < n_valid
    offs_d = tl.arange(0, D)
    offs_h = tl.arange(0, BLOCK_H)
    h_mask = offs_h < H

    # block b -> its compressed row: full_loc(b * RATIO) // RATIO off the decode snapshot.
    # There may be several query rows per request (DSpark target verification), so
    # pid_b is a QUERY row and rows_ptr maps it to the corresponding SNAPSHOT row.
    # Masked-off lanes read snapshot column 0 and are dropped by ``t_mask`` below.
    snap_row = tl.load(rows_ptr + pid_b)
    snap = tl.load(snap_ptr + snap_row * stride_sb + (offs_t * RATIO) * stride_sw,
                   mask=t_mask, other=0)
    rows = tl.maximum(snap, 0) // RATIO

    k = tl.load(pool_ptr + rows[:, None] * stride_pr + offs_d[None, :] * stride_pd,
                mask=t_mask[:, None], other=0.0)
    q = tl.load(q_ptr + pid_b * stride_qb + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd,
                mask=h_mask[:, None], other=0.0)
    w = tl.load(w_ptr + pid_b * stride_wb + offs_h * stride_wh, mask=h_mask, other=0.0).to(tl.float32)

    score = tl.dot(q, tl.trans(k))            # [BLOCK_H, BLOCK_T] fp32
    score = tl.maximum(score, 0.0) * w[:, None]
    logits = tl.sum(score, axis=0)            # [BLOCK_T]

    tl.store(out_ptrs, tl.where(t_mask, logits, float("-inf")).to(out_ptr.dtype.element_ty),
             mask=store_mask)


def indexer_decode_logits(
    q: torch.Tensor,          # [B, H, D]  bf16, one query per request
    weights: torch.Tensor,    # [B, H]
    idx_pool: torch.Tensor,   # [R, D]     bf16, this layer's paged indexer keys
    full_snap: torch.Tensor,  # [R, W]     int64, one snapshot row per request
    rows: torch.Tensor,       # [Q]        query row -> snapshot row
    valid: torch.Tensor,      # [Q]        live compressed block count per query
    n_stage: int,             # static staged width (the buffer the top-k scans)
    ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Head-reduced Lightning-Indexer logits ``[B, n_stage]`` for a decode step.

    Equivalent to the reference chain

        full_at = full_snap[rows[:, None], (arange(n_stage) * ratio)[None, :]]
        idx_kv  = idx_pool[cmp_rows(full_at, ratio).clamp_min(0)]
        score   = einsum("bhd,btd->bht", q, idx_kv).relu_() * weights[..., None]
        where(arange(n_stage) < valid, score.sum(1), -inf)

    without the ``[B, n_stage, D]`` key transient or the ``[B, H, n_stage]`` score transient,
    and doing real work only over the first ``valid`` columns.
    """
    B, H, D = q.shape
    assert weights.shape == (B, H), (weights.shape, (B, H))
    assert rows.shape == (B,), (rows.shape, (B,))
    assert idx_pool.shape[1] == D, (idx_pool.shape, D)
    assert D == triton.next_power_of_2(D), f"index_head_dim must be pow2, got {D}"

    if out is None:
        out = torch.empty((B, n_stage), dtype=idx_pool.dtype, device=q.device)
    if n_stage == 0:
        return out

    BLOCK_T = 64
    _indexer_decode_logits_kernel[(B, triton.cdiv(n_stage, BLOCK_T))](
        q, weights, idx_pool, full_snap, rows, valid, out,
        n_stage, ratio,
        q.stride(0), q.stride(1), q.stride(2),
        weights.stride(0), weights.stride(1),
        idx_pool.stride(0), idx_pool.stride(1),
        full_snap.stride(0), full_snap.stride(1),
        out.stride(0), out.stride(1),
        H=H, D=D,
        BLOCK_H=triton.next_power_of_2(H), BLOCK_T=BLOCK_T,
        num_warps=4, num_stages=2,
    )
    return out


__all__ = ["indexer_logits", "indexer_decode_logits"]
