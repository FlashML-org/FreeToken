"""Lightning-Indexer logits over PACKED index keys (DeepSeek-V4.1, Full and Reindex modes).

    logits[b, s, j] = sum_h relu(q[b, s, h, :] . k(b, pos_j)) * weights[b, s, h]

``q`` carries the ``index_n_heads`` fp4-baked index queries, ``k`` is the single shared index key of
a compressed position, gathered from the kv-source's packed pool and dequantized in-kernel. The
row of compressed position ``t`` is derived in-kernel from the request's full-token locs (the page
table row at prefill, the decode snapshot): ``locs[b, t * ratio] // ratio`` (negative = no row) --
no ``[B, T]`` row table is materialized per layer. ``weights`` folds ``softmax_scale *
n_heads**-0.5`` in.

Two position sets, one kernel:

* **full range** (Full mode): position ``j`` is the compressed position ``j``; the output is
  ``[B, S, T]`` over the staged width ``T``, but only the live tiles (columns below ``live[b, s]``
  rounded up to ``BLOCK_T``, the rest of the last tile ``-inf``) are WRITTEN. A fixed grid of worker
  programs strides over the live tiles, so both the work and the launch geometry follow the live
  history rather than the staged width; consumers (``dsv41/topk.py``) read below ``live`` only, so
  the dead area needs no fill.
* **candidate pool** (Reindex mode, Hierarchical Sparse Indexer): position ``j`` is
  ``candidates[b, s, j]`` (a compressed position, ``-1`` = empty slot); the output is ``[B, S, NC]``
  aligned with the candidate list and fully written (``-inf`` for empty / unreachable slots), so
  a top-k over it indexes straight back into the list.

Causality lives in the kernel: ``live[b, s]`` is the number of compressed positions query ``s`` may
see, and every position at or past it (or with no row) scores ``-inf``. The dot product, weighted
scores and head reduction round to bf16 at the same boundaries as the reference expression.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kvcache.dsv4.v41_row_format import RowFormat

from .row_format import load_rows

BLOCK_T = 64


@triton.jit
def _load_fp4_e8m0_rows(pool_ptr, rows, valid, D: tl.constexpr, ROW_BYTES: tl.constexpr):
    """``load_rows(..., FMT=3)`` as bf16 for ``D == 128``, with each row read as int32 words: 16 words of 8 e2m1 codes, then one word of 4 scales.
    The contiguous word axis pins the gather layout; a per-channel byte gather has no contiguous axis, so its layout (and speed) follows the Triton version."""
    W: tl.constexpr = D // 8
    base = pool_ptr.to(tl.pointer_type(tl.int32)) + tl.maximum(rows, 0).to(tl.int64) * (ROW_BYTES // 4)
    offs_w = tl.arange(0, W)
    x = tl.load(base[:, None] + offs_w[None, :], mask=valid[:, None], other=0)
    sw = tl.load(base + W, mask=valid, other=0)
    s = tl.exp2(((sw[:, None] >> (8 * (offs_w // 4))[None, :]) & 0xFF).to(tl.float32) - 127.0)
    x = tl.join(x, x >> 16)
    x = tl.join(x, x >> 8)
    x = tl.join(x, x >> 4)
    # e2m1 code -> fp32 bits: magnitude to bits 22-24, sign to bit 31 (mask 0x81C00000). That is the value times 2**-126 (code 1 is subnormal), so the multiply is exact.
    q = (((x << 28) >> 6) & -0x7E400000).to(tl.float32, bitcast=True) * 8.507059173023462e37
    return tl.reshape((q * s[:, :, None, None, None]).to(tl.bfloat16), (rows.shape[0], D))


@triton.jit
def _score_tile(
    q, w, pool_ptr, locs_row, live, pos, store_mask, out_ptrs,
    n_locs, RATIO: tl.constexpr, D: tl.constexpr, ROW_BYTES: tl.constexpr, FMT: tl.constexpr,
):
    """Score one tile of compressed positions ``pos`` against the query (``q [BLOCK_H, D]`` bf16,
    ``w [BLOCK_H]`` fp32) and store ``[BLOCK_T]`` logits (``-inf`` where unreachable)."""
    pos_ok = store_mask & (pos >= 0) & (pos < live)
    full_idx = pos * RATIO
    full = tl.load(locs_row + tl.maximum(full_idx, 0), mask=pos_ok & (full_idx < n_locs), other=-1)
    rows = tl.where(full >= 0, full // RATIO, -1).to(tl.int32)
    valid = pos_ok & (rows >= 0)
    if FMT == 3 and D == 128:
        k = _load_fp4_e8m0_rows(pool_ptr, rows, valid, D, ROW_BYTES)
    else:
        k = load_rows(pool_ptr, rows, valid, tl.arange(0, D), D, ROW_BYTES, FMT).to(tl.bfloat16)  # [BLOCK_T, D]
    score = tl.dot(q, tl.trans(k)).to(tl.bfloat16).to(tl.float32)
    score = (tl.maximum(score, 0.0) * w[:, None]).to(tl.bfloat16).to(tl.float32)
    logits = tl.sum(score, axis=0).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptrs, tl.where(valid, logits, float("-inf")), mask=store_mask)


@triton.jit
def _indexer_logits_packed_kernel(
    q_ptr, w_ptr, pool_ptr, locs_ptr, live_ptr, cand_ptr, out_ptr,
    T, NC, n_locs,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_wb, stride_ws, stride_wh,
    stride_lb, stride_lt,
    stride_vb, stride_vs,
    stride_cb, stride_cs, stride_cn,
    stride_ob, stride_os, stride_on,
    RATIO: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    FMT: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_CAND: tl.constexpr,
    N_WORKERS: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_t = tl.program_id(2)

    live = tl.load(live_ptr + pid_b * stride_vb + pid_s * stride_vs)
    locs_row = locs_ptr + pid_b * stride_lb
    out_row = out_ptr + pid_b * stride_ob + pid_s * stride_os
    offs_d = tl.arange(0, D)
    offs_h = tl.arange(0, BLOCK_H)
    h_mask = offs_h < H
    q = tl.load(
        q_ptr + pid_b * stride_qb + pid_s * stride_qs + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd,
        mask=h_mask[:, None], other=0.0,
    ).to(tl.bfloat16)
    w = tl.load(w_ptr + pid_b * stride_wb + pid_s * stride_ws + offs_h * stride_wh, mask=h_mask, other=0.0).to(tl.bfloat16).to(tl.float32)
    offs = tl.arange(0, BLOCK_T)

    if HAS_CAND:
        # one program per candidate tile; the list is fixed-width and fully written
        offs_t = pid_t * BLOCK_T + offs
        store_mask = offs_t < NC
        pos = tl.load(cand_ptr + pid_b * stride_cb + pid_s * stride_cs + offs_t * stride_cn, mask=store_mask, other=-1)
        _score_tile(q, w, pool_ptr, locs_row, live, pos, store_mask, out_row + offs_t * stride_on, n_locs, RATIO, D, ROW_BYTES, FMT)
    else:
        # worker pid_t scores live tiles pid_t, pid_t + N_WORKERS, ...; nothing past live is written
        for tile in range(pid_t, tl.cdiv(tl.minimum(live, T), BLOCK_T), N_WORKERS):
            offs_t = tile * BLOCK_T + offs
            store_mask = offs_t < T
            _score_tile(q, w, pool_ptr, locs_row, live, offs_t, store_mask, out_row + offs_t * stride_on, n_locs, RATIO, D, ROW_BYTES, FMT)


def indexer_logits_packed(
    q: torch.Tensor,           # [B, S, H, D] bf16 (fp4 round-tripped index queries)
    weights: torch.Tensor,     # [B, S, H] (softmax_scale * n_heads**-0.5 folded in)
    k_pool: torch.Tensor,      # [R, fmt.row_bytes(D)] uint8, the kv-source's packed index keys
    fmt: RowFormat,
    locs: torch.Tensor,        # [B, W] int32/int64 full-token locs of each request (page-table row / decode snapshot)
    ratio: int,                # compressed position t -> full loc locs[b, t * ratio]; its row is loc // ratio
    live: torch.Tensor,        # [B, S] int32: compressed positions query (b, s) may see
    T: int | None = None,      # staged width of the full-range output (default: W // ratio)
    candidates: torch.Tensor | None = None,  # [B, S, NC] int32 compressed positions (-1 = empty)
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Head-reduced indexer logits: ``[B, S, T]`` over the full staged range (columns below ``live``
    written, the rest untouched), or ``[B, S, NC]`` aligned with ``candidates`` (fully written).
    Unreachable / empty positions are ``-inf``."""
    B, S, H, D = q.shape
    assert locs.ndim == 2 and locs.shape[0] == B and live.shape == (B, S), (locs.shape, live.shape, (B, S))
    assert weights.shape == (B, S, H), (weights.shape, (B, S, H))
    assert D == triton.next_power_of_2(D), f"index_head_dim must be pow2, got {D}"
    assert k_pool.dtype == torch.uint8 and k_pool.shape[1] == fmt.row_bytes(D) and k_pool.is_contiguous() and k_pool.data_ptr() % 4 == 0
    assert locs.dtype in (torch.int32, torch.int64) and locs.stride(1) == 1, (locs.dtype, locs.stride())
    n_locs = locs.shape[1]
    if T is None:
        T = n_locs // ratio
    live = live.to(torch.int32)
    has_cand = candidates is not None
    if has_cand:
        assert candidates.shape[:2] == (B, S), (candidates.shape, (B, S))
        cand = candidates.to(torch.int32)
        NC = cand.shape[2]
        cs = cand.stride()
    else:
        cand, NC, cs = live, 0, (0, 0, 0)
    n_out = NC if has_cand else T
    if out is None:
        out = torch.empty((B, S, n_out), dtype=torch.float32, device=q.device)
    assert out.shape == (B, S, n_out) and out.dtype == torch.float32
    if n_out == 0 or S == 0:
        return out

    n_tiles = triton.cdiv(n_out, BLOCK_T)
    # A device-sized worker grid strides over live history without launching a
    # program for every reserved tile. Independent queries share the SM budget.
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    workers = n_tiles if has_cand else min(n_tiles, max(1, triton.cdiv(sm_count, B * S)))
    grid = (S, B, workers)
    _indexer_logits_packed_kernel[grid](
        q, weights, k_pool, locs, live, cand, out,
        T, NC, n_locs,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        weights.stride(0), weights.stride(1), weights.stride(2),
        locs.stride(0), locs.stride(1),
        live.stride(0), live.stride(1),
        cs[0], cs[1], cs[2],
        out.stride(0), out.stride(1), out.stride(2),
        RATIO=ratio, H=H, D=D, ROW_BYTES=fmt.row_bytes(D), FMT=fmt.code,
        BLOCK_H=triton.next_power_of_2(H), BLOCK_T=BLOCK_T, HAS_CAND=has_cand, N_WORKERS=workers,
        num_warps=4, num_stages=2,
    )
    return out


__all__ = ["indexer_logits_packed", "BLOCK_T"]
