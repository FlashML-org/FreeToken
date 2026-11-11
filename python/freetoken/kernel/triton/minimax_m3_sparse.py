"""Triton kernels for MiniMax-M3 block-sparse GQA attention (indexer + attend).

Ported from the vLLM reference implementation
(``vllm/models/minimax_m3/common/ops/{index_topk,sparse_attn}.py`` in the vLLM
tree -- the semantics source of truth), adapted to FreeToken's storage model:

* FreeToken's page table is page_size=1 semantics (one physical TOKEN row per
  column) and the allocator hands out page-aligned runs, so with page_size == the
  128-token sparse block each block's rows are ``base .. base+127`` where ``base =
  page_table[t, blk * 128]``. The kernels take a per-request ``block_rows`` table
  of those int32 base rows and address the FLAT slabs directly -- the index-key
  slab as ``[rows, index_dim]`` (``BSAKVCache.index_k_cache``), K/V as ``[rows,
  kv_heads, head_dim]`` (the MHA pool's storage view) -- instead of vLLM's
  ``[num_blocks, ...]`` paged layouts.
* decode is always one query token per request (no spec decode), so the
  ``decode_query_len`` machinery is dropped: ``kv_len = seq_lens[req]``.
* KV is always bf16 (FreeToken pools); the fp8-KV dequant branches are dropped.
* ``tl.dot`` needs >= 16 rows per operand, so the decode index-score kernel pads
  the 4 index heads to a masked 16-row tile (the reference relied on a fork that
  allows narrow dots).

Semantics kept verbatim from the reference:

* per-index-head (== per-KV-head) block scores: ``score[h, q, blk] = max over the
  block's 128 positions of dot(iq[q, h], ik[pos])``, causal-masked, no scale
  (only the ordering is consumed).
* forced blocks: the first ``init_blocks`` score ``1e30`` and the newest
  ``local_blocks`` score ``1e29`` (visible blocks only), so they always win the
  top-k; padding slots come back ``-1``.
* the attend visits the FIRST ``min(topk, num_visible_blocks)`` selected entries
  (the bitonic top-k sorts descending), applies the causal mask inside the newest
  block, and runs a base-2 online softmax; decode is split-K over the selected
  blocks with an LSE merge (flash-decoding).

CUDA-graph safety (decode): every kernel reads the live ``seq_lens`` from device
memory and the split/chunk counts depend only on staged shapes, so the captured
grid tracks the live position (same contract as the dsa/dsv4 backends).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.kv_rows import kv_row_scale, kv_scale_args, kv_tile
from freetoken.kvcache.format import NONE, KVFormat

# One sparse block == one KV page (the engine pins page_size to this).
SPARSE_BLOCK_SIZE = 128


def _round_up(a: int, b: int) -> int:
    return (a + b - 1) // b * b


# ---------------------------------------------------------------------------
# Bitonic top-k helpers (verbatim from the reference; layout-agnostic).
# ---------------------------------------------------------------------------
@triton.jit
def _compare_and_swap(x, ids, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)
    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(
    x, ids, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    if order == 2:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(
            tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


# ---------------------------------------------------------------------------
# Prefill index block-score kernel. score[h, token, block] = max over the
# 128-token block of (idx_q . index_k), causal-masked; blocks addressed via the
# per-request base-row table.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _index_block_score_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_ptr,  # index-K slab, row-flat: [rows, head_dim]
    score_ptr,  # [num_idx_heads, total_q, max_block(stride)]
    block_rows_ptr,  # [num_reqs, max_blocks] int32 base rows
    cu_seqlens,  # [batch+1] query start offsets
    seq_lens,  # [batch] total K length
    prefix_lens,  # [batch] context length before this chunk's queries
    num_idx_heads,
    head_dim: tl.constexpr,
    stride_q_n, stride_q_h, stride_q_d,
    stride_ik_r, stride_ik_d,
    stride_s_h, stride_s_n, stride_s_k,
    stride_br_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // num_idx_heads
    pid_h = pid_bh % num_idx_heads

    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if BLOCK_SIZE_Q * pid_q >= q_len:
        return

    q_ptrs = tl.make_block_ptr(
        base=q_ptr + seq_start * stride_q_n + pid_h * stride_q_h,
        shape=(q_len, head_dim),
        strides=(stride_q_n, stride_q_d),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, head_dim),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0,), padding_option="zero")
    q_start = prefix_len + pid_q * BLOCK_SIZE_Q

    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q + prefix_len
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, head_dim)
    br_row = block_rows_ptr + pid_b * stride_br_b
    # Causal window: only blocks up to the last query token's position.
    hi = min(seq_len, prefix_len + (pid_q + 1) * BLOCK_SIZE_Q)
    for i in tl.range(0, hi, BLOCK_SIZE_K):
        blk = i // BLOCK_SIZE_K
        base = tl.load(br_row + blk).to(tl.int64)
        base = tl.maximum(base, 0)
        pos = i + off_k
        # No masked load: pages are whole 128-row runs, so rows base..base+127
        # exist, and the index-key slab is ZERO-INITIALIZED by BSAKVCache (unlike
        # the K/V slabs) -- unwritten tail rows contribute a finite 0-dot that the
        # qk causal mask below discards. If that zero-init invariant ever changes,
        # this load (and the decode score kernel's) must gain a pos mask.
        k = tl.load(
            ik_ptr
            + (base + off_k[None, :]) * stride_ik_r
            + off_d[:, None] * stride_ik_d,
        )
        qk = tl.dot(q, k)
        if q_start < i + BLOCK_SIZE_K:
            qk = tl.where(off_q[:, None] >= pos[None, :], qk, float("-inf"))
        score = tl.max(qk, axis=1)  # [BLOCK_SIZE_Q]: max over the block
        s_ptrs = (
            score_ptr
            + pid_h * stride_s_h
            + (seq_start + pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q))
            * stride_s_n
            + blk * stride_s_k
        )
        q_store_mask = (pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) < q_len
        tl.store(s_ptrs, score, mask=q_store_mask)


# ---------------------------------------------------------------------------
# Prefill top-k over per-token block scores (verbatim reference port; the forced
# init/local blocks get score boosts, invalid slots come back -1).
# ---------------------------------------------------------------------------
@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 64}, num_warps=2, num_stages=2),
    ],
    key=["BLOCK_SIZE_T"],
)
@triton.jit(do_not_specialize_on_alignment=["prefix_lens"])
def _topk_index_kernel(
    s_ptr,  # [num_heads, total_q, max_block]
    ti_ptr,  # [num_heads, total_q, topk]
    block_size: tl.constexpr,  # sparse block size (128)
    cu_seqlens,
    prefix_lens,
    topk,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    stride_s_h, stride_s_n, stride_s_k,
    stride_ti_h, stride_ti_n, stride_ti_t,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_K > BLOCK_SIZE_T)
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q >= q_len:
        return
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    s_ptrs = (
        s_ptr
        + (seq_start + pid_q) * stride_s_n
        + pid_h * stride_s_h
        + off_k * stride_s_k
    )
    topk_score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    topk_idx = tl.full((BLOCK_SIZE_K,), 0, dtype=tl.int32)
    left_half_mask = tl.arange(0, BLOCK_SIZE_K) < BLOCK_SIZE_K // 2
    valid_blocks = (prefix_len + pid_q + block_size) // block_size
    for i in tl.range(0, valid_blocks, BLOCK_SIZE_K):
        causal_mask = i + off_k < valid_blocks
        local_mask = i + off_k >= max(0, valid_blocks - local_blocks)
        init_mask = i + off_k < init_blocks
        score = tl.load(s_ptrs, mask=causal_mask, other=-1e30).to(tl.float32)
        score = tl.where(score != score, -1e30, score)
        s_ptrs = s_ptrs + stride_s_k * BLOCK_SIZE_K
        # Force-select init (1e30) and local (1e29) blocks; local applied last so
        # it wins an overlap, matching the reference's MASK_INIT/MASK_LOCAL=False
        # order (moot for M3's init_blocks == 0).
        score = tl.where(causal_mask & init_mask, 1e30, score)
        score = tl.where(causal_mask & local_mask, 1e29, score)
        topk_score, last_topk_score = score, topk_score
        topk_idx, last_topk_idx = (tl.where(causal_mask, i + off_k + 1, 0), topk_idx)
        n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
        for j in tl.static_range(1, n_dims):
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), j, 2, n_dims
            )
        if i != 0:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, False, n_dims
            )
            topk_score_new = last_topk_score * left_half_mask + topk_score * (
                1 - left_half_mask
            )
            topk_idx_new = last_topk_idx * left_half_mask + topk_idx * (
                1 - left_half_mask
            )
            topk_score, topk_idx = _bitonic_merge(
                topk_score_new, topk_idx_new.to(tl.int32), n_dims, True, n_dims
            )
        else:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, True, n_dims
            )
    topk_mask = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    topk_idx = tl.sum(
        topk_mask[:, None]
        * tl.reshape(topk_idx - 1, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )
    ti_ptrs = (
        ti_ptr
        + (seq_start + pid_q) * stride_ti_n
        + pid_h * stride_ti_h
        + off_t * stride_ti_t
    )
    store_mask = off_t < topk
    valid_mask = off_t < valid_blocks
    topk_idx = tl.where(store_mask & valid_mask, topk_idx, -1)
    tl.store(ti_ptrs, topk_idx.to(ti_ptrs.dtype.element_ty), mask=store_mask)


# ---------------------------------------------------------------------------
# Decode index-score kernel (split-K over seq blocks; cudagraph-safe). One query
# per request; the 4 index heads ride a masked 16-row tile for tl.dot.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["num_kv_chunks"])
def _decode_index_score_kernel(
    q_ptr,  # idx_q: [num_reqs, num_idx_heads, head_dim]
    ik_ptr,  # index-K slab, row-flat: [rows, head_dim]
    score_ptr,  # [num_idx_heads, num_reqs, max_block(stride)]
    block_rows_ptr,  # [num_reqs, max_blocks] int32 base rows
    seq_lens,  # [num_reqs] (device-read live lengths)
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    init_blocks,
    local_blocks,
    stride_q_n, stride_q_h, stride_q_d,
    stride_ik_r, stride_ik_d,
    stride_s_h, stride_s_n, stride_s_k,
    stride_br_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_H: tl.constexpr,  # >= 16 (padded index-head tile)
    num_kv_chunks,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    off_h = tl.arange(0, BLOCK_SIZE_H)
    h_mask = off_h < num_idx_heads

    seq_len = tl.load(seq_lens + pid_r)
    # Padded graph rows carry zero length -> empty range, nothing stored.
    kv_len = tl.maximum(seq_len, 0)
    num_blocks = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    # Block-aligned fixed-count split: grid independent of seq_len (cuda graph).
    chunk_size_blocks = (num_blocks + num_kv_chunks - 1) // num_kv_chunks
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)
    if chunk_start_block >= chunk_end_block:
        return
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, head_dim)
    br_row = block_rows_ptr + pid_r * stride_br_b
    local_start = tl.maximum(0, num_blocks - local_blocks)
    q = tl.load(
        q_ptr
        + pid_r * stride_q_n
        + off_h[None, :] * stride_q_h
        + off_d[:, None] * stride_q_d,
        mask=h_mask[None, :],
        other=0.0,
    )  # [D, H16]
    for blk in tl.range(chunk_start_block, chunk_end_block):
        base = tl.load(br_row + blk).to(tl.int64)
        base = tl.maximum(base, 0)
        pos = blk * BLOCK_SIZE_K + off_k
        pos_mask = pos[:, None] < kv_len
        # No masked load: the index-key slab is ZERO-INITIALIZED by BSAKVCache
        # (unlike the K/V slabs), so unwritten tail rows dot to a finite 0 that
        # the pos_mask below discards. If that invariant changes, mask this load.
        k = tl.load(
            ik_ptr
            + (base + off_k[:, None]) * stride_ik_r
            + off_d * stride_ik_d,
        )  # [N, D]
        kq = tl.dot(k, q, out_dtype=tl.float32)  # [N, H16]
        kq = tl.where(pos_mask & h_mask[None, :], kq, float("-inf"))
        score = tl.max(kq, axis=0)  # [H16]: max over the block's positions
        # local wins an init/local overlap (reference order; moot for init_blocks=0).
        is_init = blk < init_blocks
        is_local = blk >= local_start
        score = tl.where(is_local, 1e29, tl.where(is_init, 1e30, score))
        tl.store(
            score_ptr + off_h * stride_s_h + pid_r * stride_s_n + blk * stride_s_k,
            score,
            mask=h_mask,
        )


# ---------------------------------------------------------------------------
# Decode top-k (split-K): per-chunk partial top-k + merge. Forced init/local
# blocks are already encoded in the scores. Verbatim reference port minus the
# spec-decode query-length math (kv_len == seq_lens[req]).
# ---------------------------------------------------------------------------
@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.jit
def _topk_index_partial_kernel(
    s_ptr,  # score: [num_idx_heads, num_reqs, max_block]
    ts_partial_ptr,  # partial scores out: [chunks, num_idx_heads, num_reqs, T]
    ti_partial_ptr,  # partial idx out (1-indexed, 0=invalid): same shape
    seq_lens,  # [num_reqs]
    block_size: tl.constexpr,  # sparse block size (128)
    topk: tl.constexpr,
    stride_s_h, stride_s_b, stride_s_k,
    stride_ts_c, stride_ts_h, stride_ts_b, stride_ts_t,
    stride_ti_c, stride_ti_h, stride_ti_b, stride_ti_t,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    tl.static_assert(topk < BLOCK_SIZE_K)
    pid_b = tl.program_id(0)  # request id
    pid_h = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    kv_len = tl.maximum(tl.load(seq_lens + pid_b), 0)
    num_blocks = (kv_len + block_size - 1) // block_size

    # Chunk split from the row's LIVE block count (not the staged table width, which
    # would degenerate short contexts to a serial scan in chunk 0) -- device-read,
    # so the captured graph's split tracks the live position.
    num_chunks = tl.num_programs(2)
    chunk_blocks = (num_blocks + num_chunks - 1) // num_chunks
    chunk_start = pid_chunk * chunk_blocks
    chunk_end = tl.minimum(chunk_start + chunk_blocks, num_blocks)
    chunk_actual = tl.maximum(chunk_end - chunk_start, 0)

    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)

    s_ptrs = (
        s_ptr
        + pid_b * stride_s_b
        + pid_h * stride_s_h
        + (chunk_start + off_k) * stride_s_k
    )

    topk_score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    topk_idx = tl.full((BLOCK_SIZE_K,), 0, dtype=tl.int32)
    left_half_mask = tl.arange(0, BLOCK_SIZE_K) < BLOCK_SIZE_K // 2

    # Streaming top-K within this chunk. tl.range(0, 0) is a no-op so empty
    # chunks (chunk_actual == 0) skip the body and store sentinel -1e30 / 0.
    for i in tl.range(0, chunk_actual, BLOCK_SIZE_K):
        mask = off_k < chunk_actual - i
        score = tl.load(s_ptrs, mask=mask, other=-1e30).to(tl.float32)
        score = tl.where(score != score, -1e30, score)
        s_ptrs = s_ptrs + stride_s_k * BLOCK_SIZE_K
        topk_score, last_topk_score = score, topk_score
        topk_idx, last_topk_idx = (
            tl.where(mask, chunk_start + i + off_k + 1, 0),  # 1-indexed global
            topk_idx,
        )
        n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
        for j in tl.static_range(1, n_dims):
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), j, 2, n_dims
            )
        if i != 0:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, False, n_dims
            )
            topk_score_new = last_topk_score * left_half_mask + topk_score * (
                1 - left_half_mask
            )
            topk_idx_new = last_topk_idx * left_half_mask + topk_idx * (
                1 - left_half_mask
            )
            topk_score, topk_idx = _bitonic_merge(
                topk_score_new, topk_idx_new.to(tl.int32), n_dims, True, n_dims
            )
        else:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, True, n_dims
            )

    # Extract first BLOCK_SIZE_T entries (top-K of this chunk after the sort).
    topk_mask_extract = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    final_score = tl.sum(
        topk_mask_extract[:, None]
        * tl.reshape(topk_score, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )
    final_idx = tl.sum(
        topk_mask_extract[:, None]
        * tl.reshape(topk_idx, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )

    # Always write all BLOCK_SIZE_T slots -- invalid slots carry -1e30 / 0
    # sentinels and lose to real scores in the merge stage.
    ts_ptrs = (
        ts_partial_ptr
        + pid_chunk * stride_ts_c
        + pid_b * stride_ts_b
        + pid_h * stride_ts_h
        + off_t * stride_ts_t
    )
    ti_ptrs = (
        ti_partial_ptr
        + pid_chunk * stride_ti_c
        + pid_b * stride_ti_b
        + pid_h * stride_ti_h
        + off_t * stride_ti_t
    )
    tl.store(ts_ptrs, final_score)
    tl.store(ti_ptrs, final_idx)


@triton.heuristics(
    {
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"]),
        "BLOCK_SIZE_K": lambda args: triton.next_power_of_2(
            args["num_topk_chunks"] * triton.next_power_of_2(args["topk"])
        ),
    }
)
@triton.jit(do_not_specialize=["num_topk_chunks"])
def _topk_index_merge_kernel(
    ts_partial_ptr,  # partial scores: [chunks, num_idx_heads, num_reqs, T]
    ti_partial_ptr,  # partial idx (1-indexed, 0=invalid): same shape
    ti_final_ptr,  # final idx (0-indexed, -1=invalid): [num_idx_heads, num_reqs, topk]
    seq_lens,  # [num_reqs]
    block_size: tl.constexpr,
    topk: tl.constexpr,
    stride_ts_c, stride_ts_h, stride_ts_b, stride_ts_t,
    stride_ti_c, stride_ti_h, stride_ti_b, stride_ti_t,
    stride_tif_h, stride_tif_b, stride_tif_t,
    num_topk_chunks,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    kv_len = tl.maximum(tl.load(seq_lens + pid_b), 0)
    num_blocks = (kv_len + block_size - 1) // block_size

    off = tl.arange(0, BLOCK_SIZE_K)
    chunk_idx = off // BLOCK_SIZE_T
    in_chunk_idx = off % BLOCK_SIZE_T
    valid = chunk_idx < num_topk_chunks

    score_offset = (
        chunk_idx * stride_ts_c
        + pid_h * stride_ts_h
        + pid_b * stride_ts_b
        + in_chunk_idx * stride_ts_t
    )
    idx_offset = (
        chunk_idx * stride_ti_c
        + pid_h * stride_ti_h
        + pid_b * stride_ti_b
        + in_chunk_idx * stride_ti_t
    )

    score = tl.load(ts_partial_ptr + score_offset, mask=valid, other=-1e30).to(
        tl.float32
    )
    score = tl.where(score != score, -1e30, score)
    idx = tl.load(ti_partial_ptr + idx_offset, mask=valid, other=0).to(tl.int32)

    n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
    for j in tl.static_range(1, n_dims):
        score, idx = _bitonic_merge(score, idx.to(tl.int32), j, 2, n_dims)
    score, idx = _bitonic_merge(score, idx.to(tl.int32), n_dims, True, n_dims)

    extract_mask = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    topk_idx_final = tl.sum(
        extract_mask[:, None]
        * tl.reshape(idx - 1, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )

    off_t = tl.arange(0, BLOCK_SIZE_T)
    tif_ptrs = (
        ti_final_ptr
        + pid_h * stride_tif_h
        + pid_b * stride_tif_b
        + off_t * stride_tif_t
    )
    store_mask = off_t < topk
    topk_idx_final = tl.where(off_t < tl.minimum(topk, num_blocks), topk_idx_final, -1)
    tl.store(
        tif_ptrs, topk_idx_final.to(ti_final_ptr.dtype.element_ty), mask=store_mask
    )


# ---------------------------------------------------------------------------
# GQA block-sparse attend, prefill. One program per (query token, kv head,
# request); the 16-head GQA group rides one tl.dot tile. Base-2 online softmax.
# ---------------------------------------------------------------------------
@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
    }
)
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    k_ptr,  # K slab, row-flat: [rows, num_kv_heads, head_dim]
    v_ptr,  # V slab, row-flat: [rows, num_kv_heads, head_dim]
    k_scale_ptr,  # [rows, num_kv_heads, 1]
    v_scale_ptr,  # [rows, num_kv_heads, 1]
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # [total_q, num_heads, head_dim]
    block_rows_ptr,  # [num_reqs, max_blocks] int32 base rows
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    stride_qn, stride_qh, stride_qd,
    stride_kr, stride_kh, stride_kd,
    stride_vr, stride_vh, stride_vd,
    stride_kss, stride_vss,
    stride_th, stride_tn, stride_tk,
    stride_on, stride_oh, stride_od,
    stride_br_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    KV_FMT: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = pid_kh * gqa_group_size
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q >= q_len:
        return
    br_row = block_rows_ptr + pid_b * stride_br_b
    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_h = tl.arange(0, BLOCK_SIZE_H)
    d_mask = off_d < head_dim
    h_mask = off_h < gqa_group_size

    t_ptr_j = t_ptr + (q_start + pid_q) * stride_tn + pid_kh * stride_th
    q_abs = prefix_len + pid_q  # absolute position of this query token
    valid_blocks = (q_abs + BLOCK_SIZE_K) // BLOCK_SIZE_K
    real_topk = tl.minimum(max_topk, valid_blocks)

    q = tl.load(
        q_ptr
        + (q_start + pid_q) * stride_qn
        + (pid_h + off_h[:, None]) * stride_qh
        + off_d[None, :] * stride_qd,
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    )  # [H, D]

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    for _ in range(real_topk):
        # `blk` is trusted in-range: selection writes exactly `real_topk` real
        # entries. A -1 padding slot here would OOB-read br_row before the base
        # clamp -- preserve that invariant when editing the top-k kernel.
        blk = tl.load(t_ptr_j).to(tl.int32)
        t_ptr_j = t_ptr_j + stride_tk
        c = blk * BLOCK_SIZE_K
        base = tl.load(br_row + blk).to(tl.int64)
        base = tl.maximum(base, 0)
        pos = c + off_n
        # causal + in-sequence: the newest block is partially visible.
        pos_mask = (pos <= q_abs) & (pos < seq_len)
        # K/V loads must be pos-masked: the masks are the correctness contract
        # for the tail page's unwritten rows (the forced local block visits the
        # partial newest page nearly every step; unmasked rows would NaN the
        # whole output row). BSAKVCache's zero-init is defense-in-depth, not a
        # substitute -- recycled pages get re-dirtied.
        k = kv_tile(
            k_ptr
            + (base + off_n[None, :]) * stride_kr
            + pid_kh * stride_kh
            + off_d[:, None] * stride_kd,
            d_mask[:, None] & pos_mask[None, :],
            KV_FMT,
        )  # [D, N]
        scale_k = kv_row_scale(k_scale_ptr + (base + off_n) * stride_kss + pid_kh, pos_mask, KV_FMT)
        qk = tl.dot(q, k) * sm_scale_log2e * scale_k[None, :]
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        # All-masked rows keep a -inf running max NaN-free.
        alpha = tl.where(m_ij == float("-inf"), 1.0, tl.exp2(m_i - m_ij))
        p = tl.where(pos_mask[None, :], tl.exp2(qk - m_ij[:, None]), 0.0)
        l_ij = tl.sum(p, axis=1)
        acc_o = acc_o * alpha[:, None]
        v = kv_tile(
            v_ptr
            + (base + off_n[:, None]) * stride_vr
            + pid_kh * stride_vh
            + off_d[None, :] * stride_vd,
            pos_mask[:, None] & d_mask[None, :],
            KV_FMT,
        )  # [N, D]
        scale_v = kv_row_scale(v_scale_ptr + (base + off_n) * stride_vss + pid_kh, pos_mask, KV_FMT)
        acc_o += tl.dot((p * scale_v[None, :]).to(v.dtype), v)
        m_i = m_ij
        lse_i = tl.where(
            m_ij == float("-inf"),
            lse_i,
            m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij),
        )
    acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
    o_ptrs = (
        o_ptr
        + (q_start + pid_q) * stride_on
        + (pid_h + off_h[:, None]) * stride_oh
        + off_d[None, :] * stride_od
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), mask=h_mask[:, None] & d_mask[None, :])


# ---------------------------------------------------------------------------
# GQA block-sparse attend, decode (split-K over the selected blocks + LSE merge;
# flash-decoding). One query per request; cudagraph-safe (device seq_lens).
# ---------------------------------------------------------------------------
@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit
def _gqa_sparse_decode_kernel(
    q_ptr,  # [num_reqs, num_heads, head_dim]
    k_ptr,  # K slab, row-flat: [rows, num_kv_heads, head_dim]
    v_ptr,  # V slab, row-flat: [rows, num_kv_heads, head_dim]
    k_scale_ptr,  # [rows, num_kv_heads, 1]
    v_scale_ptr,  # [rows, num_kv_heads, 1]
    t_ptr,  # topk_idx: [num_kv_heads, num_reqs, topk]
    o_ptr,  # partial out: [chunks, num_reqs, num_heads, head_dim]
    lse_ptr,  # partial lse (log2): [chunks, num_reqs, num_heads]
    block_rows_ptr,  # [num_reqs, max_blocks] int32 base rows
    seq_lens,  # [num_reqs]
    total_q,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    stride_qn, stride_qh, stride_qd,
    stride_kr, stride_kh, stride_kd,
    stride_vr, stride_vh, stride_vd,
    stride_kss, stride_vss,
    stride_th, stride_tn, stride_tk,
    stride_o_c, stride_o_b, stride_o_h, stride_o_d,
    stride_l_c, stride_l_b, stride_l_h,
    stride_br_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    KV_FMT: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % total_q
    pid_c = pid_bc // total_q
    pid_h = pid_kh * gqa_group_size
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_compiletime = chunk_start_topk + chunk_size_topk

    kv_len = tl.maximum(tl.load(seq_lens + pid_b), 0)

    idx_base = t_ptr + pid_kh * stride_th + pid_b * stride_tn
    num_blocks = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    real_topk = tl.minimum(max_topk, num_blocks)
    chunk_end_topk = tl.minimum(chunk_end_compiletime, real_topk)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    br_row = block_rows_ptr + pid_b * stride_br_b

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qn + pid_h * stride_qh,
        shape=(gqa_group_size, head_dim),
        strides=(stride_qh, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")

    cur_idx_ptr = idx_base + chunk_start_topk * stride_tk
    for _ in tl.range(chunk_start_topk, chunk_end_topk):
        # `blk` is trusted in-range (selection invariant; see the sequential
        # attend kernel): a -1 padding slot would OOB-read br_row pre-clamp.
        blk = tl.load(cur_idx_ptr).to(tl.int32)
        cur_idx_ptr = cur_idx_ptr + stride_tk
        c = blk * BLOCK_SIZE_K
        base = tl.load(br_row + blk).to(tl.int64)
        base = tl.maximum(base, 0)
        pos = c + off_n
        pos_mask = pos < kv_len
        # K/V loads must be pos-masked (same contract as the sequential attend
        # kernel): with K masked, the additive -inf lanes stay -inf and
        # p = exp2(-inf) = 0. Zero-init is defense-in-depth, not a substitute.
        k = kv_tile(
            k_ptr
            + (base + off_n[None, :]) * stride_kr
            + pid_kh * stride_kh
            + off_d[:, None] * stride_kd,
            d_mask[:, None] & pos_mask[None, :],
            KV_FMT,
        )
        scale_k = kv_row_scale(k_scale_ptr + (base + off_n) * stride_kss + pid_kh, pos_mask, KV_FMT)
        qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * sm_scale_log2e * scale_k[None, :]
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
        v = kv_tile(
            v_ptr
            + (base + off_n[:, None]) * stride_vr
            + pid_kh * stride_vh
            + off_d[None, :] * stride_vd,
            pos_mask[:, None] & d_mask[None, :],
            KV_FMT,
        )
        scale_v = kv_row_scale(v_scale_ptr + (base + off_n) * stride_vss + pid_kh, pos_mask, KV_FMT)
        acc_o += tl.dot((p * scale_v[None, :]).to(v.dtype), v)
        m_i = m_ij
        lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)

    # Empty chunks for active rows must store zero output; otherwise the merge
    # can hit 0 * NaN. All-empty padded rows may still produce NaNs in merge.
    scale = tl.where(lse_i > float("-inf"), tl.exp2(m_i - lse_i), tl.zeros_like(lse_i))
    acc_o = acc_o * scale[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_o_h, stride_o_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
        shape=(gqa_group_size,),
        strides=(stride_l_h,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_H,),
        order=(0,),
    )
    tl.store(lse_ptrs, lse_i.to(lse_ptr.dtype.element_ty), boundary_check=(0,))


@triton.heuristics(
    {"BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"])}
)
@triton.jit
def _merge_topk_attn_out_kernel(
    o_ptr,  # partials: [chunks, num_reqs, num_heads, head_dim]
    lse_ptr,  # partials (log2): [chunks, num_reqs, num_heads]
    out_ptr,  # merged out: [num_reqs, num_heads, head_dim]
    head_dim,
    stride_o_c, stride_o_b, stride_o_h, stride_o_d,
    stride_l_c, stride_l_b, stride_l_h,
    stride_out_n, stride_out_h, stride_out_d,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)

    off_c = tl.arange(0, NUM_TOPK_CHUNKS)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(NUM_TOPK_CHUNKS, head_dim),
        strides=(stride_o_c, stride_o_d),
        offsets=(0, 0),
        block_shape=(NUM_TOPK_CHUNKS, BLOCK_SIZE_D),
        order=(1, 0),
    )
    lse_ptrs = lse_ptr + pid_b * stride_l_b + pid_h * stride_l_h + off_c * stride_l_c
    o = tl.load(o_ptrs, boundary_check=(0, 1), padding_option="zero")
    lse = tl.load(lse_ptrs)  # empty chunks contribute -inf -> weight 0
    lse_max = tl.max(lse, axis=0)
    weights = tl.exp2(lse - lse_max)
    weights = weights / tl.sum(weights, axis=0)
    o_merged = tl.sum(o * weights[:, None], axis=0)
    out_ptrs = (
        out_ptr + pid_b * stride_out_n + pid_h * stride_out_h + off_d * stride_out_d
    )
    tl.store(out_ptrs, o_merged.to(out_ptr.dtype.element_ty), mask=off_d < head_dim)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
@torch.no_grad()
def m3_index_score_prefill(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    ik_rows: torch.Tensor,  # index-K slab, row-flat: [rows, head_dim]
    block_rows: torch.Tensor,  # [batch, max_blocks] int32 base rows
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    max_seq_len: int,
) -> torch.Tensor:
    """Per-token index scores for each visible sparse block: score ``[num_idx_heads,
    total_q, >=max_block]`` fp32, each entry the max over a 128-token block."""
    total_q, num_idx_heads, head_dim = idx_q.shape
    batch = cu_seqlens_q.shape[0] - 1
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    # Keep score strides 16-divisible to avoid Triton recompiles.
    score = torch.empty(
        (num_idx_heads, total_q, _round_up(max_block, 16)),
        dtype=torch.float32,
        device=idx_q.device,
    )
    BLOCK_SIZE_Q = 64
    grid = (triton.cdiv(max_query_len, BLOCK_SIZE_Q), batch * num_idx_heads)
    _index_block_score_kernel[grid](
        idx_q, ik_rows, score, block_rows,
        cu_seqlens_q, seq_lens, prefix_lens,
        num_idx_heads, head_dim,
        idx_q.stride(0), idx_q.stride(1), idx_q.stride(2),
        ik_rows.stride(0), ik_rows.stride(1),
        score.stride(0), score.stride(1), score.stride(2),
        block_rows.stride(0),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
    )
    return score


@torch.no_grad()
def m3_index_topk_prefill(
    score: torch.Tensor,  # [num_idx_heads, total_q, >=max_block]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Causal per-token top-k over precomputed block scores -> ``[num_idx_heads,
    total_q, topk]`` int32 block ids (-1 pad)."""
    num_idx_heads, total_q = score.shape[0], score.shape[1]
    batch = cu_seqlens_q.shape[0] - 1
    topk_idx = torch.empty(
        (num_idx_heads, total_q, topk), dtype=torch.int32, device=score.device
    )
    grid = (max_query_len, batch, num_idx_heads)
    _topk_index_kernel[grid](
        score, topk_idx,
        SPARSE_BLOCK_SIZE,
        cu_seqlens_q, prefix_lens,
        topk, init_blocks, local_blocks,
        score.stride(0), score.stride(1), score.stride(2),
        topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
    )
    return topk_idx


@torch.no_grad()
def m3_index_decode(
    idx_q: torch.Tensor,  # [num_reqs, num_idx_heads, head_dim]
    ik_rows: torch.Tensor,  # index-K slab, row-flat: [rows, head_dim]
    block_rows: torch.Tensor,  # [num_reqs, max_blocks] int32 base rows
    seq_lens: torch.Tensor,  # [num_reqs] int32 (device, live)
    max_blocks: int,  # staged block-table width (static under cuda graphs)
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Decode index block-score + top-k, both split-K (cudagraph-safe). Returns
    ``[num_idx_heads, num_reqs, topk]`` int32 (0-indexed block ids, -1 pad)."""
    num_reqs, num_idx_heads, head_dim = idx_q.shape
    score_stride = _round_up(max_blocks, 16)
    score = torch.empty(
        (num_idx_heads, num_reqs, score_stride), dtype=torch.float32, device=idx_q.device
    )
    # split-K over seq blocks; chunk count depends only on staged shapes so the
    # grid is fixed within a cuda graph.
    TARGET_GRID = 512
    MAX_NUM_KV_CHUNKS = 256
    target = max(1, min(MAX_NUM_KV_CHUNKS, TARGET_GRID // max(1, num_reqs)))
    num_kv_chunks = 1 << (target.bit_length() - 1)
    _decode_index_score_kernel[(num_reqs, num_kv_chunks)](
        idx_q, ik_rows, score, block_rows, seq_lens,
        num_idx_heads, head_dim,
        init_blocks, local_blocks,
        idx_q.stride(0), idx_q.stride(1), idx_q.stride(2),
        ik_rows.stride(0), ik_rows.stride(1),
        score.stride(0), score.stride(1), score.stride(2),
        block_rows.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        BLOCK_SIZE_H=max(16, triton.next_power_of_2(num_idx_heads)),
        num_kv_chunks=num_kv_chunks,
        num_warps=4, num_stages=2,
    )

    topk_idx = torch.empty(
        (num_idx_heads, num_reqs, topk), dtype=torch.int32, device=idx_q.device
    )
    TOPK_TARGET_GRID = 64
    MAX_NUM_TOPK_CHUNKS = 16
    topk_target = max(
        1, min(MAX_NUM_TOPK_CHUNKS, TOPK_TARGET_GRID // max(1, num_reqs * num_idx_heads))
    )
    num_topk_chunks = 1 << (topk_target.bit_length() - 1)
    block_size_t = triton.next_power_of_2(topk)
    ts_partial = torch.empty(
        (num_topk_chunks, num_idx_heads, num_reqs, block_size_t),
        dtype=torch.float32, device=idx_q.device,
    )
    ti_partial = torch.empty(
        (num_topk_chunks, num_idx_heads, num_reqs, block_size_t),
        dtype=torch.int32, device=idx_q.device,
    )
    # The chunk split is computed in-kernel from each row's LIVE block count
    # (device-read; the staged table width would serialize short contexts).
    _topk_index_partial_kernel[(num_reqs, num_idx_heads, num_topk_chunks)](
        score, ts_partial, ti_partial, seq_lens,
        SPARSE_BLOCK_SIZE, topk,
        score.stride(0), score.stride(1), score.stride(2),
        ts_partial.stride(0), ts_partial.stride(1), ts_partial.stride(2), ts_partial.stride(3),
        ti_partial.stride(0), ti_partial.stride(1), ti_partial.stride(2), ti_partial.stride(3),
        # Fixed config (no autotune: decode is captured into CUDA graphs, and
        # autotune would benchmark at run time). 128 matches the reference's
        # smallest always-valid config.
        BLOCK_SIZE_K=128,
        num_warps=4, num_stages=2,
    )
    _topk_index_merge_kernel[(num_reqs, num_idx_heads)](
        ts_partial, ti_partial, topk_idx, seq_lens,
        SPARSE_BLOCK_SIZE, topk,
        ts_partial.stride(0), ts_partial.stride(1), ts_partial.stride(2), ts_partial.stride(3),
        ti_partial.stride(0), ti_partial.stride(1), ti_partial.stride(2), ti_partial.stride(3),
        topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
        num_topk_chunks=num_topk_chunks,
        num_warps=2,
    )
    return topk_idx


@torch.no_grad()
def m3_sparse_attn_prefill(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    k_rows: torch.Tensor,  # K slab, row-flat: [rows, num_kv_heads, head_dim]
    v_rows: torch.Tensor,  # V slab, row-flat: [rows, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_rows: torch.Tensor,  # [batch, max_blocks] int32 base rows
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_format: KVFormat = NONE,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """GQA block-sparse attention over the selected blocks (prefill/extend)."""
    total_q, num_heads, head_dim = q.shape
    num_kv_heads = k_rows.shape[1]
    batch = cu_seqlens_q.shape[0] - 1
    topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    k_scale_arg, v_scale_arg, stride_kss, stride_vss = kv_scale_args(kv_format, k_rows, v_rows, k_scale, v_scale, q)
    grid = (max_query_len, num_kv_heads, batch)
    _gqa_sparse_fwd_kernel[grid](
        q, k_rows, v_rows, k_scale_arg, v_scale_arg, topk_idx, output, block_rows,
        cu_seqlens_q, seq_lens, prefix_lens,
        num_kv_heads, gqa_group_size, head_dim, topk, sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k_rows.stride(0), k_rows.stride(1), k_rows.stride(2),
        v_rows.stride(0), v_rows.stride(1), v_rows.stride(2),
        stride_kss, stride_vss,
        topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        block_rows.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        KV_FMT=kv_format.name,
        num_warps=4, num_stages=2,
    )


@torch.no_grad()
def m3_sparse_attn_decode(
    q: torch.Tensor,  # [num_reqs, num_heads, head_dim]
    k_rows: torch.Tensor,  # K slab, row-flat: [rows, num_kv_heads, head_dim]
    v_rows: torch.Tensor,  # V slab, row-flat: [rows, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, num_reqs, topk]
    block_rows: torch.Tensor,  # [num_reqs, max_blocks] int32 base rows
    seq_lens: torch.Tensor,  # [num_reqs] int32 (device, live)
    sm_scale: float,
    output: torch.Tensor,  # [num_reqs, num_heads, head_dim]
    kv_format: KVFormat = NONE,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """GQA block-sparse attention for decode (split-K over the top-k blocks +
    LSE merge; cudagraph-safe)."""
    num_reqs, num_heads, head_dim = q.shape
    num_kv_heads = k_rows.shape[1]
    max_topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    TARGET_GRID = 256
    target = max(1, min(max_topk, TARGET_GRID // max(1, num_reqs * num_kv_heads)))
    num_topk_chunks = 1 << (target.bit_length() - 1)
    o_partial = torch.empty(
        (num_topk_chunks, num_reqs, num_heads, head_dim), dtype=q.dtype, device=q.device
    )
    lse_partial = torch.empty(
        (num_topk_chunks, num_reqs, num_heads), dtype=torch.float32, device=q.device
    )
    k_scale_arg, v_scale_arg, stride_kss, stride_vss = kv_scale_args(kv_format, k_rows, v_rows, k_scale, v_scale, q)
    grid = (num_reqs * num_topk_chunks, num_kv_heads)
    _gqa_sparse_decode_kernel[grid](
        q, k_rows, v_rows, k_scale_arg, v_scale_arg, topk_idx, o_partial, lse_partial, block_rows, seq_lens,
        num_reqs, gqa_group_size, head_dim, max_topk, sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k_rows.stride(0), k_rows.stride(1), k_rows.stride(2),
        v_rows.stride(0), v_rows.stride(1), v_rows.stride(2),
        stride_kss, stride_vss,
        topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
        o_partial.stride(0), o_partial.stride(1), o_partial.stride(2), o_partial.stride(3),
        lse_partial.stride(0), lse_partial.stride(1), lse_partial.stride(2),
        block_rows.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        NUM_TOPK_CHUNKS=num_topk_chunks,
        KV_FMT=kv_format.name,
        # the decode arms raise register use
        num_warps=8 if kv_format.quantized else 4, num_stages=2,
    )
    _merge_topk_attn_out_kernel[(num_reqs, num_heads)](
        o_partial, lse_partial, output, head_dim,
        o_partial.stride(0), o_partial.stride(1), o_partial.stride(2), o_partial.stride(3),
        lse_partial.stride(0), lse_partial.stride(1), lse_partial.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        NUM_TOPK_CHUNKS=num_topk_chunks,
        num_warps=4,
    )


__all__ = [
    "SPARSE_BLOCK_SIZE",
    "m3_index_score_prefill",
    "m3_index_topk_prefill",
    "m3_index_decode",
    "m3_sparse_attn_prefill",
    "m3_sparse_attn_decode",
]
