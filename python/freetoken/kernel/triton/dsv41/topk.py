"""Exact top-k and block-candidate selection whose work follows the live history (DeepSeek-V4.1).

The indexer scores a request's compressed positions into a buffer as wide as the staged history
(``max_seq_len / ratio``, fixed while a CUDA graph is captured), but only the first ``live[row]``
columns are meaningful -- and at 1M staged positions a 4K-token request must not pay for the
other 99.6%. Every kernel here takes the live count per row on the device, keeps its output
addresses and shapes fixed, and restricts what it READS and reduces to the live prefix; nothing
past ``live`` is ever read, so the score buffer's dead area needs no initialization.

Selection is a fixed pipeline of small kernels with explicit workspace contracts:

* ``dsv41_topk``: exact top-k of ``[R, T]`` fp32 scores below ``live``. Level 0 gives every
  ``CHUNK``-wide slice of a row its own program, which radix-selects the slice's top-k into a
  ``(key, column)`` candidate workspace (a global winner beats at most k-1 columns, so it is
  inside its own slice's top-k: the union is exact). Further levels apply the same program to the
  candidates until one register-resident tile remains; the final program ranks it and emits the
  winners in ascending column order, ``-1`` padded. Slices past ``live`` exit at once and a
  slice with no more entries than ``k`` is copied through, so a short history costs a handful of
  near-empty launches. The radix primitives are the QSA block top-k's (``kernel/triton/qsa/topk``).
* ``dsv41_block_scores``: per-``block_size`` block maxima over the live prefix (the Hierarchical
  Sparse Indexer's level one), the block holding the newest live position pinned to ``+inf`` so it
  is always kept; a fixed worker grid strides over the live blocks.
* ``dsv41_candidate_blocks``: the two above plus a fixed-width expansion into the compact candidate
  list ``[R, topk_blocks * block_size]`` of compressed positions, ascending with ``-1`` in the tail.

Tie rule (documented, deterministic): scores order descending, equal scores by ascending column,
so the lowest positions win among exact ties. ``torch.topk`` leaves ties unspecified.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.qsa.topk import _BINS, _PASSES, _RADIX, _monotone_key, _resident_prefix, _tile_ranks

CHUNK = 4096  # register-resident slice per level program (fp32 keys + int32 columns)
FINAL_MAX = 4096  # the final program's resident candidate tile
BLOCK_SCORE_TILE = 512  # blocks per block-max program tile


@triton.jit
def _load_entries(src_ptr, col_ptr, base, offsets, limit, SRC_LOGITS: tl.constexpr):
    """One slice of a level's input as ``(monotone key, column)``; key 0 = dead (past ``limit``,
    ``-inf``, or a padding slot of the previous level)."""
    live = offsets < limit
    if SRC_LOGITS:
        value = tl.load(src_ptr + base + offsets, mask=live, other=-float("inf"))
        key = tl.where(live & (value > -float("inf")), _monotone_key(value), 0)
        cols = (base + offsets).to(tl.int32)
    else:
        key = tl.load(src_ptr + base + offsets, mask=live, other=0).to(tl.uint32, bitcast=True)
        cols = tl.load(col_ptr + base + offsets, mask=live, other=-1)
    return key, cols


@triton.jit
def _dsv41_topk_level_kernel(
    src_ptr, src_col_ptr, live_in_ptr, key_out_ptr, col_out_ptr, live_out_ptr,
    stride_src_row, stride_out_row, num_columns,
    TOP_K: tl.constexpr, PAD_K: tl.constexpr, CHUNK: tl.constexpr, N_SPLITS: tl.constexpr,
    SRC_LOGITS: tl.constexpr, BINS: tl.constexpr, RADIX: tl.constexpr, PASSES: tl.constexpr,
):
    """One program per (row, slice): the slice's top-k into candidate slot ``split * TOP_K``. Slot
    ``split`` is read downstream only when ``split < cdiv(live, CHUNK)``; program 0 publishes that
    live candidate count for the next level."""
    tl.static_assert(CHUNK <= 0xFFFF, "packed cumsum keeps 16 bits per half")
    tl.static_assert(TOP_K <= CHUNK, "a slice must hold its own top-k")
    row = tl.program_id(0)
    split = tl.program_id(1)
    visible = tl.maximum(tl.minimum(tl.load(live_in_ptr + row), num_columns), 0)
    if split == 0:
        tl.store(live_out_ptr + row, tl.minimum(tl.cdiv(visible, CHUNK), N_SPLITS) * TOP_K)
    base = split * CHUNK
    limit = tl.minimum(tl.maximum(visible - base, 0), CHUNK)
    if limit <= 0:
        return  # a dead slice: nothing downstream reads its slot
    slot = row.to(tl.int64) * stride_out_row + split * TOP_K
    offsets = tl.arange(0, CHUNK)
    key, cols = _load_entries(src_ptr + row.to(tl.int64) * stride_src_row, src_col_ptr + row.to(tl.int64) * stride_src_row, base, offsets, limit, SRC_LOGITS)
    if limit <= TOP_K:
        # every live entry is a winner: copy in order (dead keys stay 0 for the next level)
        inside = offsets < TOP_K
        tl.store(key_out_ptr + slot + offsets, key.to(tl.int32, bitcast=True), mask=inside)
        tl.store(col_out_ptr + slot + offsets, cols, mask=inside)
        return
    prefix, ties = _resident_prefix(key, TOP_K, BINS, RADIX, PASSES)
    rank, take, above, equal = _tile_ranks(key, prefix, ties, 0, 0)
    write = take & (rank < TOP_K)
    tl.store(key_out_ptr + slot + rank, key.to(tl.int32, bitcast=True), mask=write)
    tl.store(col_out_ptr + slot + rank, cols, mask=write)
    emitted = above + tl.minimum(equal, ties)
    pad = tl.arange(0, PAD_K)
    tl.store(key_out_ptr + slot + pad, 0, mask=(pad >= emitted) & (pad < TOP_K))


@triton.jit
def _final_tile(src_row, col_row, out_row, limit, TOP_K: tl.constexpr, BLOCK: tl.constexpr, SRC_LOGITS: tl.constexpr, BINS: tl.constexpr, RADIX: tl.constexpr, PASSES: tl.constexpr):
    tl.static_assert(BLOCK <= 0xFFFF, "packed cumsum keeps 16 bits per half")
    offsets = tl.arange(0, BLOCK)
    key, cols = _load_entries(src_row, col_row, 0, offsets, limit, SRC_LOGITS)
    prefix, ties = _resident_prefix(key, tl.minimum(limit, TOP_K), BINS, RADIX, PASSES)
    rank, take, above, equal = _tile_ranks(key, prefix, ties, 0, 0)
    tl.store(out_row + rank, cols, mask=take & (rank < TOP_K))
    return above + tl.minimum(equal, ties)


@triton.jit
def _dsv41_topk_final_kernel(
    src_ptr, src_col_ptr, live_ptr, out_ptr,
    stride_src_row, stride_out_row, num_columns,
    TOP_K: tl.constexpr, PAD_K: tl.constexpr,
    BLOCK_SMALL: tl.constexpr, BLOCK_MID: tl.constexpr, BLOCK_FULL: tl.constexpr,
    SRC_LOGITS: tl.constexpr, BINS: tl.constexpr, RADIX: tl.constexpr, PASSES: tl.constexpr,
):
    """One program per row over the last level's live candidates (or the scores themselves when the
    row fits one tile): winners at their ascending-column ranks, ``-1`` from ``emitted`` on."""
    row = tl.program_id(0)
    limit = tl.maximum(tl.minimum(tl.load(live_ptr + row), num_columns), 0)
    src_row = src_ptr + row.to(tl.int64) * stride_src_row
    col_row = src_col_ptr + row.to(tl.int64) * stride_src_row
    out_row = out_ptr + row.to(tl.int64) * stride_out_row
    emitted = 0
    if limit > 0:
        # register residency costs the whole tile, so a short row takes a narrower one
        if limit <= BLOCK_SMALL:
            emitted = _final_tile(src_row, col_row, out_row, limit, TOP_K, BLOCK_SMALL, SRC_LOGITS, BINS, RADIX, PASSES)
        elif limit <= BLOCK_MID:
            emitted = _final_tile(src_row, col_row, out_row, limit, TOP_K, BLOCK_MID, SRC_LOGITS, BINS, RADIX, PASSES)
        else:
            emitted = _final_tile(src_row, col_row, out_row, limit, TOP_K, BLOCK_FULL, SRC_LOGITS, BINS, RADIX, PASSES)
    pad = tl.arange(0, PAD_K)
    tl.store(out_row + pad, -1, mask=(pad >= emitted) & (pad < TOP_K))


MAX_K = CHUNK // 2  # a slice must shed at least half its entries per level, or the plan cannot converge


def topk_plan(columns: int, k: int) -> list[tuple[int, int]]:
    """``[(input width, n_splits), ...]`` of the reduction levels before the final tile; empty when
    the row already fits the final tile. Derived from the buffer width alone (graph-static). Each
    level maps ``width`` to ``cdiv(width, CHUNK) * k`` candidates, which shrinks the row by at least
    ``CHUNK / k >= 2`` per level for ``k <= MAX_K``; larger ``k`` would stall (``k == CHUNK`` never
    shrinks) and is rejected here, before any launch."""
    if not 0 < k <= MAX_K:
        raise ValueError(f"dsv41_topk supports 1 <= k <= {MAX_K}, got {k}")
    levels: list[tuple[int, int]] = []
    width = columns
    while width > FINAL_MAX:
        n_splits = -(-width // CHUNK)
        levels.append((width, n_splits))
        nxt = n_splits * k
        assert nxt < width, (width, k)  # guaranteed by k <= CHUNK // 2 for width > FINAL_MAX == CHUNK
        width = nxt
    return levels


def dsv41_topk(scores: torch.Tensor, live: torch.Tensor, k: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """Exact top-``k`` columns of every ``scores`` row among its first ``live[row]`` columns.

    ``scores [R, T]`` fp32 row-contiguous (columns at or past ``live`` are never read; ``-inf``
    columns are dead), ``live [R]`` int32. Returns ``out [R, k]`` int32: the winners in ascending
    column order packed at the front, ``-1`` from the ``emitted`` count on (fewer live or finite
    columns than ``k`` leave a ``-1`` tail). Ties: lower column wins."""
    assert scores.ndim == 2 and scores.dtype == torch.float32 and scores.stride(1) == 1, (scores.shape, scores.dtype)
    R, T = scores.shape
    assert live.shape == (R,), (live.shape, R)
    live = live.to(torch.int32).contiguous()
    if out is None:
        out = torch.empty((R, k), dtype=torch.int32, device=scores.device)
    assert out.shape == (R, k) and out.dtype == torch.int32 and out.stride(1) == 1
    if R == 0 or k == 0:
        return out
    if T == 0:
        out.fill_(-1)
        return out
    pad_k = triton.next_power_of_2(k)
    src, src_col, src_live, width, src_logits = scores, scores, live, T, True
    for level_width, n_splits in topk_plan(T, k):
        keys = torch.empty((R, n_splits * k), dtype=torch.int32, device=scores.device)
        cols = torch.empty((R, n_splits * k), dtype=torch.int32, device=scores.device)
        live_out = torch.empty((R,), dtype=torch.int32, device=scores.device)
        _dsv41_topk_level_kernel[(R, n_splits)](
            src, src_col, src_live, keys, cols, live_out,
            src.stride(0), keys.stride(0), level_width,
            TOP_K=k, PAD_K=pad_k, CHUNK=CHUNK, N_SPLITS=n_splits, SRC_LOGITS=src_logits,
            BINS=_BINS, RADIX=_RADIX, PASSES=_PASSES, num_warps=8, num_stages=1,
        )
        src, src_col, src_live, width, src_logits = keys, cols, live_out, n_splits * k, False
    block = triton.next_power_of_2(max(width, 1))
    _dsv41_topk_final_kernel[(R,)](
        src, src_col, src_live, out,
        src.stride(0), out.stride(0), width,
        TOP_K=k, PAD_K=pad_k,
        BLOCK_SMALL=min(1024, block), BLOCK_MID=min(4096, block), BLOCK_FULL=block,
        SRC_LOGITS=src_logits, BINS=_BINS, RADIX=_RADIX, PASSES=_PASSES, num_warps=8, num_stages=1,
    )
    return out


@triton.jit
def _dsv41_block_scores_kernel(
    scores_ptr, live_ptr, blk_ptr,
    stride_s_row, stride_b_row, num_blocks,
    BLOCK_SIZE: tl.constexpr, TILE: tl.constexpr, N_WORKERS: tl.constexpr,
):
    """Worker ``w`` of a row scores block tiles ``w, w + N_WORKERS, ...`` below the live block count:
    the block's best live position, ``+inf`` for the block holding the newest live position."""
    row = tl.program_id(0)
    worker = tl.program_id(1)
    live = tl.maximum(tl.load(live_ptr + row), 0)
    nb_live = tl.minimum(tl.cdiv(live, BLOCK_SIZE), num_blocks)
    s_row = scores_ptr + row.to(tl.int64) * stride_s_row
    b_row = blk_ptr + row.to(tl.int64) * stride_b_row
    offs_b = tl.arange(0, TILE)
    offs_p = tl.arange(0, BLOCK_SIZE)
    for tile in range(worker, tl.cdiv(nb_live, TILE), N_WORKERS):
        blocks = tile * TILE + offs_b
        b_mask = blocks < nb_live
        pos = blocks[:, None] * BLOCK_SIZE + offs_p[None, :]
        v = tl.load(s_row + pos, mask=b_mask[:, None] & (pos < live), other=-float("inf"))
        best = tl.max(v, axis=1)
        best = tl.where(blocks == nb_live - 1, float("inf"), best)
        tl.store(b_row + blocks, best, mask=b_mask)


def dsv41_block_scores(scores: torch.Tensor, live: torch.Tensor, block_size: int, out: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``(block scores [R, cdiv(T, block_size)] fp32, live blocks [R] int32)``: each live block's best
    live position (``+inf`` for the newest block); blocks at or past the live count are not written."""
    R, T = scores.shape
    nb = -(-T // block_size)
    live = live.to(torch.int32).contiguous()
    if out is None:
        out = torch.empty((R, nb), dtype=torch.float32, device=scores.device)
    live_blocks = torch.div(live + block_size - 1, block_size, rounding_mode="floor").clamp_(min=0).to(torch.int32)
    if R == 0 or nb == 0:
        return out, live_blocks
    assert block_size == triton.next_power_of_2(block_size), block_size
    sm_count = torch.cuda.get_device_properties(scores.device).multi_processor_count
    workers = min(triton.cdiv(nb, BLOCK_SCORE_TILE), max(1, triton.cdiv(sm_count, R)))
    _dsv41_block_scores_kernel[(R, workers)](
        scores, live, out, scores.stride(0), out.stride(0), nb,
        BLOCK_SIZE=block_size, TILE=BLOCK_SCORE_TILE, N_WORKERS=workers, num_warps=4,
    )
    return out, live_blocks


def dsv41_candidate_blocks(scores: torch.Tensor, live: torch.Tensor, topk_blocks: int, block_size: int) -> torch.Tensor:
    """Level one of the Hierarchical Sparse Indexer over ``[R, T]`` scores: the ``topk_blocks`` best
    live blocks (the newest always among them) expanded to the compact candidate list ``[R,
    topk_blocks * block_size]`` of compressed positions -- ascending, ``-1`` for positions at or past
    ``live`` and for unfilled blocks, all in the tail (a sorted valid prefix)."""
    R, T = scores.shape
    blk, live_blocks = dsv41_block_scores(scores, live, block_size)
    keep = dsv41_topk(blk, live_blocks, min(topk_blocks, blk.shape[1]))  # [R, KB] ascending, -1 tail
    pos = keep.to(torch.int64).unsqueeze(-1) * block_size + torch.arange(block_size, device=scores.device)
    dead = (keep < 0).unsqueeze(-1) | (pos >= live.to(torch.int64).view(R, 1, 1))
    out = torch.where(dead, -1, pos).flatten(1).to(torch.int32)
    if out.shape[1] < topk_blocks * block_size:
        out = torch.nn.functional.pad(out, (0, topk_blocks * block_size - out.shape[1]), value=-1)
    return out


__all__ = ["dsv41_topk", "dsv41_block_scores", "dsv41_candidate_blocks", "topk_plan", "CHUNK", "FINAL_MAX", "MAX_K"]
