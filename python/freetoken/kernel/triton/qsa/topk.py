"""Exact block top-k over the QSA indexer scores.

torch.topk also sorts its k winners, which the selection never uses: expand.py expands every
rank the same way and the sparse attend kernel softmaxes over the union, so only the SET
matters. One program per row runs an MSB-first radix select on the monotone uint32 image of
the fp32 scores -- ``PASSES`` ``RADIX``-bit passes, each a histogram of the columns that still
match the fixed prefix -- and one compaction pass then emits the winners in ascending column
order, -1 padded to the output width like the torch.topk path it replaces.

A row that fits one tile is held in registers across the passes (``SINGLE_TILE``); wider rows
re-read the tile per pass, which is the price of not spilling a 256 KB row.

One program per row stops scaling once a row runs to tens of thousands of columns: the whole
row goes through a single CTA. Wide buffers therefore split. Phase 1 gives each ``CHUNK``-wide
slice of a row its own program, which radix-selects the slice's own top-k into a candidate
workspace; phase 2 runs the same radix select over the union of those candidates. The global
top-k is a subset of that union -- a global winner beats at most k-1 columns, so it beats all
but at most k-1 of its own slice -- so the answer stays exact. ``_split_plan`` derives the
geometry from the buffer width alone, which is fixed when a graph is captured.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# 5-bit digits beat 8-bit ones here: tl.histogram cost grows faster with the bin count than the
# extra passes cost. 35 bits of digit cover the 32-bit key, the top pass just sees zero bits.
_RADIX = 5
_BINS = 1 << _RADIX
_PASSES = -(-32 // _RADIX)
_MAX_BLOCK_N = 4096
# A resident tile costs ~2 ns per column against ~3.2 ns for a re-read one, so both split
# phases stay resident; past 8192 the tile spills (255 registers) and that reverses.
_MAX_RESIDENT = 8192
_MIN_CHUNK = 4096


@triton.jit
def _monotone_key(value):
    """fp32 -> uint32 with the float order preserved; key 0 is reserved for a dead column."""
    bits = value.to(tl.uint32, bitcast=True)
    return tl.where((bits >> 31) == 0, bits | 0x80000000, ~bits)


@triton.jit
def _load_keys(logits_row, columns, limit):
    live = columns < limit
    value = tl.load(logits_row + columns, mask=live, other=-float("inf"))
    return tl.where(live & (value > -float("inf")), _monotone_key(value), 0)


@triton.jit
def _narrow(hist, prefix, keep, k_rem, shift, BINS: tl.constexpr):
    """One radix step: pin the digit at ``shift`` and report whether the range is settled."""
    bins = tl.arange(0, BINS)
    # Lowest bin whose strictly-greater bins no longer cover k_rem holds the k_rem-th key;
    # `above` falls with the bin id, so the predicate is an upper set.
    above = tl.sum(hist) - tl.cumsum(hist, axis=0)
    inside = above < k_rem
    bin_id = tl.min(tl.where(inside, bins, BINS - 1))
    hit = bins == bin_id
    prefix |= bin_id.to(tl.uint32) << shift
    keep |= tl.full((), BINS - 1, tl.uint32) << shift
    k_rem -= tl.sum(tl.where(hit, above, 0))
    # The whole bin fits: `prefix` is a lower bound, the ties below it all win.
    return prefix, keep, k_rem, k_rem == tl.sum(tl.where(hit, hist, 0))


@triton.jit
def _resident_prefix(key, k_eff, BINS: tl.constexpr, RADIX: tl.constexpr, PASSES: tl.constexpr):
    """Radix-select a register-resident tile; returns the winning key range and its tie budget."""
    # Invariant per pass: `k_rem` winners are still to be found inside the key range that
    # `prefix` pins on the `keep` bits, and every key above that range is already a winner.
    prefix = tl.zeros((), tl.uint32)
    keep = tl.zeros((), tl.uint32)
    k_rem = k_eff
    settled = False
    for step in tl.static_range(PASSES):
        shift = RADIX * (PASSES - 1 - step)
        if not settled:
            hist = tl.histogram(
                ((key >> shift) & (BINS - 1)).to(tl.int32),
                BINS,
                mask=(key != 0) & ((key & keep) == prefix),
            )
            prefix, keep, k_rem, settled = _narrow(hist, prefix, keep, k_rem, shift, BINS)
    return prefix, tl.where(settled, k_eff, k_rem)


@triton.jit
def _tile_ranks(key, prefix, ties, above_base, equal_base):
    """Rank every winner of one tile, and carry the running counts past it.

    One packed cumsum carries both: keys above the threshold in the low half, keys equal to it
    in the high half, so a winner's rank is ``above_before + min(equal_before, ties)``."""
    greater = ((key > prefix) & (key != 0)).to(tl.int32)
    equal = ((key == prefix) & (key != 0)).to(tl.int32)
    packed = greater | (equal << 16)
    before = tl.cumsum(packed, axis=0) - packed
    rank_equal = equal_base + (before >> 16)
    take = (greater == 1) | ((equal == 1) & (rank_equal < ties))
    rank = above_base + (before & 0xFFFF) + tl.minimum(rank_equal, ties)
    total = tl.sum(packed)
    return rank, take, above_base + (total & 0xFFFF), equal_base + (total >> 16)


@triton.jit
def _compact(
    out_row,
    key,
    columns,
    prefix,
    ties,
    above_base,
    equal_base,
    TOP_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Emit one tile's winners at their global ranks; returns the running counts after it."""
    tl.static_assert(BLOCK_N <= 0xFFFF, "packed cumsum keeps 16 bits per half")
    rank, take, above, equal = _tile_ranks(key, prefix, ties, above_base, equal_base)
    tl.store(out_row + rank, columns.to(tl.int32), mask=take & (rank < TOP_K))
    return above, equal


@triton.jit
def _qsa_block_topk_kernel(
    logits_ptr,
    visible_ptr,
    out_ptr,
    stride_logits_row,
    stride_out_row,
    num_columns,
    TOP_K: tl.constexpr,
    PAD_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SINGLE_TILE: tl.constexpr,
    BINS: tl.constexpr,
    RADIX: tl.constexpr,
    PASSES: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    limit = tl.maximum(tl.minimum(tl.load(visible_ptr + row), num_columns), 0)
    k_eff = tl.minimum(limit, TOP_K)
    logits_row = logits_ptr + row.to(tl.int64) * stride_logits_row
    out_row = out_ptr + row.to(tl.int64) * stride_out_row
    offsets = tl.arange(0, BLOCK_N)
    tiles = tl.cdiv(limit, BLOCK_N)
    if SINGLE_TILE:
        resident = _load_keys(logits_row, offsets, limit)

    emitted = 0
    if k_eff > 0:
        if SINGLE_TILE:
            prefix, ties = _resident_prefix(resident, k_eff, BINS, RADIX, PASSES)
        else:
            prefix = tl.zeros((), tl.uint32)
            keep = tl.zeros((), tl.uint32)
            k_rem = k_eff
            settled = False
            for step in tl.static_range(PASSES):
                shift = RADIX * (PASSES - 1 - step)
                if not settled:
                    hist = tl.zeros((BINS,), tl.int32)
                    for tile in range(tiles):
                        key = _load_keys(logits_row, tile * BLOCK_N + offsets, limit)
                        hist += tl.histogram(
                            ((key >> shift) & (BINS - 1)).to(tl.int32),
                            BINS,
                            mask=(key != 0) & ((key & keep) == prefix),
                        )
                    prefix, keep, k_rem, settled = _narrow(
                        hist, prefix, keep, k_rem, shift, BINS
                    )
            ties = tl.where(settled, k_eff, k_rem)

        above_base = 0
        equal_base = 0
        if SINGLE_TILE:
            above_base, equal_base = _compact(
                out_row, resident, offsets, prefix, ties, above_base, equal_base, TOP_K, BLOCK_N
            )
        else:
            for tile in range(tiles):
                columns = tile * BLOCK_N + offsets
                key = _load_keys(logits_row, columns, limit)
                above_base, equal_base = _compact(
                    out_row, key, columns, prefix, ties, above_base, equal_base, TOP_K, BLOCK_N
                )
        emitted = above_base + tl.minimum(equal_base, ties)

    pad = tl.arange(0, PAD_K)
    tl.store(out_row + pad, -1, mask=(pad >= emitted) & (pad < TOP_K))


@triton.jit
def _qsa_topk_split_kernel(
    logits_ptr,
    visible_ptr,
    key_ptr,
    col_ptr,
    stride_logits_row,
    stride_scratch_row,
    num_columns,
    TOP_K: tl.constexpr,
    PAD_K: tl.constexpr,
    CHUNK: tl.constexpr,
    BINS: tl.constexpr,
    RADIX: tl.constexpr,
    PASSES: tl.constexpr,
) -> None:
    """Phase 1: one program per (row, chunk), writing the chunk's own top-k as candidates."""
    tl.static_assert(CHUNK <= 0xFFFF, "packed cumsum keeps 16 bits per half")
    row = tl.program_id(0)
    split = tl.program_id(1)
    base = split * CHUNK
    visible = tl.maximum(tl.minimum(tl.load(visible_ptr + row), num_columns), 0)
    limit = tl.minimum(tl.maximum(visible - base, 0), CHUNK)
    k_eff = tl.minimum(limit, TOP_K)
    slot = row.to(tl.int64) * stride_scratch_row + split * TOP_K

    emitted = 0
    if k_eff > 0:
        offsets = tl.arange(0, CHUNK)
        key = _load_keys(logits_ptr + row.to(tl.int64) * stride_logits_row + base, offsets, limit)
        prefix, ties = _resident_prefix(key, k_eff, BINS, RADIX, PASSES)
        rank, take, above, equal = _tile_ranks(key, prefix, ties, 0, 0)
        write = take & (rank < TOP_K)
        tl.store(key_ptr + slot + rank, key.to(tl.int32, bitcast=True), mask=write)
        tl.store(col_ptr + slot + rank, (base + offsets).to(tl.int32), mask=write)
        emitted = above + tl.minimum(equal, ties)
    # A key of 0 is the dead-column sentinel, so the merge needs no separate count per slot.
    pad = tl.arange(0, PAD_K)
    tl.store(key_ptr + slot + pad, 0, mask=(pad >= emitted) & (pad < TOP_K))


@triton.jit
def _merge_tile(
    key_row,
    col_row,
    out_row,
    candidates,
    k_eff,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
    BINS: tl.constexpr,
    RADIX: tl.constexpr,
    PASSES: tl.constexpr,
):
    """Top-k of one resident tile of candidates; returns how many winners it wrote."""
    tl.static_assert(BLOCK <= 0xFFFF, "packed cumsum keeps 16 bits per half")
    offsets = tl.arange(0, BLOCK)
    live = offsets < candidates
    key = tl.load(key_row + offsets, mask=live, other=0).to(tl.uint32, bitcast=True)
    prefix, ties = _resident_prefix(key, k_eff, BINS, RADIX, PASSES)
    rank, take, above, equal = _tile_ranks(key, prefix, ties, 0, 0)
    column = tl.load(col_row + offsets, mask=live, other=-1)
    tl.store(out_row + rank, column, mask=take & (rank < TOP_K))
    return above + tl.minimum(equal, ties)


@triton.jit
def _qsa_topk_merge_kernel(
    visible_ptr,
    key_ptr,
    col_ptr,
    out_ptr,
    stride_scratch_row,
    stride_out_row,
    num_columns,
    TOP_K: tl.constexpr,
    PAD_K: tl.constexpr,
    CHUNK: tl.constexpr,
    N_SPLITS: tl.constexpr,
    BLOCK_SMALL: tl.constexpr,
    BLOCK_MID: tl.constexpr,
    BLOCK_FULL: tl.constexpr,
    BINS: tl.constexpr,
    RADIX: tl.constexpr,
    PASSES: tl.constexpr,
) -> None:
    """Phase 2: one program per row over the candidates phase 1 left behind."""
    row = tl.program_id(0)
    limit = tl.maximum(tl.minimum(tl.load(visible_ptr + row), num_columns), 0)
    k_eff = tl.minimum(limit, TOP_K)
    # Candidates sit chunk-major, so the splits past the visible tail are one skipped suffix
    # and the merge keeps costing what the live part of the row costs.
    candidates = tl.minimum(tl.cdiv(limit, CHUNK), N_SPLITS) * TOP_K
    key_row = key_ptr + row.to(tl.int64) * stride_scratch_row
    col_row = col_ptr + row.to(tl.int64) * stride_scratch_row
    out_row = out_ptr + row.to(tl.int64) * stride_out_row

    emitted = 0
    if k_eff > 0:
        if candidates <= TOP_K:
            # One live split: its own top-k is already the row's, ranked and packed.
            slot = tl.arange(0, PAD_K)
            inside = slot < TOP_K
            key = tl.load(key_row + slot, mask=inside, other=0)
            alive = inside & (key != 0)
            column = tl.load(col_row + slot, mask=alive, other=-1)
            tl.store(out_row + slot, column, mask=inside)
            emitted = tl.sum(alive.to(tl.int32))
        # Register residency costs the whole tile even when few candidates are live, so a
        # short row takes a narrower tile.
        elif candidates <= BLOCK_SMALL:
            emitted = _merge_tile(
                key_row, col_row, out_row, candidates, k_eff,
                TOP_K, BLOCK_SMALL, BINS, RADIX, PASSES,
            )
        elif candidates <= BLOCK_MID:
            emitted = _merge_tile(
                key_row, col_row, out_row, candidates, k_eff,
                TOP_K, BLOCK_MID, BINS, RADIX, PASSES,
            )
        else:
            emitted = _merge_tile(
                key_row, col_row, out_row, candidates, k_eff,
                TOP_K, BLOCK_FULL, BINS, RADIX, PASSES,
            )

    pad = tl.arange(0, PAD_K)
    tl.store(out_row + pad, -1, mask=(pad >= emitted) & (pad < TOP_K))


def _split_plan(columns: int, top_k: int) -> tuple[int, int] | None:
    """``(chunk, n_splits)`` for the split+merge path, or None to keep the one-program path."""
    if top_k <= 0 or columns <= _MIN_CHUNK:
        return None
    max_splits = _MAX_RESIDENT // triton.next_power_of_2(top_k)
    if max_splits < 2:
        return None
    chunk = max(_MIN_CHUNK, triton.next_power_of_2(-(-columns // max_splits)))
    if chunk > _MAX_RESIDENT:
        return None
    n_splits = -(-columns // chunk)
    # Merging n_splits*top_k candidates has to be cheaper than scanning the row once.
    if n_splits < 2 or n_splits * top_k >= columns:
        return None
    return chunk, n_splits


def qsa_block_topk_scratch_width(columns: int, top_k: int) -> int:
    """int32 columns of scratch ``qsa_block_topk`` wants per row; 0 when it needs none."""
    plan = _split_plan(columns, top_k)
    return 0 if plan is None else 2 * plan[1] * top_k


def qsa_block_topk(
    logits: torch.Tensor,
    visible: torch.Tensor,
    out: torch.Tensor,
    scratch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Top ``out.shape[1]`` columns of every ``logits`` row below ``visible``, -1 padded.

    Winners come out in ascending column order, packed at the front of the row; a row with
    fewer live columns than the width, and any column scored -inf, leaves -1 in the tail.
    ``scratch`` is the candidate workspace of the split path (see
    ``qsa_block_topk_scratch_width``); it is allocated per call when the caller passes none."""

    if logits.ndim != 2 or out.ndim != 2:
        raise ValueError("QSA block top-k takes 2-D logits and output")
    if out.shape[0] != logits.shape[0] or visible.shape != (logits.shape[0],):
        raise ValueError("QSA block top-k needs one visible count and one output row per row")
    if logits.dtype != torch.float32 or out.dtype != torch.int32:
        raise ValueError("QSA block top-k takes fp32 logits and an int32 output")
    if logits.stride(1) != 1 or out.stride(1) != 1 or visible.stride(0) != 1:
        raise ValueError("QSA block top-k needs row-contiguous logits, output and counts")
    rows, columns = logits.shape
    top_k = out.shape[1]
    if not rows or not top_k:
        return out
    pad_k = triton.next_power_of_2(top_k)
    plan = _split_plan(columns, top_k)
    if plan is None:
        block_n = min(_MAX_BLOCK_N, triton.next_power_of_2(max(columns, 1)))
        _qsa_block_topk_kernel[(rows,)](
            logits,
            visible,
            out,
            logits.stride(0),
            out.stride(0),
            columns,
            TOP_K=top_k,
            PAD_K=pad_k,
            BLOCK_N=block_n,
            SINGLE_TILE=columns <= block_n,
            BINS=_BINS,
            RADIX=_RADIX,
            PASSES=_PASSES,
            num_warps=8,
            num_stages=1,
        )
        return out

    chunk, n_splits = plan
    half = n_splits * top_k
    if scratch is None:
        scratch = torch.empty((rows, 2 * half), dtype=torch.int32, device=logits.device)
    elif (
        scratch.ndim != 2
        or scratch.shape[0] < rows
        or scratch.shape[1] < 2 * half
        or scratch.dtype != torch.int32
        or scratch.stride(1) != 1
    ):
        raise ValueError(
            f"QSA block top-k needs a row-contiguous int32 scratch of at least "
            f"[{rows}, {2 * half}], got {tuple(scratch.shape)} {scratch.dtype}"
        )
    keys = scratch[:rows, :half]
    cols = scratch[:rows, half : 2 * half]
    _qsa_topk_split_kernel[(rows, n_splits)](
        logits,
        visible,
        keys,
        cols,
        logits.stride(0),
        keys.stride(0),
        columns,
        TOP_K=top_k,
        PAD_K=pad_k,
        CHUNK=chunk,
        BINS=_BINS,
        RADIX=_RADIX,
        PASSES=_PASSES,
        num_warps=8,
        num_stages=1,
    )
    merge_block = triton.next_power_of_2(half)
    _qsa_topk_merge_kernel[(rows,)](
        visible,
        keys,
        cols,
        out,
        keys.stride(0),
        out.stride(0),
        columns,
        TOP_K=top_k,
        PAD_K=pad_k,
        CHUNK=chunk,
        N_SPLITS=n_splits,
        BLOCK_SMALL=min(1024, merge_block),
        BLOCK_MID=min(4096, merge_block),
        BLOCK_FULL=merge_block,
        BINS=_BINS,
        RADIX=_RADIX,
        PASSES=_PASSES,
        num_warps=8,
        num_stages=1,
    )
    return out


__all__ = ["qsa_block_topk", "qsa_block_topk_scratch_width"]
