"""Triton kernels for the QSA KV host offload (``kvcache/kv_host_offload.py``).

The page table, radix tree and compressed-index slab all run in LOGICAL page space; the GPU
KV buffer is an LRU cache of physical slots over the pinned host mirror. These kernels are
the fixed-shape, CUDA-graph-safe glue between the two spaces:

* :func:`write_pages` -- logical page id of every written token slot (``out_loc // page_size``).
* :func:`compact_selected_pages` -- per query row, the distinct logical pages the sparse
  selection touches (``block_table[req, indices//page_size]``), first-element padded.
* :func:`translate_table` -- logical block table -> physical slot table (post-ensure).
* :func:`mirror_store` -- write-through of this forward's K/V rows into the pinned host
  mirror (UVA stores), so a physical slot is always drop-clean.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _write_pages_kernel(out_loc, write_pages, page_size: tl.constexpr):
    i = tl.program_id(0)
    tl.store(write_pages + i, tl.load(out_loc + i) // page_size)


def write_pages(out_loc: torch.Tensor, out: torch.Tensor, page_size: int) -> None:
    """``out[i] = out_loc[i] // page_size`` for i < out.numel() (out_loc may be wider)."""
    n = out.numel()
    if n:
        _write_pages_kernel[(n,)](out_loc, out, page_size)


@triton.jit
def _compact_sel_kernel(
    indices,        # [rows, SEL] int32: selected LOGICAL token ids, -1 padded
    token_to_req,   # [rows] int32
    block_table,    # [reqs, W] int32: logical page ids
    out,            # [rows, MAXP] int32: distinct logical pages per row
    trunc,          # [1] int32: rows whose selection overflowed MAXP (0 = none dropped)
    counts,         # [rows] int32: distinct pages stored per row (<= MAXP)
    sel_width: tl.constexpr,
    width,          # W: block_table row stride
    page_size: tl.constexpr,
    maxp: tl.constexpr,      # rows of `out` per row written (<= MAXV)
    maxv: tl.constexpr,      # next_pow2(maxp)
    dummy_page,              # logical id used when a row selects nothing (never happens in practice)
):
    r = tl.program_id(0)
    req = tl.load(token_to_req + r).to(tl.int64)
    lanes = tl.arange(0, maxv)
    seen = tl.full([maxv], -1, tl.int32)
    first = tl.full((), dummy_page, tl.int32)
    count = tl.zeros((), tl.int32)
    ndrop = tl.zeros((), tl.int32)
    for j in range(sel_width):
        tok = tl.load(indices + r * sel_width + j)
        col = tok // page_size
        pg = tl.load(block_table + req * width + col, mask=tok >= 0, other=-1)
        new = (tok >= 0) & (tl.sum((seen == pg).to(tl.int32)) == 0)
        take = new & (count < maxp)
        ndrop += (new & (count >= maxp)).to(tl.int32)
        seen = tl.where((lanes == count) & take, pg, seen)
        first = tl.where((count == 0) & take, pg, first)
        count += take.to(tl.int32)
    row = tl.where(lanes < count, seen, first)
    tl.store(out + r * maxp + lanes, row, mask=lanes < maxp)
    tl.store(counts + r, count)
    tl.atomic_add(trunc, ndrop)


def compact_selected_pages(
    indices: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    out: torch.Tensor,
    page_size: int,
    dummy_page: int,
    trunc: torch.Tensor | None = None,
    counts: torch.Tensor | None = None,
) -> None:
    """Per query row, the distinct logical pages its selected tokens live on.

    ``out`` is ``[rows, maxp]`` int32; rows with fewer hits are padded with the row's first
    page (duplicates collapse in lru_ensure), or ``dummy_page`` when the row selected nothing.
    ``trunc`` (optional [1] int32, device) accumulates the TOTAL number of pages dropped
    across all rows because they exceeded ``maxp`` -- 0 means no truncation happened. ``counts`` (optional
    [rows] int32, device) receives each row's stored distinct-page count.
    """
    rows, sel = indices.shape
    maxp = out.shape[1]
    if trunc is None:
        trunc = getattr(compact_selected_pages, "_dummy", None)
        if trunc is None or trunc.device != out.device:
            trunc = compact_selected_pages._dummy = torch.zeros(1, dtype=torch.int32, device=out.device)
    if counts is None:
        c = getattr(compact_selected_pages, "_dummy_counts", None)
        if c is None or c.numel() < rows or c.device != out.device:
            c = compact_selected_pages._dummy_counts = torch.empty(
                max(rows, 4096), dtype=torch.int32, device=out.device
            )
        counts = c
    if rows:
        _compact_sel_kernel[(rows,)](
            indices, token_to_req, block_table, out, trunc, counts,
            sel, block_table.stride(0), page_size, maxp, triton.next_power_of_2(maxp),
            dummy_page,
        )


@triton.jit
def _translate_kernel(table, phys_of, out, total, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < total
    pg = tl.load(table + i, mask=mask, other=0)
    ph = tl.load(phys_of + pg, mask=mask, other=0)
    # A -1 entry means the page was not ensured (only possible for a truncated selection,
    # reported via the trunc counter): point it at slot 0 rather than indexing garbage.
    tl.store(out + i, tl.where(ph < 0, 0, ph), mask=mask)


def translate_table(table: torch.Tensor, phys_of: torch.Tensor, out: torch.Tensor) -> None:
    """``out = phys_of[table]`` elementwise over ``out.numel()`` entries (all int32)."""
    n = out.numel()
    if n:
        _translate_kernel[(triton.cdiv(n, 1024),)](table, phys_of, out, n, 1024)


@triton.jit
def _translate_slots_kernel(out_loc, phys_of, out, page_size: tl.constexpr):
    i = tl.program_id(0)
    slot = tl.load(out_loc + i)
    tl.store(out + i, tl.load(phys_of + slot // page_size) * page_size + slot % page_size)


def translate_slots(out_loc: torch.Tensor, phys_of: torch.Tensor, out: torch.Tensor, page_size: int) -> None:
    """Logical token slot -> physical token slot (page translate, offset preserved)."""
    n = out.numel()
    if n:
        _translate_slots_kernel[(n,)](out_loc, phys_of, out, page_size)


@triton.jit
def _mirror_store_kernel(
    k_bytes,      # [T, ROW] uint8 view of this layer's K rows
    v_bytes,      # [T, ROW] uint8 view of this layer's V rows
    out_loc,      # [T] int32 logical token slots
    host_ptrs,    # [2*L] int64: per dense layer the K/V host bank base addresses
    layer,
    row_bytes: tl.constexpr,
    block: tl.constexpr,
):
    t = tl.program_id(0)
    slot = tl.load(out_loc + t).to(tl.int64)
    offs = tl.arange(0, block)
    mask = offs < row_bytes
    kb = tl.load(host_ptrs + 2 * layer).to(tl.pointer_type(tl.uint8))
    vb = tl.load(host_ptrs + 2 * layer + 1).to(tl.pointer_type(tl.uint8))
    tl.store(kb + slot * row_bytes + offs, tl.load(k_bytes + t * row_bytes + offs, mask=mask), mask=mask)
    tl.store(vb + slot * row_bytes + offs, tl.load(v_bytes + t * row_bytes + offs, mask=mask), mask=mask)


def mirror_store(
    k: torch.Tensor,
    v: torch.Tensor,
    out_loc: torch.Tensor,
    host_ptrs: torch.Tensor,
    dense_layer: int,
) -> None:
    """Write-through: this layer's K/V rows of every token in ``out_loc`` into the host mirror."""
    k = k.reshape(k.shape[0], -1)
    v = v.reshape(v.shape[0], -1)
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    assert k.shape == v.shape and k.dtype == v.dtype
    t = k.shape[0]
    row_bytes = k.shape[1] * k.element_size()
    if t:
        _mirror_store_kernel[(t,)](
            k.view(torch.uint8), v.view(torch.uint8), out_loc, host_ptrs, dense_layer,
            row_bytes, triton.next_power_of_2(row_bytes),
        )




@triton.jit
def _write_pages_stride_kernel(out_loc, out, page_size: tl.constexpr):
    """One page per page_size tokens: covers every page of a run but (possibly) its last."""
    i = tl.program_id(0)
    tl.store(out + i, tl.load(out_loc + i * page_size) // page_size)


@triton.jit
def _write_pages_pad_kernel(out, nstride):
    """Pad the boundary slots with out[0]: a real logical page, so a run with fewer
    boundaries than `extra` leaves duplicates behind instead of garbage. Duplicates
    collapse inside lru_ensure."""
    i = tl.program_id(0)
    tl.store(out + nstride + i, tl.load(out + 0))


@triton.jit
def _write_pages_runs_kernel(out_loc, token_to_req, out, counter, n, nstride, extra):
    """The last token of every request run (a prefill batch concatenates runs): its page is
    the one the stride can miss. Bounded by `extra`, so the buffer stays fixed-shape."""
    i = tl.program_id(0)
    cur = tl.load(token_to_req + i)
    nxt = tl.load(token_to_req + i + 1, mask=(i + 1 < n), other=-1)
    if (i + 1 >= n) or (nxt != cur):
        slot = tl.atomic_add(counter, 1)
        if slot < extra:
            tl.store(out + nstride + slot, tl.load(out_loc + i) // page_size)


def write_pages_prefill(out_loc, token_to_req, out, counter, nstride, extra, page_size):
    """Prefill write-page set: stride samples + the last token of each request run.

    Fully device-side (no host sync): a sync here stalls the MoE prefetch pipeline that
    `--moe-prefill-overlap` depends on, which costs more than the sampling itself.
    """
    n = out_loc.numel()
    if not n:
        return
    _write_pages_stride_kernel[(nstride,)](out_loc, out, page_size)
    if extra:
        _write_pages_pad_kernel[(extra,)](out, nstride)
        counter.zero_()
        _write_pages_runs_kernel[(n,)](out_loc, token_to_req, out, counter, n, nstride, extra)


__all__ = ["write_pages", "compact_selected_pages", "translate_table", "translate_slots", "mirror_store"]
