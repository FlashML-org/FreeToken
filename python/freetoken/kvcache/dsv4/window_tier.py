"""Shared pieces of the sliding-window paged pools (DSV4, DSV41).

* ``FreeListAllocator``   -- LIFO page allocator for the window tier (unit = one P-slot page).
* ``CompressStateRing``   -- per-window-page fp32 ``kv | score`` carry ring with a scratch row.
* ``reserved_window_pages`` -- the window working set every such pool keeps for its running requests.
* ``window_state_loc``    -- the derived (never stored) ring slot of a window slot.
* ``WindowTierPagedPool`` -- the pool base: the full-loc -> window-slot mapping over the shared page
  table, the page-atomic window allocator behind the CacheManager's ``swa_pool`` duck-type, the
  in-place rebuild, and the ABC glue every such pool shares. Subclasses allocate their own tiers
  (``_alloc_buffers`` / ``_drop_buffers``), size themselves (the ``kv_cost`` family) and write rows.

``DSV4PagedKVCache`` (the origin of this code) and ``DSV41PagedKVCache`` both build on these, so a
fix to allocation, freeing or the mapping lands in one place.
"""

from __future__ import annotations

import gc
from abc import abstractmethod

import torch

from ..base import BaseKVCachePool


class FreeListAllocator:
    """LIFO free list over ``capacity // page_unit`` units; a unit's base slot is a multiple of
    ``page_unit`` (the window tier's ``G*P`` page-base invariant)."""

    def __init__(self, capacity: int, device: torch.device, page_unit: int = 1) -> None:
        assert capacity % page_unit == 0, f"capacity {capacity} must be a multiple of page_unit {page_unit}"
        self._capacity = int(capacity)
        self._page_unit = int(page_unit)
        self._device = device
        self._free = self._fresh_free()

    def _fresh_free(self) -> torch.Tensor:
        n_units = self._capacity // self._page_unit
        return torch.arange(n_units, dtype=torch.int64, device=self._device) * self._page_unit

    def alloc(self, n_units: int) -> torch.Tensor:
        """Return ``n_units`` unit base slots (each a multiple of ``page_unit``)."""
        if n_units < 0:
            raise ValueError(f"n_units must be non-negative, got {n_units}")
        if n_units > self._free.numel():
            raise RuntimeError(
                f"FreeListAllocator out of slots: requested {n_units} units, "
                f"have {self._free.numel()} (capacity {self._capacity}, unit {self._page_unit})"
            )
        if n_units == 0:
            return self._free[:0].clone()
        taken = self._free[-n_units:].clone()
        self._free = self._free[:-n_units]
        return taken

    def free(self, units: torch.Tensor) -> None:
        """Return previously-allocated unit base slots to the pool (LIFO recycle)."""
        if units.numel() == 0:
            return
        units = units.to(device=self._device, dtype=torch.int64).reshape(-1)
        self._free = torch.cat([self._free, units])

    def available(self) -> int:
        """Free capacity in slots (units * page_unit)."""
        return int(self._free.numel()) * self._page_unit

    @property
    def capacity(self) -> int:
        return self._capacity

    def reset(self) -> None:
        self._free = self._fresh_free()


class CompressStateRing:
    """Per-layer fp32 compress-state ring: ``[n_slots + 1, 2*(1+overlap)*head_dim]``.

    Last row (index ``-1``) is a permanent scratch slot, re-cleared on every write. Last dim is
    split ``kv | score``; ``set`` writes both halves.
    """

    def __init__(
        self,
        n_slots: int,
        ring_size: int,
        overlap: bool,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.ring_size = ring_size
        self.head_dim = head_dim
        self._item_size = (1 + int(overlap)) * head_dim  # width of kv (== width of score)
        last_dim = 2 * self._item_size
        self.buffer = torch.zeros((n_slots + 1, last_dim), dtype=dtype, device=device)
        self.n_slots = n_slots
        self._clear_scratch()

    def _clear_scratch(self) -> None:
        self.buffer[-1, : self._item_size].zero_()
        self.buffer[-1, self._item_size :].fill_(float("-inf"))

    @property
    def item_size(self) -> int:
        return self._item_size

    def get(self, state_loc: torch.Tensor) -> torch.Tensor:
        """Gather ``kv_score`` rows at ``state_loc`` (``-1`` -> scratch row)."""
        return self.buffer[state_loc]

    def set(self, state_loc: torch.Tensor, kv_score: torch.Tensor) -> None:
        """Scatter ``kv_score`` rows to ``state_loc`` then re-clear the scratch row."""
        self.buffer[state_loc] = kv_score
        self._clear_scratch()

    def get_blocks(self, page_base: torch.Tensor) -> torch.Tensor:
        """Batched per-row carry-block read: ``page_base`` ``[B]`` (``(window_slot // P) * ring_size``)
        -> ``[B, ring_size, 2*item]``. Distinct pages -> disjoint blocks."""
        rows = page_base[:, None] + torch.arange(self.ring_size, device=page_base.device)
        return self.buffer[rows]

    def set_blocks(self, page_base: torch.Tensor, blocks: torch.Tensor) -> None:
        """Batched per-row carry-block write of ``[B, ring_size, 2*item]``; re-clears scratch."""
        rows = page_base[:, None] + torch.arange(self.ring_size, device=page_base.device)
        self.buffer[rows] = blocks
        self._clear_scratch()


def reserved_window_pages(max_running_req: int, radix: bool) -> int:
    """Window pages the sliding pool must always keep for the concurrent working set: each
    running request's decode transients (2 per req + dummy) plus, in radix mode, PER concurrent
    request one locked live-tail page AND a retained (soft-pinned) prompt-end window -- the
    window is 2 pages here because the retention gap page-aligns to a whole extra page at
    P == window."""
    return 2 * (max_running_req + 1) + (3 * max_running_req if radix else 0) + 1


def window_state_loc(window_slot: torch.Tensor, ring_size: int, P: int) -> torch.Tensor:
    """Ring slot of a window slot: ``(ws // P) * ring_size + ws % ring_size``; ``-1`` stays ``-1``
    (the scratch row). ``ring_size | P`` keeps distinct pages on disjoint blocks."""
    pages = torch.div(window_slot, P, rounding_mode="floor")
    loc = pages * ring_size + (window_slot % ring_size)
    return torch.where(window_slot < 0, torch.full_like(loc, -1), loc)


class WindowTierPagedPool(BaseKVCachePool):
    """A paged KV pool whose window tier is page-bound to the shared full-token page table.

    Subclasses set ``P`` (the window page), ``_device``, ``_dtype`` and ``sizes`` (with
    ``full_token`` and ``n_win_slots``) before calling ``_alloc_buffers``, which must create
    ``self.full_to_window`` (``[full_token + 1]`` int64, ``-1`` = unmapped, the trailing row a
    permanent ``-1`` sentinel so a gather at ``-1`` is safe) and ``self.window_pool`` (one tensor
    per layer, indexed by window slot). Scatter paths must never see a negative slot.

    ShadowRadix layering: the shared page table is the virtual full-token coordinate; the pool
    projects it into physical tiers. The window tier is the managed "second currency" -- token-face
    signatures (what the generic CacheManager speaks), PAGE-ATOMIC internals (window pages are 1:1
    page-bound to full pages; the per-page state ring requires it). ``alloc_swa`` receives whole
    ascending pages and free paths are page-complete by construction -- asserted here, not assumed.
    """

    swa_paged = True
    # the tier buffers are bound into per-forward model scratch, invalid after a realloc
    needs_rebind_on_rebuild = True

    P: int
    _device: torch.device
    _dtype: torch.dtype
    sizes: object
    full_to_window: torch.Tensor
    window_pool: list

    # ----- tiers (subclass) -----
    @abstractmethod
    def _alloc_buffers(self) -> None: ...

    @abstractmethod
    def _drop_buffers(self) -> None:
        """Release every tier buffer before a rebuild reallocates them."""

    # ----- full-loc translation (gather-only -1 safety; see the class docstring) -----
    def translate_full_to_window(self, full_locs: torch.Tensor) -> torch.Tensor:
        # int64 gather indices: the shared page_table stores full locs as int32
        return self.full_to_window[full_locs.to(dtype=torch.int64)]

    @staticmethod
    def cmp_rows(full_locs: torch.Tensor, ratio: int) -> torch.Tensor:
        """The compressed row of a full loc: ``full_loc // ratio`` (negative stays negative)."""
        return torch.div(full_locs.to(dtype=torch.int64), ratio, rounding_mode="floor")

    def bind_window_pages(self, full_page_base: int, window_page_base: int) -> None:
        assert full_page_base % self.P == 0 and window_page_base % self.P == 0
        self.full_to_window[full_page_base : full_page_base + self.P] = torch.arange(
            window_page_base, window_page_base + self.P, dtype=torch.int64, device=self._device
        )

    def unbind_window_pages(self, full_locs: torch.Tensor) -> None:
        self.full_to_window[full_locs[full_locs >= 0]] = -1

    @staticmethod
    def state_loc(window_slot: torch.Tensor, ring_size: int, P: int) -> torch.Tensor:
        return window_state_loc(window_slot, ring_size, P)

    # ----- generic swa_pool duck-type (the CacheManager plug-in surface) -----
    @property
    def sliding_window_size(self) -> int:
        return self.P

    @property
    def prefill_chunk_budget(self) -> int:
        return self._chunk_budget

    def _init_paged_state(self, max_running_req: int, radix: bool) -> None:
        """Build the pool-owned window free list + the tail dummy binding. The LAST full page and
        LAST window page are the reserved dummy region: page_table's dummy row points at
        ``full_token - P`` (the generic ``fill_(num_tokens)`` convention), permanently bound so
        graph-padded rows scatter to a real slot."""
        P = self.P
        self._paged_params = (int(max_running_req), bool(radix))
        self.full_to_window.fill_(-1)
        self._win_alloc = FreeListAllocator(self.sizes.n_win_slots - P, self._device, page_unit=P)
        self.bind_window_pages(self.sizes.full_token - P, self.sizes.n_win_slots - P)
        # Chunk cap: a batched prefill holds the whole chunk's window live at once (sliding frees
        # only between chunks; peak ~2x the chunk), so reserve the concurrent working set and
        # halve the rest.
        n_win_pages = (self.sizes.n_win_slots // P) - 1
        reserved = reserved_window_pages(max_running_req, radix)
        self._chunk_budget = max(P, (n_win_pages - reserved) // 2 * P)

    @property
    def window_pages(self) -> int:
        return self.sizes.n_win_pages - 1

    @property
    def swa_num_tokens(self) -> int:
        # Allocatable window slots + 1: the generic capacity convention reserves slot 0 as a
        # sentinel (cap == swa_num_tokens - 1); here the reserved unit is the tail dummy page,
        # already excluded from the free list, so +1 re-encodes the same cap.
        return (self.sizes.n_win_slots - self.P) + 1

    def swa_available_size(self) -> int:
        return int(self._win_alloc.available())

    def alloc_swa(self, full_indices: torch.Tensor) -> None:
        """Bind one window page per incoming FULL page. ``full_indices`` must be whole ascending
        pages (the ``_page_to_token`` expansion); the in-page offsets are preserved
        (``window_slot = wbase + pos % P``), which the state ring's page-block layout requires."""
        n = int(full_indices.numel())
        if n == 0:
            return
        P = self.P
        assert n % P == 0, f"alloc_swa needs whole pages, got {n} slots"
        fi = full_indices.to(device=self._device, dtype=torch.int64).view(-1, P)
        fbases = fi[:, 0]
        assert torch.equal(fi, fbases[:, None] + torch.arange(P, device=self._device)), "alloc_swa pages must be contiguous ascending"
        wbases = self._win_alloc.alloc(fbases.numel())  # raises when exhausted (caller gated)
        offsets = torch.arange(P, dtype=torch.int64, device=self._device)
        self.full_to_window[(fbases[:, None] + offsets).flatten()] = (wbases[:, None] + offsets).flatten()

    def free_swa(self, full_indices: torch.Tensor) -> None:
        """Return the window pages backing these FULL locs and unbind the mapping. Page-atomic: the
        incoming locs must cover each touched page completely (guaranteed by the padded finish
        tails / aligned frontiers / page-aligned tree values). Idempotent over already unbound
        (slid / tombstoned) pages."""
        if full_indices.numel() == 0:
            return
        P = self.P
        fi = full_indices.to(device=self._device, dtype=torch.int64)
        fi = fi[fi >= 0]
        if fi.numel() == 0:
            return
        fbases, counts = torch.unique(torch.div(fi, P, rounding_mode="floor") * P, return_counts=True)
        assert bool((counts == P).all()), f"free_swa got partial pages (counts {counts[counts != P].tolist()[:4]})"
        ws = self.full_to_window[fbases]
        live = ws[ws >= 0]
        offsets = torch.arange(P, dtype=torch.int64, device=self._device)
        self.full_to_window[(fbases[:, None] + offsets).flatten()] = -1
        if live.numel():
            self._win_alloc.free(torch.div(live, P, rounding_mode="floor") * P)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        return self.full_to_window[kv_indices.to(dtype=torch.int64)]

    # ----- rebuild / page table -----
    def rebuild(self, sizes) -> None:
        """In-place resize to ``sizes`` (identity-preserving; free-before-alloc). The manager's
        tree / page bookkeeping reset is the scheduler's generic cache_manager.rebuild; the engine
        re-attaches the page table via attach_page_table afterwards."""
        assert self._paged_params is not None, "rebuild before _init_paged_state"
        self.sizes = sizes
        self._drop_buffers()
        self.full_to_window = None  # type: ignore[assignment]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._alloc_buffers()
        self._init_paged_state(*self._paged_params)

    def attach_page_table(self, page_table: torch.Tensor) -> None:
        # the model reads full locs through full_loc_map; under the shared route that IS the page table
        self.full_loc_map = page_table

    # ----- BaseKVCachePool glue: the window tier is the K == V latent cache -----
    def k_cache(self, index: int) -> torch.Tensor:
        return self.window_pool[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self.window_pool[index]

    def store_kv(self, k, v, out_loc, layer_id) -> None:
        self.store_window(k, layer_id, out_loc)

    @abstractmethod
    def store_window(self, kv: torch.Tensor, layer_id: int, window_slot: torch.Tensor) -> None: ...

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype


__all__ = [
    "FreeListAllocator",
    "CompressStateRing",
    "WindowTierPagedPool",
    "reserved_window_pages",
    "window_state_loc",
]
