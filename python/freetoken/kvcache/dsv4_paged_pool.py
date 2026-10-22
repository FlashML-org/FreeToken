"""DSV4 paged KV pools (sglang-style management, FreeToken byte layout).

Four buffer families, sized from a budget (the cost model) not ``num_requests``:

* ``window_pool[L]``  -- all layers; the 128-sliding KV ring, page-granular.
* ``cmp_pool[L]``     -- ratio>0 layers; compressed KV, per-block.
* ``idx_pool[L]``     -- ratio==4 layers; Lightning-Indexer compressed keys.
* ``state_ring[L]``   -- ratio>0 layers; the per-window-page compress-state ring
  (fp32, ``kv|score`` split), with index ``-1`` a permanent scratch slot.

The KV/compressed/indexer pools are bf16 (the fp8/fp4 quant is an in-place round-trip already
baked into the bf16 value, so ``index_select`` staging is byte-exact); only the compress-state
ring is fp32.

``state_loc`` is DERIVED from a window slot, never stored:
    state_loc = where(ws < 0, -1, (ws // P) * ring_size + ws % ring_size)
``ring_size | P`` so distinct pages map to disjoint ring blocks.
"""

from __future__ import annotations

import torch

from freetoken.utils import init_logger

from .base import BaseKVCachePool
from .dsv4_cost_model import (
    DSV4PoolSizes,
    dsv4_kv_unit_bytes,
    dsv4_window_unit_bytes,
    ring_size_for_ratio,
)


logger = init_logger(__name__)


# LIFO free-list allocator for the window tier: one allocated unit spans ``page_unit = P``
# slots, so a unit base is always a multiple of ``P`` (the spec's ``G*P`` page-base
# invariant). The cmp/idx tiers have no allocator -- their rows are ``full_loc // ratio``
# pure arithmetic.
class FreeListAllocator:
    def __init__(
        self,
        capacity: int,
        device: torch.device,
        page_unit: int = 1,
    ) -> None:
        assert capacity % page_unit == 0, (
            f"capacity {capacity} must be a multiple of page_unit {page_unit}"
        )
        self._capacity = int(capacity)
        self._page_unit = int(page_unit)
        self._device = device
        # Unit base slots: [0, page_unit, 2*page_unit, ...]. LIFO -> pop the tail.
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

    Last row (index ``-1``) is a permanent scratch slot, re-cleared on every
    write. Last dim is split ``kv | score``; ``set_state`` writes both halves.
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
        # +1 trailing scratch row at index -1.
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
        """Batched per-row carry-block read. ``page_base`` is ``[B]`` (the ring block base
        row per row, == ``(window_slot // P) * ring_size``). Returns ``[B, ring_size, 2*item]``
        — each row's whole rolling carry block. Distinct pages -> disjoint blocks (isolation)."""
        rows = page_base[:, None] + torch.arange(self.ring_size, device=page_base.device)
        return self.buffer[rows]

    def set_blocks(self, page_base: torch.Tensor, blocks: torch.Tensor) -> None:
        """Batched per-row carry-block write: ``buffer[base+arange(ring_size)] = blocks[row]``.
        ``page_base`` ``[B]``, ``blocks`` ``[B, ring_size, 2*item]``. Re-clears scratch."""
        rows = page_base[:, None] + torch.arange(self.ring_size, device=page_base.device)
        self.buffer[rows] = blocks
        self._clear_scratch()


class DSV4PagedKVCache(BaseKVCachePool):
    def __init__(
        self,
        sizes: DSV4PoolSizes,
        args,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        P: int = 128,
        n_scratch: int = 1,
    ) -> None:
        assert dtype == torch.bfloat16, "KV pools are bf16 (fp4/fp8 is an in-place round-trip)"
        self.args = args
        self.sizes = sizes
        self._device = device
        self._dtype = dtype
        self.P = P
        self._n_layers = args.n_layers
        self.head_dim = args.head_dim
        self.index_head_dim = args.index_head_dim
        # The checkpoint can ship one extra trailing ratio (44 for 43 layers); the model only uses
        # the first n_layers, so truncate to match.
        self.compress_ratios = tuple(args.compress_ratios)[: self._n_layers]
        assert len(self.compress_ratios) == self._n_layers
        # Scratch rows appended to each cmp/idx pool tensor BEYOND the allocator's capacity
        # (never handed out): batched decode routes each row whose compressed block did NOT
        # complete this step to its own scratch row ``cmp_scratch_base + row`` (a discarded
        # write), so the masked per-row scatter is graph-safe (no host sync, no -1 index, no
        # cross-row collision). One per running request row.
        self.n_scratch = int(n_scratch)

        # Logical full-loc currency: ONE mapping from the virtual full-token index space to window
        # slots; cmp/idx rows derive arithmetically (full_loc // ratio), the state ring off the
        # window slot. Unmapped = -1. The trailing row is a PERMANENT -1 sentinel, so a fancy-index
        # gather at -1 lands on it and returns -1 (gather-safe); scatter paths must never see a
        # negative (index_copy_ raises OOB, it does not wrap).
        for ratio in set(self.compress_ratios):
            assert ratio == 0 or P % ratio == 0, f"P={P} must be divisible by ratio {ratio}"
        self._paged_params: tuple[int, bool] | None = None  # (_init_paged_state args, for rebuild)
        self._alloc_buffers()

    def _alloc_buffers(self) -> None:
        """(Re)allocate every physical buffer for the CURRENT ``self.sizes``. Shared by __init__
        and the in-place ``rebuild`` (identity-preserving, like HybridSWAKVCache.rebuild -- the
        CacheManager/engine/ctx all hold THIS object)."""
        sizes, device, dtype = self.sizes, self._device, self._dtype
        self.cmp_scratch_base = []
        self.idx_scratch_base = []
        self.full_to_window = torch.full(
            (sizes.full_token + 1,), -1, dtype=torch.int64, device=device
        )

        # The ONE slot map (table_idx, pos) -> full loc: the shared page_table, attached by the
        # engine policy. Window slots come from ``full_to_window``; cmp/idx rows are arithmetic.
        if not hasattr(self, "full_loc_map"):
            self.full_loc_map: torch.Tensor | None = None

        # Window KV: every layer.
        self.window_pool: list[torch.Tensor] = [
            torch.zeros(sizes.n_win_slots, self.head_dim, device=device, dtype=dtype)
            for _ in range(self._n_layers)
        ]

        # Compressed KV / Indexer KV / compress-state ring: per ratio-class.
        # ``state_ring`` is the ATTENTION compressor's ring (head_dim). Ratio-4
        # layers additionally own an ``indexer_state_ring`` (index_head_dim) for
        # the indexer's own compressor -- a separate pool, no collision.
        self.cmp_pool: list[torch.Tensor | None] = []
        self.idx_pool: list[torch.Tensor | None] = []
        self.state_ring: list[CompressStateRing | None] = []
        self.indexer_state_ring: list[CompressStateRing | None] = []
        for L in range(self._n_layers):
            ratio = self.compress_ratios[L]
            if ratio == 0:
                self.cmp_pool.append(None)
                self.idx_pool.append(None)
                self.state_ring.append(None)
                self.indexer_state_ring.append(None)
                self.cmp_scratch_base.append(None)
                self.idx_scratch_base.append(None)
                continue

            self.cmp_scratch_base.append(sizes.cmp_blocks[L])
            self.cmp_pool.append(
                torch.zeros(
                    sizes.cmp_blocks[L] + self.n_scratch, self.head_dim, device=device, dtype=dtype
                )
            )
            if ratio == 4:
                self.idx_scratch_base.append(sizes.idx_blocks[L])
                self.idx_pool.append(
                    torch.zeros(
                        sizes.idx_blocks[L] + self.n_scratch,
                        self.index_head_dim,
                        device=device,
                        dtype=dtype,
                    )
                )
                # Indexer compressor ring: index_head_dim, overlap, ring_size=8.
                self.indexer_state_ring.append(
                    CompressStateRing(
                        n_slots=sizes.idx_state_slots[L],
                        ring_size=ring_size_for_ratio(4),
                        overlap=True,
                        head_dim=self.index_head_dim,
                        device=device,
                    )
                )
            else:
                self.idx_pool.append(None)
                self.indexer_state_ring.append(None)
                self.idx_scratch_base.append(None)

            self.state_ring.append(
                CompressStateRing(
                    n_slots=sizes.state_slots[L],
                    ring_size=ring_size_for_ratio(ratio),
                    overlap=(ratio == 4),
                    head_dim=self.head_dim,
                    device=device,
                )
            )

    # ----- full-loc translation (gather-only -1 safety; see field comment) -----
    def translate_full_to_window(self, full_locs: torch.Tensor) -> torch.Tensor:
        # int64 gather indices: the shared page_table stores full locs as int32.
        return self.full_to_window[full_locs.to(dtype=torch.int64)]

    @staticmethod
    def cmp_rows(full_locs: torch.Tensor, ratio: int) -> torch.Tensor:
        return torch.div(full_locs.to(dtype=torch.int64), ratio, rounding_mode="floor")

    def bind_window_pages(self, full_page_base: int, window_page_base: int) -> None:
        assert full_page_base % self.P == 0 and window_page_base % self.P == 0
        self.full_to_window[full_page_base : full_page_base + self.P] = torch.arange(
            window_page_base, window_page_base + self.P, dtype=torch.int64, device=self._device
        )

    def unbind_window_pages(self, full_locs: torch.Tensor) -> None:
        self.full_to_window[full_locs[full_locs >= 0]] = -1

    # ----- generic swa_pool duck-type (the CacheManager plug-in surface) -----
    # ShadowRadix layering: the shared page_table is the virtual full-token coordinate; this pool
    # projects it into the physical tiers. The window tier is the managed "second currency" --
    # token-face signatures (what the generic CacheManager speaks), PAGE-ATOMIC internals (window
    # pages are 1:1 page-bound to full pages; the per-page state ring requires it). alloc_swa
    # receives whole ascending pages (allocate_paged's _page_to_token expansion) and free paths
    # are page-complete by construction (padded finish tails, align_down frontiers, page-aligned
    # tree nodes) -- asserted here, not assumed.
    swa_paged = True

    @property
    def sliding_window_size(self) -> int:
        return self.P

    @property
    def prefill_chunk_budget(self) -> int:
        return self._chunk_budget

    # ----- engine-facing rebuild surface -----
    # The tier buffers are bound into per-forward model scratch, invalid after a realloc.
    needs_rebind_on_rebuild = True

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .dsv4_cost_model import _dsv4_swa_ratio, _dsv4_window_floor_pages
        from .dsv4_cost_model import dsv4_auto_cost_model

        dsv4_args = config.model_config.dsv4_args
        P = dsv4_args.window_size
        floor = _dsv4_window_floor_pages(config, P)
        per_page, fixed, min_reserve_tokens = dsv4_auto_cost_model(
            dsv4_args, _dsv4_swa_ratio(config), floor, P=P, n_scratch=config.max_running_req + 1
        )
        return per_page, fixed, config.page_size, min_reserve_tokens

    @classmethod
    def solve_num_pages(cls, config, available_memory: int) -> int:
        # Solve the largest budget-respecting anchor with the exact per-tier byte model.
        # num_pages is in P (window) units and anchors full_token = num_pages*P (the FULL
        # cmp/idx tiers). The window working-set floor is honored in PAGES and the total is
        # byte-checked here. A budget too small for even the minimal working set raises a
        # graceful config error, not a late OOM.
        from freetoken.utils import mem_GB

        from .dsv4_cost_model import _dsv4_pool_sizes, _dsv4_swa_ratio, _dsv4_window_floor_pages
        from .dsv4_cost_model import dsv4_pool_bytes, dsv4_solve_num_pages

        dsv4_args = config.model_config.dsv4_args
        P = dsv4_args.window_size
        num_pages = config.num_page_override
        if num_pages is None:
            sizes = dsv4_solve_num_pages(
                available_memory, dsv4_args, _dsv4_swa_ratio(config),
                floor_win_pages=_dsv4_window_floor_pages(config, P), P=P,
                n_scratch=config.max_running_req + 1,
            )
            # The solver fits PHYSICAL pages to memory; one is the dummy page, so the
            # usable (advertised) count is one less.
            num_pages = sizes.full_token // P - 1
        else:
            # Fail at config time with guidance: a below-floor pool would otherwise boot
            # (dsv4_pool_sizes caps the window at num_pages) and die at runtime alloc.
            floor = _dsv4_window_floor_pages(config, P)
            if num_pages < floor:
                raise ValueError(
                    f"--num-pages {num_pages} ({num_pages * P} tokens) is below the DSV4 "
                    f"window working-set floor {floor} pages ({floor * P} tokens); raise "
                    f"--num-pages or lower max_running_req/max_seq_len"
                )
            sizes = _dsv4_pool_sizes(config, num_pages + 1)  # +1 for dummy page
        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        real = dsv4_pool_bytes(sizes, dsv4_args, config.max_running_req + 1)
        logger.info(
            f"Allocating {num_pages * P} tokens for DSV4 KV cache "
            f"({sizes.n_win_pages} window pages), total = {mem_GB(real)}"
        )
        return num_pages

    @classmethod
    def min_kv_tokens(cls, config) -> int:
        # The full anchor must cover the window working-set floor (full >= window always), so
        # that floor -- the value validate_rebuild enforces -- is the pool's floor in tokens.
        from .dsv4_cost_model import _dsv4_window_floor_pages

        P = config.model_config.dsv4_args.window_size
        return _dsv4_window_floor_pages(config, P) * P

    def validate_rebuild(
        self, config, *, num_pages: int | None, target_moe: int, per_expert_bytes: int,
        baseline_free: int, weights_bytes: int, current_num_pages: int,
        extra_fixed_bytes: int = 0, extra_note: str = "",
        num_swa_pages: int | None = None, **targets,
    ) -> None:
        from freetoken.engine.cache_budget import net_cache_budget_bytes
        from freetoken.utils import mem_GB

        from .base import CacheRebuildRejected
        from .dsv4_cost_model import _dsv4_pool_sizes, _dsv4_window_floor_pages
        from .dsv4_cost_model import dsv4_pool_bytes

        dsv4_args = config.model_config.dsv4_args
        if num_pages is not None:
            floor = _dsv4_window_floor_pages(config, dsv4_args.window_size)
            if num_pages < floor:
                raise CacheRebuildRejected(
                    f"num_pages {num_pages} is below the DSV4 window working-set floor {floor} "
                    f"(max_running_req={config.max_running_req}); admission would deadlock"
                )
        if num_pages is not None or num_swa_pages is not None:
            # Size the pool a KV/window rebuild would build: the target anchor (or current) with
            # the target window (or current), computed BEFORE the config is mutated.
            target_pages = num_pages if num_pages is not None else current_num_pages
            kv_sizes = _dsv4_pool_sizes(
                config, target_pages + 1, num_swa_pages=num_swa_pages
            )  # +1 for dummy page
        else:
            # MoE-only rebuild keeps the CURRENT pool: budget-check against its live sizes
            # (reflects DSV4_FORCE_SMALL_POOL and the physical dummy page).
            kv_sizes = self.sizes
        # The rebuilds are free-before-alloc, so the whole budget is available (no fixed
        # cache term); an unfit request must still reject BEFORE the teardown.
        budget = net_cache_budget_bytes(config.memory_ratio, baseline_free, weights_bytes, 0)
        need = target_moe * per_expert_bytes + dsv4_pool_bytes(
            kv_sizes, dsv4_args, config.max_running_req + 1
        )
        if need > budget:
            kv_part = f"kv={num_pages} P-pages" if num_pages is not None else "kv=current pool"
            raise CacheRebuildRejected(
                f"requested cache (moe={target_moe} slots, {kv_part}) needs "
                f"{mem_GB(need)} > budget {mem_GB(budget)}; old cache kept, still serving"
            )

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        from .dsv4_cost_model import _dsv4_pool_sizes

        # +1 for the dummy page
        self.rebuild(_dsv4_pool_sizes(config, num_pages + 1, num_swa_pages=num_swa_pages))

    def attach_page_table(self, page_table: torch.Tensor) -> None:
        # The model reads full locs through full_loc_map; under the shared route that IS the
        # page table. The attention backend re-allocates its decode snapshot on re-capture.
        self.full_loc_map = page_table

    def unit_bytes(self) -> tuple[int, int]:
        # No measurable flat buffer (owned paged pool): the full (cmp/idx + mapping) and window
        # (sliding KV + state rings) per-token costs come from the per-tier cost model.
        return dsv4_kv_unit_bytes(self.args, self.P), dsv4_window_unit_bytes(self.args, self.P)

    def rebuild(self, sizes) -> None:
        """In-place resize to ``sizes`` (identity-preserving; free-before-alloc). The manager's
        tree/page bookkeeping reset is the scheduler's generic cache_manager.rebuild; the engine
        re-attaches the page table via attach_page_table afterwards."""
        import gc

        assert self._paged_params is not None, "rebuild before _init_paged_state"
        self.sizes = sizes
        self.window_pool = self.cmp_pool = self.idx_pool = None
        self.state_ring = self.indexer_state_ring = None
        self.full_to_window = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._alloc_buffers()
        self._init_paged_state(*self._paged_params)

    def _init_paged_state(self, max_running_req: int, radix: bool) -> None:
        """Build the pool-owned window free-list + the tail dummy binding. The LAST full page and
        LAST window page are the reserved dummy region: page_table's dummy row points at
        ``full_token - P`` (== the generic ``fill_(num_tokens)`` convention with num_tokens = the
        allocatable token count), permanently bound so graph-padded rows scatter to a real slot."""
        from .dsv4_cost_model import dsv4_reserved_window_pages
        P = self.P
        self._paged_params = (int(max_running_req), bool(radix))
        self.full_to_window.fill_(-1)
        self._win_alloc = FreeListAllocator(self.sizes.n_win_slots - P, self._device, page_unit=P)
        self.bind_window_pages(self.sizes.full_token - P, self.sizes.n_win_slots - P)
        # Chunk cap: a batched prefill holds the whole chunk's window live at once (sliding frees
        # only between chunks; peak ~2x the chunk), so reserve the concurrent working set and
        # halve the rest -- the same formula the bespoke manager used.
        n_win_pages = (self.sizes.n_win_slots // P) - 1
        reserved = dsv4_reserved_window_pages(max_running_req, radix)
        self._chunk_budget = max(P, (n_win_pages - reserved) // 2 * P)

    @property
    def swa_num_tokens(self) -> int:
        # Allocatable window slots + 1: the generic capacity convention reserves slot 0 as a
        # sentinel (cap == swa_num_tokens - 1); DSV4's reserved unit is the tail dummy page,
        # already excluded from the free-list, so +1 re-encodes the same cap.
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
        assert torch.equal(fi, fbases[:, None] + torch.arange(P, device=self._device)), (
            "alloc_swa pages must be contiguous ascending"
        )
        wbases = self._win_alloc.alloc(fbases.numel())  # raises when exhausted (caller gated)
        offsets = torch.arange(P, dtype=torch.int64, device=self._device)
        self.full_to_window[(fbases[:, None] + offsets).flatten()] = (
            wbases[:, None] + offsets
        ).flatten()

    def free_swa(self, full_indices: torch.Tensor) -> None:
        """Return the window pages backing these FULL locs and unbind the mapping. Page-atomic:
        the incoming locs must cover each touched page completely (guaranteed by the padded
        finish tails / aligned frontiers / page-aligned tree values). Idempotent over already
        unbound (slid/tombstoned) pages."""
        if full_indices.numel() == 0:
            return
        P = self.P
        fi = full_indices.to(device=self._device, dtype=torch.int64)
        fi = fi[fi >= 0]
        if fi.numel() == 0:
            return
        fbases, counts = torch.unique(
            torch.div(fi, P, rounding_mode="floor") * P, return_counts=True
        )
        assert bool((counts == P).all()), (
            f"free_swa got partial pages (counts {counts[counts != P].tolist()[:4]})"
        )
        ws = self.full_to_window[fbases]
        live = ws[ws >= 0]
        offsets = torch.arange(P, dtype=torch.int64, device=self._device)
        self.full_to_window[(fbases[:, None] + offsets).flatten()] = -1
        if live.numel():
            self._win_alloc.free(torch.div(live, P, rounding_mode="floor") * P)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        return self.full_to_window[kv_indices.to(dtype=torch.int64)]

    # ----- state_loc derivation (vectorized, LongTensor in/out) -----
    @staticmethod
    def state_loc(window_slot: torch.Tensor, ring_size: int, P: int) -> torch.Tensor:
        pages = torch.div(window_slot, P, rounding_mode="floor")
        loc = pages * ring_size + (window_slot % ring_size)
        return torch.where(window_slot < 0, torch.full_like(loc, -1), loc)

    def ring_size(self, layer_id: int) -> int:
        return ring_size_for_ratio(self.compress_ratios[layer_id])

    # ----- compress-state ring accessors -----
    def get_state(self, layer_id: int, state_loc: torch.Tensor) -> torch.Tensor:
        ring = self.state_ring[layer_id]
        assert ring is not None, f"layer {layer_id} (ratio 0) has no compress-state ring"
        return ring.get(state_loc)

    def set_state(self, layer_id: int, state_loc: torch.Tensor, kv_score: torch.Tensor) -> None:
        ring = self.state_ring[layer_id]
        assert ring is not None, f"layer {layer_id} (ratio 0) has no compress-state ring"
        ring.set(state_loc, kv_score)

    # ----- specialized writes -----
    def store_window(self, k: torch.Tensor, layer_id: int, window_slot: torch.Tensor) -> None:
        self.window_pool[layer_id].index_copy_(0, window_slot, k.to(self._dtype))

    def store_compressed(self, kv: torch.Tensor, layer_id: int, cmp_slot: torch.Tensor) -> None:
        pool = self.cmp_pool[layer_id]
        assert pool is not None, f"layer {layer_id} (ratio 0) has no compressed pool"
        pool.index_copy_(0, cmp_slot, kv.to(self._dtype))

    def store_indexer(self, k: torch.Tensor, layer_id: int, idx_slot: torch.Tensor) -> None:
        pool = self.idx_pool[layer_id]
        assert pool is not None, f"layer {layer_id} has no indexer pool (only ratio-4)"
        pool.index_copy_(0, idx_slot, k.to(self._dtype))

    def total_bytes(self) -> int:
        n = self.full_to_window.numel() * self.full_to_window.element_size()
        n += sum(t.numel() * t.element_size() for t in self.window_pool)
        n += sum(t.numel() * t.element_size() for t in self.cmp_pool if t is not None)
        n += sum(t.numel() * t.element_size() for t in self.idx_pool if t is not None)
        n += sum(
            r.buffer.numel() * r.buffer.element_size()
            for r in self.state_ring
            if r is not None
        )
        n += sum(
            r.buffer.numel() * r.buffer.element_size()
            for r in self.indexer_state_ring
            if r is not None
        )
        return int(n)

    # ----- BaseKVCachePool interface -----
    def k_cache(self, index: int) -> torch.Tensor:
        return self.window_pool[index]

    def v_cache(self, index: int) -> torch.Tensor:  # MLA: K == V (single latent)
        return self.window_pool[index]

    def store_kv(self, k, v, out_loc, layer_id) -> None:
        # Thin window-write shim for ABC compat; DSV4 writes via the specialized
        # setters above. K == V (single latent), so the window slot is out_loc.
        self.store_window(k, layer_id, out_loc)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def num_layers(self) -> int:
        return self._n_layers


__all__ = ["CompressStateRing", "DSV4PagedKVCache"]
