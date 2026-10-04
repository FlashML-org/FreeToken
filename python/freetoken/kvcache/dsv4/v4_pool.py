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

from .v4_cost_model import (
    DSV4PoolSizes,
    dsv4_kv_unit_bytes,
    dsv4_window_unit_bytes,
    ring_size_for_ratio,
)
# The window-tier building blocks and the pool base live in window_tier.py (shared with the DSV41
# pool); the first two are re-exported here for the DSV4 callers and tests that import them from
# this module.
from .window_tier import CompressStateRing, FreeListAllocator, WindowTierPagedPool  # noqa: F401


logger = init_logger(__name__)


class DSV4PagedKVCache(WindowTierPagedPool):
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

    # ----- engine-facing rebuild surface -----
    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .v4_cost_model import _dsv4_swa_ratio, _dsv4_window_floor_pages
        from .v4_cost_model import dsv4_auto_cost_model

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

        from .v4_cost_model import _dsv4_pool_sizes, _dsv4_swa_ratio, _dsv4_window_floor_pages
        from .v4_cost_model import dsv4_pool_bytes, dsv4_solve_num_pages

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
    def window_spec(cls, config):
        from ..base import WindowPoolSpec
        from .v4_cost_model import _dsv4_window_floor_pages

        P = config.model_config.dsv4_args.window_size
        return WindowPoolSpec(P, _dsv4_window_floor_pages(config, P) - 1)

    @classmethod
    def min_kv_tokens(cls, config) -> int:
        # The full anchor must cover the window working-set floor (full >= window always), so
        # that floor -- the value validate_rebuild enforces -- is the pool's floor in tokens.
        from .v4_cost_model import _dsv4_window_floor_pages

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

        from ..base import CacheRebuildRejected
        from .v4_cost_model import _dsv4_pool_sizes, _dsv4_window_floor_pages
        from .v4_cost_model import dsv4_pool_bytes

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
        from .v4_cost_model import _dsv4_pool_sizes

        # +1 for the dummy page
        self.rebuild(_dsv4_pool_sizes(config, num_pages + 1, num_swa_pages=num_swa_pages))

    def unit_bytes(self) -> tuple[int, int]:
        # No measurable flat buffer (owned paged pool): the full (cmp/idx + mapping) and window
        # (sliding KV + state rings) per-token costs come from the per-tier cost model.
        return dsv4_kv_unit_bytes(self.args, self.P), dsv4_window_unit_bytes(self.args, self.P)

    def _drop_buffers(self) -> None:
        self.window_pool = self.cmp_pool = self.idx_pool = None
        self.state_ring = self.indexer_state_ring = None

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

    @property
    def num_layers(self) -> int:
        return self._n_layers


__all__ = ["CompressStateRing", "DSV4PagedKVCache"]
