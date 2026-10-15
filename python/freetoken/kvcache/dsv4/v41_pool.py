"""DSV41 paged KV pool (DeepSeek-V4.1): packed byte tiers over the shared page table.

Tiers, sized from a budget (``v41_cost_model``) not from ``num_requests``:

* ``window_pool[L]``      -- every layer; the P-sliding KV ring in ``win_fmt`` (fp8) rows, page-granular.
* ``main_pool[src]``      -- per kv-source layer; compressed KV latents in ``main_fmt`` (fp4) rows.
* ``idx_pool[src]``       -- per kv-source layer; indexer keys in ``idx_fmt`` (fp4) rows.
* ``state_ring[src]``     -- ratio>1 sources; the per-window-page compress-state ring (fp32 ``kv|score``).

Consumer layers (Reindex / Reuse modes) alias their source's pools: ``main_pool_of(layer)``.

Addressing is the shared ``WindowTierPagedPool``'s (``kvcache/dsv4/window_tier.py``, also under the DSV4
pool): the page table maps ``(table_idx, pos)`` to a full-token loc; ``full_to_window`` maps that to a
window slot (page-bound by the free list); the main / index rows are pure arithmetic ``full_loc //
ratio``; the ring slot derives from the window slot. Scratch rows past every main / index pool's
arithmetic range give a graph-safe destination for the decode rows whose compressed group did not
complete this step.

Rows are written through ``pack_rows`` (quantize + scatter) and read inside the attention /
indexer kernels; ``read_*`` helpers dequantize for tests and torch reference paths.
"""

from __future__ import annotations

import torch

from freetoken.utils import init_logger

from .v41_cost_model import (
    IDX_FMT,
    MAIN_FMT,
    WIN_FMT,
    DSV41PoolSizes,
    _idx_row_bytes,
    _main_row_bytes,
    _win_row_bytes,
    dsv41_kv_unit_bytes,
    dsv41_window_unit_bytes,
    private_window_layer_ids,
)
from .window_tier import CompressStateRing, WindowTierPagedPool

logger = init_logger(__name__)


class DSV41PagedKVCache(WindowTierPagedPool):
    win_fmt = WIN_FMT
    main_fmt = MAIN_FMT
    idx_fmt = IDX_FMT

    def __init__(
        self,
        sizes: DSV41PoolSizes,
        args,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        n_scratch: int = 1,
    ) -> None:
        for fmt, dim in ((self.win_fmt, args.head_dim), (self.main_fmt, args.head_dim), (self.idx_fmt, args.index_head_dim)):
            fmt.validate_dim(dim)
        for src in args.backbone_kv_sources:
            if args.window_size % args.compress_ratios[src]:
                raise ValueError(f"window {args.window_size} must be a multiple of compress ratio {args.compress_ratios[src]}")
        self.args = args
        self.private_window_layer_ids = private_window_layer_ids(args)
        self.sizes = sizes
        self._device = device
        self._dtype = dtype  # the model's compute dtype; the tiers themselves are packed bytes
        self.P = args.window_size
        self.head_dim = args.head_dim
        self.index_head_dim = args.index_head_dim
        self.n_scratch = int(n_scratch)
        self._paged_params: tuple[int, bool] | None = None
        self.full_loc_map: torch.Tensor | None = None
        self._alloc_buffers()

    # ----- buffers -----
    def _alloc_buffers(self) -> None:
        sizes, args, dev = self.sizes, self.args, self._device
        self.full_to_window = torch.full((sizes.full_token + 1,), -1, dtype=torch.int64, device=dev)
        # shared layers: the page-bound window pool; private layers: one ring of P slots per
        # page-table row (n_scratch rows, the dummy included), addressed row * P + pos % P
        self.window_pool: list[torch.Tensor] = [
            torch.zeros(self.n_scratch * self.P if self.is_private_window(l) else sizes.n_win_slots, _win_row_bytes(args), dtype=torch.uint8, device=dev)
            for l in range(args.n_layers)
        ]
        self.main_pool: dict[int, torch.Tensor] = {}
        self.idx_pool: dict[int, torch.Tensor] = {}
        self.state_ring: dict[int, CompressStateRing] = {}
        self.scratch_base: dict[int, int] = {}
        for src in args.backbone_kv_sources:
            rows = sizes.main_rows[src]
            self.scratch_base[src] = rows
            self.main_pool[src] = torch.zeros(rows + self.n_scratch, _main_row_bytes(args), dtype=torch.uint8, device=dev)
            self.idx_pool[src] = torch.zeros(sizes.idx_rows[src] + self.n_scratch, _idx_row_bytes(args), dtype=torch.uint8, device=dev)
            if src in sizes.state_slots:
                self.state_ring[src] = CompressStateRing(
                    n_slots=sizes.state_slots[src], ring_size=args.compress_ratios[src], overlap=False,
                    head_dim=args.head_dim, device=dev,
                )

    def total_bytes(self) -> int:
        n = self.full_to_window.numel() * self.full_to_window.element_size()
        n += sum(t.numel() for t in self.window_pool)
        n += sum(t.numel() for t in self.main_pool.values())
        n += sum(t.numel() for t in self.idx_pool.values())
        n += sum(r.buffer.numel() * r.buffer.element_size() for r in self.state_ring.values())
        return int(n)

    @property
    def prefix_replay_tokens(self) -> int:
        """Prompt-tail tokens a prefix hit must leave to the prefill: bounded replay runs the decoder on
        the prompt's last window from that window's encoder outputs, which no cache keeps."""
        return self.P if self.private_window_layer_ids else 0

    # ----- private window rings -----
    def is_private_window(self, layer_id: int) -> bool:
        return layer_id in self.private_window_layer_ids

    def ring_slots(self, table_rows: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Ring slots of ``positions`` (int64, broadcastable with ``table_rows``) for private window layers;
        negative positions map to ``-1``."""
        slots = table_rows.to(torch.int64) * self.P + positions % self.P
        return torch.where(positions < 0, -1, slots)

    # ----- source aliasing -----
    def source_of(self, layer_id: int) -> int:
        src = self.args.roles[layer_id].kv_source
        assert src is not None, f"layer {layer_id} is window-only"
        return src

    def main_pool_of(self, layer_id: int) -> torch.Tensor:
        return self.main_pool[self.source_of(layer_id)]

    def idx_pool_of(self, layer_id: int) -> torch.Tensor:
        return self.idx_pool[self.source_of(layer_id)]

    def ratio_of(self, layer_id: int) -> int:
        return self.args.roles[layer_id].ratio

    # ----- engine-facing sizing / rebuild surface -----
    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .v41_cost_model import _dsv41_swa_ratio, _dsv41_window_floor_pages, dsv41_auto_cost_model

        args = config.model_config.dsv41_args
        P = args.window_size
        per_page, fixed, min_reserve = dsv41_auto_cost_model(
            args, _dsv41_swa_ratio(config), _dsv41_window_floor_pages(config, args), P, n_scratch=config.max_running_req + 1
        )
        return per_page, fixed, config.page_size, min_reserve

    @classmethod
    def solve_num_pages(cls, config, available_memory: int) -> int:
        from freetoken.utils import mem_GB

        from .v41_cost_model import (
            _dsv41_pool_sizes,
            _dsv41_swa_ratio,
            _dsv41_window_floor_pages,
            dsv41_pool_bytes,
            dsv41_solve_num_pages,
        )

        args = config.model_config.dsv41_args
        P = args.window_size
        num_pages = config.num_page_override
        if num_pages is None:
            sizes = dsv41_solve_num_pages(
                available_memory, args, _dsv41_swa_ratio(config), _dsv41_window_floor_pages(config, args), P,
                n_scratch=config.max_running_req + 1,
            )
            num_pages = sizes.full_token // P - 1  # one physical page is the dummy
        else:
            floor = _dsv41_window_floor_pages(config, args)
            if num_pages < floor:
                raise ValueError(
                    f"--num-pages {num_pages} ({num_pages * P} tokens) is below the DSV41 window working-set "
                    f"floor {floor} pages ({floor * P} tokens); raise --num-pages or lower max_running_req/max_seq_len"
                )
            sizes = _dsv41_pool_sizes(config, num_pages + 1)
        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        real = dsv41_pool_bytes(sizes, args, config.max_running_req + 1)
        logger.info(
            f"Allocating {num_pages * P} tokens for DSV41 KV cache ({sizes.n_win_pages} window pages), total = {mem_GB(real)}"
        )
        return num_pages

    @classmethod
    def window_spec(cls, config):
        from ..base import WindowPoolSpec
        from .v41_cost_model import _dsv41_window_floor_pages

        args = config.model_config.dsv41_args
        return WindowPoolSpec(args.window_size, _dsv41_window_floor_pages(config, args) - 1)

    @classmethod
    def min_kv_tokens(cls, config) -> int:
        from .v41_cost_model import _dsv41_window_floor_pages

        args = config.model_config.dsv41_args
        return _dsv41_window_floor_pages(config, args) * args.window_size

    def validate_rebuild(
        self, config, *, num_pages: int | None, target_moe: int, per_expert_bytes: int,
        baseline_free: int, weights_bytes: int, current_num_pages: int,
        extra_fixed_bytes: int = 0, extra_note: str = "",
        num_swa_pages: int | None = None, **targets,
    ) -> None:
        from freetoken.engine.cache_budget import net_cache_budget_bytes
        from freetoken.utils import mem_GB

        from ..base import CacheRebuildRejected
        from .v41_cost_model import _dsv41_pool_sizes, _dsv41_window_floor_pages, dsv41_pool_bytes

        if num_pages is not None:
            floor = _dsv41_window_floor_pages(config, self.args)
            if num_pages < floor:
                raise CacheRebuildRejected(
                    f"num_pages {num_pages} is below the DSV41 window working-set floor {floor} "
                    f"(max_running_req={config.max_running_req}); admission would deadlock"
                )
        if num_pages is not None or num_swa_pages is not None:
            target_pages = num_pages if num_pages is not None else current_num_pages
            kv_sizes = _dsv41_pool_sizes(config, target_pages + 1, num_swa_pages=num_swa_pages)
        else:
            kv_sizes = self.sizes
        budget = net_cache_budget_bytes(config.memory_ratio, baseline_free, weights_bytes, 0)
        need = target_moe * per_expert_bytes + dsv41_pool_bytes(kv_sizes, self.args, config.max_running_req + 1)
        if need > budget:
            kv_part = f"kv={num_pages} P-pages" if num_pages is not None else "kv=current pool"
            raise CacheRebuildRejected(
                f"requested cache (moe={target_moe} slots, {kv_part}) needs {mem_GB(need)} > budget {mem_GB(budget)}; "
                "old cache kept, still serving"
            )

    def rebuild_from_config(self, config, num_pages: int, *, num_swa_pages: int | None = None) -> None:
        from .v41_cost_model import _dsv41_pool_sizes

        self.rebuild(_dsv41_pool_sizes(config, num_pages + 1, num_swa_pages=num_swa_pages))

    def _drop_buffers(self) -> None:
        self.window_pool = self.main_pool = self.idx_pool = self.state_ring = None  # type: ignore[assignment]

    def unit_bytes(self) -> tuple[int, int]:
        return dsv41_kv_unit_bytes(self.args, self.P), dsv41_window_unit_bytes(self.args, self.P)

    # ----- writes (quantize + scatter) -----
    def store_window(self, kv: torch.Tensor, layer_id: int, window_slot: torch.Tensor) -> None:
        from freetoken.kernel.triton.dsv41.pack import pack_rows

        pack_rows(kv, self.win_fmt, pool=self.window_pool[layer_id], row_ids=window_slot)

    def store_main(self, latent: torch.Tensor, source: int, rows: torch.Tensor) -> None:
        from freetoken.kernel.triton.dsv41.pack import pack_rows

        pack_rows(latent, self.main_fmt, pool=self.main_pool[source], row_ids=rows)

    def store_index(self, k: torch.Tensor, source: int, rows: torch.Tensor) -> None:
        from freetoken.kernel.triton.dsv41.pack import pack_rows

        pack_rows(k, self.idx_fmt, pool=self.idx_pool[source], row_ids=rows)

    # ----- reads (gather + dequantize; tests and torch reference paths) -----
    def read_window(self, layer_id: int, window_slot: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.dsv41.pack import unpack_rows

        return unpack_rows(self.window_pool[layer_id], self.win_fmt, self.head_dim, window_slot)

    def read_main(self, source: int, rows: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.dsv41.pack import unpack_rows

        return unpack_rows(self.main_pool[source], self.main_fmt, self.head_dim, rows)

    def read_index(self, source: int, rows: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.dsv41.pack import unpack_rows

        return unpack_rows(self.idx_pool[source], self.idx_fmt, self.index_head_dim, rows)

    # ----- compress-state ring accessors -----
    def get_state(self, source: int, state_loc: torch.Tensor) -> torch.Tensor:
        return self.state_ring[source].get(state_loc)

    def set_state(self, source: int, state_loc: torch.Tensor, kv_score: torch.Tensor) -> None:
        self.state_ring[source].set(state_loc, kv_score)

    @property
    def num_layers(self) -> int:
        return self.args.n_layers


__all__ = ["DSV41PagedKVCache"]
