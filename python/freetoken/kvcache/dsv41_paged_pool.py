"""V4.1 source-owned compressed/index keys on the shared page-table allocator."""

from __future__ import annotations

import torch

from .dsv4_paged_pool import CompressStateRing, DSV4PagedKVCache
from .dsv4_cost_model import _dsv4_window_floor_pages
from .dsv41_cost_model import (
    _dsv41_pool_sizes, dsv41_auto_cost_model, dsv41_pool_bytes,
    dsv41_solve_num_pages, dsv41_unit_bytes, source_layers,
)
from .dsv41_layout import dsv41_row_bytes


class DSV41PagedKVCache(DSV4PagedKVCache):
    def __init__(self, sizes, args, device, dtype=torch.bfloat16, P=128, n_scratch=1,
                 *, kv_quant="none"):
        dsv41_row_bytes(args, kv_quant)
        if dtype != torch.bfloat16:
            raise ValueError("DeepSeek-V4.1 KV pools require logical dtype bfloat16")
        self._kv_quant = kv_quant
        super().__init__(sizes, args, device, dtype, P, n_scratch)

    @property
    def kv_quant(self):
        return self._kv_quant

    def _alloc_buffers(self):
        sizes, device, dtype = self.sizes, self._device, self._dtype
        row_bytes = dsv41_row_bytes(self.args, self.kv_quant)
        if self.kv_quant == "fp8-fp4":
            dtype = torch.uint8
            window_width, cmp_width, idx_width = row_bytes
        else:
            window_width, cmp_width, idx_width = (width // 2 for width in row_bytes)
        self.kv_sources, self.index_sources = source_layers(self.args)
        self.full_to_window = torch.full((sizes.full_token + 1,), -1, dtype=torch.int64, device=device)
        if not hasattr(self, "full_loc_map"):
            self.full_loc_map = None
        self.window_pool = [torch.zeros(sizes.n_win_slots, window_width, device=device, dtype=dtype)
                            for _ in range(self._n_layers)]
        self.cmp_pool = [None] * self._n_layers
        self.idx_pool = [None] * self._n_layers
        self.state_ring = [None] * self._n_layers
        self.indexer_state_ring = [None] * self._n_layers
        self.cmp_scratch_base = [None] * self._n_layers
        self.idx_scratch_base = [None] * self._n_layers
        for layer in self.args.kv_source_layers:
            count = sizes.cmp_blocks[layer]
            self.cmp_scratch_base[layer] = count
            self.idx_scratch_base[layer] = count
            self.cmp_pool[layer] = torch.zeros(count + self.n_scratch, cmp_width, device=device, dtype=dtype)
            self.idx_pool[layer] = torch.zeros(count + self.n_scratch, idx_width, device=device, dtype=dtype)
            if self.compress_ratios[layer] == 2:
                self.state_ring[layer] = CompressStateRing(sizes.state_slots[layer], 2, False,
                                                          self.head_dim, device)

    def _store_rows(self, pool, rows, values):
        if pool is None:
            raise ValueError("Only a DeepSeek-V4.1 source layer owns compressed KV rows")
        if (values.ndim != 2 or rows.ndim != 1 or values.shape[0] != rows.numel()
                or values.shape[1] != pool.shape[1]):
            raise ValueError("DeepSeek-V4.1 KV row shape does not match its storage layout")
        if self.kv_quant == "fp8-fp4":
            if values.dtype != torch.uint8:
                raise ValueError("DeepSeek-V4.1 fp8-fp4 KV writes require packed uint8 rows")
        else:
            if not values.is_floating_point():
                raise ValueError("DeepSeek-V4.1 unquantized KV writes require floating-point rows")
            values = values.to(self._dtype)
        pool.index_copy_(0, rows, values)

    def store_window(self, k, layer_id, window_slot):
        self._store_rows(self.window_pool[layer_id], window_slot, k)

    def store_compressed(self, kv, layer_id, cmp_slot):
        self._store_rows(self.cmp_pool[layer_id], cmp_slot, kv)

    def store_indexer(self, k, layer_id, idx_slot):
        self._store_rows(self.idx_pool[layer_id], idx_slot, k)

    def ring_size(self, layer_id):
        ring = self.state_ring[layer_id]
        if ring is None:
            raise ValueError(f"Layer {layer_id} has no pending compression state")
        return ring.ring_size

    @classmethod
    def kv_cost(cls, config):
        return dsv41_auto_cost_model(config)

    @classmethod
    def solve_num_pages(cls, config, available_memory):
        dsv41_row_bytes(config.model_config.dsv41_args, getattr(config, "kv_quant", "none"))
        if config.num_page_override is None:
            return dsv41_solve_num_pages(config, available_memory)
        pages = config.num_page_override
        if pages < cls.min_kv_tokens(config) // config.page_size:
            raise ValueError("--num-pages is below the DeepSeek-V4.1 window working-set floor")
        return pages

    @classmethod
    def min_kv_tokens(cls, config):
        P = config.model_config.dsv41_args.window_size
        return _dsv4_window_floor_pages(config, P) * P

    def unit_bytes(self):
        return dsv41_unit_bytes(self.args, self.P, self.kv_quant)

    def _validate_rebuild_format(self, config):
        from .base import CacheRebuildRejected

        if getattr(config, "kv_quant", "none") != self.kv_quant:
            raise CacheRebuildRejected("Changing DeepSeek-V4.1 KV storage format requires a model restart")

    def rebuild_from_config(self, config, num_pages, *, num_swa_pages=None):
        self._validate_rebuild_format(config)
        self.rebuild(_dsv41_pool_sizes(config, num_pages + 1, num_swa_pages))

    def validate_rebuild(self, config, *, num_pages, target_moe, per_expert_bytes,
                         baseline_free, weights_bytes, current_num_pages,
                         extra_fixed_bytes=0, extra_note="", num_swa_pages=None, **targets):
        from freetoken.engine.cache_budget import net_cache_budget_bytes
        from .base import CacheRebuildRejected

        self._validate_rebuild_format(config)
        if num_pages is not None and num_pages * self.P < self.min_kv_tokens(config):
            raise CacheRebuildRejected("DeepSeek-V4.1 KV pool is below its window working-set floor")
        sizes = self.sizes
        if num_pages is not None or num_swa_pages is not None:
            sizes = _dsv41_pool_sizes(config, (num_pages if num_pages is not None else current_num_pages) + 1,
                                     num_swa_pages)
        budget = net_cache_budget_bytes(config.memory_ratio, baseline_free, weights_bytes, extra_fixed_bytes)
        need = target_moe * per_expert_bytes + dsv41_pool_bytes(
            sizes, self.args, config.max_running_req + 1, self.kv_quant)
        if need > budget:
            raise CacheRebuildRejected(f"Requested V4.1 cache needs {need} bytes, exceeding budget {budget}")
