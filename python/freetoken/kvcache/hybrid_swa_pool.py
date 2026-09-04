from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from freetoken.distributed import get_tp_info
from freetoken.models.config import KVCacheGroupSpec
from freetoken.utils import align_ceil, div_even

from .base import BaseKVCachePool
from .quant import NONE, KVQuantSpec
from .quant_storage import QuantizedKVStorageMixin


@dataclass(frozen=True)
class _LayerRef:
    group: str
    index: int


@dataclass(frozen=True)
class _KVGroupStorage:
    buffer: torch.Tensor
    k_buffer: torch.Tensor
    v_buffer: torch.Tensor
    storage_shape: tuple[int, int, int]
    # Per-block scales for an 8-bit group; None when the group stores the compute dtype.
    scale_buffer: torch.Tensor | None = None
    k_scale: torch.Tensor | None = None
    v_scale: torch.Tensor | None = None


class HybridSWAKVCache(QuantizedKVStorageMixin, BaseKVCachePool):
    """SGLang-style wrapper for hybrid full/SWA attention KV storage."""

    def __init__(
        self,
        groups: Sequence[KVCacheGroupSpec],
        num_layers: int,
        num_full_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        num_swa_tokens: int | None = None,
        quant: KVQuantSpec = NONE,
    ) -> None:
        self._quant = quant
        specs = {group.name: group for group in groups if group.num_layers > 0}
        if set(specs) != {"full", "swa"}:
            raise ValueError(f"HybridSWAKVCache requires full and swa groups, got {sorted(specs)}")

        self._num_layers = num_layers
        self._device = device
        self._dtype = dtype
        self._full_num_tokens = num_full_pages * page_size
        self._swa_num_tokens = num_swa_tokens if num_swa_tokens is not None else self._full_num_tokens
        self._page_size = page_size
        # Global-paged SWA (== sglang SWAKVPool): a swa pool reached through a dense full->swa
        # slot mapping + an independent swa free-list. Used by BOTH SWA cache paths -- naive
        # (NaivePrefixCache, no reuse) and radix (SWARadixCache, cross-request reuse) -- which
        # differ only in the cache object. Always paged; the mapping/free-list are built below
        # (translate/store_kv unconditionally index full_to_swa_index_mapping).
        self._swa_paged = True

        tp_size = get_tp_info().size
        self.full_kv_pool = self._allocate_group(
            specs["full"],
            tp_size=tp_size,
            outer_size=num_full_pages,
            inner_size=page_size,
            dtype=dtype,
            device=device,
        )
        self.swa_kv_pool = self._allocate_group(
            specs["swa"],
            tp_size=tp_size,
            outer_size=self._swa_num_tokens,
            inner_size=1,
            dtype=dtype,
            device=device,
        )
        self._storages = {
            "full": self.full_kv_pool,
            "swa": self.swa_kv_pool,
        }
        self.layers_mapping = self._build_layers_mapping(num_layers, specs)
        if self._swa_paged:
            self._init_swa_paged_state()

    def _allocate_group(
        self,
        spec: KVCacheGroupSpec,
        tp_size: int,
        outer_size: int,
        inner_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> _KVGroupStorage:
        local_kv_heads = div_even(spec.num_kv_heads, tp_size, allow_replicate=True)
        # Last axis: logical head_dim for 8-bit, packed byte count for sub-byte.
        # See mha_pool.MHAKVCache.__init__ for the same fix and its rationale.
        last_dim = self._quant.physical_head_dim(spec.head_dim) if self._quant.enabled else spec.head_dim
        shape = (2, spec.num_layers, outer_size, inner_size, local_kv_heads, last_dim)
        buffer = torch.empty(shape, device=device, dtype=self._buffer_dtype(dtype))
        scales = self._alloc_scales(shape, device)
        return _KVGroupStorage(
            buffer=buffer,
            k_buffer=buffer[0],
            v_buffer=buffer[1],
            storage_shape=(outer_size * inner_size, local_kv_heads, last_dim),
            scale_buffer=scales,
            k_scale=None if scales is None else scales[0],
            v_scale=None if scales is None else scales[1],
        )

    @staticmethod
    def _build_layers_mapping(
        num_layers: int, specs: dict[str, KVCacheGroupSpec]
    ) -> tuple[_LayerRef, ...]:
        mapping: list[_LayerRef | None] = [None] * num_layers
        for group_name in ("full", "swa"):
            for local_index, layer_id in enumerate(specs[group_name].layer_ids):
                if layer_id < 0 or layer_id >= num_layers:
                    raise ValueError(f"KV layer id {layer_id} is outside [0, {num_layers})")
                if mapping[layer_id] is not None:
                    raise ValueError(f"KV layer id {layer_id} appears in more than one group")
                mapping[layer_id] = _LayerRef(group=group_name, index=local_index)

        missing = [layer_id for layer_id, ref in enumerate(mapping) if ref is None]
        if missing:
            raise ValueError(f"KV layer ids missing from full/swa groups: {missing}")
        return tuple(ref for ref in mapping if ref is not None)

    def is_full_layer(self, layer_id: int) -> bool:
        return self.layers_mapping[layer_id].group == "full"

    def is_swa_layer(self, layer_id: int) -> bool:
        return self.layers_mapping[layer_id].group == "swa"

    def group_of(self, layer_id: int) -> str:
        return self.layers_mapping[layer_id].group

    def translate_loc_from_full_to_swa(self, out_loc: torch.Tensor) -> torch.Tensor:
        # Global-paged SWA: a full slot's swa slot is read from the dense mapping (0 = no live
        # swa slot). Computed/reused full slots' swa entries are written by alloc_swa before any
        # store/gather; out-of-window slots map back to 0.
        return self.full_to_swa_index_mapping[out_loc.to(torch.int64)].to(torch.int32)

    # ---- Option A: global-paged SWA allocator (== sglang SWATokenToKVPoolAllocator) ----

    def _init_swa_paged_state(self) -> None:
        """(Re)build the global-paged SWA allocator state: the dense full->swa slot mapping
        (slot 0 = 'no live SWA slot' sentinel) and the swa free-list. Called from __init__ and
        from rebuild() so the mapping + free-list always match the freshly (re)allocated swa
        buffer (idle-only on rebuild)."""
        n = self._full_num_tokens
        ps = self._page_size
        dev = self._device
        # Every full slot maps to 0 (= no live swa slot) until alloc_swa writes it. The
        # trailing -1 lets a -1 "last_loc" map to -1 (matches sglang allocator/swa.py).
        self.full_to_swa_index_mapping = torch.cat(
            [
                torch.zeros(n + ps, dtype=torch.int64, device=dev),
                torch.tensor([-1], dtype=torch.int64, device=dev),
            ]
        )
        # swa slots 1.._swa_num_tokens-1 are allocatable; slot 0 is the reserved sentinel.
        self._swa_free = torch.arange(1, self._swa_num_tokens, dtype=torch.int32, device=dev)

    def alloc_swa(self, full_indices: torch.Tensor) -> None:
        """Allocate one swa slot per full slot and record full->swa in the mapping. The caller
        must check ``swa_available_size()`` first. Granularity-agnostic: at page_size>1 the
        caller hands whole pages (allocate_paged's _page_to_token expansion) and the finish
        path returns the padding slots with the page (see CacheManager._swa_padded_tail)."""
        assert self._swa_paged, "alloc_swa requires the global-paged SWA mode"
        n = int(full_indices.numel())
        if n == 0:
            return
        if n > self._swa_free.numel():
            raise RuntimeError(f"SWA pool exhausted: need {n}, have {int(self._swa_free.numel())}")
        swa = self._swa_free[:n]
        self._swa_free = self._swa_free[n:]
        self.full_to_swa_index_mapping[full_indices.to(torch.int64)] = swa.to(torch.int64)

    def free_swa(self, full_indices: torch.Tensor) -> None:
        """Return the swa slots backing ``full_indices`` to the free-list and reset their
        mapping entries to the 0 sentinel. Never touches the full pool; idempotent over the
        sentinel (already-freed entries are filtered by ``> 0``). == sglang free_swa."""
        assert self._swa_paged, "free_swa requires the global-paged SWA mode"
        if full_indices.numel() == 0:
            return
        fi = full_indices.to(torch.int64)
        swa = self.full_to_swa_index_mapping[fi]
        swa = swa[swa > 0]
        if swa.numel():
            self._swa_free = torch.cat([self._swa_free, swa.to(torch.int32)])
        self.full_to_swa_index_mapping[fi] = 0

    def swa_available_size(self) -> int:
        return int(self._swa_free.numel())

    @property
    def swa_paged(self) -> bool:
        return self._swa_paged

    def k_cache(self, index: int) -> torch.Tensor:
        ref = self.layers_mapping[index]
        return self._storages[ref.group].k_buffer[ref.index]

    def v_cache(self, index: int) -> torch.Tensor:
        ref = self.layers_mapping[index]
        return self._storages[ref.group].v_buffer[ref.index]

    def k_scale(self, index: int) -> torch.Tensor | None:
        ref = self.layers_mapping[index]
        scales = self._storages[ref.group].k_scale
        return None if scales is None else scales[ref.index]

    def v_scale(self, index: int) -> torch.Tensor | None:
        ref = self.layers_mapping[index]
        scales = self._storages[ref.group].v_scale
        return None if scales is None else scales[ref.index]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        ref = self.layers_mapping[layer_id]
        storage = self._storages[ref.group]
        indices = out_loc
        if ref.group == "swa":
            indices = self.translate_loc_from_full_to_swa(out_loc)
        scale_shape = (storage.storage_shape[0], storage.storage_shape[1], -1)
        self._store_kv_into(
            storage.k_buffer[ref.index].view(storage.storage_shape),
            storage.v_buffer[ref.index].view(storage.storage_shape),
            None if storage.k_scale is None else storage.k_scale[ref.index].view(scale_shape),
            None if storage.v_scale is None else storage.v_scale[ref.index].view(scale_shape),
            indices,
            k,
            v,
        )

    @property
    def full_num_tokens(self) -> int:
        return self._full_num_tokens

    @property
    def swa_num_tokens(self) -> int:
        return self._swa_num_tokens

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @staticmethod
    def _group_geometry(group: _KVGroupStorage) -> tuple:
        # Everything the realloc needs that does NOT pin the old buffer alive: layer count,
        # kv heads, head_dim, device, dtype. (Plain ints + device/dtype handles, no tensor.)
        _, num_layers, _old_outer, _old_inner, local_kv_heads, head_dim = group.buffer.shape
        return (num_layers, local_kv_heads, head_dim, group.buffer.device, group.buffer.dtype)

    def _alloc_group(self, geom: tuple, outer_size: int, inner_size: int) -> _KVGroupStorage:
        # Only the outer (page/token) dimension changes; the rest comes from ``geom``.
        num_layers, local_kv_heads, head_dim, device, dtype = geom
        last_dim = self._quant.physical_head_dim(head_dim) if self._quant.enabled else head_dim
        shape = (2, num_layers, outer_size, inner_size, local_kv_heads, last_dim)
        buffer = torch.empty(shape, device=device, dtype=dtype)
        scales = self._alloc_scales(shape, device)
        return _KVGroupStorage(
            buffer=buffer,
            k_buffer=buffer[0],
            v_buffer=buffer[1],
            storage_shape=(outer_size * inner_size, local_kv_heads, last_dim),
            scale_buffer=scales,
            k_scale=None if scales is None else scales[0],
            v_scale=None if scales is None else scales[1],
        )

    def rebuild(self, num_full_pages: int, num_swa_tokens: int | None = None) -> None:
        """Reallocate both group buffers IN PLACE for new sizes.

        ``page_size`` (the full group's inner dim) and group geometry are read from the
        existing buffers; ``layers_mapping`` is unchanged. Object identity is preserved.
        """
        page_size = self.full_kv_pool.buffer.shape[3]
        self._full_num_tokens = num_full_pages * page_size
        self._swa_num_tokens = num_swa_tokens if num_swa_tokens is not None else self._full_num_tokens
        # Capture geometry, then DROP all references to the old buffers before allocating
        # the replacements so empty_cache() can actually reclaim them. Otherwise the old
        # and new KV buffers are live simultaneously and the rebuild can OOM even when the
        # target geometry alone would fit.
        full_geom = self._group_geometry(self.full_kv_pool)
        swa_geom = self._group_geometry(self.swa_kv_pool)
        self.full_kv_pool = None
        self.swa_kv_pool = None
        self._storages = {}
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
            torch.cuda.empty_cache()
        self.full_kv_pool = self._alloc_group(full_geom, outer_size=num_full_pages, inner_size=page_size)
        self.swa_kv_pool = self._alloc_group(swa_geom, outer_size=self._swa_num_tokens, inner_size=1)
        self._storages = {"full": self.full_kv_pool, "swa": self.swa_kv_pool}
        if self._swa_paged:
            # Reset the full->swa mapping (all 0) + swa free-list (all free) to the new
            # geometry, atomic with the buffer realloc. The fresh empty SWA radix tree built
            # by CacheManager.rebuild is then consistent by construction (idle-only).
            self._init_swa_paged_state()

    @classmethod
    def kv_cost(cls, config, *, num_swa_pages: int | None = None) -> tuple[int, int, int, int]:
        """Full groups ride cache_per_page; the window pool is either a pinned absolute
        window (fixed) or ratio x full (per-page) + the concurrency floor reservation, and
        the dense full_to_swa mapping (8B int64/slot) always scales with the full pages.
        ``num_swa_pages`` is THIS family's own keyword: a rebuild's target window, priced
        before the config override is written; None reads config's truth (its override,
        else the ratio)."""
        from .base import spec_kv_bytes_per_token

        swa_pin = (
            num_swa_pages if num_swa_pages is not None
            else config.swa_num_pages_override
        )
        cache_per_page = 0
        fixed_cache_size = 0
        for spec in config.model_config.kv_cache_group_specs():
            per_token = spec_kv_bytes_per_token(spec, config)
            if not spec.is_swa:
                cache_per_page += per_token * config.page_size
                continue
            cache_per_page += 8 * config.page_size  # full_to_swa_index_mapping bytes/page
            if config.cache_type != "swa_radix":
                # naive swa pool (see _naive_swa_num_tokens): fixed concurrency x window.
                fixed_cache_size += per_token * _naive_swa_num_tokens(config)
            elif swa_pin is not None:
                # pinned window == _swa_paged_num_tokens(override): max(floor, pin) + 1.
                fixed_cache_size += per_token * (
                    max(_swa_pool_floor(config), int(swa_pin)) + 1
                )
            else:
                cache_per_page += int(per_token * config.page_size * config.swa_full_tokens_ratio)
                fixed_cache_size += per_token * _swa_pool_floor(config)  # concurrency floor
        return cache_per_page, fixed_cache_size, config.page_size, 0

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        # radix sizes the window by ratio (cross-request reuse), naive by concurrency x window.
        num_swa_tokens = (
            _swa_paged_num_tokens(config, num_pages + 1, num_swa_pages=num_swa_pages)
            if config.cache_type == "swa_radix"
            else _naive_swa_num_tokens(config)
        )
        # +1 for the dummy page (matches create_kvcache_pool)
        self.rebuild(num_full_pages=num_pages + 1, num_swa_tokens=num_swa_tokens)

    def unit_bytes(self) -> tuple[int, int]:
        def group_bytes(group: _KVGroupStorage) -> int:
            total = group.buffer.numel() * group.buffer.element_size()
            if group.scale_buffer is not None:
                total += group.scale_buffer.numel() * group.scale_buffer.element_size()
            return int(total)

        full = self.full_kv_pool.buffer
        full_tokens = int(full.shape[2]) * int(full.shape[3])
        return (
            group_bytes(self.full_kv_pool) // full_tokens,
            group_bytes(self.swa_kv_pool) // self._swa_num_tokens,
        )


# ---- SWA pool sizing (pure arithmetic; the pool family's geometry formulas) ----


def _naive_swa_num_tokens(config) -> int:
    """Naive SWA pool = one window+headroom buffer per concurrent request. Sized so a request's
    whole swa footprint (up to max_forward_len) fits without prefill-time out-of-window freeing;
    the decode driver still bounds it during generation. == sglang's concurrency x window cap on
    the shared paged pool (memory-efficient naive with prefill-time freeing is a follow-up)."""
    swa_group = config.model_config.swa_attention_group()
    assert swa_group is not None
    width = align_ceil(swa_group.sliding_window + config.max_forward_len + 1, 32)
    return (config.max_running_req + 1) * width


def _swa_per_req_swa_floor(config) -> int:
    """One request's NON-EVICTABLE swa while it decodes, in tokens:

      - the trailing window the prefill-boundary commit locks -- window + retain gap, page-rounded
        (_cache_req_swa splits the committed node there so inc_lock pins that and no more);
      - its own decode tail up to the first out-of-window free: the driver runs only every
        _SWA_EVICTION_INTERVAL forwards and floors its frees at the committed length, so the
        request grows window + 2 pages + one interval before it can reclaim anything.

    Neither is reachable by evict_swa, so the pool must hold both for every running request."""
    from freetoken.scheduler.cache import _SWA_EVICTION_INTERVAL, _SWA_RETAIN_GAP

    window = next(g.sliding_window for g in config.model_config.kv_cache_group_specs() if g.is_swa)
    ps = config.page_size
    locked = ((window + _SWA_RETAIN_GAP + ps - 1) // ps) * ps
    floor = locked + window + _SWA_EVICTION_INTERVAL + 2 * ps
    if getattr(config, "special_token_ckpt", False):
        floor += window + _SWA_RETAIN_GAP + _SWA_EVICTION_INTERVAL
    return floor


def _swa_pool_floor(config) -> int:
    """The swa pool's hard floor: max_running_req x the per-request non-evictable footprint.
    Admission gates only the incoming chunk (PrefillAdder's need_swa is one window) and never
    reserves the decode growth of the requests already running, so a full batch can drive the
    pool to zero -- and the alloc_swa that then raises has no handler above it."""
    return config.max_running_req * _swa_per_req_swa_floor(config)


def _swa_paged_num_tokens(config, num_full_pages: int, num_swa_pages: int | None = None) -> int:
    """SWA pool size = max(concurrency floor, target tokens) + 1 (slot-0 sentinel). The target is
    a pinned absolute window (``num_swa_pages`` if given, else swa_num_pages_override, usable
    tokens), else ratio x full-pool tokens. The floor keeps a full batch always fitting;
    < 1.0 ratio trades reuse for memory."""
    floor = _swa_pool_floor(config)
    override = num_swa_pages if num_swa_pages is not None else config.swa_num_pages_override
    if override is not None:
        return max(floor, int(override)) + 1
    full_tokens = num_full_pages * config.page_size
    return max(floor, int(config.swa_full_tokens_ratio * full_tokens)) + 1
