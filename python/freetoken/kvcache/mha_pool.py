from __future__ import annotations

from typing import ClassVar, Sequence

import torch
from freetoken.distributed import get_tp_info
from freetoken.utils import div_even

from .base import BaseKVCachePool
from .format import NONE, ROW_SCALE_FORMATS, KVFormat


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.

    ``layer_ids`` lets the pool back only a *subset* of the model's layers while
    callers keep indexing by their global ``layer_id``. Hybrid models (e.g. the
    Qwen3.5 GatedDeltaNet/full-attention stack) interleave linear-attention layers
    that hold no paged KV; passing the full-attention layer ids here allocates one
    storage slab per KV layer (not per model layer) and remaps the global id to its
    dense slot, avoiding a multiple-x over-allocation of unused slabs.
    """

    kv_formats: ClassVar[frozenset[str]] = ROW_SCALE_FORMATS

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_ids: Sequence[int] | None = None,
        kv_format: KVFormat = NONE,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._num_layers = num_layers
        self.kv_format = kv_format
        self._dtype = dtype
        self._head_dim = head_dim
        if layer_ids is None:
            num_storage_layers = num_layers
            self._layer_map: list[int] | None = None
        else:
            num_storage_layers = len(layer_ids)
            layer_map = [-1] * num_layers
            for dense, global_id in enumerate(layer_ids):
                if global_id < 0 or global_id >= num_layers:
                    raise ValueError(f"KV layer id {global_id} outside [0, {num_layers})")
                layer_map[global_id] = dense
            self._layer_map = layer_map
        self._device = device
        self._alloc(num_storage_layers, num_pages, page_size, local_kv_heads)

    def _alloc(self, num_storage_layers: int, num_pages: int, page_size: int, local_kv_heads: int) -> None:
        fmt = self.kv_format
        elem_dim = fmt.elem_dim(self._head_dim)
        # zeros: an unwritten e4m3 code can be NaN
        new = torch.zeros if fmt.quantized else torch.empty
        self._kv_buffer = new(
            (2, num_storage_layers, num_pages, page_size, local_kv_heads, elem_dim),
            device=self._device,
            dtype=fmt.elem_tensor_dtype(self._dtype),
        )
        self._scale_buffer = None
        if fmt.scale_dtype is not None:
            self._scale_buffer = torch.zeros(
                (2, num_storage_layers, num_pages * page_size, local_kv_heads, fmt.scale_dim(self._head_dim)),
                device=self._device,
                dtype=fmt.scale_store_dtype,
            )
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._storage_shape = (num_pages * page_size, local_kv_heads, elem_dim)

    def rebuild(self, num_pages: int) -> None:
        """Reallocate the KV buffer for ``num_pages`` pages IN PLACE.

        Storage layers, page_size and kv heads are taken from the existing buffer and
        head_dim from the pool; only the page count changes. Views and ``_storage_shape`` are
        refreshed. Object identity is preserved so cached backend references stay valid.
        """
        _, num_storage_layers, _old_pages, page_size, local_kv_heads, _elem_dim = self._kv_buffer.shape
        device = self._device
        self._k_buffer = None
        self._v_buffer = None
        self._kv_buffer = None
        self._scale_buffer = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        self._alloc(num_storage_layers, num_pages, page_size, local_kv_heads)

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token

        per_token = sum(
            spec_kv_bytes_per_token(spec, config, getattr(config, "kv_format", NONE))
            for spec in config.model_config.kv_cache_group_specs()
            if not spec.is_swa
        )
        return per_token * config.page_size, 0, config.page_size, 0

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        self.rebuild(num_pages + 1)  # +1 for the dummy page (matches create_kvcache_pool)

    def unit_bytes(self) -> tuple[int, int]:
        buf = self._kv_buffer
        tokens = int(buf.shape[2]) * int(buf.shape[3])
        total = int(buf.numel() * buf.element_size())
        if self._scale_buffer is not None:
            total += int(self._scale_buffer.numel() * self._scale_buffer.element_size())
        return total // tokens, 0

    def _dense(self, layer_id: int) -> int:
        if self._layer_map is None:
            return layer_id
        dense = self._layer_map[layer_id]
        if dense < 0:
            raise KeyError(f"layer {layer_id} has no paged KV storage")
        return dense

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[self._dense(index)]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[self._dense(index)]

    def k_scale(self, index: int) -> torch.Tensor | None:
        return None if self._scale_buffer is None else self._scale_buffer[0, self._dense(index)]

    def v_scale(self, index: int) -> torch.Tensor | None:
        return None if self._scale_buffer is None else self._scale_buffer[1, self._dense(index)]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        dense = self._dense(layer_id)
        k_cache = self._k_buffer[dense].view(self._storage_shape)
        v_cache = self._v_buffer[dense].view(self._storage_shape)
        if self.kv_format.quantized:
            from freetoken.kernel.triton.kv_rows import store_kv_rows

            store_kv_rows(self.kv_format, k, v, out_loc, k_cache, v_cache, self.k_scale(layer_id), self.v_scale(layer_id))
            return

        from freetoken.kernel import store_cache

        store_cache(
            k_cache=k_cache,
            v_cache=v_cache,
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
