from __future__ import annotations

from typing import Sequence

import torch
from freetoken.distributed import get_tp_info
from freetoken.utils import div_even

from .base import BaseKVCachePool
from .quant import NONE, KVQuantSpec
from .quant_storage import QuantizedKVStorageMixin


class MHAKVCache(QuantizedKVStorageMixin, BaseKVCachePool):
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
        quant: KVQuantSpec = NONE,
    ) -> None:
        self._quant = quant
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._num_layers = num_layers
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
        self._compute_dtype = dtype
        # The last (head_dim) axis of the byte buffer is logical head_dim for 8-bit
        # schemes, and ``physical_head_dim(logical)`` for sub-byte ones -- the bytes
        # pack multiple values per byte. The kernel sees the LOGICAL extent as
        # ``head_dim`` and unpacks inside the load.
        self._head_dim = head_dim
        last_dim = quant.physical_head_dim(head_dim) if quant.enabled else head_dim
        kv_shape = (2, num_storage_layers, num_pages, page_size, local_kv_heads, last_dim)
        self._kv_buffer = torch.empty(
            kv_shape, device=device, dtype=self._buffer_dtype(dtype)
        )
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._scale_buffer = self._alloc_scales(kv_shape, device)
        self._k_scale = self._scale_buffer[0] if self._scale_buffer is not None else None
        self._v_scale = self._scale_buffer[1] if self._scale_buffer is not None else None
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, last_dim)

    def rebuild(self, num_pages: int) -> None:
        """Reallocate the KV buffer for ``num_pages`` pages IN PLACE.

        Geometry (storage layers, page_size, kv heads, head_dim) is taken from the
        existing buffer; only the page count changes. Views and ``_storage_shape`` are
        refreshed. Object identity is preserved so cached backend references stay valid.
        """
        _, num_storage_layers, _old_pages, page_size, local_kv_heads, _physical = self._kv_buffer.shape
        dtype = self._kv_buffer.dtype
        device = self._device
        self._k_buffer = None
        self._v_buffer = None
        self._kv_buffer = None
        # Drop the scale slab too before reallocating, for the same reason the KV slab is
        # dropped: holding the old one alive can OOM a rebuild the target size would fit.
        self._k_scale = None
        self._v_scale = None
        self._scale_buffer = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        last_dim = self._quant.physical_head_dim(self._head_dim) if self._quant.enabled else self._head_dim
        kv_shape = (2, num_storage_layers, num_pages, page_size, local_kv_heads, last_dim)
        self._kv_buffer = torch.empty(kv_shape, device=device, dtype=dtype)
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._scale_buffer = self._alloc_scales(kv_shape, device)
        self._k_scale = self._scale_buffer[0] if self._scale_buffer is not None else None
        self._v_scale = self._scale_buffer[1] if self._scale_buffer is not None else None
        self._storage_shape = (num_pages * page_size, local_kv_heads, last_dim)

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token

        per_token = sum(
            spec_kv_bytes_per_token(spec, config)
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
        total = buf.numel() * buf.element_size()
        if self._scale_buffer is not None:
            total += self._scale_buffer.numel() * self._scale_buffer.element_size()
        return int(total) // tokens, 0

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
        return None if self._k_scale is None else self._k_scale[self._dense(index)]

    def v_scale(self, index: int) -> torch.Tensor | None:
        return None if self._v_scale is None else self._v_scale[self._dense(index)]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        dense = self._dense(layer_id)
        scale_shape = (self._storage_shape[0], self._storage_shape[1], -1)
        self._store_kv_into(
            self._k_buffer[dense].view(self._storage_shape),
            self._v_buffer[dense].view(self._storage_shape),
            None if self._k_scale is None else self._k_scale[dense].view(scale_shape),
            None if self._v_scale is None else self._v_scale[dense].view(scale_shape),
            out_loc,
            k,
            v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def compute_dtype(self) -> torch.dtype:
        return self._compute_dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
