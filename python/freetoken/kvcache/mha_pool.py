from __future__ import annotations

from typing import Sequence

import torch
from freetoken.distributed import get_tp_info
from freetoken.utils import div_even

from .base import BaseKVCachePool, KVStorageDescriptor, kv_storage_descriptor


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
        storage_type=None,
    ) -> None:
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
        if storage_type is None:
            descriptor = KVStorageDescriptor(
                "bf16" if dtype == torch.bfloat16 else "fp16", payload_dtype=dtype
            )
            self._storage_dtype = dtype
        else:
            descriptor = kv_storage_descriptor(type("KVConfig", (), {"kv_storage_type": storage_type})(), head_dim=head_dim)
            self._storage_dtype = descriptor.payload_dtype
        self._descriptor = descriptor
        self._generation = 0
        shape = (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim)
        if descriptor.is_quantized:
            descriptor.validate_head_dim(head_dim)
            self._kv_buffer = None
            self._k_buffer = torch.empty(shape, device=device, dtype=torch.int8)[0]
            self._v_buffer = torch.empty(shape, device=device, dtype=torch.int8)[1]
            scale_shape = (*shape[:4], local_kv_heads, head_dim // descriptor.block_size)
            scales = torch.empty(scale_shape, device=device, dtype=descriptor.scale_dtype)
            self._k_scales = scales[0]
            self._v_scales = scales[1]
            self._zero_dummy_page()
        else:
            self._kv_buffer = torch.empty(shape, device=device, dtype=self._storage_dtype)
            self._k_buffer = self._kv_buffer[0]
            self._v_buffer = self._kv_buffer[1]
            self._k_scales = self._v_scales = None
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def rebuild(self, num_pages: int) -> None:
        """Reallocate the KV buffer for ``num_pages`` pages IN PLACE.

        Geometry (storage layers, page_size, kv heads, head_dim) is taken from the
        existing buffer; only the page count changes. Views and ``_storage_shape`` are
        refreshed. Object identity is preserved so cached backend references stay valid.
        """
        old = self._k_buffer
        num_storage_layers, _old_pages, page_size, local_kv_heads, head_dim = old.shape
        device = self._device
        self._k_buffer = None
        self._v_buffer = None
        self._kv_buffer = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        shape = (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim)
        if self._descriptor.is_quantized:
            self._k_buffer = torch.empty(shape, device=device, dtype=torch.int8)[0]
            self._v_buffer = torch.empty(shape, device=device, dtype=torch.int8)[1]
            scale_shape = (*shape[:4], local_kv_heads, head_dim // self._descriptor.block_size)
            scales = torch.empty(scale_shape, device=device, dtype=self._descriptor.scale_dtype)
            self._k_scales = scales[0]
            self._v_scales = scales[1]
            self._zero_dummy_page()
        else:
            self._kv_buffer = torch.empty(shape, device=device, dtype=self._storage_dtype)
            self._k_buffer = self._kv_buffer[0]
            self._v_buffer = self._kv_buffer[1]
            self._k_scales = self._v_scales = None
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)
        self._generation += 1

    def _zero_dummy_page(self) -> None:
        """Initialize reserved dummy page, including Q8 payload and scales."""
        if not self._descriptor.is_quantized:
            return
        self._k_buffer[:, -1].zero_()
        self._v_buffer[:, -1].zero_()
        self._k_scales[:, -1].zero_()
        self._v_scales[:, -1].zero_()

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
        tokens = int(self._k_buffer.shape[1]) * int(self._k_buffer.shape[2])
        total = self._k_buffer.numel() * self._k_buffer.element_size()
        if self._descriptor.is_quantized:
            total += self._k_scales.numel() * self._k_scales.element_size()
        total *= 2
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

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        if self._descriptor.is_quantized:
            from freetoken.kernel.triton.q8_kv import store_q8_cache

            # Qwen3.5 projection path keeps KV heads flattened as [tokens,
            # heads * head_dim]. Q8 storage quantizes one row per head, so
            # restore its explicit row geometry at cache boundary. reshape is
            # view-only for normal contiguous projection output.
            kv_heads, head_dim = self._storage_shape[1:]
            flat_width = kv_heads * head_dim
            if k.ndim == 2 and v.ndim == 2 and k.shape[1] == flat_width and v.shape[1] == flat_width:
                k = k.reshape(-1, kv_heads, head_dim)
                v = v.reshape(-1, kv_heads, head_dim)

            dense = self._dense(layer_id)
            store_q8_cache(
                k_payload=self._k_buffer[dense].view(self._storage_shape),
                v_payload=self._v_buffer[dense].view(self._storage_shape),
                k_scales=self._k_scales[dense].view(-1, self._storage_shape[1], self._storage_shape[2] // 32),
                v_scales=self._v_scales[dense].view(-1, self._storage_shape[1], self._storage_shape[2] // 32),
                indices=out_loc,
                k=k,
                v=v,
            )
            return

        from freetoken.kernel import store_cache

        dense = self._dense(layer_id)
        store_cache(
            k_cache=self._k_buffer[dense].view(self._storage_shape),
            v_cache=self._v_buffer[dense].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._descriptor.payload_dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def storage_descriptor(self) -> KVStorageDescriptor:
        return self._descriptor

    @property
    def pointer_generation(self) -> int:
        return self._generation

    @property
    def is_quantized(self) -> bool:
        return self._descriptor.is_quantized

    def k_cache_view(self, index: int):
        if not self.is_quantized:
            return self.k_cache(index)
        from .base import QuantizedKVView

        dense = self._dense(index)
        return QuantizedKVView(
            self._k_buffer[dense].view(self._storage_shape),
            self._k_scales[dense].view(-1, self._storage_shape[1], self._storage_shape[2] // 32),
            self._descriptor,
        )

    def v_cache_view(self, index: int):
        if not self.is_quantized:
            return self.v_cache(index)
        from .base import QuantizedKVView

        dense = self._dense(index)
        return QuantizedKVView(
            self._v_buffer[dense].view(self._storage_shape),
            self._v_scales[dense].view(-1, self._storage_shape[1], self._storage_shape[2] // 32),
            self._descriptor,
        )
