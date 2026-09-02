from __future__ import annotations

from typing import Sequence

import torch
from freetoken.distributed import get_tp_info
from freetoken.utils import div_even

from .base import BaseKVCachePool


def _kv_store_dtype(dtype: torch.dtype, kv_quant: str) -> torch.dtype:
    """Storage dtype of the KV buffer for a quantization mode."""
    if kv_quant == "none":
        return dtype
    if kv_quant == "fp8":
        from freetoken.kernel.triton.kv_quant import kv_codes_dtype

        return kv_codes_dtype()
    raise ValueError(f"unknown kv_quant {kv_quant!r}")


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

    ``kv_quant="fp8"`` halves the cache: rows become e4m3 codes and every
    ``(token, slab, layer, kv head)`` row carries one fp32 scale (see
    :mod:`freetoken.kernel.triton.kv_quant`). The codes buffer keeps the exact same
    shape as the 16-bit one, so ``k_cache``/``v_cache`` and every index into them are
    unchanged -- only the element type, and ``store_kv``'s write path, differ.
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
        kv_quant: str = "none",
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._num_layers = num_layers
        self.kv_quant = kv_quant
        self._compute_dtype = dtype
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
        self._alloc(num_pages, page_size, num_storage_layers, local_kv_heads, head_dim)

    def _alloc(
        self,
        num_pages: int,
        page_size: int,
        num_storage_layers: int,
        local_kv_heads: int,
        head_dim: int,
    ) -> None:
        """Allocate the code buffer (and, when quantized, the scale buffer).

        A quantized buffer is zero-filled -- e4m3 has NaN bit patterns, so an
        unwritten slot (the dummy page, a padded request's row) must not read back as
        one. The 16-bit buffer keeps ``torch.empty``: it is bytes-sized, never
        interpreted, and the memset would cost real startup time on a large cache.
        """
        shape = (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim)
        if self.kv_quant == "fp8":
            from freetoken.kernel.triton.kv_quant import alloc_codes

            self._kv_buffer = alloc_codes(shape, self._device)
            self._scale_buffer = torch.zeros(
                (2, num_storage_layers, num_pages * page_size, local_kv_heads),
                device=self._device,
                dtype=torch.float32,
            )
        else:
            self._kv_buffer = torch.empty(
                shape, device=self._device, dtype=self._compute_dtype
            )
            self._scale_buffer = None
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def rebuild(self, num_pages: int) -> None:
        """Reallocate the KV buffer for ``num_pages`` pages IN PLACE.

        Geometry (storage layers, page_size, kv heads, head_dim) is taken from the
        existing buffer; only the page count changes. Views and ``_storage_shape`` are
        refreshed. Object identity is preserved so cached backend references stay valid.
        """
        _, num_storage_layers, _old_pages, page_size, local_kv_heads, head_dim = self._kv_buffer.shape
        device = self._device
        self._k_buffer = None
        self._v_buffer = None
        self._kv_buffer = None
        self._scale_buffer = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        self._alloc(num_pages, page_size, num_storage_layers, local_kv_heads, head_dim)

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
        kv = int(buf.numel() * buf.element_size()) // tokens
        if self._scale_buffer is not None:
            sc = self._scale_buffer
            kv += int(sc.numel() * sc.element_size()) // tokens
        return kv, 0

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
        if self._scale_buffer is None:
            return None
        return self._scale_buffer[0][self._dense(index)]

    def v_scale(self, index: int) -> torch.Tensor | None:
        if self._scale_buffer is None:
            return None
        return self._scale_buffer[1][self._dense(index)]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        dense = self._dense(layer_id)
        if self.kv_quant == "fp8":
            from freetoken.kernel.triton.kv_quant import quantize_kv_to_cache

            quantize_kv_to_cache(
                k=k,
                v=v,
                out_loc=out_loc,
                k_cache=self._k_buffer[dense].view(self._storage_shape),
                v_cache=self._v_buffer[dense].view(self._storage_shape),
                k_scale=self._scale_buffer[0][dense],
                v_scale=self._scale_buffer[1][dense],
            )
            return
        from freetoken.kernel import store_cache

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
        """The COMPUTE dtype (kvcache/base.py): what ``store_kv`` receives and what
        backends size their scratch with -- still 16-bit on an fp8 pool."""
        return self._compute_dtype

    @property
    def store_dtype(self) -> torch.dtype:
        """Element type of ``_kv_buffer``: fp8/uint8 codes when quantized, else the
        compute dtype. Reading the buffer itself needs THIS, and a backend that cannot
        apply the row scales never gets a quantized pool (supports_fp8_kv gate)."""
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
