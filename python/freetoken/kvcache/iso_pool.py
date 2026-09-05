"""ISOKVCache: paged KV pool storing K/V quantized to ISO3/ISO4 blocks.

Same page/layer plumbing as MHAKVCache, but each token row holds packed
IsoQuant blocks (kernel/iso.py) instead of bf16 values:
per head vector of head_dim values -> head_dim//128 blocks of 50 B (iso3) or
68 B (iso4). That is 3.125 / 4.25 bits per value vs 16 for bf16.

Reads happen exclusively through the custom attention kernels
(kernel/csrc/jit/iso_attention.cu) via the matching attention backend
(freetoken.attention.iso); writes go through iso_store_cache
(quantize-on-write for decode, deferred bulk store after extend attention).
"""

from __future__ import annotations

from typing import Sequence

import torch
from freetoken.utils import div_even
from freetoken.distributed import get_tp_info

from .base import BaseKVCachePool


class ISOKVCache(BaseKVCachePool):
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
        iso_fmt: str = "iso3",
    ) -> None:
        from freetoken.kernel.iso import packed_row_bytes

        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._iso_fmt = iso_fmt
        self._head_dim = head_dim
        self._local_kv_heads = local_kv_heads
        row_bytes = packed_row_bytes(head_dim, iso_fmt)
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
        self._kv_buffer = torch.empty(
            (2, num_storage_layers, num_pages, page_size, local_kv_heads, row_bytes),
            device=device,
            dtype=torch.uint8,
        )
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads * row_bytes)

    @property
    def iso_fmt(self) -> str:
        return self._iso_fmt

    def rebuild(self, num_pages: int) -> None:
        """Reallocate the packed KV buffer for ``num_pages`` pages IN PLACE
        (object identity preserved, see MHAKVCache.rebuild)."""
        _, num_storage_layers, _old_pages, page_size, local_kv_heads, row_bytes = (
            self._kv_buffer.shape
        )
        device = self._device
        self._k_buffer = None
        self._v_buffer = None
        self._kv_buffer = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        self._kv_buffer = torch.empty(
            (2, num_storage_layers, num_pages, page_size, local_kv_heads, row_bytes),
            device=device,
            dtype=torch.uint8,
        )
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._storage_shape = (num_pages * page_size, local_kv_heads * row_bytes)

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from freetoken.kernel.iso import packed_row_bytes

        fmt = getattr(config, "kv_cache_iso", "iso3")
        per_token = 0
        for spec in config.model_config.kv_cache_group_specs():
            if spec.is_swa:
                continue
            local_heads = div_even(
                spec.num_kv_heads, config.tp_info.size, allow_replicate=True
            )
            per_token += (
                2  # K + V
                * packed_row_bytes(spec.head_dim, fmt)
                * local_heads
                * spec.num_layers
            )
        return per_token * config.page_size, 0, config.page_size, 0

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        self.rebuild(num_pages + 1)  # +1 for the dummy page (matches create_kvcache_pool)

    def unit_bytes(self) -> tuple[int, int]:
        buf = self._kv_buffer
        tokens = int(buf.shape[2]) * int(buf.shape[3])
        return int(buf.numel()) // tokens, 0  # uint8: numel == bytes

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
        from freetoken.kernel.iso import iso_store_cache

        if self._device.type != "cuda":
            raise NotImplementedError("ISOKVCache.store_kv requires a CUDA device")

        dense = self._dense(layer_id)
        n = k.shape[0]
        iso_store_cache(
            self._k_buffer[dense].view(self._storage_shape),
            self._v_buffer[dense].view(self._storage_shape),
            out_loc,
            k.reshape(n, -1),
            v.reshape(n, -1),
            self._local_kv_heads,
            self._head_dim,
            self._iso_fmt,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
