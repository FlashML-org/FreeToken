"""Scale-buffer bookkeeping shared by the quantizable KV pools.

A quantized pool allocates, alongside each K/V slab, a scale slab with the same logical
shape but the last dimension divided by :data:`~freetoken.kvcache.quant.BLOCK`. Packed
formats additionally shrink the K/V slab's physical last dimension. The slabs must be
allocated, rebuilt and freed together, and ``store_kv`` routes to the quantizing kernel
instead of the byte-copy one -- that is all this mixin owns. The pools keep their own
geometry and indexing.
"""

from __future__ import annotations

import torch

from .quant import NONE, SCALE_DTYPE, KVQuantSpec


class QuantizedKVStorageMixin:
    """Allocation + store routing for pools whose K/V slabs may be compact.

    Subclasses set ``self._quant`` before allocating and call :meth:`_alloc_scales` for
    each K/V buffer they create. ``_quant`` defaulting to the unquantized spec keeps
    pools that never opt in behaving exactly as before.
    """

    _quant: KVQuantSpec = NONE

    @property
    def quant(self) -> KVQuantSpec:
        return self._quant

    def _buffer_dtype(self, compute_dtype: torch.dtype) -> torch.dtype:
        """Element dtype for a K/V slab under the active scheme."""
        return self._quant.storage_dtype if self._quant.enabled else compute_dtype

    def _buffer_shape(self, kv_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Slab shape for a logical KV shape (last dim halves when packed)."""
        return self._quant.storage_shape(kv_shape) if self._quant.enabled else kv_shape

    def _alloc_scales(self, kv_shape: tuple[int, ...], device: torch.device) -> torch.Tensor | None:
        """Scale slab matching a ``[2, layers, ..., heads, head_dim]`` logical K/V buffer.

        None when unquantized -- callers store that verbatim and the attention path reads
        it as "no scales", which is what selects the bf16 kernel branch. ``kv_shape`` is
        the unpacked (element-counted) geometry: the scale extent is D // BLOCK regardless
        of how the slab packs its elements.
        """
        if not self._quant.enabled:
            return None
        return torch.empty(
            self._quant.scale_shape(kv_shape), device=device, dtype=SCALE_DTYPE
        )

    def _store_kv_into(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale: torch.Tensor | None,
        v_scale: torch.Tensor | None,
        indices: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Write one layer's K/V, quantizing on the way in when the pool is compact."""
        if not self._quant.enabled:
            from freetoken.kernel import store_cache

            store_cache(k_cache=k_cache, v_cache=v_cache, indices=indices, k=k, v=v)
            return

        from freetoken.kernel.triton.kv_quant import store_kv_quant

        heads, storage_head_dim = k_cache.shape[-2:]
        head_dim = storage_head_dim * self._quant.elements_per_byte
        store_kv_quant(
            k_cache,
            k_scale,
            v_cache,
            v_scale,
            indices,
            k.view(-1, heads, head_dim),
            v.view(-1, heads, head_dim),
            self._quant,
        )


__all__ = ["QuantizedKVStorageMixin"]
