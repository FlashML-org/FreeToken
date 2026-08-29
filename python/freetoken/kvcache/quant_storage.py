"""Scale-buffer bookkeeping shared by the quantizable KV pools.

A quantized pool allocates, alongside each K/V slab, a scale slab with the same shape
but the last dimension divided by :data:`~freetoken.kvcache.quant.BLOCK`. The two must
be allocated, rebuilt and freed together, and ``store_kv`` has to route to the
quantizing kernel instead of the byte-copy one -- that is all this mixin owns. The pools
keep their own geometry and indexing.
"""

from __future__ import annotations

import torch

from .quant import NONE, SCALE_DTYPE, KVQuantSpec


class QuantizedKVStorageMixin:
    """Allocation + store routing for pools whose K/V slabs may be 8-bit.

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

    def _alloc_scales(self, kv_shape: tuple[int, ...], device: torch.device) -> torch.Tensor | None:
        """Scale slab matching a ``[2, layers, ..., heads, head_dim]`` K/V buffer.

        None when unquantized -- callers store that verbatim and the attention path reads
        it as "no scales", which is what selects the bf16 kernel branch.
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
        """Write one layer's K/V, quantizing on the way in when the pool is 8-bit."""
        if not self._quant.enabled:
            from freetoken.kernel import store_cache

            store_cache(k_cache=k_cache, v_cache=v_cache, indices=indices, k=k, v=v)
            return

        from freetoken.kernel.triton.kv_quant import store_kv_quant

        # The input ``k``/``v`` are flattened to ``[N, num_kv_heads * head_dim]``
        # by the model code (Qwen3.5 + others fuse the kv projection and reshape
        # to a single ``kv_attn_dim`` axis). Recover ``num_kv_heads`` and the
        # LOGICAL head_dim from the cache, then the actual head_dim is the
        # logical one (not the kv_attn_dim).
        kv_heads = k_cache.shape[-2]
        # k.shape[-1] is num_kv_heads * head_dim. The model's head_dim is
        # recoverable from the cache: its last axis (D_PHYSICAL) is logical head_dim
        # for 8-bit or packed for sub-byte. Invert: logical = physical * 8 / bits.
        kv_attn_dim = k.shape[-1]
        if self._quant.enabled and self._quant.layout != "q8":
            head_dim = k_cache.shape[-1] * 8 // self._quant.bits
        else:
            head_dim = k_cache.shape[-1]
        if head_dim <= 0 or kv_attn_dim % head_dim != 0:
            raise AssertionError(
                f"can't recover head_dim: k.shape[-1]={kv_attn_dim}, cache last-dim={k_cache.shape[-1]}, "
                f"bits={self._quant.bits if self._quant.enabled else 8}"
            )
        # The cache's last axis is the LOGICAL head_dim (D_PHYSICAL for 8-bit is
        # identical to logical); kv_heads is num_kv_heads. Use them directly.
        store_kv_quant(
            k_cache,
            k_scale,
            v_cache,
            v_scale,
            indices,
            k.view(-1, kv_heads, head_dim),
            v.view(-1, kv_heads, head_dim),
            self._quant,
        )


__all__ = ["QuantizedKVStorageMixin"]
