"""GQA block-sparse (BSA) KV pool: paged GQA K/V + the indexer's index-key slab.

MiniMax-M3's sparse layers attend only the top-k 128-token blocks a lightning
indexer selects, so the pool is the MHA pool (standard paged GQA K/V, page_size
== the sparse block size, one page per block) plus one bf16 index-key row per
token per SPARSE layer (``index_head_dim`` wide; M3 has a single shared index
key head). Slot order = the backend's sparse-layer order, same convention as
``DSAKVCache``'s full-indexer slots.

Storage lives here -- not in the attention backend -- so the engine's rebuild
path (fresh allocation, object identity preserved, views re-derived per
forward) and the KV cost model (which budgets the index-K bytes off the
attention-group spec via ``spec_kv_bytes_per_token``) stay correct by
construction. Both slabs resize atomically in ``rebuild`` so the allocator can
never hand out a row one slab has and the other lacks (DSAKVCache precedent).
"""

from __future__ import annotations

from typing import Sequence

import torch

from .format import NONE, KVFormat
from .mha_pool import MHAKVCache


class BSAKVCache(MHAKVCache):
    """MHA paged pool + the block-sparse index-key slab.

    ``index_k_cache(slot)`` is row-flat ``[num_pages * page_size,
    index_head_dim]`` addressed by the same physical token rows as the K/V
    slabs (the shared page table is page_size=1 semantics), so the backend's
    block addressing -- base row of block ``b`` = ``page_table[t, b *
    page_size]`` -- serves K/V and index keys alike.
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
        index_head_dim: int,
        num_index_layers: int,
        layer_ids: Sequence[int] | None = None,
        kv_format: KVFormat = NONE,
    ) -> None:
        self._index_head_dim = index_head_dim
        self._num_index_layers = num_index_layers
        self._page_size = page_size
        # Index keys ride the compute dtype (bf16/fp16 -- the model's index_q/index_k
        # are engine-dtype and store_index_k's index_put_ raises on a mismatch). The
        # KV cost model budgets 2 bytes/token/index-layer for this slab
        # (base.spec_kv_bytes_per_token); keep the two in lockstep.
        assert dtype.itemsize == 2, (
            f"BSA index slab budgets 2 bytes/token (spec_kv_bytes_per_token); got {dtype}"
        )
        self._index_dtype = dtype
        super().__init__(
            num_kv_heads=num_kv_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            num_pages=num_pages,
            page_size=page_size,
            dtype=dtype,
            device=device,
            layer_ids=layer_ids,
            kv_format=kv_format,
        )
        self._zero_kv_slabs()
        self._alloc_index_slab(num_pages)

    def _zero_kv_slabs(self) -> None:
        # Defense-in-depth: the attend kernels pos-mask every K/V load (the real
        # fix for torch.empty's recycled NaN/Inf bit patterns), but a zeroed slab
        # keeps any future unmasked read finite instead of model-poisoning. One
        # memset per (re)allocation -- negligible.
        self._kv_buffer.zero_()

    def _alloc_index_slab(self, num_pages: int) -> None:
        # ZERO-initialized -- the index-score kernels load index keys unmasked and
        # rely on unwritten tail rows dotting to a finite 0 (see the invariant
        # comments in kernel/triton/minimax_m3_sparse.py).
        self._index_k_buffer = torch.zeros(
            self._num_index_layers,
            num_pages * self._page_size,
            self._index_head_dim,
            dtype=self._index_dtype,
            device=self._device,
        )

    def rebuild(self, num_pages: int) -> None:
        # Free the index slab BEFORE the K/V realloc (super().rebuild frees + syncs +
        # empty_cache), then re-derive it at the new page count. If the index-slab
        # alloc itself fails (OOM), null the K/V slab too and re-raise: a pool with a
        # grown K/V slab and no index slab would mis-serve silently, and the engine's
        # rebuild path pre-validates the budget so this is already exceptional.
        self._index_k_buffer = None
        super().rebuild(num_pages)
        self._zero_kv_slabs()
        try:
            self._alloc_index_slab(num_pages)
        except Exception:
            self._kv_buffer = None
            self._k_buffer = None
            self._v_buffer = None
            self._scale_buffer = None
            raise

    def unit_bytes(self) -> tuple[int, int]:
        # The index slab rides the same token budget as the K/V slabs; each slab's
        # per-token cost is floor-divided on its own, matching the cost model's terms.
        kv, swa = super().unit_bytes()
        idx = self._index_k_buffer
        tokens = int(self._kv_buffer.shape[2]) * int(self._kv_buffer.shape[3])
        return kv + int(idx.numel() * idx.element_size()) // tokens, swa

    def index_k_cache(self, slot: int) -> torch.Tensor:
        """Row-flat index keys for a sparse-layer slot: ``[rows, index_head_dim]``."""
        return self._index_k_buffer[slot]

    def store_index_k(self, k: torch.Tensor, out_loc: torch.Tensor, slot: int) -> None:
        self._index_k_buffer[slot][out_loc] = k


__all__ = ["BSAKVCache"]
