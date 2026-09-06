"""MLA latent-KV pools (GLM-5.2 / DeepSeek-style latent attention).

``MLAKVCache`` is the MHA pool's sibling for latent-KV models: ONE slab holding the
per-token latent ``ckv (kv_lora_rank) | kpe (qk_rope_head_dim)`` -- there is no
separate V (``v_cache`` aliases ``k_cache``, same convention as dsv4_paged_pool's
single-latent tiers). ``DSAKVCache`` extends it with the DeepSeek-Sparse-Attention
index-key slab: one ``index_head_dim``-wide bf16 row per token per full-indexer
layer, addressed by the SAME physical rows as the latent slab (GLM-5.2, page 1;
``KpoolDSAKVCache`` overrides the geometry to a 1/ratio shadow). ``rebuild``
resizes ALL slabs atomically so the allocator can never hand out a slot one slab
has and the other lacks.

Storage lives here -- not in the attention backend -- so the engine's rebuild path
(``MHAKVCache.rebuild``-shaped: fresh allocation, object identity preserved, views
re-derived by callers per forward) and the KV cost model (which budgets the index-K
bytes off the attention-group spec) stay correct by construction.
"""

from __future__ import annotations

import torch

from .base import BaseKVCachePool


class MLAKVCache(BaseKVCachePool):
    """Paged latent-KV pool: ``[1, num_layers, num_pages, page_size, 1, latent_dim]``.

    The leading singleton keeps the buffer shape-compatible with MHAKVCache's
    (tokens = shape[2] * shape[3]).

    ``layer_ids`` backs only a SUBSET of the model's layers (hybrid linear x
    MLA/DSA) while callers keep addressing by GLOBAL layer id -- the same remap
    contract as MHAKVCache; None keeps the identity mapping.
    """

    def __init__(
        self,
        latent_dim: int,
        num_layers: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_ids: "tuple[int, ...] | None" = None,
    ) -> None:
        self._latent_dim = latent_dim
        if layer_ids is None:
            self._num_layers = num_layers
            self._layer_index: dict[int, int] | None = None
        else:
            self._num_layers = len(layer_ids)
            self._layer_index = {int(g): i for i, g in enumerate(layer_ids)}
        self._page_size = page_size
        self._dtype = dtype
        self._device = device
        self._alloc(num_pages)

    def _local_layer(self, layer_id: int) -> int:
        return layer_id if self._layer_index is None else self._layer_index[layer_id]

    def _alloc(self, num_pages: int) -> None:
        self._num_pages = num_pages
        self._kv_buffer = torch.empty(
            (1, self._num_layers, num_pages, self._page_size, 1, self._latent_dim),
            device=self._device,
            dtype=self._dtype,
        )

    # -- views (addressed by GLOBAL layer id; remapped when layer_ids was given) --
    def k_cache(self, layer_id: int) -> torch.Tensor:
        """Paged latent view ``[num_pages, page_size, latent_dim]``."""
        return self._kv_buffer[0, self._local_layer(layer_id)].view(
            self._num_pages, self._page_size, -1
        )

    def v_cache(self, layer_id: int) -> torch.Tensor:
        # MLA: K == V (single latent); same buffer, dsv4_paged_pool precedent.
        return self.k_cache(layer_id)

    def latent_rows(self, layer_id: int) -> torch.Tensor:
        """Row-flat latent view ``[num_pages * page_size, latent_dim]``."""
        return self._kv_buffer[0, self._local_layer(layer_id)].view(-1, self._latent_dim)

    # -- writes -----------------------------------------------------------------
    def store_kv(
        self,
        c_kv: torch.Tensor,
        k_rope: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        """Scatter this forward's latent rows: ``c_kv`` [T, kv_lora_rank] and
        ``k_rope`` [T, qk_rope_head_dim] land in the row's two halves.

        Two narrow ``index_put_`` scatters. TODO: fuse into kernel/csrc
        store.cu (two-width store).
        """
        rows = self.latent_rows(layer_id)
        split = rows.shape[1] - k_rope.shape[-1]
        rows[out_loc, :split] = c_kv
        rows[out_loc, split:] = k_rope

    def rebuild(self, num_pages: int) -> None:
        """In-place resize (frees the old slab first; object identity preserved --
        callers re-derive views per forward, same contract as MHAKVCache.rebuild)."""
        self._kv_buffer = None
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
            torch.cuda.empty_cache()
        self._alloc(num_pages)

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token

        per_token = sum(
            spec_kv_bytes_per_token(spec, config)
            for spec in config.model_config.kv_cache_group_specs()
        )
        return per_token * config.page_size, 0, config.page_size, 0

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        self.rebuild(num_pages + 1)  # +1 for the dummy page (matches create_kvcache_pool)

    def unit_bytes(self) -> tuple[int, int]:
        buf = self._kv_buffer
        return int(buf.numel() * buf.element_size()) // (self._num_pages * self._page_size), 0

    # -- pool properties ----------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers


class DSAKVCache(MLAKVCache):
    """MLA latent pool + the DSA index-key slab (one row per token per full-indexer
    layer, bf16, slot order = the backend's full-layer order)."""

    def __init__(
        self,
        latent_dim: int,
        num_layers: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        index_head_dim: int,
        num_index_layers: int,
        layer_ids: "tuple[int, ...] | None" = None,
    ) -> None:
        self._index_head_dim = index_head_dim
        self._num_index_layers = num_index_layers
        super().__init__(
            latent_dim, num_layers, num_pages, page_size, dtype, device,
            layer_ids=layer_ids,
        )

    def _index_rows(self, num_pages: int) -> int:
        """Index-slab row count: one row per token (KpoolDSAKVCache overrides to the
        1/ratio shadow + scratch layout)."""
        return num_pages * self._page_size

    def _alloc(self, num_pages: int) -> None:
        # Both slabs in one allocation step so rebuild can never leave one grown
        # and the other stale.
        super()._alloc(num_pages)
        # bf16 == the 2 bytes/token/layer the KV cost model budgets for this slab
        # (cache_status._kv_cost_model); keep the two in lockstep.
        self._index_k_buffer = torch.zeros(
            self._num_index_layers,
            self._index_rows(num_pages),
            self._index_head_dim,
            dtype=torch.bfloat16,
            device=self._device,
        )

    def rebuild(self, num_pages: int) -> None:
        self._index_k_buffer = None
        super().rebuild(num_pages)

    def unit_bytes(self) -> tuple[int, int]:
        # The index slab rides the same token budget as the latent slab; each slab's per-token
        # cost is floor-divided on its own, matching the cost model's two separate terms.
        kv, swa = super().unit_bytes()
        idx = self._index_k_buffer
        tokens = self._num_pages * self._page_size
        return kv + int(idx.numel() * idx.element_size()) // tokens, swa

    def index_k_cache(self, slot: int) -> torch.Tensor:
        """Row-flat index keys for a full-indexer layer slot: ``[rows, index_head_dim]``."""
        return self._index_k_buffer[slot]

    def store_index_k(self, k: torch.Tensor, out_loc: torch.Tensor, slot: int) -> None:
        self._index_k_buffer[slot][out_loc] = k


class KpoolDSAKVCache(DSAKVCache):
    """DSAKVCache with the index slab as a 1/ratio SHADOW of the KV pages, plus
    per-request scratch rows and tail rings.

    A pooled entry lives at ``token_slot // index_ratio`` (``page_size %
    index_ratio == 0``), so compressed rows follow page sharing/eviction for
    free. Rows whose pool does not close write to the request's scratch row
    instead (scoring never reads it). The tail rings hold the in-progress
    pool's raw K + gate at ``pos % index_ratio`` for pools straddling two
    forwards; never cleared -- prefix-cache resume points are page-aligned,
    so a stale ring is never read.
    """

    def __init__(self, *args, num_req_slots: int, index_ratio: int, **kwargs) -> None:
        self._num_req_slots = num_req_slots
        self._index_ratio = index_ratio
        super().__init__(*args, **kwargs)
        assert self._page_size % index_ratio == 0, (
            f"kpool needs page_size ({self._page_size}) divisible by "
            f"index_ratio ({index_ratio})"
        )

    def _index_rows(self, num_pages: int) -> int:
        # 1/ratio shadow of every token slot + one scratch row per request slot.
        return num_pages * self._page_size // self._index_ratio + self._num_req_slots

    @property
    def cmp_scratch_base(self) -> int:
        """First scratch row (== shadow row count); request ``table_idx`` offsets it."""
        return self._num_pages * self._page_size // self._index_ratio

    def _alloc(self, num_pages: int) -> None:
        super()._alloc(num_pages)
        self._tail_k = torch.zeros(
            self._num_index_layers, self._num_req_slots, self._index_ratio,
            self._index_head_dim, dtype=torch.bfloat16, device=self._device,
        )
        self._tail_gate = torch.zeros_like(self._tail_k)

    def rebuild(self, num_pages: int) -> None:
        self._tail_k = None
        self._tail_gate = None
        super().rebuild(num_pages)

    def tail_k(self, slot: int) -> torch.Tensor:
        """Tail raw keys for an indexer layer slot: ``[num_req_slots, ratio, head_dim]``."""
        return self._tail_k[slot]

    def tail_gate(self, slot: int) -> torch.Tensor:
        return self._tail_gate[slot]


__all__ = ["MLAKVCache", "DSAKVCache", "KpoolDSAKVCache"]
