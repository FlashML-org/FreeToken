"""MLA latent-KV pools (GLM-5.2 / DeepSeek-style latent attention).

``MLAKVCache`` is the MHA pool's sibling for latent-KV models: ONE slab holding the
per-token latent ``ckv (kv_lora_rank) | kpe (qk_rope_head_dim)`` -- there is no
separate V (``v_cache`` aliases ``k_cache``, same convention as dsv4_paged_pool's
single-latent tiers). ``DSAKVCache`` extends it with the DeepSeek-Sparse-Attention
index-key slab: one ``index_head_dim``-wide bf16 row per token per full-indexer
layer, addressed by the SAME physical rows as the latent slab (page_size == 1), and
``rebuild`` resizes BOTH slabs atomically so the allocator can never hand out a slot
one slab has and the other lacks.

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
    """

    def __init__(
        self,
        latent_dim: int,
        num_layers: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_ids=None,
    ) -> None:
        self._latent_dim = latent_dim
        # Hybrid linear-attention models (GLM-5.3: 34 KDA + 11 MLA/DSA) back only the
        # full-attention layers here; ``layer_ids`` remaps global layer id -> dense slab
        # (MHAKVCache pattern), so the (majority) linear layers cost no latent slab.
        if layer_ids is None:
            self._layer_map = None
        else:
            num_layers = len(layer_ids)
            m = [-1] * (max(layer_ids) + 1)
            for dense, g in enumerate(layer_ids):
                m[g] = dense
            self._layer_map = m
        self._num_layers = num_layers
        self._page_size = page_size
        self._dtype = dtype
        self._device = device
        self._alloc(num_pages)

    def _dense(self, layer_id: int) -> int:
        return layer_id if self._layer_map is None else self._layer_map[layer_id]

    def _alloc(self, num_pages: int) -> None:
        self._num_pages = num_pages
        self._kv_buffer = torch.empty(
            (1, self._num_layers, num_pages, self._page_size, 1, self._latent_dim),
            device=self._device,
            dtype=self._dtype,
        )

    # -- views ------------------------------------------------------------------
    def k_cache(self, layer_id: int) -> torch.Tensor:
        """Paged latent view ``[num_pages, page_size, latent_dim]``."""
        return self._kv_buffer[0, self._dense(layer_id)].view(self._num_pages, self._page_size, -1)

    def v_cache(self, layer_id: int) -> torch.Tensor:
        # MLA: K == V (single latent); same buffer, dsv4_paged_pool precedent.
        return self.k_cache(layer_id)

    def latent_rows(self, layer_id: int) -> torch.Tensor:
        """Row-flat latent view ``[num_pages * page_size, latent_dim]``."""
        return self._kv_buffer[0, self._dense(layer_id)].view(-1, self._latent_dim)

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

        v0: two narrow ``index_put_`` scatters. TODO: generalize kernel/csrc
        store.cu to a two-width fused store and route this through it.
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
        layer_ids=None,
        index_kpool: int = 1,
    ) -> None:
        self._index_head_dim = index_head_dim
        self._num_index_layers = num_index_layers
        self._index_kpool = index_kpool
        super().__init__(latent_dim, num_layers, num_pages, page_size, dtype, device, layer_ids=layer_ids)

    def _alloc(self, num_pages: int) -> None:
        # Both slabs in one allocation step: rebuild can never leave the pool with a
        # grown latent slab and a stale index slab (the OOB class this type exists for).
        super()._alloc(num_pages)
        # bf16 == the 2 bytes/token/layer the KV cost model budgets for this slab
        # (cache_status._kv_cost_model); keep the two in lockstep.
        self._index_k_buffer = torch.zeros(
            self._num_index_layers,
            num_pages * self._page_size,
            self._index_head_dim,
            dtype=torch.bfloat16,
            device=self._device,
        )
        # k-pool (GLM-5.3): raw gate-score slab (same row addressing as the key slab) and
        # the pooled-key slab, where pool p's key is stored at the PHYSICAL ROW OF ITS FIRST
        # TOKEN (rows[kpool*p]); scoring gathers via the strided pool-row snapshot.
        if self._index_kpool > 1:
            self._gate_buffer = torch.zeros_like(self._index_k_buffer)
            self._pool_k_buffer = torch.zeros_like(self._index_k_buffer)
        else:
            self._gate_buffer = None
            self._pool_k_buffer = None

    def rebuild(self, num_pages: int) -> None:
        self._index_k_buffer = None
        self._gate_buffer = None
        self._pool_k_buffer = None
        super().rebuild(num_pages)

    def unit_bytes(self) -> tuple[int, int]:
        # The index slab rides the same token budget as the latent slab; each slab's per-token
        # cost is floor-divided on its own, matching the cost model's two separate terms.
        kv, swa = super().unit_bytes()
        idx = self._index_k_buffer
        tokens = self._num_pages * self._page_size
        per = int(idx.numel() * idx.element_size()) // tokens
        # k-pool slabs (gate + pooled keys) ride the same token budget -- count them or the
        # sizing pass over-allocates and the alloc OOMs (caught by the dummy e2e run).
        if self._gate_buffer is not None:
            per += int(self._gate_buffer.numel() * self._gate_buffer.element_size()) // tokens
            per += int(self._pool_k_buffer.numel() * self._pool_k_buffer.element_size()) // tokens
        return kv + per, swa

    def index_k_cache(self, slot: int) -> torch.Tensor:
        """Row-flat index keys for a full-indexer layer slot: ``[rows, index_head_dim]``."""
        return self._index_k_buffer[slot]

    def store_index_k(self, k: torch.Tensor, out_loc: torch.Tensor, slot: int) -> None:
        self._index_k_buffer[slot][out_loc] = k

    # ----- k-pool slabs (GLM-5.3; None-guarded when index_kpool == 1) -----
    def gate_cache(self, slot: int) -> torch.Tensor:
        return self._gate_buffer[slot]

    def store_gate(self, g: torch.Tensor, out_loc: torch.Tensor, slot: int) -> None:
        self._gate_buffer[slot][out_loc] = g

    def pool_k_cache(self, slot: int) -> torch.Tensor:
        return self._pool_k_buffer[slot]

    def store_pool_k(self, k: torch.Tensor, rows: torch.Tensor, slot: int) -> None:
        """Scatter pooled keys at the pools' first-token physical rows (int64 rows)."""
        self._pool_k_buffer[slot][rows] = k


__all__ = ["MLAKVCache", "DSAKVCache"]
