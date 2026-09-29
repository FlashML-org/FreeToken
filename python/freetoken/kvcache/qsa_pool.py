"""QSA compressed-block sparse KV pool: paged GQA K/V + compressed index keys + pending ring.

Qwen3.8-Flash-Next scores whole ``index_ratio``-token groups instead of single tokens, so
its indexer slab holds ONE compressed key row per group, addressed by ``slot //
index_ratio``. Because ``page_size % index_ratio == 0``, a group's tokens always live in one
page at consecutive slots, which makes that division well-defined: the compressed rows are a
1/ratio shadow of the K/V pages and follow page sharing and eviction for free -- no
allocator, no free, no clear (SGLang qsa_kv_pool / vLLM compressed-region precedent).

Two tiers ride alongside the shadow slab and are NOT per-token:
- ``pending_ring``: the last ``ring_capacity`` pre-RoPE index keys of each running request (sized by ``ring_capacity_for``), indexed by ``Req.table_idx``. A group that straddles two forwards (chunked prefill, and
  every decode step) reads its already-consumed members from here. Never cleared: a new
  tenant of a table_idx starts at a group boundary (cached_len is 0 or a page multiple), so
  its first closing group takes every member from its own forward.
- scratch rows at ``cmp_scratch_base``: one row per request slot, the write target for rows
  whose group does not close in this forward, so the compress kernel scatters unconditionally
  with no negative index and no cross-row conflict (DSV4 precedent).

The slab is amortized into the per-token KV price (``unit_bytes``); the ring and scratch are
fixed and priced through ``kv_cost``'s ``fixed_cache_size``.

Placed on the host (``--kv-placement host``), the K/V slabs live in pinned host memory and only
the index tiers stay on the device, so the context is bounded by host RAM. Decode reads its
selection in place over PCIe. A prefill cannot: every query row reads its own selection, so a
chunk would move each cached token over the bus hundreds of times. It stages the batch's pages
into a device window once per layer instead (``staging_window``), a fixed device cost.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from freetoken.utils import init_logger, mem_GB

from .base import KVPlacement
from .mha_pool import MHAKVCache

logger = init_logger(__name__)

# The index tiers are always 2-byte (compute dtype); spec_kv_bytes_per_token budgets the same.
_INDEX_DTYPE_BYTES = 2
# Tokens of one layer's K/V a host-placed pool stages on the device for a prefill. Wider costs
# VRAM the expert cache could hold; a context past it is attended in several passes.
STAGING_TOKENS = 1 << 17
# t/h/w int32 rope position kept per KV slot on mrope models
_ROPE_POS_BYTES = 3 * 4


class QSAKVCache(MHAKVCache):
    """MHA paged pool + the compressed index-key slab + the per-request pending ring.

    ``cmp_k_cache(slot)`` is row-flat ``[num_pages * page_size // index_ratio + num_req_slots,
    index_head_dim]``: row ``r < cmp_scratch_base`` holds the compressed key of the token group
    whose K/V slots are ``[r * index_ratio, (r + 1) * index_ratio)``, and the rows from
    ``cmp_scratch_base`` on are the per-request-slot scratch sinks. ``slot`` is the sparse
    layer's order in the attention backend, same convention as BSAKVCache/DSAKVCache.
    """

    @classmethod
    def ring_capacity_for(cls, index_ratio: int, num_speculative_tokens: int = 0) -> int:
        """Ring depth: one row per pending position, keyed ``position % capacity``; spec decode widens by the draft depth (vLLM sizing)."""
        return index_ratio * math.ceil((index_ratio + num_speculative_tokens) / index_ratio)

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
        index_ratio: int,
        num_req_slots: int,
        ring_capacity: int | None = None,
        layer_ids: Sequence[int] | None = None,
        mrope: bool = False,
        placement: KVPlacement = "device",
        num_qo_heads: int = 0,
        prefill_rows: int = 0,
    ) -> None:
        """``num_qo_heads`` and ``prefill_rows`` size a host-placed pool's fold accumulator: the
        query heads of one attention layer, and the most rows one forward attends."""
        if placement == "host" and (num_qo_heads < 1 or prefill_rows < 1):
            raise ValueError(
                "a host-placed QSA pool needs num_qo_heads and prefill_rows for its fold accumulator"
            )
        if index_ratio < 1 or page_size % index_ratio != 0:
            # slot // index_ratio only names one group when a group never straddles a page.
            raise ValueError(
                f"QSA needs page_size ({page_size}) divisible by index_ratio ({index_ratio})"
            )
        if ring_capacity is None:
            ring_capacity = self.ring_capacity_for(index_ratio)
        if ring_capacity < index_ratio:
            # A closing group reads up to index_ratio - 1 past members plus this forward's.
            raise ValueError(
                f"QSA needs ring_capacity ({ring_capacity}) >= index_ratio ({index_ratio})"
            )
        # Index keys ride the compute dtype (the model's index_k is engine-dtype). The KV cost
        # model budgets 2 bytes per token per index layer for the slab
        # (base.spec_kv_bytes_per_token); keep the two in lockstep.
        assert dtype.itemsize == _INDEX_DTYPE_BYTES, (
            f"QSA index slab budgets 2 bytes/token (spec_kv_bytes_per_token); got {dtype}"
        )
        self._index_head_dim = index_head_dim
        self._num_index_layers = num_index_layers
        self._index_ratio = index_ratio
        self._num_req_slots = num_req_slots
        self._ring_capacity = ring_capacity
        self._index_dtype = dtype
        self._page_size = page_size
        self._mrope = mrope
        super().__init__(
            num_kv_heads=num_kv_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            num_pages=num_pages,
            page_size=page_size,
            dtype=dtype,
            device=device,
            layer_ids=layer_ids,
            placement=placement,
        )
        self._zero_kv_slabs()
        self._alloc_index_tiers(num_pages)
        self._staging: torch.Tensor | None = None
        self._fold: tuple[torch.Tensor, torch.Tensor] | None = None
        if placement == "host":
            self._staging = self._alloc_staging_window()
            self._fold = self._alloc_fold_workspace(prefill_rows, num_qo_heads, head_dim)
            logger.info(
                f"K/V in pinned host memory: {mem_GB(self.host_bytes)}; the device keeps the "
                f"index tiers, a {STAGING_TOKENS}-token staging window and a {prefill_rows}-row "
                "fold accumulator"
            )

    def _alloc_staging_window(self) -> torch.Tensor:
        _, _, _, page_size, kv_heads, head_dim = self._kv_buffer.shape
        return torch.zeros(
            (2, STAGING_TOKENS // page_size, page_size, kv_heads, head_dim),
            dtype=self._kv_buffer.dtype,
            device=self._device,
        )

    def _alloc_fold_workspace(
        self, rows: int, heads: int, head_dim: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """float32 output and log2-sum-exp rows a prefill spanning several windows folds its
        passes into; held for the pool's life so the deepest prefill allocates nothing more."""
        return (
            torch.zeros((rows, heads, head_dim), dtype=torch.float32, device=self._device),
            torch.zeros((2, rows, heads), dtype=torch.float32, device=self._device),
        )

    def _zero_kv_slabs(self) -> None:
        # Defense-in-depth: the attend kernels pos-mask every K/V load (the real fix for
        # torch.empty's recycled NaN/Inf bit patterns), but a zeroed slab keeps any future
        # unmasked read finite instead of model-poisoning. One memset per (re)allocation.
        self._kv_buffer.zero_()

    def _alloc_index_tiers(self, num_pages: int) -> None:
        # ZERO-initialized: the score kernel reads whole rows of blocks unmasked and relies on
        # never-written tail rows dotting to a finite 0. Written rows are never cleared again,
        # so the kernel must clamp visible blocks to kvlen // index_ratio.
        self._cmp_scratch_base = num_pages * self._page_size // self._index_ratio
        self._cmp_k_buffer = torch.zeros(
            self._num_index_layers,
            self._cmp_scratch_base + self._num_req_slots,
            self._index_head_dim,
            dtype=self._index_dtype,
            device=self._device,
        )
        self._pending_ring = torch.zeros(
            self._num_req_slots,
            self._num_index_layers,
            self._ring_capacity,
            self._index_head_dim,
            dtype=self._index_dtype,
            device=self._device,
        )
        # 3-axis rope position of every stored token: a compressed group ropes at its first token, which under mrope is not derivable from the logical position
        self._rope_positions = (
            torch.zeros(num_pages * self._page_size, 3, dtype=torch.int32, device=self._device)
            if self._mrope
            else None
        )

    def rebuild(self, num_pages: int) -> None:
        # Free the index tiers BEFORE the K/V realloc (super().rebuild frees + syncs +
        # empty_cache), then re-derive them at the new page count. If the index alloc itself
        # fails (OOM), null the K/V slab too and re-raise: a pool with a grown K/V slab and no
        # index slab would mis-serve silently. Rebuild is idle-only, so zeroing the ring here
        # cannot drop a live request's pending members.
        self._cmp_k_buffer = None
        self._pending_ring = None
        self._rope_positions = None
        super().rebuild(num_pages)
        self._zero_kv_slabs()
        try:
            self._alloc_index_tiers(num_pages)
        except Exception:
            self._kv_buffer = None
            self._k_buffer = None
            self._v_buffer = None
            raise

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        """Device bytes only: host-placed K/V costs none per token, and its staging window and
        fold accumulator are fixed."""
        from .base import spec_index_bytes_per_token, spec_kv_slab_bytes_per_token
        from freetoken.attention import AttnType

        num_req_slots = config.max_running_req + 1
        per_token = 0
        fixed = 0
        for spec in config.model_config.kv_cache_group_specs():
            if spec.is_swa:
                continue
            slab = spec_kv_slab_bytes_per_token(spec, config)
            per_token += spec_index_bytes_per_token(spec)
            if config.kv_placement == "host":
                fixed += slab * STAGING_TOKENS // spec.num_layers
                # float32: the output rows [rows, heads, head_dim] and two lse rows [rows, heads]
                fixed += config.max_forward_len * config.model_config.num_qo_heads * (spec.head_dim + 2) * 4
            else:
                per_token += slab
            if spec.attn_type is AttnType.QSA:
                # One index-key row = all index layers at one position.
                row = spec.index_head_dim * spec.num_index_layers * _INDEX_DTYPE_BYTES
                fixed += num_req_slots * row * (cls.ring_capacity_for(spec.index_ratio) + 1)
                if config.model_config.model_is_mrope:
                    per_token += _ROPE_POS_BYTES
        return per_token * config.page_size, fixed, config.page_size, 0

    def unit_bytes(self) -> tuple[int, int]:
        # Only the shadow slab scales with pages, and only its non-scratch rows; the ring, the
        # scratch rows and a host pool's staging window are the fixed term kv_cost reports.
        kv, swa = super().unit_bytes()
        tokens = int(self._kv_buffer.shape[2]) * int(self._kv_buffer.shape[3])
        slab = (
            self._num_index_layers
            * self._cmp_scratch_base
            * self._index_head_dim
            * self._index_dtype.itemsize
        )
        return kv + slab // tokens + (_ROPE_POS_BYTES if self._mrope else 0), swa

    def staging_window(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Device K and V pages ``[pages, page_size, kv_heads, head_dim]`` a host-placed pool's
        prefill gathers one layer's pages into; shared by every layer, one at a time."""
        assert self._staging is not None, "only a host-placed pool stages its K/V"
        return self._staging[0], self._staging[1]

    def fold_workspace(self, rows: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """float32 ``total [rows, heads, head_dim]``, ``total_lse`` and ``part_lse [rows, heads]``
        a host-placed prefill folds its window passes into; the caller initializes them."""
        assert self._fold is not None, "only a host-placed pool folds window passes"
        total, lse = self._fold
        assert rows <= total.shape[0], f"{rows} rows exceed the {total.shape[0]} prefill rows"
        return total[:rows], lse[0, :rows], lse[1, :rows]

    @property
    def host_bytes(self) -> int:
        """Pinned host bytes the K/V slabs hold; zero on the device."""
        if self.placement == "device":
            return 0
        return self._kv_buffer.numel() * self._kv_buffer.element_size()

    def cmp_k_cache(self, slot: int) -> torch.Tensor:
        """Compressed index keys of one sparse layer: ``[rows, index_head_dim]``."""
        return self._cmp_k_buffer[slot]

    def pending_ring(self, slot: int) -> torch.Tensor:
        """One sparse layer's pending ring: ``[num_req_slots, ring_capacity, index_head_dim]``."""
        return self._pending_ring[:, slot]

    @property
    def rope_positions(self) -> torch.Tensor:
        """``[num_tokens, 3]`` int32 t/h/w rope position per KV slot (written by the QSA backend)."""
        assert self._rope_positions is not None, "rope positions are only kept on mrope models"
        return self._rope_positions

    @property
    def cmp_scratch_base(self) -> int:
        """First scratch row of ``cmp_k_cache``; row ``cmp_scratch_base + table_idx`` sinks a
        forward whose group does not close."""
        return self._cmp_scratch_base

    @property
    def index_ratio(self) -> int:
        return self._index_ratio

    @property
    def index_head_dim(self) -> int:
        return self._index_head_dim

    @property
    def ring_capacity(self) -> int:
        return self._ring_capacity

    @property
    def num_req_slots(self) -> int:
        return self._num_req_slots


__all__ = ["QSAKVCache"]
