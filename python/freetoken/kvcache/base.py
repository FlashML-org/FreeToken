from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import torch
from freetoken.utils import div_even, init_logger, mem_GB

logger = init_logger(__name__)


class CacheRebuildRejected(Exception):
    """A runtime cache rebuild was rejected BEFORE any destructive free (e.g. the
    requested geometry does not fit). The old caches are intact and serving continues --
    this is recoverable, unlike a failure after the free."""


def spec_kv_bytes_per_token(spec, config) -> int:
    """One paged-KV group's bytes per token: (1|2 slabs) x head_dim x local kv heads x dtype
    x layers, plus the bf16 DSA index-key slab when the spec carries indexer dims. Pure
    per-spec arithmetic -- pool families compose it over THEIR OWN groups; no family
    branching here. (2 bytes/elem == the torch.bfloat16 dsa_pool.DSAKVCache._alloc
    hardcodes; keep the two in lockstep if the slab dtype ever changes.)"""
    per_token = (
        (1 if spec.mla else 2)  # MLA latent groups store one slab (V aliases K)
        * spec.head_dim
        * div_even(spec.num_kv_heads, config.tp_info.size, allow_replicate=True)
        * config.dtype.itemsize
        * spec.num_layers
    )
    index_bytes = spec.index_head_dim * spec.num_index_layers * 2
    if spec.attn_type.value == "qsa":
        index_bytes *= spec.index_num_kv_heads
        if index_bytes % spec.index_compress_ratio:
            raise ValueError("QSA index bytes must divide evenly by its compression ratio")
        index_bytes //= spec.index_compress_ratio
    return per_token + index_bytes


class BaseKVCachePool(ABC):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used.
    """

    # Pools whose buffers are bound into per-forward model scratch (DSV4's tiers) need the
    # model re-bound after a rebuild; the engine asks before it resizes.
    needs_rebind_on_rebuild: ClassVar[bool] = False

    # ---- sizing/cost classmethods: run BEFORE the pool exists (startup budget solve,
    # --moe-cache-auto). The engine measures memory and passes bytes in; each pool family
    # implements kv_cost for ITS OWN buffers only (the engine sums families, e.g. adds
    # LinearStatePool.state_pool_bytes for hybrid-GDN models); solve_num_pages and
    # validate_rebuild are family-agnostic templates over cls.kv_cost.

    @classmethod
    @abstractmethod
    def kv_cost(cls, config, **kwargs) -> tuple[int, int, int, int]:
        """``(cache_per_page, fixed_cache_size, page_tokens, min_reserve_tokens)`` for this
        pool family's OWN buffers, priced from the group specs / model payload. No other
        pool's terms (GDN state, MoE) belong here. Extra keywords are each family's own
        convention -- the base declares none; window-bearing pools take
        ``num_swa_pages=`` (a rebuild's target window, priced before the config override
        is written)."""

    @classmethod
    def solve_num_pages(cls, config, available_memory: int) -> int:
        """Largest usable page count fitting ``available_memory`` (bytes measured by the
        engine: memory_ratio x baseline free minus resident weights and any sibling pool's
        fixed cost), honoring ``config.num_page_override``."""
        cache_per_page, fixed_cache_size, _, _ = cls.kv_cost(config)
        num_pages = config.num_page_override
        if num_pages is None:
            num_pages = (available_memory - fixed_cache_size) // cache_per_page
        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        real_kv_size = num_pages * cache_per_page + fixed_cache_size
        logger.info(
            f"Allocating {num_pages * config.page_size} tokens for KV cache, "
            f"K + V = {mem_GB(real_kv_size)}"
        )
        return num_pages

    @classmethod
    def min_kv_tokens(cls, config) -> int:
        # rebuild rejects num_pages <= 0, so the floor is one page's worth of tokens.
        return int(config.page_size)

    def validate_rebuild(
        self, config, *, num_pages: int | None, target_moe: int, per_expert_bytes: int,
        baseline_free: int, weights_bytes: int, current_num_pages: int,
        extra_fixed_bytes: int = 0, extra_note: str = "", **targets,
    ) -> None:
        """Budget fit-check for a runtime rebuild target, BEFORE any destructive free.
        The engine supplies the memory account (baseline/weights, the MoE terms, and any
        sibling pool's fixed bytes at ITS target, e.g. the GDN state pool); the pool
        answers whether its own target geometry fits. Raises CacheRebuildRejected.

        The base names no family-specific target: the engine forwards the whole request
        bag through ``**targets``, overrides declare the keys they consume (DSV4:
        num_swa_pages/moe_cache_size), and this template hands each family's kv_cost
        exactly the non-None keys its signature declares -- so a new pool family adds a
        sizing target by declaring it on ITS kv_cost, with no base-signature churn."""
        import inspect

        from freetoken.engine.cache_budget import net_cache_budget_bytes, required_bytes

        target_pages = num_pages if num_pages is not None else current_num_pages
        cost_params = inspect.signature(type(self).kv_cost).parameters
        cost_kwargs = {
            k: v for k, v in targets.items() if v is not None and k in cost_params
        }
        cache_per_page, fixed_cache_size, _, _ = type(self).kv_cost(config, **cost_kwargs)
        budget = net_cache_budget_bytes(
            config.memory_ratio, baseline_free, weights_bytes,
            fixed_cache_size + extra_fixed_bytes,
        )
        need = required_bytes(target_moe, target_pages, per_expert_bytes, cache_per_page)
        if need > budget:
            raise CacheRebuildRejected(
                f"requested cache (moe={target_moe} slots, kv={target_pages} pages{extra_note}) "
                f"needs {mem_GB(need)} > budget {mem_GB(budget)}; old cache kept, still serving"
            )

    @abstractmethod
    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        """Resize to ``num_pages`` USABLE pages in place, identity-preserving (callers keep
        holding this object and re-derive views per forward). The dummy page and every
        secondary tier (window pool, index slab, state rings) are the pool's own to derive.
        Uniform-pool families (full/MLA/DSA) size from ``num_pages x page_size`` alone and
        ignore ``num_swa_pages``; window-bearing pools (SWA hybrid, DSV4) take the target
        window explicitly here, falling back to config's override/ratio when None."""

    def attach_page_table(self, page_table: torch.Tensor) -> None:
        """Called once at engine init and again on every page-table realloc. Pools routed by
        the shared table but deriving reads through their own mapping (DSV4) re-point here."""

    @abstractmethod
    def unit_bytes(self) -> tuple[int, int]:
        """``(kv_bytes_per_token, swa_bytes_per_token)`` of the live pool -- the per-unit VRAM
        cost the cache sliders denominate in. 0 for a tier this pool does not own."""

    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...


@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    cached_len: int

    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor: ...


class SizeInfo(NamedTuple):
    evictable_size: int
    protected_size: int

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size


class InsertResult(NamedTuple):
    cached_len: int  # length already in cache before insertion (should be freed)
    handle: BaseCacheHandle  # cache handle for the inserted prefix


class MatchResult(NamedTuple):
    cuda_handle: BaseCacheHandle
    # Hybrid (GDN) models: the restored GDN state snapshot slot for this prefix (None = cold /
    # non-hybrid). Surfaced by HybridRadixCache via CacheManager.match_req.
    mamba_value: int | None = None
    # TODO: support HiCache


class BasePrefixCache(ABC):
    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        Lock or unlock a cache handle.
        This operation will not modify the cache, but change the size info only.
        When a handle is locked, it cannot be evicted.
        Handles must be locked before the previously-returned tensor of `match_prefix` is used.
        Otherwise it may be evicted by calling evict.

        Args:
            handle (BaseCacheHandle): The cache handle to lock or unlock.
            unlock (bool): Whether to unlock the handle. Defaults to False.
        """

    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """
        Match prefix and return the indices of the matched prefix in the cache.
        This operation will not modify the cache.
        The returned indices is only safe to use when the handle is locked.

        Args:
            input_ids (torch.Tensor): The input ids to match. Shape: (seq_len,)
        Returns:
            MatchResult: The match result containing the cache handles.
        """

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """
        Insert a new prefix into the cache.
        This operation will modify the cache.
        Args:
            input_ids (torch.Tensor): The input ids to insert. Shape: (seq_len,)
            indices (torch.Tensor): The indices to store the new prefix. Shape: (seq_len,)

        Returns:
            InsertResult: The result of the insertion.
        """

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        """
        Evict some prefixes from the cache to free up space.
        This operation will modify the cache.
        Note that evict 0 is always safe and does nothing.
        Note that the actual evict size may be larger than the requested size.
        Args:
            size (int): The size to evict.

        Returns:
            torch.Tensor: The indices evicted. Shape: (evict_size,)
        Raises:
            RuntimeError: If the requested size is larger than the evictable size.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the cache manager and the underlying cache."""

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo:
        """Get the size information of the cache."""

    @abstractmethod
    def check_integrity(self) -> None:
        """Check the integrity of the cache. Raise an error if the cache is corrupted."""
