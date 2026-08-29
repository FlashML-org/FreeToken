from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from freetoken.utils import Registry

if TYPE_CHECKING:
    import torch
    from freetoken.models import ModelConfig

from .base import (
    BaseCacheHandle,
    BaseKVCachePool,
    BasePrefixCache,
    MatchResult,
    SizeInfo,
)


class CacheManagerCreator(Protocol):
    def __call__(self, device: torch.device) -> BasePrefixCache: ...


SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


def resolve_pool_class(model_config: ModelConfig) -> type[BaseKVCachePool]:
    """attn_type -> KV pool family, the dispatch shared by ``create_kv_pool`` and the
    engine's pre-pool sizing calls (the classmethod cost/solve surface). Driven by the
    group-spec walk (same source as the backend capability matrix); getattr fallbacks
    cover duck-typed test configs that don't implement it."""
    from freetoken.attention import AttnType

    specs_fn = getattr(model_config, "kv_cache_group_specs", None)
    if specs_fn is None:
        if getattr(model_config, "dsv4_args", None) is not None:
            from .dsv4_paged_pool import DSV4PagedKVCache

            return DSV4PagedKVCache
        from .mha_pool import MHAKVCache

        return MHAKVCache
    types = {spec.attn_type for spec in specs_fn()}
    if AttnType.DSV4 in types:
        from .dsv4_paged_pool import DSV4PagedKVCache

        return DSV4PagedKVCache
    if AttnType.SWA in types:
        from .hybrid_swa_pool import HybridSWAKVCache

        return HybridSWAKVCache
    if AttnType.DSA in types:
        from .dsa_pool import DSAKVCache

        return DSAKVCache
    if AttnType.MLA in types:
        from .dsa_pool import MLAKVCache

        return MLAKVCache
    if AttnType.QSA in types:
        from .qsa_pool import QSAKVCache

        return QSAKVCache
    if AttnType.BSA in types:
        from .bsa_pool import BSAKVCache

        return BSAKVCache
    from .mha_pool import MHAKVCache

    return MHAKVCache


def create_kv_pool(config, num_pages: int, device: torch.device, dtype: torch.dtype):
    """Build the engine's KV pool for ``num_pages`` USABLE pages (the dummy page and every
    secondary tier -- window pool, index slab, state rings -- are derived here or inside
    the pool). Single factory entry for all pool families, DSV4 included."""
    from .dsv4_cost_model import _dsv4_pool_sizes
    from .hybrid_swa_pool import _naive_swa_num_tokens, _swa_paged_num_tokens
    from .dsv4_paged_pool import DSV4PagedKVCache

    model_config = config.model_config
    if resolve_pool_class(model_config) is DSV4PagedKVCache:
        # DSV4 is driven by the generic CacheManager over the shared page table; the pool is
        # the only DSV4-specific piece (the swa_pool plug-in: window tier + cmp/idx/state
        # shadows). Sizing reads dsv4_args, never the group spec.
        pool = DSV4PagedKVCache(
            sizes=_dsv4_pool_sizes(config, num_pages + 1),  # +1 for dummy page
            args=model_config.dsv4_args,
            device=device,
            dtype=dtype,
            P=model_config.dsv4_args.window_size,
            n_scratch=config.max_running_req + 1,
        )
        pool._init_paged_state(config.max_running_req, config.cache_type != "naive")
        return pool

    num_swa_tokens = None
    # Both the naive and radix SWA paths share the global-paged swa pool; radix sizes it by
    # ratio (cross-request reuse), naive by concurrency x window.
    if model_config.has_swa_attention:
        num_swa_tokens = (
            _swa_paged_num_tokens(config, num_pages + 1)
            if config.cache_type == "swa_radix"
            else _naive_swa_num_tokens(config)
        )
    return create_kvcache_pool(
        model_config=model_config,
        num_pages=num_pages + 1,  # +1 for dummy page
        page_size=config.page_size,
        num_swa_tokens=num_swa_tokens,
        device=device,
        dtype=dtype,
        num_req_slots=config.max_running_req + 1,  # + 1 for the dummy request row
        quant=getattr(config, "kv_quant", None),
    )


def create_kvcache_pool(
    model_config: ModelConfig,
    num_pages: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
    num_swa_tokens: int | None = None,
    num_req_slots: int | None = None,
    quant=None,
) -> BaseKVCachePool:
    from .quant import NONE

    quant = quant if quant is not None else NONE
    if model_config.has_swa_attention:
        from .hybrid_swa_pool import HybridSWAKVCache

        return HybridSWAKVCache(
            groups=model_config.kv_cache_group_specs(),
            num_layers=model_config.num_layers,
            num_full_pages=num_pages,
            page_size=page_size,
            num_swa_tokens=num_swa_tokens,
            device=device,
            dtype=dtype,
            quant=quant,
        )

    from .mha_pool import MHAKVCache

    # Hybrid linear-attention models (e.g. Qwen3.5 GatedDeltaNet) only store paged KV
    # for their full-attention layers; the linear layers keep a separate recurrent
    # state. Back just those layers and remap their global ids to dense storage slots
    # so we don't over-allocate slabs for the (majority) linear layers.
    layer_ids: tuple[int, ...] | None = None
    num_kv_heads = model_config.num_kv_heads
    head_dim = model_config.head_dim
    if model_config.has_linear_attention:
        specs = [s for s in model_config.kv_cache_group_specs() if s.num_layers > 0]
        assert len(specs) == 1, f"expected one paged-KV group, got {[s.name for s in specs]}"
        spec = specs[0]
        layer_ids = spec.layer_ids
        num_kv_heads = spec.num_kv_heads
        head_dim = spec.head_dim

    # Latent-KV MLA models declare it on their single full-attention group: they get
    # the latent pool (one slab, V aliases K), plus the DSA index-key slab when the
    # spec carries indexer dims. The same spec fields drive the KV cost model, so the
    # factory and the budget can never disagree.
    kv_specs = model_config.kv_cache_group_specs()

    # GQA block-sparse (MiniMax-M3): one full-attention group carrying the index dims
    # with mla=False -> the MHA pool plus the index-key slab. The same spec fields
    # drive the KV cost model, so the factory and the budget can never disagree.
    from freetoken.attention import AttnType as _AttnType

    if len(kv_specs) == 1 and kv_specs[0].attn_type == _AttnType.BSA:
        from .bsa_pool import BSAKVCache

        spec = kv_specs[0]
        return BSAKVCache(
            num_kv_heads=spec.num_kv_heads,
            num_layers=model_config.num_layers,
            head_dim=spec.head_dim,
            num_pages=num_pages,
            page_size=page_size,
            dtype=dtype,
            device=device,
            index_head_dim=spec.index_head_dim,
            num_index_layers=spec.num_index_layers,
        )

    # QSA (Qwen3.8-Flash-Next): the same GQA group, but it stores one index key per
    # index_ratio tokens and adds per-request tiers sized by the concurrency, not the pages.
    # layer_ids is mandatory here -- the model is hybrid-linear, and letting MHAKVCache back
    # all num_layers would allocate K/V slabs for the GDN layers too.
    if len(kv_specs) == 1 and kv_specs[0].attn_type == _AttnType.QSA:
        from .qsa_pool import QSAKVCache

        spec = kv_specs[0]
        if num_req_slots is None:
            raise ValueError("QSA pools need num_req_slots (max_running_req + 1)")
        return QSAKVCache(
            num_kv_heads=spec.num_kv_heads,
            num_layers=model_config.num_layers,
            head_dim=spec.head_dim,
            num_pages=num_pages,
            page_size=page_size,
            dtype=dtype,
            device=device,
            index_head_dim=spec.index_head_dim,
            num_index_layers=spec.num_index_layers,
            index_ratio=spec.index_ratio,
            num_req_slots=num_req_slots,
            layer_ids=spec.layer_ids,
        )

    if len(kv_specs) == 1 and kv_specs[0].mla:
        from .dsa_pool import DSAKVCache, MLAKVCache

        spec = kv_specs[0]
        if spec.index_head_dim > 0 and spec.num_index_layers > 0:
            return DSAKVCache(
                latent_dim=spec.head_dim,
                num_layers=model_config.num_layers,
                num_pages=num_pages,
                page_size=page_size,
                dtype=dtype,
                device=device,
                index_head_dim=spec.index_head_dim,
                num_index_layers=spec.num_index_layers,
            )
        return MLAKVCache(
            latent_dim=spec.head_dim,
            num_layers=model_config.num_layers,
            num_pages=num_pages,
            page_size=page_size,
            dtype=dtype,
            device=device,
        )

    return MHAKVCache(
        num_kv_heads=num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_layers,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
        layer_ids=layer_ids,
        quant=quant,
    )


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache(device: torch.device, page_size: int | None = None):
    from .naive_cache import NaivePrefixCache

    return NaivePrefixCache(device=device)  # naive has no page arithmetic


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(device: torch.device, page_size: int | None = None):
    from .radix_cache import RadixPrefixCache

    return RadixPrefixCache(device=device, page_size=page_size)


# NOTE: "hybrid_radix" is NOT registered as a user-facing --cache-type. It is the internal
# materialization of "radix" for hybrid GDN models (cross-request GDN-state reuse), produced by
# _resolve_cache_type and built directly in CacheManager._make_prefix_cache (HybridRadixCache
# needs page_size). Users pick "radix" (the concept) or "naive"; the engine picks hybrid_radix.


def create_prefix_cache(
    device: torch.device, type: str, page_size: int | None = None
) -> BasePrefixCache:
    return SUPPORTED_CACHE_MANAGER[type](device, page_size=page_size)


__all__ = [
    "create_kv_pool",
    "create_kvcache_pool",
    "create_prefix_cache",
    "resolve_pool_class",
    "BaseKVCachePool",
    "BaseCacheHandle",
    "BasePrefixCache",
    "SizeInfo",
    "MatchResult",
    "SUPPORTED_CACHE_MANAGER",
]
