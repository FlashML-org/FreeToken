"""Pool-family sizing classmethods: dispatch and parity with the pure cost functions.

The engine sizes KV before the pool exists through resolve_pool_class(...).kv_cost /
solve_num_pages; these pins prove the family dispatch follows the group-spec attn_type
and that each family's arithmetic is the same pure function the old engine-side policy
used (byte-identical by delegation)."""

from types import SimpleNamespace

import pytest

from freetoken.attention import AttnType
from freetoken.kvcache import resolve_pool_class
from freetoken.models.config import KVCacheGroupSpec


def _spec(name, attn_type, *, mla=False, index_head_dim=0, sliding_window=None):
    return KVCacheGroupSpec(
        name=name, layer_ids=(0, 1), num_kv_heads=2, head_dim=64,
        sliding_window=sliding_window, mla=mla, index_head_dim=index_head_dim,
        num_index_layers=2 if index_head_dim else 0, attn_type=attn_type,
    )


def _model_config(specs, **attrs):
    mc = SimpleNamespace(has_swa_attention=False, dsv4_args=None, **attrs)
    mc.kv_cache_group_specs = lambda: specs
    return mc


def test_resolve_pool_class_follows_attn_type():
    from freetoken.kvcache.dsa_pool import DSAKVCache, MLAKVCache
    from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
    from freetoken.kvcache.mha_pool import MHAKVCache

    cases = [
        ((_spec("full", AttnType.FULL),), MHAKVCache),
        ((_spec("full", AttnType.FULL), _spec("swa", AttnType.SWA, sliding_window=128)),
         HybridSWAKVCache),
        ((_spec("full", AttnType.MLA, mla=True),), MLAKVCache),
        ((_spec("full", AttnType.DSA, mla=True, index_head_dim=128),), DSAKVCache),
        ((_spec("dsv4", AttnType.DSV4, sliding_window=128),), DSV4PagedKVCache),
    ]
    for specs, expected in cases:
        assert resolve_pool_class(_model_config(specs)) is expected

    # duck-typed configs without the spec walk: dsv4_args marks DSV4, else generic
    assert resolve_pool_class(SimpleNamespace(dsv4_args=object())) is DSV4PagedKVCache
    assert resolve_pool_class(SimpleNamespace()) is MHAKVCache


def _generic_config(num_page_override=None):
    mc = _model_config((_spec("full", AttnType.FULL),))
    mc.linear_attention_group = lambda: None
    return SimpleNamespace(
        model_config=mc, page_size=16, dtype=SimpleNamespace(itemsize=2),
        tp_info=SimpleNamespace(size=1), cache_type="radix",
        swa_full_tokens_ratio=1.0, swa_num_pages_override=None,
        num_page_override=num_page_override, max_running_req=4, max_seq_len=1024,
    )


def test_generic_kv_cost_and_solve_parity():
    from freetoken.kvcache.base import spec_kv_bytes_per_token
    from freetoken.kvcache.mha_pool import MHAKVCache

    config = _generic_config()
    (spec,) = config.model_config.kv_cache_group_specs()
    per_page = spec_kv_bytes_per_token(spec, config) * config.page_size
    # 2 slabs x head_dim x kv_heads x itemsize x layers -- the uniform-slab price
    assert per_page == 2 * 64 * 2 * 2 * 2 * config.page_size
    fixed = 0  # uniform slab: no fixed tier
    assert MHAKVCache.kv_cost(config) == (per_page, fixed, config.page_size, 0)
    # solve = (available - fixed) // per_page, override wins verbatim
    assert MHAKVCache.solve_num_pages(config, available_memory=per_page * 100 + fixed) == 100
    assert MHAKVCache.solve_num_pages(_generic_config(num_page_override=7), 0) == 7
    assert MHAKVCache.min_kv_tokens(config) == config.page_size


def _dsv4_config(num_page_override=None):
    from freetoken.models.deepseek_v4.args import DeepseekV4Args

    args = DeepseekV4Args()
    return SimpleNamespace(
        model_config=SimpleNamespace(dsv4_args=args),
        page_size=args.window_size, max_running_req=2, max_seq_len=4096, cache_type="radix",
        swa_full_tokens_ratio=1.0, swa_num_pages_override=None,
        num_page_override=num_page_override,
    )


def test_dsv4_kv_cost_and_floor_parity():
    from freetoken.kvcache.dsv4_cost_model import _dsv4_swa_ratio, _dsv4_window_floor_pages
    from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache

    pytest.importorskip("freetoken.kvcache.dsv4_cost_model")
    from freetoken.kvcache.dsv4_cost_model import dsv4_auto_cost_model

    config = _dsv4_config()
    P = 128
    floor = _dsv4_window_floor_pages(config, P)
    per_page, fixed, min_reserve = dsv4_auto_cost_model(
        config.model_config.dsv4_args, _dsv4_swa_ratio(config), floor, P=P,
        n_scratch=config.max_running_req + 1,
    )
    assert DSV4PagedKVCache.kv_cost(config) == (per_page, fixed, P, min_reserve)
    assert DSV4PagedKVCache.min_kv_tokens(config) == floor * P

    # an explicit --num-pages below the window working-set floor is a config-time error
    with pytest.raises(ValueError, match="window working-set floor"):
        DSV4PagedKVCache.solve_num_pages(_dsv4_config(num_page_override=1), 0)


def test_startup_kv_budget_composition():
    # The engine-side budget must stay ratio*old - (old - new); a sign/order slip here
    # mis-sizes every model's startup KV pool with the whole CPU suite green.
    from freetoken.engine.engine import _startup_kv_budget

    assert _startup_kv_budget(0.9, 1000, 400) == int(0.9 * 1000) - 600
    assert _startup_kv_budget(1.0, 1000, 1000) == 1000


def test_generic_validate_rebuild_budget_check():
    from freetoken.kvcache.base import CacheRebuildRejected
    from freetoken.kvcache.mha_pool import MHAKVCache

    config = _generic_config()
    per_page, fixed, _, _ = MHAKVCache.kv_cost(config)
    pool = object.__new__(MHAKVCache)  # generic validate_rebuild reads no instance state
    budget = per_page * 50 + fixed  # memory_ratio=1.0: exactly 50 pages fit

    def check(pages, baseline):
        pool.validate_rebuild(
            config, num_pages=pages, num_swa_pages=None,
            target_moe=0, per_expert_bytes=0, baseline_free=baseline,
            weights_bytes=0, current_num_pages=10,
        )

    object.__setattr__(config, "memory_ratio", 1.0)
    check(50, budget)  # fits exactly
    with pytest.raises(CacheRebuildRejected, match="old cache kept"):
        check(51, budget)
    # num_pages=None budgets the CURRENT page count
    check(None, per_page * 10 + fixed)


def test_dsv4_validate_rebuild_floor():
    from freetoken.kvcache.base import CacheRebuildRejected
    from freetoken.kvcache.dsv4_cost_model import _dsv4_window_floor_pages
    from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache

    config = _dsv4_config()
    pool = object.__new__(DSV4PagedKVCache)  # floor path reads no instance state
    floor = _dsv4_window_floor_pages(config, config.page_size)
    with pytest.raises(CacheRebuildRejected, match="working-set floor"):
        pool.validate_rebuild(
            config, num_pages=floor - 1, num_swa_pages=None,
            target_moe=0, per_expert_bytes=0, baseline_free=0,
            weights_bytes=0, current_num_pages=100,
        )


def test_rebuild_from_config_explicit_num_swa_pages_wins():
    # The explicit parameter must take precedence over config.swa_num_pages_override --
    # the engine's override write would otherwise mask a dropped parameter forever.
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache.dsv4_cost_model import _dsv4_pool_sizes
    from freetoken.kvcache.hybrid_swa_pool import _swa_paged_num_tokens, _swa_pool_floor

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

    swa_cfg = _generic_config()
    swa_specs = (
        _spec("full", AttnType.FULL),
        _spec("swa", AttnType.SWA, sliding_window=32),
    )
    swa_cfg.model_config.kv_cache_group_specs = lambda: swa_specs
    object.__setattr__(swa_cfg, "swa_num_pages_override", 999_999)
    floor = _swa_pool_floor(swa_cfg)
    explicit = 2 * floor  # above the floor so the pin is what we read back
    assert _swa_paged_num_tokens(swa_cfg, 11, num_swa_pages=explicit) == explicit + 1
    assert _swa_paged_num_tokens(swa_cfg, 11) == 999_999 + 1  # override path unchanged

    from freetoken.kvcache.dsv4_cost_model import _dsv4_window_floor_pages

    d_cfg = _dsv4_config()
    d_floor = _dsv4_window_floor_pages(d_cfg, d_cfg.page_size)
    pin = d_floor + 3  # above the floor so the pin (not the floor) is what we read back
    a = _dsv4_pool_sizes(d_cfg, pin + 40, num_swa_pages=pin)
    object.__setattr__(d_cfg, "swa_num_pages_override", 4)
    b = _dsv4_pool_sizes(d_cfg, pin + 40, num_swa_pages=pin)
    # +1 = the sentinel window page (same convention as _swa_paged_num_tokens's +1 slot)
    assert a.n_win_pages == b.n_win_pages == pin + 1  # explicit target wins over the override


def test_create_kv_pool_builds_the_right_family():
    import torch

    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache import create_kv_pool
    from freetoken.kvcache.hybrid_swa_pool import _swa_paged_num_tokens
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
    from freetoken.kvcache.mha_pool import MHAKVCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

    mc = SimpleNamespace(
        has_swa_attention=False, has_linear_attention=False, dsv4_args=None,
        num_layers=2, num_kv_heads=1, head_dim=8,
    )
    mc.kv_cache_group_specs = lambda: (_spec("full", AttnType.FULL),)
    config = SimpleNamespace(
        model_config=mc, page_size=1, cache_type="radix", max_running_req=2,
        swa_full_tokens_ratio=1.0, swa_num_pages_override=None, max_seq_len=64,
    )
    pool = create_kv_pool(config, num_pages=8, device=torch.device("cpu"), dtype=torch.bfloat16)
    assert isinstance(pool, MHAKVCache)

    import dataclasses

    swa_specs = (
        dataclasses.replace(_spec("full", AttnType.FULL), layer_ids=(0,)),
        dataclasses.replace(_spec("swa", AttnType.SWA, sliding_window=4), layer_ids=(1,)),
    )
    mc2 = SimpleNamespace(
        has_swa_attention=True, has_linear_attention=False, dsv4_args=None, num_layers=2,
    )
    mc2.kv_cache_group_specs = lambda: swa_specs
    config2 = SimpleNamespace(
        model_config=mc2, page_size=1, cache_type="swa_radix", max_running_req=2,
        swa_full_tokens_ratio=0.5, swa_num_pages_override=None, max_seq_len=64,
        max_extend_tokens=64,
    )
    pool2 = create_kv_pool(config2, num_pages=8, device=torch.device("cpu"), dtype=torch.bfloat16)
    assert isinstance(pool2, HybridSWAKVCache)
    assert pool2.swa_num_tokens == _swa_paged_num_tokens(config2, 9)


def test_linear_state_pool_prices_itself():
    # The GDN state pool is a sibling pool: the KV family's kv_cost excludes it and the
    # engine adds state_pool_bytes -- the sum must equal the old single-walk total.
    from freetoken.kvcache.linear_state_pool import (
        _linear_pool_num_slots, linear_state_bytes_per_req, state_pool_bytes,
    )
    from freetoken.models.config import LinearGatedDeltaGroupConfig

    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(1, 3), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    config = _generic_config()
    config.linear_state_cache_ratio = 0.5
    config.cache_type = "hybrid_radix"
    config.model_config.linear_attention_group = lambda: group
    per_req = linear_state_bytes_per_req(group, config.tp_info.size, config.dtype)
    assert state_pool_bytes(config) == per_req * _linear_pool_num_slots(config)
    assert state_pool_bytes(config, num_slots=7) == per_req * 7
    # models without a linear group price to zero
    config.model_config.linear_attention_group = lambda: None
    assert state_pool_bytes(config) == 0


def test_validate_rebuild_targets_flow_by_kv_cost_signature():
    """The base template hands each family's kv_cost exactly the non-None target keys its
    signature declares: hybrid's pinned window flows through and changes the verdict;
    uniform pools never see keys they don't declare (no TypeError on the same call)."""
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache.base import CacheRebuildRejected
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache, _swa_pool_floor
    from freetoken.kvcache.mha_pool import MHAKVCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

    config = _generic_config()
    specs = (
        _spec("full", AttnType.FULL),
        _spec("swa", AttnType.SWA, sliding_window=32),
    )
    config.model_config.kv_cache_group_specs = lambda: specs
    object.__setattr__(config, "cache_type", "swa_radix")
    object.__setattr__(config, "memory_ratio", 1.0)

    def check(cls, budget_tokens_kwargs, **targets):
        pool = object.__new__(cls)
        per_page, fixed, _, _ = cls.kv_cost(config, **budget_tokens_kwargs)
        pool.validate_rebuild(
            config, num_pages=10, target_moe=0, per_expert_bytes=0,
            baseline_free=10 * per_page + fixed, weights_bytes=0,
            current_num_pages=10, **targets,
        )

    # exactly-fitting budget passes with the pin priced in ...
    pin = 2 * _swa_pool_floor(config)
    check(HybridSWAKVCache, {"num_swa_pages": pin}, num_swa_pages=pin)
    # ... and the same budget WITHOUT the pin priced in rejects: the pin reached kv_cost
    with pytest.raises(CacheRebuildRejected):
        check(HybridSWAKVCache, {}, num_swa_pages=pin)

    # uniform pool: stray family keys are filtered, never a TypeError
    plain = _generic_config()
    object.__setattr__(plain, "memory_ratio", 1.0)
    per_page = MHAKVCache.kv_cost(plain)[0]
    pool = object.__new__(MHAKVCache)
    pool.validate_rebuild(
        plain, num_pages=10, target_moe=0, per_expert_bytes=0,
        baseline_free=10 * per_page, weights_bytes=0, current_num_pages=10,
        num_swa_pages=None, future_family_key=None,
    )
