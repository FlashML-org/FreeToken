from __future__ import annotations

import torch

from freetoken.distributed import set_tp_info, try_get_tp_info


def _init_tp() -> None:
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _mha_pool(num_pages=4):
    from freetoken.kvcache.mha_pool import MHAKVCache

    _init_tp()
    return MHAKVCache(
        num_kv_heads=8, num_layers=3, head_dim=64,
        num_pages=num_pages, page_size=16, dtype=torch.float16, device=torch.device("cpu"),
    )


def test_mha_kv_cache_rebuild_changes_pages_preserves_identity():
    pool = _mha_pool()
    buf_id_before = id(pool)
    assert pool._kv_buffer.shape == (2, 3, 4, 16, 8, 64)

    pool.rebuild(10)

    assert id(pool) == buf_id_before  # same object
    assert pool._kv_buffer.shape == (2, 3, 10, 16, 8, 64)
    # views and storage shape refreshed
    assert pool._k_buffer.shape == (3, 10, 16, 8, 64)
    assert pool._v_buffer.shape == (3, 10, 16, 8, 64)
    assert pool._storage_shape == (10 * 16, 8, 64)
    # k_cache view derives from the new buffer
    assert pool.k_cache(0).shape == (10, 16, 8, 64)


def test_mha_rebuild_from_config_adds_the_dummy_page():
    # The engine speaks USABLE pages; the +1 dummy page is the pool's own.
    pool = _mha_pool()
    pool.rebuild_from_config(config=None, num_pages=10)
    assert pool._kv_buffer.shape[2] == 11
    # per-token cost is page-count invariant
    assert pool.unit_bytes() == (2 * 3 * 8 * 64 * 2, 0)


def test_mla_and_dsa_rebuild_from_config_and_unit_bytes():
    from freetoken.kvcache.dsa_pool import DSAKVCache, MLAKVCache

    latent, idx_dim, layers, n_idx = 80, 32, 2, 1
    mla = MLAKVCache(latent_dim=latent, num_layers=layers, num_pages=8, page_size=1,
                     dtype=torch.bfloat16, device=torch.device("cpu"))
    mla.rebuild_from_config(config=None, num_pages=20)
    assert mla.latent_rows(0).shape[0] == 21  # 20 usable + 1 dummy page
    assert mla.unit_bytes() == (layers * latent * 2, 0)

    dsa = DSAKVCache(latent_dim=latent, num_layers=layers, num_pages=8, page_size=1,
                     dtype=torch.bfloat16, device=torch.device("cpu"),
                     index_head_dim=idx_dim, num_index_layers=n_idx)
    dsa.rebuild_from_config(config=None, num_pages=20)
    assert dsa.latent_rows(0).shape[0] == 21 and dsa.index_k_cache(0).shape[0] == 21
    # the index slab's per-token bytes ride on top of the latent slab's, each floored on its own
    assert dsa.unit_bytes() == (layers * latent * 2 + n_idx * idx_dim * 2, 0)


def _hybrid_groups():
    from freetoken.models.config import KVCacheGroupSpec

    full = KVCacheGroupSpec(name="full", layer_ids=(0, 2), num_kv_heads=8, head_dim=64, sliding_window=None)
    swa = KVCacheGroupSpec(name="swa", layer_ids=(1,), num_kv_heads=8, head_dim=64, sliding_window=128)
    return [full, swa]


def test_hybrid_swa_rebuild_resizes_both_groups_preserves_identity():
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache

    _init_tp()
    pool = HybridSWAKVCache(
        groups=_hybrid_groups(), num_layers=3, num_full_pages=4, page_size=16,
        num_swa_tokens=32, dtype=torch.float16, device=torch.device("cpu"),
    )
    pool_id = id(pool)
    mapping_before = pool.layers_mapping
    assert pool.full_kv_pool.buffer.shape == (2, 2, 4, 16, 8, 64)
    assert pool.swa_kv_pool.buffer.shape == (2, 1, 32, 1, 8, 64)

    pool.rebuild(num_full_pages=10, num_swa_tokens=48)

    assert id(pool) == pool_id
    assert pool.layers_mapping is mapping_before  # mapping unchanged
    assert pool.full_kv_pool.buffer.shape == (2, 2, 10, 16, 8, 64)
    assert pool.swa_kv_pool.buffer.shape == (2, 1, 48, 1, 8, 64)
    assert pool._full_num_tokens == 10 * 16
    assert pool._swa_num_tokens == 48
    assert pool.full_kv_pool.storage_shape == (10 * 16, 8, 64)
    assert pool.swa_kv_pool.storage_shape == (48, 8, 64)
    # storages dict points at the new pools
    assert pool._storages["full"] is pool.full_kv_pool
    assert pool._storages["swa"] is pool.swa_kv_pool


def _swa_config(cache_type: str):
    """The EngineConfig slice HybridSWAKVCache.rebuild_from_config reads."""
    from types import SimpleNamespace

    groups = _hybrid_groups()
    return SimpleNamespace(
        page_size=16,
        cache_type=cache_type,
        max_running_req=2,
        max_forward_len=64,
        swa_full_tokens_ratio=0.5,
        swa_num_pages_override=None,
        model_config=SimpleNamespace(
            kv_cache_group_specs=lambda: groups,
            swa_attention_group=lambda: groups[1],
        ),
    )


def test_hybrid_swa_rebuild_from_config_derives_the_window_per_cache_type():
    """The window size the engine used to compute for the pool: ratio x full for radix,
    concurrency x window for naive."""
    from freetoken.kvcache.hybrid_swa_pool import _naive_swa_num_tokens, _swa_paged_num_tokens
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache

    _init_tp()
    for cache_type, expected in (
        ("swa_radix", lambda cfg: _swa_paged_num_tokens(cfg, 11)),
        ("naive", _naive_swa_num_tokens),
    ):
        pool = HybridSWAKVCache(
            groups=_hybrid_groups(), num_layers=3, num_full_pages=4, page_size=16,
            num_swa_tokens=32, dtype=torch.float16, device=torch.device("cpu"),
        )
        config = _swa_config(cache_type)
        pool.rebuild_from_config(config, 10)
        assert pool.full_kv_pool.buffer.shape[2] == 11  # 10 usable + 1 dummy page
        assert pool.swa_num_tokens == expected(config)
        assert pool.swa_kv_pool.buffer.shape[2] == pool.swa_num_tokens
        assert pool.unit_bytes() == (2 * 2 * 8 * 64 * 2, 2 * 1 * 8 * 64 * 2)


def test_linear_state_pool_rebuild_resizes_preserves_identity_and_dtypes():
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.config import LinearGatedDeltaGroupConfig

    _init_tp()
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0, 1, 2), num_key_heads=4, num_value_heads=8,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    pool = LinearStatePool(group=group, num_slots=10, dtype=torch.bfloat16, device=torch.device("cpu"))
    pid = id(pool)
    conv_dtype, rec_dtype = pool.conv_states.dtype, pool.recurrent_states.dtype
    _, _, conv_dim, km1 = pool.conv_states.shape
    _, _, v_heads, k_dim, v_dim = pool.recurrent_states.shape
    assert pool.num_slots == 10 and pool.num_free_slots == 9  # slot 0 is the padding sink

    pool.rebuild(25)

    assert id(pool) == pid  # identity preserved (ctx.linear_state_pool stays valid)
    assert pool.num_slots == 25 and pool.num_free_slots == 24
    assert sorted(pool._free_slots) == list(range(1, 25))
    assert pool.conv_states.shape == (3, 25, conv_dim, km1)
    assert pool.recurrent_states.shape == (3, 25, v_heads, k_dim, v_dim)
    assert pool.conv_states.dtype == conv_dtype  # conv keeps model dtype (bf16)
    assert pool.recurrent_states.dtype == rec_dtype  # recurrent keeps fp32

    pool.rebuild(6)  # shrink also works
    assert pool.num_slots == 6 and pool.num_free_slots == 5


def test_dsv4_rebuild_from_config_builds_the_pool_sizes_and_attaches_the_table():
    """DSV4 turns (config, usable pages) into a full DSV4PoolSizes itself -- the derivation
    the engine used to do -- and re-points full_loc_map at the shared page table."""
    from types import SimpleNamespace

    from freetoken.kvcache.dsv4_cost_model import _dsv4_pool_sizes
    from freetoken.kvcache.dsv4_cost_model import (
        dsv4_kv_unit_bytes,
        dsv4_pool_sizes,
        dsv4_window_unit_bytes,
    )
    from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache
    from freetoken.models.deepseek_v4.args import DeepseekV4Args

    P, mrr = 128, 1
    args = DeepseekV4Args(
        n_layers=4, compress_ratios=(0, 4, 128, 4), max_seq_len=512,
        head_dim=64, index_head_dim=32, window_size=P,
    )
    config = SimpleNamespace(
        max_seq_len=512, page_size=P, max_running_req=mrr, cache_type="swa_radix",
        swa_full_tokens_ratio=0.5, swa_num_pages_override=None,
        model_config=SimpleNamespace(dsv4_args=args),
    )
    pool = DSV4PagedKVCache(
        sizes=dsv4_pool_sizes(num_pages=4, args=args, swa_ratio=0.5, P=P),
        args=args, device=torch.device("cpu"), dtype=torch.bfloat16, P=P, n_scratch=mrr + 1,
    )
    pool._init_paged_state(mrr, True)  # the engine's create_kv_pool step
    pool.rebuild_from_config(config, 15)

    assert pool.sizes == _dsv4_pool_sizes(config, 16)  # 15 usable + 1 dummy page
    assert pool.sizes.full_token == 16 * P
    assert pool.window_pool[0].shape[0] == pool.sizes.n_win_slots
    assert pool.unit_bytes() == (dsv4_kv_unit_bytes(args, P), dsv4_window_unit_bytes(args, P))

    page_table = torch.zeros((mrr + 1, 64), dtype=torch.int32)
    pool.attach_page_table(page_table)
    assert pool.full_loc_map is page_table


def test_dsv4_refresh_seq_state_tracks_page_table_width():
    """A GROWING DSV4 rebuild raises max_seq_len; page_table (and thus the scheduler's
    token_pool) must track it, or admission accepts a request whose decode positions index
    the boot-width token_pool out of bounds. Mirrors Engine._refresh_seq_state."""
    from types import SimpleNamespace

    from freetoken.engine.engine import Engine
    from freetoken.utils import align_ceil
    from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache
    from freetoken.scheduler.table import TableManager

    P, mrr = 128, 4
    config = SimpleNamespace(
        max_seq_len=65536, page_size=P, max_running_req=mrr,
        model_config=SimpleNamespace(dsv4_args=SimpleNamespace(window_size=P)),
    )
    # attach_page_table only re-points the pool's full_loc_map; the decode snapshot belongs to
    # the attention backend, which re-allocates it from the engine ceiling on re-capture. The
    # real method on an uninitialized pool: no buffers needed, just the field it re-points.
    pool = object.__new__(DSV4PagedKVCache)
    pool.full_loc_map = None
    eng = SimpleNamespace(
        num_pages=489, device=torch.device("cpu"),
        ctx=SimpleNamespace(page_table=None),
        dummy_req=SimpleNamespace(table_idx=mrr),
        kv_cache=pool,
    )
    eng.max_seq_len = min(config.max_seq_len, eng.num_pages * P)
    eng.page_table = torch.zeros((mrr + 1, align_ceil(eng.max_seq_len, 32)), dtype=torch.int32)
    tm = TableManager(mrr, eng.page_table)

    for target in (519, 489, 400, 519):  # grow, shrink back, shrink, grow again
        eng.num_pages = target
        Engine._refresh_seq_state(eng, config)
        # the scheduler's rebuild_cache re-point (owned-KV branch)
        if tm.page_table is not eng.page_table:
            tm.rebuild(eng.page_table)
        assert eng.max_seq_len <= tm.token_pool.shape[1]
        assert eng.ctx.page_table is eng.page_table
        # the pool reads full locs through the live table
        assert eng.kv_cache.full_loc_map is eng.page_table
        # the highest column any decode step writes is max_seq_len - 1
        tm.token_pool[(torch.tensor([0]), torch.tensor([eng.max_seq_len - 1]))] = 1


def test_every_kv_pool_answers_the_sizing_surface():
    """The engine drives pre-pool sizing through these classmethods on whatever pool family
    the model resolved to. DSV4 must carry its OWN overrides (getattr would resolve to the
    base class and silently misprice the window floor if an override vanished)."""
    from freetoken.kvcache.dsa_pool import DSAKVCache, MLAKVCache
    from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
    from freetoken.kvcache.mha_pool import MHAKVCache

    for cls in (MHAKVCache, MLAKVCache, DSAKVCache, HybridSWAKVCache, DSV4PagedKVCache):
        for hook in ("kv_cost", "solve_num_pages", "min_kv_tokens", "validate_rebuild"):
            assert callable(getattr(cls, hook, None)), f"{cls.__name__} is missing {hook}"
    for hook in ("kv_cost", "solve_num_pages", "min_kv_tokens", "validate_rebuild"):
        assert hook in DSV4PagedKVCache.__dict__, f"DSV4 lost its {hook} override"


def test_every_kv_pool_answers_the_rebuild_surface():
    """The engine drives every pool through this one surface; a pool that skips an
    implementation must fail loudly at import/instantiation, not at rebuild time."""
    from freetoken.kvcache.dsa_pool import DSAKVCache, MLAKVCache
    from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
    from freetoken.kvcache.mha_pool import MHAKVCache

    for cls in (MHAKVCache, MLAKVCache, DSAKVCache, HybridSWAKVCache, DSV4PagedKVCache):
        assert not cls.__abstractmethods__, f"{cls.__name__}: {sorted(cls.__abstractmethods__)}"
    # Only DSV4 rebinds the model and re-points a page table; the rest inherit the defaults.
    assert DSV4PagedKVCache.needs_rebind_on_rebuild
    assert not any(
        cls.needs_rebind_on_rebuild
        for cls in (MHAKVCache, MLAKVCache, DSAKVCache, HybridSWAKVCache)
    )
    assert _mha_pool().attach_page_table(torch.zeros(1)) is None  # base no-op
