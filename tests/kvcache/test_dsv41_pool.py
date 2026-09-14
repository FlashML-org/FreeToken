import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from freetoken.kvcache.dsv41_cost_model import (
    _dsv41_pool_sizes, dsv41_auto_cost_model, dsv41_pool_sizes, dsv41_pool_bytes,
    dsv41_solve_num_pages, dsv41_unit_bytes, source_layers,
)
from freetoken.kvcache.dsv41_layout import dsv41_row_bytes
from freetoken.kvcache.dsv41_paged_pool import DSV41PagedKVCache


def args():
    return SimpleNamespace(n_layers=5, compress_ratios=(0, 2, 2, 1, 1),
                           kv_source_layers=(1, 3), index_source_layers=(1, 3, 4),
                           head_dim=32, index_head_dim=32, window_size=128)


@pytest.mark.parametrize("kv_quant", ["none", "fp8-fp4"])
def test_only_sources_allocate_compressed_keys_and_pool_bytes_are_exact(kv_quant):
    a = args()
    sizes = dsv41_pool_sizes(13, a, .4)
    pool = DSV41PagedKVCache(sizes, a, torch.device("cpu"), n_scratch=3, kv_quant=kv_quant)
    assert pool.kv_sources == (None, 1, 1, 3, 3)
    assert pool.index_sources == (None, 1, 1, 3, 4)
    assert [i for i, p in enumerate(pool.cmp_pool) if p is not None] == [1, 3]
    assert [i for i, p in enumerate(pool.idx_pool) if p is not None] == [1, 3]
    assert [i for i, p in enumerate(pool.state_ring) if p is not None] == [1]
    assert pool.total_bytes() == dsv41_pool_bytes(sizes, a, 3, kv_quant)
    assert pool.dtype == torch.bfloat16
    assert pool.state_ring[1].buffer.dtype == torch.float32
    expected_widths = (32, 32, 32) if kv_quant == "none" else (33, 18, 17)
    stored = (pool.window_pool[0], pool.cmp_pool[1], pool.idx_pool[1])
    assert tuple(t.shape[1] for t in stored) == expected_widths
    storage_dtype = torch.bfloat16 if kv_quant == "none" else torch.uint8
    assert all(t.dtype == storage_dtype for t in stored)


@pytest.mark.parametrize("kv_quant", ["none", "fp8-fp4"])
def test_two_request_pending_pairs_have_disjoint_state_after_page_recycling(kv_quant):
    a = args()
    pool = DSV41PagedKVCache(dsv41_pool_sizes(8, a, 1), a, torch.device("cpu"), kv_quant=kv_quant)
    pool._init_paged_state(2, True)
    full = torch.arange(256)
    pool.alloc_swa(full)
    window = pool.translate_full_to_window(torch.tensor([0, 128]))
    state = pool.state_loc(window, 2, 128)
    expected = torch.stack((torch.full((64,), 3.), torch.full((64,), -7.)))
    pool.set_state(1, state, expected)
    assert state[0] != state[1]
    torch.testing.assert_close(pool.get_state(1, state), expected)
    pool.free_swa(full[:128])
    pool.alloc_swa(torch.arange(256, 384))
    assert pool.translate_full_to_window(torch.tensor([0])).item() == -1
    torch.testing.assert_close(pool.get_state(1, state[1:]), expected[1:])


@pytest.mark.parametrize("field,value", [("compress_ratios", (0, 2, 1, 1, 1)),
                                         ("kv_source_layers", (1, 4)),
                                         ("index_source_layers", (3, 4))])
def test_reject_malformed_source_geometry(field, value):
    a = args()
    setattr(a, field, value)
    with pytest.raises(ValueError):
        source_layers(a)


@pytest.mark.parametrize("kv_quant", ["none", "fp8-fp4"])
def test_solver_respects_exact_source_owned_cost_and_maximal_fit(kv_quant):
    a = args()
    config = SimpleNamespace(model_config=SimpleNamespace(dsv41_args=a), max_seq_len=8192,
                             max_running_req=2, cache_type="swa_radix", swa_full_tokens_ratio=.2,
                             swa_num_pages_override=None, kv_quant=kv_quant)
    budget = dsv41_pool_bytes(_dsv41_pool_sizes(config, 64), a, 3, kv_quant) + 100
    usable = dsv41_solve_num_pages(config, budget)
    assert dsv41_pool_bytes(_dsv41_pool_sizes(config, usable + 1), a, 3, kv_quant) <= budget
    assert dsv41_pool_bytes(_dsv41_pool_sizes(config, usable + 2), a, 3, kv_quant) > budget


@pytest.mark.parametrize("ratio", [.001, .2, .333, 1.])
@pytest.mark.parametrize("window_pages", [None, 28, 57])
@pytest.mark.parametrize("kv_quant", ["none", "fp8-fp4"])
def test_auto_budget_includes_dummy_floor_and_window_rounding(ratio, window_pages, kv_quant):
    a = args()
    config = SimpleNamespace(model_config=SimpleNamespace(dsv41_args=a), max_seq_len=1 << 20,
                             max_running_req=2, cache_type="swa_radix", swa_full_tokens_ratio=ratio,
                             swa_num_pages_override=window_pages, kv_quant=kv_quant)
    per_page, fixed, P, reserve = dsv41_auto_cost_model(config)
    for usable in (reserve // P, 41, 100, 8192):
        exact = dsv41_pool_bytes(_dsv41_pool_sizes(config, usable + 1), a, 3, kv_quant)
        assert usable * per_page + fixed >= exact


@pytest.mark.parametrize("kv_quant", ["none", "fp8-fp4"])
def test_rebuild_budget_keeps_auxiliary_fixed_allocation(kv_quant):
    from freetoken.kvcache.base import CacheRebuildRejected

    a = args()
    sizes = dsv41_pool_sizes(32, a, .5)
    pool = DSV41PagedKVCache(sizes, a, torch.device("cpu"), n_scratch=3, kv_quant=kv_quant)
    config = SimpleNamespace(model_config=SimpleNamespace(dsv41_args=a), max_seq_len=4096,
                             max_running_req=2, cache_type="swa_radix", swa_full_tokens_ratio=.5,
                             swa_num_pages_override=None, memory_ratio=1., kv_quant=kv_quant)
    kwargs = dict(num_pages=None, target_moe=0, per_expert_bytes=0,
                  baseline_free=pool.total_bytes() + 4096, weights_bytes=0, current_num_pages=31)
    pool.validate_rebuild(config, extra_fixed_bytes=4096, **kwargs)
    with pytest.raises(CacheRebuildRejected, match="exceeding budget"):
        pool.validate_rebuild(config, extra_fixed_bytes=4097, **kwargs)


@pytest.mark.parametrize("kv_quant", ["none", "fp8-fp4"])
def test_smallest_solved_pool_honors_usable_page_floor(kv_quant):
    a = args()
    config = SimpleNamespace(model_config=SimpleNamespace(dsv41_args=a), max_seq_len=8192,
                             max_running_req=2, cache_type="swa_radix", swa_full_tokens_ratio=.2,
                             swa_num_pages_override=None, kv_quant=kv_quant)
    floor = DSV41PagedKVCache.min_kv_tokens(config) // a.window_size
    budget = dsv41_pool_bytes(_dsv41_pool_sizes(config, floor + 1), a, 3, kv_quant)
    assert dsv41_solve_num_pages(config, budget) == floor
    with pytest.raises(ValueError, match="window working set"):
        dsv41_solve_num_pages(config, budget - 1)


def _target_config(tokens, ratio, kv_quant="none"):
    from freetoken.models.deepseek_v41.config import parse_config

    fixture = Path(__file__).parents[1] / "models/fixtures/deepseek_v41_nvfp4_config.json"
    return SimpleNamespace(model_config=parse_config(json.loads(fixture.read_text())),
                           max_seq_len=tokens, max_running_req=2, cache_type="swa_radix",
                           swa_full_tokens_ratio=ratio, swa_num_pages_override=None, kv_quant=kv_quant)


@pytest.mark.parametrize("tokens,ratio,window_pages,bf16_bytes,packed_bytes", [
    (32768, .2, 52, 379_465_736, 171_409_848),
    (131072, .2, 205, 1_500_745_736, 677_061_048),
    (1048576, .2, 1639, 11_997_630_472, 5_412_839_864),
    (1048576, .02, 164, 4_228_132_872, 1_389_134_264),
    (1048576, .01, 82, 3_796_201_480, 1_165_443_512),
])
@pytest.mark.parametrize("kv_quant", ["none", "fp8-fp4"])
def test_target_kv_budget_dimensions(tokens, ratio, window_pages, bf16_bytes, packed_bytes, kv_quant):
    config = _target_config(tokens, ratio, kv_quant)
    sizes = _dsv41_pool_sizes(config, tokens // 128 + 1)
    assert sizes.n_win_pages == window_pages
    expected_bytes = bf16_bytes if kv_quant == "none" else packed_bytes
    assert dsv41_pool_bytes(sizes, config.model_config.dsv41_args, 3, kv_quant) == expected_bytes
    expected_units = (3208, 41152) if kv_quant == "none" else (898, 21312)
    assert dsv41_unit_bytes(config.model_config.dsv41_args, 128, kv_quant) == expected_units


@pytest.mark.parametrize("kv_quant", ["none", "fp8-fp4"])
def test_target_long_context_auto_plan_fits_conservative_30_gib_baseline(kv_quant):
    from freetoken.engine.cache_budget import resolve_moe_cache_auto
    from freetoken.moe.expert_banks import bank_bytes_estimate

    config = _target_config(1048576, .02, kv_quant)
    bank_bytes = bank_bytes_estimate(config.model_config)
    assert bank_bytes == 306_063_605_760
    a = config.model_config.dsv41_args
    assert sum(a.engram_num_embeddings) * (a.engram_head_dim + a.engram_head_dim // 32) == 202_758_032_400
    expert_bytes = bank_bytes // (40 * 384)
    per_page, fixed, P, floor = dsv41_auto_cost_model(config)
    weights = 12_190_720_448 + 25_166_848
    slots, pages, overlap = resolve_moe_cache_auto(
        baseline_free=30 << 30, weights_bytes=weights, memory_ratio=.9,
        cache_per_page=per_page, fixed_cache_size=fixed, per_expert_bytes=expert_bytes,
        num_experts=384, total_experts=40 * 384, prefill_overlap=False,
        kv_reserve_tokens=max(1048576, floor), page_size=P, max_slots=None,
    )
    assert slots >= 384 and pages * P >= 1048576 and not overlap
    exact_kv = dsv41_pool_bytes(_dsv41_pool_sizes(config, pages + 1), a, 3, kv_quant)
    assert weights + slots * expert_bytes + exact_kv <= int(.9 * (30 << 30))


def _packed_config():
    return SimpleNamespace(model_config=SimpleNamespace(dsv41_args=args()), max_seq_len=8192,
                           max_running_req=2, cache_type="swa_radix", swa_full_tokens_ratio=.5,
                           swa_num_pages_override=None, kv_quant="fp8-fp4", page_size=128,
                           num_page_override=None, memory_ratio=1.)


def test_factory_builds_inline_packed_rows_and_prices_the_dummy_page():
    from freetoken.kvcache import create_kv_pool

    config = _packed_config()
    pool = create_kv_pool(config, 31, torch.device("cpu"), torch.bfloat16)
    assert pool.kv_quant == "fp8-fp4"
    assert pool.sizes.full_token == 4096
    assert pool.sizes.n_win_pages == 21
    assert pool.n_scratch == 3
    assert pool.total_bytes() == 702_554
    assert pool.total_bytes() == dsv41_pool_bytes(pool.sizes, pool.args, 3, "fp8-fp4")
    dummy_full = torch.arange(pool.sizes.full_token - 128, pool.sizes.full_token)
    dummy_window = torch.arange(pool.sizes.n_win_slots - 128, pool.sizes.n_win_slots)
    torch.testing.assert_close(pool.translate_full_to_window(dummy_full), dummy_window)
    assert pool.full_to_window[-1] == -1
    assert torch.count_nonzero(pool.window_pool[0][dummy_window]) == 0


def _byte_rows(width, starts):
    return ((torch.arange(width)[None, :] + torch.tensor(starts)[:, None]) % 256).to(torch.uint8)


def test_inline_codes_and_scales_follow_window_reuse_without_touching_other_request():
    from freetoken.kvcache import create_kv_pool

    pool = create_kv_pool(_packed_config(), 31, torch.device("cpu"), torch.bfloat16)
    pool.alloc_swa(torch.arange(256))
    slots = pool.translate_full_to_window(torch.tensor([0, 128]))
    first = _byte_rows(33, [200, 127])
    pool.store_window(first, 0, slots)
    torch.testing.assert_close(pool.window_pool[0][slots], first)
    pool.free_swa(torch.arange(128))
    pool.alloc_swa(torch.arange(256, 384))
    recycled = pool.translate_full_to_window(torch.tensor([256]))
    assert recycled[0] == slots[0]
    replacement = _byte_rows(33, [251])
    pool.store_window(replacement, 0, recycled)
    torch.testing.assert_close(pool.window_pool[0][recycled], replacement)
    torch.testing.assert_close(pool.window_pool[0][slots[1:]], first[1:])
    assert pool.full_to_window[0] == -1


@pytest.mark.parametrize("layer", [1, 3])
@pytest.mark.parametrize("tier,width", [("compressed", 18), ("indexer", 17)])
def test_packed_source_rows_and_each_decode_scratch_are_independent(layer, tier, width):
    from freetoken.kvcache import create_kv_pool

    pool = create_kv_pool(_packed_config(), 31, torch.device("cpu"), torch.bfloat16)
    backing = pool.cmp_pool[layer] if tier == "compressed" else pool.idx_pool[layer]
    base = pool.cmp_scratch_base[layer] if tier == "compressed" else pool.idx_scratch_base[layer]
    writer = getattr(pool, "store_" + tier)
    main_rows = torch.tensor([0, base - 1])
    main_values = _byte_rows(width, [17, 201])
    writer(main_values, layer, main_rows)
    scratch_rows = base + torch.arange(3)
    scratch_values = _byte_rows(width, [77, 154, 231])
    writer(scratch_values, layer, scratch_rows)
    torch.testing.assert_close(backing[main_rows], main_values)
    torch.testing.assert_close(backing[scratch_rows], scratch_values)
    assert torch.count_nonzero(backing[1]) == 0
    other = 3 if layer == 1 else 1
    other_backing = pool.cmp_pool[other] if tier == "compressed" else pool.idx_pool[other]
    assert torch.count_nonzero(other_backing) == 0


@pytest.mark.parametrize("method,layer,width", [("store_window", 0, 33),
                                               ("store_compressed", 1, 18),
                                               ("store_indexer", 1, 17)])
@pytest.mark.parametrize("bad", ["dtype", "width", "row_count", "rank"])
def test_packed_writes_reject_malformed_rows_before_modifying_storage(method, layer, width, bad):
    a = args()
    pool = DSV41PagedKVCache(dsv41_pool_sizes(8, a, 1), a, torch.device("cpu"), kv_quant="fp8-fp4")
    values = _byte_rows(width, [200])
    if bad == "dtype":
        values = values.to(torch.bfloat16)
    elif bad == "width":
        values = values[:, :-1]
    elif bad == "row_count":
        values = values.expand(2, -1)
    else:
        values = values.unsqueeze(0)
    with pytest.raises(ValueError, match="row|uint8"):
        getattr(pool, method)(values, layer, torch.tensor([0]))
    assert torch.count_nonzero(pool.window_pool[0]) == 0
    assert torch.count_nonzero(pool.cmp_pool[1]) == 0
    assert torch.count_nonzero(pool.idx_pool[1]) == 0


@pytest.mark.parametrize("method,width", [("store_compressed", 18), ("store_indexer", 17)])
def test_reusing_layer_cannot_write_a_source_pool(method, width):
    a = args()
    pool = DSV41PagedKVCache(dsv41_pool_sizes(8, a, 1), a, torch.device("cpu"), kv_quant="fp8-fp4")
    with pytest.raises(ValueError, match="source layer"):
        getattr(pool, method)(_byte_rows(width, [1]), 2, torch.tensor([0]))


def test_packed_rebuild_reallocates_every_inline_row_and_preserves_storage_format():
    from freetoken.kvcache import create_kv_pool
    from freetoken.kvcache.base import CacheRebuildRejected

    config = _packed_config()
    pool = create_kv_pool(config, 31, torch.device("cpu"), torch.bfloat16)
    table = torch.zeros(3, 8192, dtype=torch.int32)
    pool.attach_page_table(table)
    for pages in (63, 31):
        previous_window, previous_cmp, previous_index = pool.window_pool[0], pool.cmp_pool[1], pool.idx_pool[1]
        previous_window.fill_(231)
        previous_cmp.fill_(154)
        previous_index.fill_(77)
        pool.rebuild_from_config(config, pages)
        assert pool.window_pool[0] is not previous_window
        assert pool.cmp_pool[1] is not previous_cmp
        assert pool.idx_pool[1] is not previous_index
        assert pool.full_loc_map is table
        assert pool.kv_quant == "fp8-fp4" and pool.dtype == torch.bfloat16
        assert pool.total_bytes() == dsv41_pool_bytes(pool.sizes, pool.args, 3, "fp8-fp4")
        assert torch.count_nonzero(pool.window_pool[0]) == 0
        assert torch.count_nonzero(pool.cmp_pool[1]) == 0
        assert torch.count_nonzero(pool.idx_pool[1]) == 0
        assert pool.cmp_scratch_base[1] == pool.sizes.full_token // 2
        assert pool.idx_scratch_base[3] == pool.sizes.full_token
        dummy = torch.tensor([pool.sizes.full_token - 128])
        assert pool.translate_full_to_window(dummy).item() == pool.sizes.n_win_slots - 128
    previous_window = pool.window_pool[0]
    config.kv_quant = "none"
    with pytest.raises(CacheRebuildRejected, match="restart"):
        pool.rebuild_from_config(config, 63)
    with pytest.raises(CacheRebuildRejected, match="restart"):
        pool.validate_rebuild(config, num_pages=63, target_moe=0, per_expert_bytes=0,
                              baseline_free=1 << 30, weights_bytes=0, current_num_pages=31)
    assert pool.window_pool[0] is previous_window
    with pytest.raises(AttributeError):
        pool.kv_quant = "none"


@pytest.mark.parametrize("field", ["head_dim", "index_head_dim"])
def test_packed_layout_rejects_incompatible_channel_blocks(field):
    a = args()
    setattr(a, field, 72)
    with pytest.raises(ValueError, match="divisible by 32"):
        dsv41_row_bytes(a, "fp8-fp4")
    with pytest.raises(ValueError, match="divisible by 32"):
        DSV41PagedKVCache(dsv41_pool_sizes(8, a, 1), a, torch.device("cpu"), kv_quant="fp8-fp4")


@pytest.mark.parametrize("kv_quant", ["fp8", "nvfp4", "invalid"])
def test_pool_rejects_other_storage_formats(kv_quant):
    a = args()
    with pytest.raises(ValueError, match="storage format"):
        DSV41PagedKVCache(dsv41_pool_sizes(8, a, 1), a, torch.device("cpu"), kv_quant=kv_quant)
