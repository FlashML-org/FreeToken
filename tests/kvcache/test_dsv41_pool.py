"""DSV41 paged KV pool + cost model (CPU, no model): args validation, per-source tiers, sizing /
byte accounting, the window free-list duck-type, full-loc translation; packed writes on CUDA."""

from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.dsv4.v41_cost_model import (
    dsv41_cache_per_page,
    dsv41_kv_unit_bytes,
    dsv41_pool_bytes,
    dsv41_pool_sizes,
    dsv41_solve_num_pages,
    dsv41_window_unit_bytes,
)
from freetoken.kvcache.dsv4.v41_pool import DSV41PagedKVCache
from freetoken.kvcache.dsv4.v41_row_format import FP4_E4M3_B16, FP4_E8M0_B32, FP8_E8M0_B32
from freetoken.models.deepseek_v41.args import DeepseekV41Args

DEVICE = torch.device("cpu")
P = 128
# a 10-layer miniature of V4.1's layout: 2 window-only, ratio-2 encoder (source 2, source 5), ratio-1 decoder (source 7)
RATIOS = (0, 0, 2, 2, 2, 2, 2, 1, 1, 1)
SOURCES = (2, 5, 7)


def _args(**over) -> DeepseekV41Args:
    base = dict(n_layers=10, head_dim=512, index_head_dim=128, window_size=P, compress_ratios=RATIOS,
                kv_source_layers=SOURCES, swa_decoder_replay="exact")
    base.update(over)
    return DeepseekV41Args(**base)


def _pool(num_pages=8, swa_ratio=0.5, n_scratch=1, **over):
    args = _args(**over)
    sizes = dsv41_pool_sizes(num_pages, args, swa_ratio, P)
    return DSV41PagedKVCache(sizes, args, DEVICE, n_scratch=n_scratch), sizes, args


def test_window_control_uses_pool_units_and_restores_concrete_capacity():
    from types import SimpleNamespace

    from freetoken.attention import AttnType
    from freetoken.kvcache.cache_status import (
        _supports_swa_ratio, compute_cache_floors, compute_cache_pools,
    )
    from freetoken.kvcache.dsv4.v41_cost_model import _dsv41_pool_sizes
    from freetoken.scheduler.scheduler import Scheduler
    from freetoken.server.api_server import CacheRebuildRequest, _resolve_num_swa_pages, cache_geometry
    from freetoken.server.stats import _swa_page_size

    pool, _, args = _pool(num_pages=64)
    config = SimpleNamespace(
        page_size=P, max_running_req=1, max_seq_len=1024, cache_type="swa_radix",
        swa_full_tokens_ratio=0.5, swa_num_pages_override=None,
        model_config=SimpleNamespace(
            dsv4_args=None, dsv41_args=_args(), has_swa_attention=False,
            kv_cache_group_specs=lambda: [SimpleNamespace(attn_type=AttnType.DSV41)],
        ),
    )
    engine = SimpleNamespace(
        config=config, kv_cache=pool, num_pages=63,
        moe_offload_cache=None, linear_state_pool=None,
    )
    assert _supports_swa_ratio(config)
    assert compute_cache_pools(engine)["swa_page_size"] == P
    assert compute_cache_pools(engine)["num_swa_pages"] == 31
    floor = _dsv41_pool_sizes(config, 64, num_swa_pages=1).n_win_pages - 1
    assert compute_cache_floors(engine)["swa_tokens"] == floor * P
    prior = Scheduler._current_cache_geometry(SimpleNamespace(engine=engine, config=config))
    assert prior["num_swa_pages"] == 31
    state = SimpleNamespace(config=config, cache_pools=compute_cache_pools(engine))
    req = CacheRebuildRequest(num_pages=64, swa_full_tokens_ratio=0.5)
    assert _resolve_num_swa_pages(state, req) == 32
    assert _swa_page_size(config) == P
    state.stats = SimpleNamespace(kv_total_pages=63, mamba_total_slots=0)
    state.last_rebuild = {"num_pages": 64, "num_swa_pages": 32}
    assert cache_geometry(state)["swa_full_tokens_ratio"] == 0.5
    pool._init_paged_state(1, True)
    pool.rebuild_from_config(config, 63, num_swa_pages=24)
    assert pool.window_pages == 24
    pool.rebuild_from_config(config, prior["num_pages"], num_swa_pages=prior["num_swa_pages"])
    assert pool.window_pages == 31


def test_pool_reads_sources_and_rings_from_args():
    pool, _, _ = _pool()
    assert [pool.source_of(l) for l in range(2, 10)] == [2, 2, 2, 5, 5, 7, 7, 7]
    assert set(pool.state_ring) == {2, 5} and pool.state_ring[2].ring_size == 2
    with pytest.raises(ValueError):  # the window is not a multiple of a compress ratio
        _pool(window_size=P - 1)
    with pytest.raises(ValueError):  # an indexer key the fp4 row format cannot hold
        _pool(index_head_dim=48)


@pytest.mark.parametrize("replay", ["bounded", "exact"])
def test_v41_flash_row_bytes_and_private_window_layers(replay):
    """The tech report's headline: 3 ratio-2 encoder sources + 1 ratio-1 decoder source = 890 B/token."""
    ratios = (0, 0) + (2,) * 18 + (1,) * 20
    args = DeepseekV41Args(n_layers=40, head_dim=512, index_head_dim=128, window_size=128, compress_ratios=ratios,
                           kv_source_layers=(2, 8, 14, 20), swa_decoder_replay=replay)
    pool = DSV41PagedKVCache(dsv41_pool_sizes(4, args, 1.0, 128), args, DEVICE)
    win, main, idx = pool.window_pool[0].shape[1], pool.main_pool[2].shape[1], pool.idx_pool[2].shape[1]
    assert (win, main, idx) == (528, 288, 68)
    assert pool.state_ring[2].buffer.shape[1] * 4 == 4096
    assert sum((main + idx) // pool.ratio_of(s) for s in args.backbone_kv_sources) == 890
    assert dsv41_kv_unit_bytes(args, 128) == 890 + 8  # + the int64 full->window map slot
    assert pool.private_window_layer_ids == (tuple(range(20, 40)) if replay == "bounded" else ())


def test_pool_tiers_per_source_and_aliasing():
    pool, sizes, args = _pool()
    assert len(pool.window_pool) == 10 and pool.window_pool[0].shape == (sizes.n_win_slots, 528)
    assert set(pool.main_pool) == set(SOURCES) == set(pool.idx_pool)
    assert set(pool.state_ring) == {2, 5}  # the ratio-1 decoder source carries no partial group
    assert pool.main_pool[2].shape == (sizes.full_token // 2 + 1, 288)
    assert pool.main_pool[7].shape == (sizes.full_token + 1, 288)
    assert pool.idx_pool[5].shape == (sizes.full_token // 2 + 1, 68)
    assert pool.main_pool_of(4) is pool.main_pool[2] and pool.idx_pool_of(9) is pool.idx_pool[7]
    assert pool.scratch_base[2] == sizes.full_token // 2
    with pytest.raises(AssertionError):
        pool.main_pool_of(0)
    ring = pool.state_ring[2]
    assert ring.ring_size == 2 and ring.item_size == 512 and ring.buffer.shape == (sizes.state_slots[2] + 1, 1024)
    assert torch.all(ring.buffer[-1, :512] == 0) and torch.all(torch.isneginf(ring.buffer[-1, 512:]))


def test_pool_bytes_match_allocation_and_solver_respects_budget():
    pool, sizes, args = _pool(num_pages=16, swa_ratio=0.25)
    assert pool.total_bytes() == dsv41_pool_bytes(sizes, args, n_scratch=1)
    assert dsv41_cache_per_page(args, 0.25, P) > 0
    assert dsv41_window_unit_bytes(args, P) == -(-(10 * P * 528 + 2 * 2 * 4096) // P)
    budget = 64 << 20
    solved = dsv41_solve_num_pages(budget, args, 0.25, floor_win_pages=4, P=P, n_scratch=3)
    assert dsv41_pool_bytes(solved, args, 3) <= budget
    # one more page, sized the way the solver sizes (window = max(floor, ceil(ratio * pages))), overflows
    more = solved.full_token // P + 1
    bigger = dsv41_pool_sizes(more, args, 0.25, P, n_win_pages=max(4, (round(0.25 * more * P) + P - 1) // P))
    assert dsv41_pool_bytes(bigger, args, 3) > budget
    assert solved.n_win_pages >= 4
    with pytest.raises(ValueError):
        dsv41_solve_num_pages(1 << 10, args, 0.25, floor_win_pages=4, P=P)


def test_translation_state_loc_and_cmp_rows():
    pool, sizes, args = _pool(num_pages=16)
    pool.bind_window_pages(full_page_base=0, window_page_base=2 * P)
    pool.bind_window_pages(full_page_base=3 * P, window_page_base=0)
    assert pool.translate_full_to_window(torch.tensor([0, 1, 127])).tolist() == [2 * P, 2 * P + 1, 2 * P + 127]
    assert pool.translate_full_to_window(torch.tensor([3 * P + 5])).item() == 5
    assert pool.translate_full_to_window(torch.tensor([P + 7, -1])).tolist() == [-1, -1]
    full = torch.tensor([0, 1, 2, 127, 128, 4 * P - 1])
    assert pool.cmp_rows(full, 2).tolist() == [0, 0, 1, 63, 64, 2 * P - 1]
    assert pool.cmp_rows(full, 1).tolist() == full.tolist()
    assert pool.cmp_rows(torch.tensor([-1]), 2).item() < 0
    top = pool.cmp_rows(torch.tensor([sizes.full_token - 1]), 2).item()
    assert top < pool.scratch_base[2]
    ws = pool.translate_full_to_window(torch.tensor([127, 3 * P]))
    assert DSV41PagedKVCache.state_loc(ws, 2, P).tolist() == [2 * 2 + 1, 0]
    assert DSV41PagedKVCache.state_loc(torch.tensor([-1]), 2, P).item() == -1


def _expand(bases):
    return (torch.tensor(bases, dtype=torch.int64)[:, None] + torch.arange(P)).flatten()


def test_swa_duck_type_alloc_free_and_dummy():
    pool, sizes, _ = _pool(num_pages=8)
    pool._init_paged_state(max_running_req=2, radix=True)
    cap = sizes.n_win_slots - P
    assert pool.swa_available_size() == cap and pool.swa_num_tokens - 1 == cap
    assert pool.sliding_window_size == P and pool.prefill_chunk_budget >= P
    pool.alloc_swa(_expand([0, 2 * P]))
    assert pool.swa_available_size() == cap - 2 * P
    ws = pool.translate_loc_from_full_to_swa(torch.arange(2 * P, 3 * P))
    assert (ws >= 0).all() and int(ws[0]) % P == 0 and torch.equal(ws - ws[0], torch.arange(P))
    pool.free_swa(_expand([0]))
    pool.free_swa(_expand([0]))  # idempotent
    assert pool.swa_available_size() == cap - P
    with pytest.raises(AssertionError):
        pool.free_swa(torch.arange(2 * P, 2 * P + 5))
    dummy = pool.translate_loc_from_full_to_swa(torch.arange(sizes.full_token - P, sizes.full_token))
    assert int(dummy[0]) == sizes.n_win_slots - P
    assert pool.k_cache(3) is pool.window_pool[3] and pool.num_layers == 10 and pool.unit_bytes()[0] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="packed writes run through triton")
def test_packed_writes_round_trip_on_cuda():
    from kernels.test_dsv41_pack import reference_roundtrip  # tests/ is on sys.path under pytest

    args = _args()
    sizes = dsv41_pool_sizes(4, args, 0.5, P)
    pool = DSV41PagedKVCache(sizes, args, torch.device("cuda"), n_scratch=2)
    kv = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([0, 130, 5], device="cuda")
    pool.store_window(kv, 4, slots)
    assert torch.equal(pool.read_window(4, slots), reference_roundtrip(kv, FP8_E8M0_B32))
    pool.store_main(kv, 2, slots)
    assert torch.equal(pool.read_main(2, slots), reference_roundtrip(kv, FP4_E4M3_B16))
    k = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    pool.store_index(k, 7, slots)
    assert torch.equal(pool.read_index(7, slots), reference_roundtrip(k, FP4_E8M0_B32))
    pool.store_kv(kv, kv, slots, 0)
    assert torch.equal(pool.read_window(0, slots), reference_roundtrip(kv, FP8_E8M0_B32))
