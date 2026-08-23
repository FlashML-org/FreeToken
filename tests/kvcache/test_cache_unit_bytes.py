"""Unit tests for compute_cache_unit_bytes / compute_cache_floors / compute_cache_status_meta
-- the per-unit cache VRAM cost measurement + per-pool rebuild floors the scheduler emits on
the ("meta", …) ack. Uses tiny CPU tensors + duck-typed pool stand-ins so it exercises the
real byte arithmetic without a GPU or a model load."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kvcache.cache_status import (
    compute_cache_floors,
    compute_cache_status_meta,
    compute_cache_unit_bytes,
)
from freetoken.kvcache.dsv4_cost_model import _dsv4_window_floor_pages


def _init_tp() -> None:
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _mha_engine(layers=4, pages=100, page_size=16, kv_heads=2, head_dim=64, dtype=torch.bfloat16):
    from freetoken.kvcache.mha_pool import MHAKVCache

    _init_tp()
    eng = SimpleNamespace(
        kv_cache=MHAKVCache(
            num_kv_heads=kv_heads, num_layers=layers, head_dim=head_dim, num_pages=pages,
            page_size=page_size, dtype=dtype, device=torch.device("cpu"),
        ),
        moe_offload_cache=None,
        linear_state_pool=None,
        config=None,
    )
    return eng


def test_kv_bytes_per_token_from_mha_buffer():
    layers, kv_heads, head_dim, itemsize = 4, 2, 64, 2  # bf16
    eng = _mha_engine(layers=layers, kv_heads=kv_heads, head_dim=head_dim)
    ub = compute_cache_unit_bytes(eng)
    # bytes/token = total / (pages*page_size) = 2 * layers * kv_heads * head_dim * itemsize.
    assert ub["kv_bytes_per_token"] == 2 * layers * kv_heads * head_dim * itemsize
    assert ub["moe_bytes_per_expert"] == 0
    assert ub["mamba_bytes_per_slot"] == 0


def test_kv_and_swa_bytes_per_token_from_hybrid_pools():
    # HybridSWAKVCache owns two pools: the full one denominates kv_bytes_per_token (over its
    # own layers), the window one swa_bytes_per_token (token-granular).
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
    from freetoken.models.config import KVCacheGroupSpec

    _init_tp()
    kv_heads, head_dim = 4, 32
    pool = HybridSWAKVCache(
        groups=[
            KVCacheGroupSpec(name="full", layer_ids=(0, 2), num_kv_heads=kv_heads,
                             head_dim=head_dim, sliding_window=None),
            KVCacheGroupSpec(name="swa", layer_ids=(1,), num_kv_heads=kv_heads,
                             head_dim=head_dim, sliding_window=128),
        ],
        num_layers=3, num_full_pages=50, page_size=16, num_swa_tokens=512,
        dtype=torch.bfloat16, device=torch.device("cpu"),
    )
    eng = SimpleNamespace(
        kv_cache=pool, moe_offload_cache=None, linear_state_pool=None, config=None
    )
    ub = compute_cache_unit_bytes(eng)
    assert ub["kv_bytes_per_token"] == 2 * 2 * kv_heads * head_dim * 2  # 2 full layers
    assert ub["swa_bytes_per_token"] == 2 * 1 * kv_heads * head_dim * 2  # 1 swa layer


def test_moe_bytes_per_expert_sums_bank_rows():
    # bank_caches: name -> (cache_size, *row_shape); one slot = summed row bytes over banks.
    cache_size = 8
    gate_up = torch.empty((cache_size, 512), dtype=torch.bfloat16)  # 512 * 2 = 1024 B/row
    down = torch.empty((cache_size, 256), dtype=torch.bfloat16)  # 256 * 2 = 512 B/row
    eng = SimpleNamespace(
        kv_cache=None,
        moe_offload_cache=SimpleNamespace(bank_caches={"gate_up": gate_up, "down": down}),
        linear_state_pool=None,
    )
    ub = compute_cache_unit_bytes(eng)
    assert ub["moe_bytes_per_expert"] == 1024 + 512
    assert ub["kv_bytes_per_token"] == 0


def test_mamba_bytes_per_slot_from_pool_method():
    eng = SimpleNamespace(
        kv_cache=None,
        moe_offload_cache=None,
        linear_state_pool=SimpleNamespace(bytes_per_slot=lambda: 524288),
    )
    ub = compute_cache_unit_bytes(eng)
    assert ub["mamba_bytes_per_slot"] == 524288


def test_all_units_present_for_hybrid_moe_model():
    eng = _mha_engine()
    eng.moe_offload_cache = SimpleNamespace(
        bank_caches={"w": torch.empty((4, 100), dtype=torch.bfloat16)}
    )
    eng.linear_state_pool = SimpleNamespace(bytes_per_slot=lambda: 999)
    ub = compute_cache_unit_bytes(eng)
    assert ub["kv_bytes_per_token"] > 0
    assert ub["moe_bytes_per_expert"] == 100 * 2
    assert ub["mamba_bytes_per_slot"] == 999


def test_dense_model_all_zero():
    eng = SimpleNamespace(kv_cache=None, moe_offload_cache=None, linear_state_pool=None)
    assert compute_cache_unit_bytes(eng) == {
        "kv_bytes_per_token": 0,
        "moe_bytes_per_expert": 0,
        "mamba_bytes_per_slot": 0,
        "swa_bytes_per_token": 0,
    }


def test_never_raises_on_bad_pool():
    # A pool whose accessor raises must degrade that unit to 0, not propagate (readiness path).
    class Boom:
        def bytes_per_slot(self):
            raise RuntimeError("boom")

    eng = SimpleNamespace(
        kv_cache=object(), moe_offload_cache=object(), linear_state_pool=Boom()
    )
    ub = compute_cache_unit_bytes(eng)
    assert ub == {
        "kv_bytes_per_token": 0,
        "moe_bytes_per_expert": 0,
        "mamba_bytes_per_slot": 0,
        "swa_bytes_per_token": 0,
    }


# ---------------------------------------------------------------------------
# compute_cache_floors / compute_cache_status_meta
# ---------------------------------------------------------------------------


def _config(page_size=16, max_running_req=4, cache_type="hybrid_radix", num_experts=128):
    return SimpleNamespace(
        page_size=page_size,
        max_running_req=max_running_req,
        cache_type=cache_type,
        model_config=SimpleNamespace(
            num_experts=num_experts, dsv4_args=None, has_swa_attention=False
        ),
    )


def test_floors_hybrid_moe_model():
    # KV floor = one page's tokens (rebuild rejects num_pages <= 0); MoE floor = one layer's
    # experts (_require_offload_cache_size); mamba floor = _linear_pool_min_slots - 1 usable
    # (hybrid_radix: 4 slots per running request non-evictable, padding sink excluded).
    eng = SimpleNamespace(
        config=_config(),
        moe_offload_cache=object(),
        linear_state_pool=object(),
        kv_cache=None,
    )
    assert compute_cache_floors(eng) == {
        "kv_tokens": 16,
        "moe_experts": 128,
        "mamba_slots": 4 * 4,  # 4*max_running_req + 1 physical -> -1 padding = 16 usable
        "swa_tokens": 0,  # not a radix-SWA model (cache_type=hybrid_radix, no has_swa_attention)
    }


def test_floors_naive_gdn_and_dense():
    # naive (non-hybrid_radix) GDN pool: physical floor mr+1 -> mr usable.
    eng = SimpleNamespace(
        config=_config(cache_type="naive"), moe_offload_cache=None, linear_state_pool=object(),
        kv_cache=None,
    )
    assert compute_cache_floors(eng)["mamba_slots"] == 4
    # Dense non-hybrid model: no MoE offload cache, no GDN pool -> both floors 0.
    dense = SimpleNamespace(
        config=_config(), moe_offload_cache=None, linear_state_pool=None, kv_cache=None
    )
    floors = compute_cache_floors(dense)
    assert floors["moe_experts"] == 0 and floors["mamba_slots"] == 0
    assert floors["kv_tokens"] == 16


def test_floors_dsv4_reports_real_window_floor():
    # DSV4 (owned-KV) reports its true window working-set floor via the KV policy -- not the
    # bogus 0 the removed owns_kv_cache branch used to report (the G-floor fix).
    P = 128
    cfg = SimpleNamespace(
        page_size=P,
        max_running_req=4,
        max_seq_len=4096,
        cache_type="radix",
        model_config=SimpleNamespace(
            dsv4_args=SimpleNamespace(window_size=P), has_swa_attention=False
        ),
    )
    from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache

    eng = SimpleNamespace(config=cfg, moe_offload_cache=None, linear_state_pool=None)
    eng.kv_cache = object.__new__(DSV4PagedKVCache)  # min_kv_tokens is a classmethod
    floors = compute_cache_floors(eng)
    assert floors["kv_tokens"] == _dsv4_window_floor_pages(cfg, P) * P
    assert floors["kv_tokens"] > 0
    # The window (swa) floor is reported too, in tokens (window pages x P) -- not the bogus 0 the
    # radix-only branch used to give for DSV4.
    assert floors["swa_tokens"] == _dsv4_window_floor_pages(cfg, P) * P


def test_floors_missing_config_all_zero():
    # No config at all (defensive) -> all zero, never raises.
    assert compute_cache_floors(SimpleNamespace(config=None)) == {
        "kv_tokens": 0,
        "moe_experts": 0,
        "mamba_slots": 0,
        "swa_tokens": 0,
    }


def test_status_meta_bundles_units_free_vram_and_floors():
    # Duck-typed engine: unit bytes measured, the pool budget degrades to 0 (no
    # _post_weights_free baseline captured — real Engine.__init__ records it after weights
    # load, before any cache pool), floors derived from config -- all in one payload and
    # never raising.
    eng = _mha_engine()
    eng.config = _config()
    meta = compute_cache_status_meta(eng)
    assert meta["kv_bytes_per_token"] > 0
    assert meta["free_vram_bytes"] == 0  # no baseline on the fake engine
    assert meta["floors"] == {"kv_tokens": 16, "moe_experts": 0, "mamba_slots": 0, "swa_tokens": 0}


def test_status_meta_reports_post_weights_baseline():
    # The budget is the value Engine.__init__ recorded, verbatim -- not a query-time reading.
    eng = _mha_engine()
    eng.config = _config()
    eng._post_weights_free = 6 * 2 ** 30
    meta = compute_cache_status_meta(eng)
    assert meta["free_vram_bytes"] == 6 * 2 ** 30


def test_compute_cache_pools_reads_load_time_allocations():
    from freetoken.kvcache.cache_status import compute_cache_pools

    eng = SimpleNamespace(
        num_pages=90112,
        config=SimpleNamespace(
            page_size=1,
            max_extend_tokens=8192,
            model_config=SimpleNamespace(dsv4_args=None, has_swa_attention=False),
        ),
        kv_cache=SimpleNamespace(prefill_chunk_budget=8064),
        moe_offload_cache=SimpleNamespace(cache_size=526, bank_caches={}),
        # num_slots includes the reserved padding sink; API units are usable slots.
        linear_state_pool=SimpleNamespace(num_slots=65),
    )
    assert compute_cache_pools(eng) == {
        "num_pages": 90112,
        "page_size": 1,
        "moe_cache_size": 526,
        "num_mamba_slots": 64,
        "swa_page_size": 0,
        "num_swa_pages": 0,
        "requested_prefill_tokens": 8192,
        "pool_prefill_cap_tokens": 8064,
        "effective_prefill_tokens": 8064,
        "swa_capacity_source": "none",
        "prefill_limiting_reason": "swa_pool",
    }


def test_compute_cache_pools_zero_for_missing_pools():
    from freetoken.kvcache.cache_status import compute_cache_pools

    eng = SimpleNamespace(num_pages=0, config=None, moe_offload_cache=None, linear_state_pool=None)
    assert compute_cache_pools(eng) == {
        "num_pages": 0,
        "page_size": 0,
        "moe_cache_size": 0,
        "num_mamba_slots": 0,
        "swa_page_size": 0,
        "num_swa_pages": 0,
        "requested_prefill_tokens": 0,
        "pool_prefill_cap_tokens": 0,
        "effective_prefill_tokens": 0,
        "swa_capacity_source": "none",
        "prefill_limiting_reason": "none",
    }


def test_cache_status_reports_effective_prefill_clamp():
    from freetoken.cache_report import format_cache_status

    rendered = format_cache_status({
        "state": "serving",
        "geometry": {
            "requested_prefill_tokens": 8192,
            "pool_prefill_cap_tokens": 8064,
            "effective_prefill_tokens": 8064,
            "swa_capacity_source": "explicit",
            "prefill_limiting_reason": "swa_pool",
        },
    })
    assert (
        "prefill=8064 tok (requested 8192, pool cap 8064, "
        "source explicit, reason swa_pool)"
    ) in rendered


def test_cache_status_always_reports_equal_and_generic_prefill_geometry():
    from freetoken.cache_report import format_cache_status

    equal = format_cache_status({"geometry": {
        "requested_prefill_tokens": 8192,
        "pool_prefill_cap_tokens": 8192,
        "effective_prefill_tokens": 8192,
        "swa_capacity_source": "derived",
        "prefill_limiting_reason": "requested_and_swa_pool",
    }})
    assert (
        "requested 8192, pool cap 8192, source derived, reason requested_and_swa_pool"
    ) in equal

    generic = format_cache_status({"geometry": {
        "requested_prefill_tokens": 8192,
        "pool_prefill_cap_tokens": 0,
        "effective_prefill_tokens": 8192,
        "swa_capacity_source": "none",
        "prefill_limiting_reason": "requested_limit",
    }})
    assert "requested 8192, pool cap none, source none, reason requested_limit" in generic


def test_status_meta_includes_pools():
    eng = _mha_engine()
    eng.config = _config()
    eng.num_pages = 100
    meta = compute_cache_status_meta(eng)
    assert meta["pools"]["num_pages"] == 100
    assert meta["pools"]["page_size"] == 16


def test_status_meta_reports_effective_derived_dsv4_ratio(monkeypatch):
    import freetoken.kvcache.cache_status as cache_status

    monkeypatch.setattr(cache_status, "compute_cache_unit_bytes", lambda engine: {})
    monkeypatch.setattr(cache_status, "compute_cache_floors", lambda engine: {})
    config = SimpleNamespace(
        page_size=128, max_extend_tokens=24576, max_running_req=4,
        cache_type="swa_radix", swa_capacity_source="derived",
        swa_full_tokens_ratio=0.2, memory_ratio=0.9,
        model_config=SimpleNamespace(
            dsv4_args=SimpleNamespace(window_size=128), has_swa_attention=False,
        ),
    )
    engine = SimpleNamespace(
        config=config, num_pages=4096,
        kv_cache=SimpleNamespace(
            sizes=SimpleNamespace(n_win_pages=408), prefill_chunk_budget=24576,
        ),
        moe_offload_cache=None, linear_state_pool=None,
        _post_weights_free=0, _baseline_free=0, _weights_bytes=0,
    )

    meta = compute_cache_status_meta(engine)
    assert meta["pools"]["num_swa_pages"] == 407
    assert meta["swa_full_tokens_ratio"] == pytest.approx(407 / 4096)
