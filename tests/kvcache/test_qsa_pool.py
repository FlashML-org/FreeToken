"""QSAKVCache tiers (paged K/V, compressed index slab, pending ring, scratch).

Pins the three things the QSA kernels and the startup budget both depend on: the compressed
slab is a 1/index_ratio shadow of the K/V pages with the scratch rows behind it, the ring and
scratch are fixed (concurrency-sized) and priced apart from the per-token slider, and the K/V
slabs cover the sparse layers only. The PLE conv history rides the GDN slots, so it must
follow every slot operation and show up in the state-pool byte account.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention import AttnType
from freetoken.kvcache.base import spec_kv_bytes_per_token
from freetoken.kvcache.qsa_pool import QSAKVCache
from freetoken.models.config import KVCacheGroupSpec

DEV = torch.device("cpu")

# Qwen3.8-Flash-Next: 48 layers, every 4th is QSA; 2 kv heads x 256, indexer 128 wide, ratio 4.
FULL_LAYER_IDS = tuple(range(3, 48, 4))
REAL_KV_BYTES = 2 * 256 * 2 * 2 * 12
REAL_INDEX_BYTES = 128 * 12 * 2 // 4


@pytest.fixture(autouse=True)
def _tp(monkeypatch):
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "freetoken.kvcache.mha_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


def _pool(num_pages=4, page_size=64, index_ratio=4, num_req_slots=4, ring_capacity=None):
    return QSAKVCache(
        num_kv_heads=2,
        num_layers=8,
        head_dim=64,
        num_pages=num_pages,
        page_size=page_size,
        dtype=torch.bfloat16,
        device=DEV,
        index_head_dim=32,
        num_index_layers=4,
        index_ratio=index_ratio,
        num_req_slots=num_req_slots,
        ring_capacity=ring_capacity,
        layer_ids=(1, 3, 5, 7),
    )


def _spec(*, index_ratio=4, attn_type=AttnType.QSA, num_kv_heads=2, head_dim=64,
          index_head_dim=32, num_index_layers=4, layer_ids=(1, 3, 5, 7)):
    return KVCacheGroupSpec(
        name="full",
        layer_ids=layer_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        sliding_window=None,
        index_head_dim=index_head_dim,
        num_index_layers=num_index_layers,
        index_ratio=index_ratio,
        attn_type=attn_type,
    )


def _config(spec, *, page_size=64, max_running_req=3):
    mc = SimpleNamespace(num_layers=8, has_swa_attention=False, has_linear_attention=True)
    mc.kv_cache_group_specs = lambda: (spec,)
    return SimpleNamespace(
        model_config=mc,
        page_size=page_size,
        dtype=torch.bfloat16,
        tp_info=SimpleNamespace(size=1),
        max_running_req=max_running_req,
    )


# --------------------------------------------------------------------- slab / ring geometry


def test_slab_ring_and_scratch_shapes():
    pool = _pool(num_pages=4)
    # 4 pages x 64 tokens / ratio 4 = 64 shadow rows, then one scratch row per request slot
    assert pool.cmp_scratch_base == 64
    assert pool.cmp_k_cache(0).shape == (64 + 4, 32)
    assert pool.cmp_k_cache(3).shape == (64 + 4, 32)
    assert pool.pending_ring(0).shape == (4, QSAKVCache.ring_capacity_for(4), 32)
    assert pool.cmp_k_cache(0).abs().sum().item() == 0.0
    assert pool.k_cache(1).shape == (4, 64, 2, 64)


def test_kv_slabs_cover_sparse_layers_only():
    # Copying the BSA branch (no layer_ids) would back all 8 model layers instead of 4.
    pool = _pool()
    assert pool._kv_buffer.shape[1] == 4
    pool.k_cache(7)
    with pytest.raises(KeyError):
        pool.k_cache(0)


def test_ring_capacity_and_ratio_are_parameters():
    pool = _pool(num_pages=8, index_ratio=2, num_req_slots=3, ring_capacity=6)
    assert pool.index_ratio == 2 and pool.ring_capacity == 6
    assert pool.cmp_scratch_base == 8 * 64 // 2
    assert pool.pending_ring(0).shape == (3, 6, 32)


def test_ring_capacity_formula_and_floor():
    assert QSAKVCache.ring_capacity_for(4) == 4
    assert QSAKVCache.ring_capacity_for(8) == 8
    assert QSAKVCache.ring_capacity_for(4, num_speculative_tokens=3) == 8
    with pytest.raises(ValueError, match="ring_capacity"):
        _pool(ring_capacity=2, index_ratio=4)


def test_group_must_not_straddle_a_page():
    with pytest.raises(ValueError, match="divisible"):
        _pool(page_size=6, index_ratio=4)


def test_index_slab_needs_a_two_byte_dtype():
    with pytest.raises(AssertionError, match="2 bytes"):
        QSAKVCache(
            num_kv_heads=2, num_layers=8, head_dim=64, num_pages=4, page_size=64,
            dtype=torch.float32, device=DEV, index_head_dim=32, num_index_layers=4,
            index_ratio=4, num_req_slots=4, layer_ids=(1, 3, 5, 7),
        )


def test_shadow_row_is_shared_by_a_whole_group():
    pool = _pool()
    cmp = pool.cmp_k_cache(2)
    row = torch.randn(32, dtype=torch.bfloat16)
    for slot in (64, 65, 66, 67):
        assert slot // pool.index_ratio == 16
    cmp[16] = row
    assert torch.equal(pool.cmp_k_cache(2)[16], row)
    # the other sparse layers keep their own rows
    assert pool.cmp_k_cache(1)[16].abs().sum().item() == 0.0


def test_rebuild_resizes_every_tier_and_keeps_identity():
    pool = _pool(num_pages=4)
    ident = id(pool)
    pool.rebuild(16)
    assert id(pool) == ident
    assert pool.k_cache(1).shape == (16, 64, 2, 64)
    assert pool.cmp_scratch_base == 16 * 64 // 4
    assert pool.cmp_k_cache(0).shape == (16 * 64 // 4 + 4, 32)
    assert pool.pending_ring(3).shape == (4, pool.ring_capacity, 32)
    assert pool._kv_buffer.shape[1] == 4  # sparse-layer slabs survive the resize
    pool.k_cache(7)


# ------------------------------------------------------------------------------ budgeting


def test_spec_bytes_per_token_divides_the_index_slab():
    spec = _spec(num_kv_heads=2, head_dim=256, index_head_dim=128, num_index_layers=12,
                 layer_ids=FULL_LAYER_IDS)
    config = _config(spec)
    assert spec_kv_bytes_per_token(spec, config) == REAL_KV_BYTES + REAL_INDEX_BYTES
    assert spec_kv_bytes_per_token(spec, config) == 24576 + 768

    # BSA/DSA keep one index row per token (ratio 1)
    bsa = _spec(num_kv_heads=2, head_dim=256, index_head_dim=128, num_index_layers=12,
                layer_ids=FULL_LAYER_IDS, index_ratio=1, attn_type=AttnType.BSA)
    assert spec_kv_bytes_per_token(bsa, config) == REAL_KV_BYTES + 128 * 12 * 2


def test_kv_cost_prices_ring_and_scratch_as_fixed():
    spec = _spec()
    config = _config(spec, max_running_req=3)
    per_page, fixed, page_tokens, min_reserve = QSAKVCache.kv_cost(config)
    assert per_page == spec_kv_bytes_per_token(spec, config) * 64
    assert page_tokens == 64 and min_reserve == 0
    row = 32 * 4 * 2
    assert fixed == 4 * row * (QSAKVCache.ring_capacity_for(4) + 1)


def test_unit_bytes_matches_the_cost_model():
    spec = _spec()
    config = _config(spec)
    pool = _pool()
    kv_bytes, swa_bytes = pool.unit_bytes()
    assert swa_bytes == 0
    # the scratch rows and the ring must NOT inflate the per-token slider
    assert kv_bytes == spec_kv_bytes_per_token(spec, config)
    assert kv_bytes * 64 == QSAKVCache.kv_cost(config)[0]


def test_resolve_pool_class_and_factory():
    from freetoken.kvcache import create_kvcache_pool, resolve_pool_class

    spec = _spec()
    mc = SimpleNamespace(
        num_layers=8, has_swa_attention=False, has_linear_attention=True,
        num_kv_heads=2, head_dim=64, dsv4_args=None,
    )
    mc.kv_cache_group_specs = lambda: (spec,)
    assert resolve_pool_class(mc) is QSAKVCache

    pool = create_kvcache_pool(
        mc, num_pages=4, page_size=64, dtype=torch.bfloat16, device=DEV, num_req_slots=4
    )
    assert isinstance(pool, QSAKVCache)
    assert pool._kv_buffer.shape[1] == 4  # not the model's 8 layers
    assert pool.cmp_k_cache(0).shape == (4 * 64 // 4 + 4, 32)

    with pytest.raises(ValueError, match="num_req_slots"):
        create_kvcache_pool(mc, num_pages=4, page_size=64, dtype=torch.bfloat16, device=DEV)
