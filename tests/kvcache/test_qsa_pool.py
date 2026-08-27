from __future__ import annotations

import torch

from freetoken.distributed.info import DistributedInfo
from freetoken.kvcache.qsa_pool import QSAKVCache


def _pool(monkeypatch, pages=3):
    monkeypatch.setattr(
        "freetoken.kvcache.mha_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )
    return QSAKVCache(
        num_kv_heads=2,
        num_layers=6,
        head_dim=16,
        num_pages=pages,
        page_size=64,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        index_num_kv_heads=1,
        index_head_dim=8,
        compress_ratio=4,
        layer_ids=(1, 5),
    )


def test_qsa_pool_geometry_and_cost(monkeypatch):
    pool = _pool(monkeypatch)
    assert pool.k_cache(1).shape == (3, 64, 2, 16)
    assert pool.k_cache(5).shape == (3, 64, 2, 16)
    assert pool.compressed_k_cache(1).shape == (3 * 16, 1, 8)
    kv, swa = pool.unit_bytes()
    assert kv == 2 * 2 * 2 * 16 * 2 + 2 * 1 * 8 * 2 // 4
    assert swa == 0


def test_qsa_pool_rebuild_and_compressed_rows(monkeypatch):
    pool = _pool(monkeypatch)
    keys = torch.arange(16, dtype=torch.bfloat16).view(2, 1, 8)
    pool.store_compressed_k(keys, torch.tensor([0, 17]), layer_id=5)
    assert torch.equal(pool.compressed_k_cache(5)[17], keys[1])
    pool.rebuild(5)
    assert pool.k_cache(1).shape[0] == 5
    assert pool.compressed_k_cache(5).shape == (5 * 16, 1, 8)


def test_qsa_pending_ring_validates_logical_positions(monkeypatch):
    pool = _pool(monkeypatch)
    pool.ensure_pending_capacity(4)
    positions = torch.tensor([5, 6, 7])
    keys = torch.randn(3, 1, 8, dtype=torch.bfloat16)
    rope = torch.tensor([[5, 5, 5], [6, 7, 8], [7, 9, 11]])
    pool.store_pending(5, 2, positions, keys, rope)
    assert torch.equal(pool.pending_group(5, 2, positions), keys)
    assert torch.equal(pool.pending_rope_group(5, 2, positions), rope)
