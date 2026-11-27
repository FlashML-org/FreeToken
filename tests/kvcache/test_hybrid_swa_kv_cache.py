from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _init_tp() -> None:
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _hybrid_model_config():
    from freetoken.models.config import (
        FullAttentionGroupConfig,
        ModelConfig,
        RotaryConfig,
        SWAAttentionGroupConfig,
    )

    full_rope = RotaryConfig(
        head_dim=512,
        rotary_dim=256,
        max_position=4096,
        base=1_000_000.0,
        scaling=None,
    )
    swa_rope = RotaryConfig(
        head_dim=256,
        rotary_dim=256,
        max_position=4096,
        base=10_000.0,
        scaling=None,
    )
    return ModelConfig(
        num_layers=6,
        num_qo_heads=16,
        num_kv_heads=8,
        head_dim=256,
        hidden_size=4096,
        vocab_size=32000,
        intermediate_size=8192,
        rms_norm_eps=1e-6,
        rotary_config=swa_rope,
        hidden_act="gelu_tanh",
        tie_word_embeddings=True,
        num_experts=128,
        num_experts_per_tok=8,
        moe_intermediate_size=704,
        norm_topk_prob=True,
        model_type="hybrid_swa_moe",
        architectures=["HybridSWAModel"],
        moe_enabled=True,
        attention_groups=(
            FullAttentionGroupConfig(
                name="full",
                layer_ids=(2, 5),
                num_kv_heads=2,
                head_dim=512,
                rotary_config=full_rope,
                k_eq_v=True,
            ),
            SWAAttentionGroupConfig(
                name="swa",
                layer_ids=(0, 1, 3, 4),
                num_kv_heads=8,
                head_dim=256,
                rotary_config=swa_rope,
                sliding_window=1024,
            ),
        ),
    )


def _kv_group_specs():
    from freetoken.models.config import KVCacheGroupSpec

    return (
        KVCacheGroupSpec(
            name="full",
            layer_ids=(2, 5),
            num_kv_heads=2,
            head_dim=512,
            sliding_window=None,
        ),
        KVCacheGroupSpec(
            name="swa",
            layer_ids=(0, 1, 3, 4),
            num_kv_heads=8,
            head_dim=256,
            sliding_window=1024,
        ),
    )


def _patch_tp(monkeypatch) -> None:
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "freetoken.kvcache.hybrid_swa_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


def test_hybrid_swa_cache_maps_layers_to_physical_groups(monkeypatch):
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache

    _patch_tp(monkeypatch)

    pool = HybridSWAKVCache(
        groups=_kv_group_specs(),
        num_layers=6,
        num_full_pages=4,
        page_size=16,
        num_swa_tokens=32,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    assert pool.full_num_tokens == 64
    assert pool.swa_num_tokens == 32
    assert pool.group_of(2) == "full"
    assert pool.group_of(0) == "swa"
    assert pool.is_full_layer(5)
    assert not pool.is_full_layer(4)
    assert pool.k_cache(2).shape == (4, 16, 2, 512)
    assert pool.v_cache(2).shape == (4, 16, 2, 512)
    assert pool.k_cache(0).shape == (32, 1, 8, 256)
    assert pool.v_cache(0).shape == (32, 1, 8, 256)


def test_create_kvcache_pool_uses_hybrid_swa_cache(monkeypatch):
    from freetoken.kvcache import create_kvcache_pool
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache

    _patch_tp(monkeypatch)

    pool = create_kvcache_pool(
        model_config=_hybrid_model_config(),
        num_pages=4,
        page_size=16,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    assert isinstance(pool, HybridSWAKVCache)
    assert pool.group_of(2) == "full"
    assert pool.group_of(0) == "swa"


def test_adjust_config_resolves_swa_cache_type():
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.engine import _adjust_config
    from freetoken.scheduler.config import SchedulerConfig

    _init_tp()
    # `--cache-type radix` on a SWA model materializes to the global-paged SWA radix cache
    # (SWARadixCache, cross-request reuse).
    radix = SchedulerConfig(
        model_path="/unused",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        cache_type="radix",
    )
    radix.__dict__["model_config"] = _hybrid_model_config()
    _adjust_config(radix)
    assert radix.attention_backend == "triton"
    assert radix.cache_type == "swa_radix"

    # `--cache-type naive` stays 'naive' (NaivePrefixCache, no reuse) on the same paged pool.
    naive = SchedulerConfig(
        model_path="/unused",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        cache_type="naive",
    )
    naive.__dict__["model_config"] = _hybrid_model_config()
    _adjust_config(naive)
    assert naive.attention_backend == "triton"
    assert naive.cache_type == "naive"


def test_unit_bytes_counts_window_state_rows():
    _init_tp()
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
    from freetoken.models.config import KVCacheGroupSpec, WindowStateSpec

    groups = (KVCacheGroupSpec(name="full", layer_ids=(1,), num_kv_heads=1, head_dim=8, sliding_window=None),
              KVCacheGroupSpec(name="swa", layer_ids=(0,), num_kv_heads=1, head_dim=8, sliding_window=16))
    state = WindowStateSpec(name="conv", shape=(8,), layer_ids=(0, 1), window=3)
    pool = HybridSWAKVCache(groups=groups, num_layers=2, num_full_pages=64, page_size=1, dtype=torch.bfloat16,
                            device=torch.device("cpu"), num_swa_tokens=32, window_states=(state,), num_state_rows=(5,))
    kv, swa = pool.unit_bytes()
    # per window token: K + V of one 8-wide head in bf16, plus 2 layers x 5 rows x 8 bf16 over the 32 window tokens
    assert kv == 2 * 8 * 2
    assert swa == 2 * 8 * 2 + 2 * 5 * 8 * 2 // 32
