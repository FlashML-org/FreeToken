from __future__ import annotations

from types import SimpleNamespace

from freetoken.attention import AttnType


def test_kv_usage_pages_excludes_evictable_prefix_cache():
    # page_usage now lives on the CacheManagerLike interface (polymorphic vs DSV4); test the
    # generic formula there. The unbound method works on a duck-typed namespace.
    from freetoken.scheduler.cache import CacheManager

    cache_manager = SimpleNamespace(
        num_pages=100,
        page_size=4,
        free_slots=[0] * 20,
        is_hybrid=False,
        is_swa=False,
        prefix_cache=SimpleNamespace(
            size_info=SimpleNamespace(evictable_size=120, protected_size=40)
        ),
    )

    used_pages, total_pages = CacheManager.page_usage(cache_manager)

    assert used_pages == 50
    assert total_pages == 100


def test_swa_token_usage_counts_window_pool():
    from freetoken.scheduler.scheduler import Scheduler

    # swa_paged model: 129-token pool -> 128 allocatable (sentinel excluded from total);
    # 64 available (free + evictable tree) -> 64 used.
    cache_manager = SimpleNamespace(
        swa_paged=True,
        swa_pool=SimpleNamespace(swa_num_tokens=129),
        swa_available_size=64,
    )
    assert Scheduler._swa_token_usage(SimpleNamespace(cache_manager=cache_manager)) == (64, 128)

    # non-SWA models report None (no swa field in logs/stats).
    plain = SimpleNamespace(cache_manager=SimpleNamespace(swa_paged=False))
    assert Scheduler._swa_token_usage(plain) is None


def _fake_engine(swa=True, moe=True, mamba=True):
    import torch

    # Superset fake engine driving the REAL compute_cache_pools/compute_cache_unit_bytes:
    # KV 64 pages x 16 tokens x 1 MiB/token = 1.00 GiB; swa 512 tokens x 2 MiB = 1.00 GiB;
    # mamba 64 slots x 16 MiB = 1.00 GiB; MoE 24 slots of 8x16 experts, 4 KiB/slot.
    return SimpleNamespace(
        num_pages=64,
        config=SimpleNamespace(
            page_size=16,
            cache_type="swa_radix",
            max_running_req=1,
            model_config=SimpleNamespace(
                dsv4_args=None, has_swa_attention=swa,
                kv_cache_group_specs=lambda: [SimpleNamespace(attn_type=AttnType.SWA, is_swa=True, sliding_window=128)] if swa else [],
                swa_attention_group=lambda: SimpleNamespace(sliding_window=128),
            ),
        ),
        kv_cache=SimpleNamespace(
            window_pages=512, swa_num_tokens=513, unit_bytes=lambda: (1 << 20, 1 << 21)
        ),
        moe_offload_cache=SimpleNamespace(
            cache_size=24, num_layers=8, num_experts=16,
            bank_caches={"w": torch.zeros((24, 1024), dtype=torch.float32)},
        ) if moe else None,
        linear_state_pool=SimpleNamespace(num_slots=65, bytes_per_slot=lambda: 1 << 24)
        if mamba else None,
    )


def test_log_cache_geometry_reports_all_pools(monkeypatch):
    import freetoken.scheduler.scheduler as sched_mod
    from freetoken.scheduler.scheduler import Scheduler

    lines: list[str] = []
    monkeypatch.setattr(sched_mod.logger, "info_rank0", lines.append)
    Scheduler._log_cache_geometry(SimpleNamespace(engine=_fake_engine()), "Cache rebuilt")
    line = lines[-1]
    assert "Cache rebuilt: KV 64 pages (1024 tokens, 1.00 GiB)" in line
    assert "swa 512 pages (512 tokens, 1.00 GiB)" in line
    assert "mamba 64 slots (1.00 GiB)" in line
    assert "MoE cache 24/128 (0.00 GiB)" in line


def test_log_cache_geometry_plain_model_kv_only(monkeypatch):
    import freetoken.scheduler.scheduler as sched_mod
    from freetoken.scheduler.scheduler import Scheduler

    lines: list[str] = []
    monkeypatch.setattr(sched_mod.logger, "info_rank0", lines.append)
    engine = _fake_engine(swa=False, moe=False, mamba=False)
    Scheduler._log_cache_geometry(SimpleNamespace(engine=engine), "Cache rebuilt")
    line = lines[-1]
    assert "Cache rebuilt: KV 64 pages (1024 tokens, 1.00 GiB)" in line
    assert "swa" not in line and "mamba" not in line and "MoE" not in line
