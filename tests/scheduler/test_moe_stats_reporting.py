from types import SimpleNamespace
from typing import Any


def test_moe_decode_stats_log_only_after_request_finishes(monkeypatch):
    import freetoken.scheduler.scheduler as scheduler_module
    from freetoken.scheduler.scheduler import Scheduler

    expected = {
        "requested_rows": 470,
        "miss_rows": 80,
        "hit_rows": 390,
        "bytes_h2d": 1234,
    }
    cache = SimpleNamespace(
        collect_stats=True,
        decode_miss_stats=lambda: expected,
    )
    scheduler: Any = Scheduler.__new__(Scheduler)
    scheduler.engine = SimpleNamespace(moe_offload_cache=cache)
    scheduler.config = SimpleNamespace(tp_info=SimpleNamespace(rank=0))
    logged = []
    monkeypatch.setattr(scheduler_module.logger, "info_rank0", logged.append)

    scheduler._report_moe_decode_stats(set())
    assert logged == []

    scheduler._report_moe_decode_stats({object()})
    assert len(logged) == 1
    assert "cumulative" in logged[0]
    assert "rank-local" in logged[0]
    assert "'bytes_h2d': 1234" in logged[0]

    scheduler.config.tp_info.rank = 1
    scheduler._report_moe_decode_stats({object()})
    assert len(logged) == 1
