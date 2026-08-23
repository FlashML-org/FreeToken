"""The cache-rebuild path where it has an in-process seam: the scheduler's idle gate and the
pool/table re-point it performs. The destructive orchestration underneath (graph teardown, pool
resize, page-table refresh, graph re-capture) has no seam worth stubbing and is covered against a
real server by tests/e2e/test_cache_rebuild.py; the maintenance state machine the HTTP layer runs
on top lives in tests/server/test_rebuild_maintenance.py."""

from __future__ import annotations

import torch


def _page_table(max_running_reqs: int, width: int) -> torch.Tensor:
    return torch.zeros((max_running_reqs + 1, width), dtype=torch.int32, device=torch.device("cpu"))


def _setup_context(page_size: int) -> None:
    """Initialize global context if not already done."""
    from freetoken.core import Context, get_global_ctx, set_global_ctx

    try:
        get_global_ctx()
    except AssertionError:
        # Create minimal context
        ctx = Context(page_size=page_size)
        set_global_ctx(ctx)


def test_cache_manager_rebuild_resets_pages_and_prefix():
    from freetoken.scheduler.cache import CacheManager

    _setup_context(page_size=2)

    pt = _page_table(4, 64)
    cm = CacheManager(num_pages=8, page_size=2, page_table=pt, type="radix")
    # mutate state so we can prove rebuild resets it
    cm.free_slots = cm.free_slots[:3]

    new_pt = _page_table(4, 128)
    cm.rebuild(num_pages=20, page_table=new_pt)

    assert cm.num_pages == 20
    assert cm.page_table is new_pt
    assert cm.free_slots.tolist() == [i * 2 for i in range(20)]
    assert cm.prefix_cache.size_info.total_size == 0
    cm.check_integrity()  # must pass: free_pages(20) + cache_pages(0) == num_pages(20)


def test_table_manager_rebuild_reallocs_token_pool_and_frees_slots():
    from freetoken.scheduler.table import TableManager

    pt = _page_table(4, 64)
    tm = TableManager(max_running_reqs=4, page_table=pt)
    tm.allocate(); tm.allocate()  # consume 2 slots

    new_pt = _page_table(4, 128)
    tm.rebuild(new_pt)

    assert tm.page_table is new_pt
    assert tm.token_pool.shape == new_pt.shape
    assert tm.available_size == 4  # all slots free again


def _stub_scheduler(*, prefill_runnable: bool, decode_runnable: bool, pending: object | None):
    """A Scheduler shell (no __init__/GPU) wired just enough to drive normal_loop's
    rebuild-drain branch. _execute_pending_rebuild is replaced with a recorder."""
    from types import SimpleNamespace

    from freetoken.scheduler.scheduler import Scheduler

    sched = Scheduler.__new__(Scheduler)
    sched.prefill_manager = SimpleNamespace(runnable=prefill_runnable)
    sched.decode_manager = SimpleNamespace(runnable=decode_runnable)
    sched._pending_rebuild = pending
    sched.receive_msg = lambda blocking: []
    sched._schedule_next_batch = lambda: None
    sched._process_last_data = lambda data: None
    calls = []

    def _exec():
        calls.append(True)
        sched._pending_rebuild = None

    sched._execute_pending_rebuild = _exec
    return sched, calls


def test_normal_loop_executes_pending_rebuild_when_idle():
    # Non-overlap mode (DISABLE_OVERLAP_SCHEDULING) must drain a queued rebuild at the idle
    # safe point, else it hangs until the HTTP request times out.
    from freetoken.scheduler.scheduler import Scheduler

    sched, calls = _stub_scheduler(prefill_runnable=False, decode_runnable=False, pending=object())
    Scheduler.normal_loop(sched)
    assert calls == [True]
    assert sched._pending_rebuild is None


def test_normal_loop_defers_pending_rebuild_while_busy():
    # A queued rebuild must NOT run while prefill/decode is still in flight.
    from freetoken.scheduler.scheduler import Scheduler

    pending = object()
    sched, calls = _stub_scheduler(prefill_runnable=False, decode_runnable=True, pending=pending)
    Scheduler.normal_loop(sched)
    assert calls == []
    assert sched._pending_rebuild is pending  # still queued
































def test_rebuild_cache_refreshes_prefill_budget(monkeypatch):
    # A rebuild that shrank the DSV4 window pool must shrink Scheduler.prefill_budget to the new
    # prefill_chunk_budget, or the next long prompt is chunked against the stale (larger) cap.
    from types import SimpleNamespace

    from freetoken.scheduler.scheduler import Scheduler

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)

    from freetoken.scheduler.cache import CacheManager

    pool = SimpleNamespace(prefill_chunk_budget=5000, swa_paged=False)
    page_table = _page_table(4, 64)
    cache_manager = CacheManager(
        num_pages=8, page_size=2, page_table=page_table, type="radix", swa_pool=pool
    )

    sched = Scheduler.__new__(Scheduler)
    sched.prefill_manager = SimpleNamespace(runnable=False)
    sched.decode_manager = SimpleNamespace(runnable=False)
    sched.device = torch.device("cpu")
    sched.config = SimpleNamespace(tp_info=SimpleNamespace(size=1), max_extend_tokens=100_000)
    def rebuild_runtime_cache(**kw):
        # DSV4 pools rebuild in place; their capability changes without replacing the object held
        # by CacheManager.
        pool.prefill_chunk_budget = 1000

    sched.engine = SimpleNamespace(
        rebuild_runtime_cache=rebuild_runtime_cache, num_pages=8, page_table=page_table
    )
    # A window-only rebuild leaves the generic page table and token pool in place.
    sched.table_manager = SimpleNamespace(page_table=page_table)
    sched.cache_manager = cache_manager
    sched.prefill_budget = min(sched.config.max_extend_tokens, cache_manager.prefill_chunk_budget)
    assert sched.prefill_budget == 5000

    Scheduler.rebuild_cache(sched, num_swa_pages=10)
    assert cache_manager.prefill_chunk_budget == 1000
    assert sched.prefill_budget == 1000  # tracks the shrunk cap, not the stale 5000


def test_destructive_swa_rollback_restores_prior_capacity_source():
    from types import SimpleNamespace

    from freetoken.scheduler.scheduler import Scheduler

    sched = Scheduler.__new__(Scheduler)
    sched._pending_rebuild = SimpleNamespace(
        request_id="r1", moe_cache_size=None, num_pages=None,
        num_mamba_slots=None, num_swa_pages=10,
    )
    sched.config = SimpleNamespace(
        tp_info=SimpleNamespace(size=1), swa_capacity_source="derived"
    )
    sched.engine = SimpleNamespace(rebuild_teardown_started=False)
    snapshot = {
        "moe_cache_size": 4602, "num_pages": 4096, "num_mamba_slots": None,
        "num_swa_pages": 407, "requested_prefill_tokens": 24576,
        "pool_prefill_cap_tokens": 24576, "effective_prefill_tokens": 24576,
        "swa_capacity_source": "derived",
        "prefill_limiting_reason": "requested_and_swa_pool",
    }
    sched._current_cache_geometry = lambda: dict(snapshot)
    calls = []

    def rebuild_cache(**targets):
        calls.append(targets)
        object.__setattr__(sched.config, "swa_capacity_source", "explicit")
        if len(calls) == 1:
            sched.engine.rebuild_teardown_started = True
            raise RuntimeError("injected post-teardown failure")

    replies = []
    sched.rebuild_cache = rebuild_cache
    sched._log_cache_geometry = lambda event: None
    sched._reply_rebuild = lambda request_id, status, error=None: replies.append(
        (request_id, status, error)
    )

    Scheduler._execute_pending_rebuild(sched)

    assert calls == [
        {"moe_cache_size": None, "num_pages": None, "num_mamba_slots": None,
         "num_swa_pages": 10},
        {"num_swa_pages": 407},
    ]
    assert sched.config.swa_capacity_source == "derived"
    assert replies[0][1] == "rejected"
