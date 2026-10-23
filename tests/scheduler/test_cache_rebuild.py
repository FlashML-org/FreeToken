"""The cache-rebuild path where it has an in-process seam: the scheduler's idle gate and the
pool/table re-point it performs. The destructive orchestration underneath (graph teardown, pool
resize, page-table refresh, graph re-capture) has no seam worth stubbing and is covered against a
real server by tests/e2e/test_cache_rebuild.py; the maintenance state machine the HTTP layer runs
on top lives in tests/server/test_rebuild_maintenance.py. Also the /v1/cache/flush path:
CacheManager.flush and the scheduler's deferred-flush gate."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
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
    sched._pending_flush = None
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

    sched = Scheduler.__new__(Scheduler)
    sched.prefill_manager = SimpleNamespace(runnable=False)
    sched.decode_manager = SimpleNamespace(runnable=False)
    sched.device = torch.device("cpu")
    sched.config = SimpleNamespace(tp_info=SimpleNamespace(size=1), max_extend_tokens=100_000)
    sched.engine = SimpleNamespace(
        rebuild_runtime_cache=lambda **kw: None, num_pages=32, page_table=None
    )
    # engine.page_table unchanged across the (stubbed) rebuild -> no token_pool re-point.
    sched.table_manager = SimpleNamespace(page_table=None)
    # DSV4-like manager: prefill_chunk_budget tracks the (about-to-shrink) window pool; no shared
    # page table, so rebuild_cache's prefix-cache rebuild branch is skipped.
    cache_manager = SimpleNamespace(
        prefill_chunk_budget=5000, rebuild=lambda *a: None, check_integrity=lambda: None)
    sched.cache_manager = cache_manager
    sched.table_manager.rebuild = lambda pt: None
    sched.table_manager.token_pool = None
    sched.prefill_budget = min(sched.config.max_extend_tokens, cache_manager.prefill_chunk_budget)
    assert sched.prefill_budget == 5000

    cache_manager.prefill_chunk_budget = 1000  # the (stubbed) engine rebuild shrank the pool
    Scheduler.rebuild_cache(sched, num_pages=16)
    assert sched.prefill_budget == 1000  # tracks the shrunk cap, not the stale 5000


def _flush_manager(cache_type: str):
    from freetoken.scheduler.cache import CacheManager

    ps, num_pages = 2, 16
    pt = _page_table(4, num_pages * ps)
    if cache_type == "hybrid_radix":
        from freetoken.kvcache.linear_state_pool import LinearStatePool
        from freetoken.models.config import LinearGatedDeltaGroupConfig

        g = LinearGatedDeltaGroupConfig(
            name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
            key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
        )
        pool = LinearStatePool(group=g, num_slots=8, dtype=torch.bfloat16,
                               device=torch.device("cpu"), tp_size=1)
        return CacheManager(num_pages, ps, pt, cache_type, linear_state_pool=pool)
    if cache_type == "swa_radix":
        from freetoken.distributed import set_tp_info, try_get_tp_info
        from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
        from freetoken.models.config import KVCacheGroupSpec

        if try_get_tp_info() is None:
            set_tp_info(rank=0, size=1)

        groups = (
            KVCacheGroupSpec(name="full", layer_ids=(1,), num_kv_heads=1, head_dim=8,
                             sliding_window=None),
            KVCacheGroupSpec(name="swa", layer_ids=(0,), num_kv_heads=1, head_dim=8,
                             sliding_window=ps),
        )
        pool = HybridSWAKVCache(
            groups=groups, num_layers=2, num_full_pages=num_pages, page_size=ps,
            dtype=torch.bfloat16, device=torch.device("cpu"), num_swa_tokens=num_pages * ps,
        )
        return CacheManager(num_pages, ps, pt, cache_type, swa_pool=pool, sliding_window_size=ps)
    return CacheManager(num_pages, ps, pt, cache_type)


def _insert(cm, ids: list[int]) -> None:
    """Commit ``ids`` to the tree through the tree's own insert, returning duplicate slots like cache_req does."""
    ids = torch.tensor(ids, dtype=torch.int32)
    kv = cm._page_to_token(cm._allocate(len(ids) // cm.page_size))
    if cm.is_swa:
        cm.swa_pool.alloc_swa(kv)
        _, dup = cm.prefix_cache.insert(ids, kv)
        cm._free_swa(dup)
        cm._free(dup)
    elif cm.is_hybrid:
        matched, _ = cm.prefix_cache.insert(ids, kv, cm.linear_state_pool.alloc(1)[0])
        cm._free(kv[:matched])
    else:
        cm._free(kv[: cm.prefix_cache.insert_prefix(ids, kv).cached_len])


@pytest.mark.parametrize("cache_type", ["radix", "hybrid_radix", "swa_radix"])
def test_cache_manager_flush_returns_every_pool_slot(cache_type):
    _setup_context(page_size=2)
    cm = _flush_manager(cache_type)
    # The shared [1..4] prefix becomes an internal node, so flush must also evict a parent left childless.
    _insert(cm, [1, 2, 3, 4, 5, 6, 7, 8])
    _insert(cm, [1, 2, 3, 4, 20, 21, 22, 23])
    _insert(cm, [30, 31])
    cm.check_integrity()

    assert cm.flush() == 14
    assert len(cm.free_slots) == cm.num_pages
    if cm.is_hybrid:
        assert cm.linear_state_pool.num_free_slots == 8 - 1  # slot 0 is padding
    if cm.is_swa:
        assert cm.swa_pool.swa_available_size() == cm.swa_pool.swa_num_tokens - 1  # slot 0 is the sentinel
    cm.check_integrity()
    pending = SimpleNamespace(input_ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int32), input_len=9)
    assert cm.match_req(pending).cuda_handle.cached_len == 0


@pytest.mark.parametrize("loop", ["normal", "overlap"])
def test_deferred_flush_runs_at_idle_and_times_out_while_busy(monkeypatch, loop):
    from freetoken.message import FlushCacheBackendMsg
    from freetoken.scheduler import scheduler as scheduler_mod
    from freetoken.scheduler.scheduler import Scheduler

    clock = [0.0]
    monkeypatch.setattr(scheduler_mod, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    sched, _ = _stub_scheduler(prefill_runnable=False, decode_runnable=False, pending=None)
    sched.prefill_manager.pending_list = []
    sched.decode_manager.running_reqs = set()
    sched.config = SimpleNamespace(tp_info=SimpleNamespace(size=1))
    sched._last_data = None
    sched.stream = SimpleNamespace(wait_stream=lambda stream: None)
    sched.engine = SimpleNamespace(stream=None)
    flushed = []
    sched.cache_manager = SimpleNamespace(flush=lambda: flushed.append(True) or 0, check_integrity=lambda: None)
    sent = []
    sched.send_result = lambda msgs: sent.extend((m.request_id, m.status) for m in msgs)
    blocked = []
    sched.receive_msg = lambda blocking: blocked.append(blocking) or []

    def step(busy: bool) -> None:
        if loop == "normal":
            sched.decode_manager.runnable = busy
            Scheduler.normal_loop(sched)
        else:
            # Only the in-flight batch keeps the overlap scheduler busy here.
            Scheduler.overlap_loop(sched, object() if busy else None)

    Scheduler._process_one_msg(sched, FlushCacheBackendMsg(request_id="a", timeout=10.0))
    Scheduler._process_one_msg(sched, FlushCacheBackendMsg(request_id="b", timeout=10.0))
    assert sent == [("b", "busy")]

    # Busy before the deadline: still waiting.
    clock[0] = 5.0
    step(busy=True)
    assert len(sent) == 1 and not flushed

    # Idle: runs without new traffic, so the loop must not block on receive.
    step(busy=False)
    assert blocked[-1] is False
    assert flushed == [True] and sent[-1] == ("a", "ok")
    assert sched._pending_flush is None

    # Busy past the deadline: fails and clears the slot.
    Scheduler._process_one_msg(sched, FlushCacheBackendMsg(request_id="c", timeout=10.0))
    clock[0] = 15.0
    step(busy=True)
    assert sent[-1] == ("c", "timeout")
    assert flushed == [True] and sched._pending_flush is None

    # timeout=0 never waits: busy fails in the same iteration, idle flushes in the same iteration.
    Scheduler._process_one_msg(sched, FlushCacheBackendMsg(request_id="d", timeout=0.0))
    step(busy=True)
    assert sent[-1] == ("d", "busy") and sched._pending_flush is None
    Scheduler._process_one_msg(sched, FlushCacheBackendMsg(request_id="e", timeout=0.0))
    step(busy=False)
    assert sent[-1] == ("e", "ok") and flushed == [True, True]
