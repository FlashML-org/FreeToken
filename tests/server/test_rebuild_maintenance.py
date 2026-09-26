"""State-machine tests for the runtime cache-rebuild maintenance gate (api_server).

dispatch_rebuild latches maintenance_state to "rebuilding"; the only ways back to a definite
state are the scheduler's reply (via FrontendManager._resolve_rebuild), a dispatch-time error,
or the liveness watchdog on a crash. These tests pin all three edge paths so a rebuild can
never wedge the server in "rebuilding" forever:

  1. dispatch exception   -> gate rolls back to "serving" (the scheduler never got the request)
  2. HTTP wait timeout    -> stays "rebuilding" on purpose, but the late reply still resolves it
  3. scheduler crash      -> the watchdog latches "failed", wakes the in-flight rebuild waiter
                             (so it fails promptly, not on timeout), and a buffered reply can't
                             undo the latch

They use a lightweight fake state exposing only the handful of attributes the code touches, so
no ZMQ / listener task / GPU is needed. _resolve_rebuild is exercised as an unbound method on
the fake — the real method logic, a stand-in ``self``.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient

from freetoken.server.api_server import FrontendManager, dispatch_rebuild
from freetoken.server.supervisor import (
    BackendHandle,
    LoadProgress,
    run_backend_supervisor,
)


class _FakeState:
    """Stand-in for FrontendManager exposing exactly what dispatch_rebuild / _resolve_rebuild /
    fail_pending_rebuilds read: rebuild_futures, maintenance_state, fatal_error, last_rebuild,
    the event loop (_loop, for cross-thread future resolution), and an async send_one delegating
    to an injected impl (so a test can make the enqueue succeed or raise)."""

    def __init__(self, send_impl, *, maintenance_state="serving", fatal_error=None):
        self.rebuild_futures: dict = {}
        self.maintenance_state = maintenance_state
        self.fatal_error = fatal_error
        self.last_rebuild = None
        self._loop = None
        self._send_impl = send_impl

    async def send_one(self, msg):
        await self._send_impl(msg)


def _reply(request_id, status, **over):
    base = dict(
        request_id=request_id,
        status=status,
        moe_cache_size=0,
        num_pages=0,
        mamba_slots=0,
        num_swa_pages=0,
        error=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_dispatch_exception_returns_to_serving():
    """Enqueue failure (e.g. a transient ZMQ error): the scheduler never received the request,
    so the gate must roll back to serving — not latch "rebuilding" with no reply ever coming."""

    async def boom(_msg):
        raise RuntimeError("zmq push failed")

    state = _FakeState(boom)

    async def _run():
        return await dispatch_rebuild(state, moe_cache_size=8, num_pages=None)

    result = asyncio.run(_run())
    assert result["status"] == "failed"
    assert "zmq push failed" in result["error"]
    assert state.maintenance_state == "serving"
    assert state.rebuild_futures == {}  # no dangling future leaked


def test_timeout_stays_rebuilding_then_late_reply_resolves():
    """The HTTP wait timing out deliberately leaves the gate "rebuilding" (the scheduler may
    still be mid-recapture). That is not a dead end: the eventual reply resolves it."""

    async def ok(_msg):
        return None  # enqueue succeeds, but nothing ever resolves the future -> timeout

    state = _FakeState(ok)

    async def _run():
        return await dispatch_rebuild(state, moe_cache_size=8, num_pages=None, timeout=0.01)

    result = asyncio.run(_run())
    assert result["status"] == "timeout"
    assert state.maintenance_state == "rebuilding"  # still gated on purpose
    assert state.rebuild_futures == {}  # cancelled future dropped, not leaked

    # The late reply is what un-wedges it -> definite "serving".
    FrontendManager._resolve_rebuild(state, _reply(result["request_id"], "ok", num_pages=1024))
    assert state.maintenance_state == "serving"
    assert state.last_rebuild["num_pages"] == 1024


def test_resolve_failed_latches_failed():
    state = _FakeState(None, maintenance_state="rebuilding")
    FrontendManager._resolve_rebuild(state, _reply("r1", "failed", error="OOM during recapture"))
    assert state.maintenance_state == "failed"
    assert state.last_rebuild["error"] == "OOM during recapture"


def test_resolve_nonfatal_statuses_keep_serving():
    # ok / busy / rejected / unsupported all leave the prior cache intact -> keep serving.
    for status in ("ok", "busy", "rejected", "unsupported"):
        state = _FakeState(None, maintenance_state="rebuilding")
        FrontendManager._resolve_rebuild(state, _reply("r1", status))
        assert state.maintenance_state == "serving", status


def test_late_reply_cannot_resurrect_a_fatal_latch():
    """A worker crash latched "failed" (the watchdog). A buffered "ok" reply that raced the
    crash must NOT reopen the gate — a dead backend cannot serve — yet it still wakes the
    waiter blocked on that request_id (the reply path) and is recorded for observability."""

    async def _run():
        state = _FakeState(None, maintenance_state="failed", fatal_error="scheduler exited")
        # A caller is still parked on this request_id's future (the "late reply" the name
        # promises): _resolve_rebuild must wake it even though the gate stays latched.
        fut = asyncio.get_running_loop().create_future()
        state.rebuild_futures["r1"] = fut

        FrontendManager._resolve_rebuild(state, _reply("r1", "ok", num_pages=2048))

        assert state.maintenance_state == "failed"          # gate stays latched
        assert fut.done() and fut.result()["num_pages"] == 2048  # waiter still woken
        assert state.rebuild_futures == {}                  # future consumed, not leaked
        # The reply is still recorded for observability; the gate stays latched.
        assert state.last_rebuild["num_pages"] == 2048

    asyncio.run(_run())


def test_crash_during_rebuild_latches_failed_via_watchdog():
    """Scheduler-crash path, end to end and cross-thread: a worker dies while a rebuild is in
    flight ("rebuilding"). No reply ever arrives, but the liveness watchdog (on its own thread)
    fires on_failure, which mirrors production _on_failure — it (a) drives the gate to a definite
    "failed" and (b) wakes the in-flight rebuild waiter so dispatch_rebuild returns "failed"
    PROMPTLY, instead of stranding the caller until its full (here 30 s) timeout."""

    class Proc:
        name = "freetoken-TP0-scheduler"

        def __init__(self):
            self._alive = True

        def is_alive(self):
            return self._alive

    async def _run():
        proc = Proc()
        q: "queue.Queue" = queue.Queue()
        q.put("scheduler ready")
        handle = BackendHandle(ack_queue=q, processes=[proc], expected_acks=1)

        async def ok(_msg):
            return None  # enqueue succeeds; the backend then crashes without ever replying

        state = _FakeState(ok, maintenance_state="serving")
        state._loop = asyncio.get_running_loop()
        ready_evt = threading.Event()

        def on_ready():
            state.maintenance_state = "serving"
            ready_evt.set()

        # Production _on_failure: latch failed AND wake any pending rebuild waiter — called here
        # from the supervisor thread, so fail_pending_rebuilds must marshal onto the loop.
        def on_failure(message):
            state.fatal_error = message
            state.maintenance_state = "failed"
            FrontendManager.fail_pending_rebuilds(state, message)

        sup = threading.Thread(
            target=run_backend_supervisor,
            args=(handle, LoadProgress(), on_ready),
            kwargs={"on_failure": on_failure, "poll": 0.01},
            daemon=True,
        )
        sup.start()
        # Let the supervisor drain to readiness and enter the post-ready watch loop.
        while not ready_evt.is_set():
            await asyncio.sleep(0.005)
        assert state.maintenance_state == "serving"

        # A rebuild is now in flight: a real pending future, gate latched "rebuilding".
        task = asyncio.create_task(
            dispatch_rebuild(state, moe_cache_size=8, num_pages=None, timeout=30.0)
        )
        await asyncio.sleep(0)
        assert state.rebuild_futures          # a waiter is parked
        assert state.maintenance_state == "rebuilding"

        proc._alive = False  # …and the scheduler crashes mid-rebuild

        # The waiter is woken promptly (well under the 30 s timeout) with a failed result.
        result = await asyncio.wait_for(task, timeout=5.0)
        assert result["status"] == "failed"
        assert "scheduler" in result["error"]
        assert state.maintenance_state == "failed"  # escaped "rebuilding" to a definite state
        assert "scheduler" in state.fatal_error
        assert state.rebuild_futures == {}          # waiter resolved and cleared, not leaked
        sup.join(timeout=1.0)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# The maintenance gate seen from the request side: what a client hits while the server is
# loading or rebuilding, on both the OpenAI generation routes and the rebuild route itself.
# ---------------------------------------------------------------------------


def test_openai_gate_message_is_loading_aware():
    from freetoken.server.openai_api import _maintenance_gate

    assert _maintenance_gate(SimpleNamespace(maintenance_state="serving")) is None
    loading = _maintenance_gate(SimpleNamespace(maintenance_state="loading"))
    assert loading is not None and loading.status_code == 503
    assert b"loading" in loading.body.lower()
    rebuild = _maintenance_gate(SimpleNamespace(maintenance_state="rebuilding"))
    assert rebuild is not None and rebuild.status_code == 503
    assert b"rebuild" in rebuild.body.lower()
    failed = _maintenance_gate(SimpleNamespace(maintenance_state="failed"))
    assert failed is not None and failed.status_code == 503
    # A state object without the attribute defaults to serving (defensive, never blocks).
    assert _maintenance_gate(SimpleNamespace()) is None


def test_cache_rebuild_guarded_during_loading():
    import freetoken.server.api_server as api

    prev = api._GLOBAL_STATE
    api._GLOBAL_STATE = SimpleNamespace(
        maintenance_state="loading",
        rebuild_futures={},
        last_rebuild=None,
    )
    try:
        client = TestClient(api.app)
        r = client.post("/v1/cache/rebuild", json={})
        assert r.status_code == 503
        assert "loading" in r.json().get("error", "").lower()
    finally:
        api._GLOBAL_STATE = prev


def test_cache_rebuild_timeout_keeps_gate_closed():
    # On HTTP timeout the scheduler may still be mid-rebuild, so the endpoint must NOT
    # reopen the maintenance gate -- it stays "rebuilding" until the real reply arrives.
    import asyncio
    from types import SimpleNamespace

    from freetoken.server import api_server
    from freetoken.server.api_server import CacheRebuildRequest, cache_rebuild

    sent = []

    async def send_one(msg):
        sent.append(msg)

    state = SimpleNamespace(
        maintenance_state="serving", rebuild_futures={}, last_rebuild=None, send_one=send_one
    )
    api_server._GLOBAL_STATE = state
    try:
        resp = asyncio.run(
            cache_rebuild(CacheRebuildRequest(moe_cache_size=8, timeout=0.05))
        )  # future never resolves -> times out
    finally:
        api_server._GLOBAL_STATE = None

    assert resp.status_code == 504
    assert state.maintenance_state == "rebuilding"  # gate stays closed
    assert state.rebuild_futures == {}  # cancelled future dropped, no leak
    assert len(sent) == 1  # the rebuild request was still dispatched to the backend


def test_cache_rebuild_send_failure_rolls_back_gate():
    # If dispatching the rebuild message fails, the scheduler never received it and the engine
    # is untouched -- the gate must roll back to serving, not latch maintenance forever.
    import asyncio
    from types import SimpleNamespace

    from freetoken.server import api_server
    from freetoken.server.api_server import CacheRebuildRequest, cache_rebuild

    async def boom(msg):
        raise RuntimeError("zmq down")

    state = SimpleNamespace(
        maintenance_state="serving", rebuild_futures={}, last_rebuild=None, send_one=boom
    )
    api_server._GLOBAL_STATE = state
    try:
        resp = asyncio.run(cache_rebuild(CacheRebuildRequest(moe_cache_size=8, timeout=5.0)))
    finally:
        api_server._GLOBAL_STATE = None

    assert resp.status_code == 503
    assert state.maintenance_state == "serving"  # rolled back, not latched
    assert state.rebuild_futures == {}  # dangling future cleaned up


def test_cache_rebuild_request_rejects_unknown_mode():
    # The public request model only accepts the implemented mode; "drain" is deferred and
    # must fail fast at the validation layer (422), not reach the scheduler.
    import pytest
    from pydantic import ValidationError

    from freetoken.server.api_server import CacheRebuildRequest

    assert CacheRebuildRequest(mode="if_idle").mode == "if_idle"
    with pytest.raises(ValidationError):
        CacheRebuildRequest(mode="drain")


from pydantic import ValidationError  # noqa: E402

from freetoken.server.api_server import CacheRebuildRequest  # noqa: E402


def test_rebuild_timeout_bounds():
    import pytest

    for bad in (0, -5, 3601, float("nan")):
        with pytest.raises(ValidationError):
            CacheRebuildRequest(timeout=bad)
    assert CacheRebuildRequest(timeout=3600).timeout == 3600
    assert CacheRebuildRequest().timeout == 300.0
