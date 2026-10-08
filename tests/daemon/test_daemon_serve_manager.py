from __future__ import annotations

import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from freetoken.daemon import proxy
from freetoken.daemon.accounting import (
    AccountingOutbox,
    AccountingOutboxError,
    AccountingPrepareError,
)
from freetoken.daemon.app import build_app
from freetoken.daemon.logring import LogRing
from freetoken.daemon.pidfile import ServeState, ServeStateStore
from freetoken.daemon.serve_manager import Conflict, ExitInfo, ServeManager, spawn_serve


# --------------------------------------------------------------------------- test doubles


class FakeChild:
    def __init__(self, pid: int, *, adopted: bool = False):
        self.pid = pid
        self.starttime = 42
        self.log_path = None
        self.adopted = adopted
        self.reaped = threading.Event()
        self.tailer = None
        self.closed = False
        self._exit = threading.Event()
        self._info = ExitInfo(0, "exited")

    def wait(self) -> ExitInfo:
        self._exit.wait()
        return self._info

    def poll(self):
        return self._info if self._exit.is_set() else None

    def close(self):
        self.closed = True

    def die(self, code: int = 0, source: str = "exited"):
        self._info = ExitInfo(code, source)
        self._exit.set()


class Spawner:
    def __init__(self):
        self.children: list[FakeChild] = []
        self.calls: list[tuple] = []
        self.api_keys: list[str | None] = []
        self._pid = 1000
        self.gate: threading.Event | None = None

    def __call__(self, model, port, args, api_key=None) -> FakeChild:
        self.calls.append((model, port, list(args)))
        self.api_keys.append(api_key)
        if self.gate is not None:
            self.gate.wait()
        self._pid += 1
        c = FakeChild(self._pid)
        self.children.append(c)
        return c

    def by_pid(self, pid: int) -> FakeChild:
        return next(c for c in self.children if c.pid == pid)


def wait_until(pred, timeout=3.0, interval=0.005):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(interval)
    return False


def make_manager(
    tmp_path,
    spawner,
    *,
    signal_fn=None,
    grace_s=0.2,
    auto_restart=False,
    adopt_fn=None,
    prepare_stop=None,
    read_stats=lambda port: {
        "requests": {"promptTokensTotal": 0, "completionTokensTotal": 0},
        "uptimeS": 0,
        "reachable": True,
    },
    accounting_outbox=None,
    wall_now=time.time,
    reap_wait_s=1.0,
):
    ring = LogRing()
    store = ServeStateStore(str(tmp_path / "serve.json"))
    mgr = ServeManager(
        ring,
        store,
        spawn_fn=spawner,
        adopt_fn=adopt_fn,
        tailer_factory=None,
        signal_fn=signal_fn or (lambda pid, sig: None),
        grace_s=grace_s,
        reap_wait_s=reap_wait_s,
        apply_oom=False,
        auto_restart=auto_restart,
        prepare_stop=prepare_stop,
        read_stats=read_stats,
        accounting_outbox=accounting_outbox,
        wall_now=wall_now,
    )
    return mgr, store, ring


# --------------------------------------------------------------------------- start / idempotency


def test_start_reports_running(tmp_path):
    sp = Spawner()
    mgr, store, _ = make_manager(tmp_path, sp)
    res = mgr.start("modelA", 1919, ["--x"])
    assert res["idempotent"] is False
    st = mgr.status()
    assert st["running"] and st["model"] == "modelA" and st["port"] == 1919
    assert st["pid"] == res["pid"]
    # persisted for re-adoption
    saved = store.load()
    assert saved is not None and saved.pid == res["pid"] and saved.args == ["--x"]


def test_idempotent_start_same_params_does_not_respawn(tmp_path):
    sp = Spawner()
    mgr, _, _ = make_manager(tmp_path, sp)
    r1 = mgr.start("m", 1919, [])
    r2 = mgr.start("m", 1919, [])
    assert r2["idempotent"] is True
    assert r1["pid"] == r2["pid"]
    assert len(sp.calls) == 1


def test_conflicting_start_raises(tmp_path):
    sp = Spawner()
    mgr, _, _ = make_manager(tmp_path, sp)
    mgr.start("m", 1919, [])
    with pytest.raises(Conflict):
        mgr.start("other", 1919, [])
    with pytest.raises(Conflict):
        mgr.start("m", 1919, ["--diff"])  # same model/port, different args


def test_concurrent_identical_starts_spawn_once(tmp_path):
    sp = Spawner()
    sp.gate = threading.Event()  # hold the first spawn open to widen the race window
    mgr, _, _ = make_manager(tmp_path, sp)
    results = {}

    def go(tag):
        results[tag] = mgr.start("m", 1919, [])

    a = threading.Thread(target=go, args=("a",))
    a.start()
    assert wait_until(lambda: len(sp.calls) == 1)  # A entered spawn
    b = threading.Thread(target=go, args=("b",))
    b.start()
    time.sleep(0.05)  # give B time to hit the _starting gate and block
    sp.gate.set()
    a.join(3)
    b.join(3)
    assert len(sp.calls) == 1  # B awaited A instead of spawning a second serve
    assert results["a"]["pid"] == results["b"]["pid"]
    assert results["a"]["idempotent"] != results["b"]["idempotent"]  # one real, one idempotent


# --------------------------------------------------------------------------- stop / escalation


def test_stop_sigterm_then_child_exits_no_sigkill(tmp_path):
    sp = Spawner()
    sent = []

    def sig(pid, s):
        sent.append(s)
        if s == signal.SIGTERM:
            sp.by_pid(pid).die(0)  # graceful exit on SIGTERM

    mgr, store, _ = make_manager(tmp_path, sp, signal_fn=sig)
    mgr.start("m", 1919, [])
    res = mgr.stop()
    assert res["stopped"] is True
    kill_signal = getattr(signal, "SIGKILL", None)
    assert kill_signal is None or kill_signal not in sent  # exited within grace → no escalation
    assert wait_until(lambda: mgr.status()["running"] is False)
    assert mgr.status()["lastExitReason"] == "stopped"
    assert store.load() is None  # state cleared on stop


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="POSIX signal escalation")
def test_stop_escalates_to_sigkill_when_grace_elapses(tmp_path):
    sp = Spawner()
    sent = []

    def sig(pid, s):
        sent.append(s)
        if s == signal.SIGKILL:
            sp.by_pid(pid).die(-9, "signalled")  # only dies once force-killed

    mgr, _, _ = make_manager(tmp_path, sp, signal_fn=sig, grace_s=0.15)
    mgr.start("m", 1919, [])
    mgr.stop()
    assert signal.SIGTERM in sent and signal.SIGKILL in sent
    assert wait_until(lambda: mgr.status()["running"] is False)


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="POSIX signal escalation")
def test_stop_fails_closed_when_child_survives_sigkill(tmp_path):
    sp = Spawner()
    sent = []
    mgr, _, _ = make_manager(
        tmp_path,
        sp,
        signal_fn=lambda pid, sig: sent.append(sig),
        grace_s=0.01,
        reap_wait_s=0.01,
    )
    mgr.start("m", 1919, [])

    with pytest.raises(RuntimeError, match="did not exit after SIGKILL"):
        mgr.stop()

    assert sent == [signal.SIGTERM, signal.SIGKILL]
    assert mgr.status()["running"] is True
    with pytest.raises(Conflict, match="serve already running"):
        mgr.start("other", 1919, [])
    assert len(sp.calls) == 1
    sp.children[0].die(-9, "signalled")
    assert wait_until(lambda: mgr.status()["running"] is False)


def test_stop_when_idle_is_noop(tmp_path):
    sp = Spawner()
    mgr, _, _ = make_manager(tmp_path, sp)
    assert mgr.stop() == {"stopped": True, "already": True, "accounting": None}


def _sealed(instance_id="engine-1"):
    return {
        "instance_id": instance_id,
        "model_id": "m",
        "prompt_tokens_total": 123,
        "completion_tokens_total": 45,
        "uptime_s": 9.75,
        "drain_complete": True,
    }


def test_stop_prepares_and_persists_sealed_receipt_before_signal(tmp_path):
    sp = Spawner()
    events = []

    class RecordingOutbox:
        def persist(self, receipt):
            events.append(("persist", receipt))
            return receipt

        def pending(self):
            return []

        def ack(self, receipt_id):
            return {"acked": True, "already": False, "receiptId": receipt_id}

    def prepare(port):
        events.append(("prepare", port))
        return _sealed()

    def sig(pid, s):
        events.append(("signal", s))
        if s == signal.SIGTERM:
            sp.by_pid(pid).die(0)

    mgr, _, _ = make_manager(
        tmp_path,
        sp,
        signal_fn=sig,
        prepare_stop=prepare,
        accounting_outbox=RecordingOutbox(),
        wall_now=lambda: 123.456,
    )
    mgr.start("m", 1919, [])
    result = mgr.stop()
    assert [event[0] for event in events] == ["prepare", "persist", "signal"]
    receipt = result["accounting"]
    assert receipt == events[1][1]
    assert receipt["createdAt"] == 123456
    assert receipt["instanceId"] == "engine-1"
    assert receipt["promptTokensTotal"] == 123
    assert receipt["completionTokensTotal"] == 45
    assert receipt["uptimeS"] == 9
    assert receipt["drainComplete"] is True
    assert receipt["degraded"] is False and receipt["bestEffort"] is False


def test_outbox_failure_never_signals_and_engine_remains_running(tmp_path):
    sp = Spawner()
    sent = []
    outbox = AccountingOutbox(
        str(tmp_path / "outbox"), replace_fn=lambda src, dst: (_ for _ in ()).throw(OSError("full"))
    )
    mgr, _, _ = make_manager(
        tmp_path,
        sp,
        signal_fn=lambda pid, s: sent.append(s),
        prepare_stop=lambda port: _sealed(),
        accounting_outbox=outbox,
    )
    mgr.start("m", 1919, [])
    with pytest.raises(AccountingOutboxError, match="full"):
        mgr.stop()
    assert sent == []
    assert mgr.status()["running"] is True


def test_prepare_failure_is_fail_closed_unless_force_is_explicit(tmp_path):
    sp = Spawner()
    sent = []

    def prepare(port):
        raise AccountingPrepareError("HTTP 503")

    def sig(pid, s):
        sent.append(s)
        if s == signal.SIGTERM:
            sp.by_pid(pid).die(0)

    mgr, _, _ = make_manager(tmp_path, sp, signal_fn=sig, prepare_stop=prepare)
    mgr.start("m", 1919, [])
    with pytest.raises(AccountingPrepareError, match="503"):
        mgr.stop()
    assert sent == [] and mgr.status()["running"] is True

    result = mgr.stop(force=True)
    receipt = result["accounting"]
    assert sent == [signal.SIGTERM]
    assert receipt["degraded"] is True and receipt["bestEffort"] is True
    assert receipt["drainComplete"] is False
    assert receipt["reason"] == "prepare-stop-failed"


def test_legacy_engine_gets_explicit_best_effort_receipt(tmp_path):
    sp = Spawner()

    def sig(pid, s):
        if s == signal.SIGTERM:
            sp.by_pid(pid).die(0)

    mgr, _, _ = make_manager(tmp_path, sp, signal_fn=sig)
    mgr.start("m", 1919, [])
    receipt = mgr.stop()["accounting"]
    assert receipt["reason"] == "legacy-engine"
    assert receipt["degraded"] is True and receipt["bestEffort"] is True
    assert receipt["drainComplete"] is False
    assert receipt["promptTokensTotal"] == 0 and receipt["completionTokensTotal"] == 0


def test_legacy_stats_failure_is_fail_closed_unless_force_is_explicit(tmp_path):
    sp = Spawner()
    sent = []

    def sig(pid, s):
        sent.append(s)
        if s == signal.SIGTERM:
            sp.by_pid(pid).die(0)

    mgr, _, _ = make_manager(tmp_path, sp, signal_fn=sig, read_stats=None)
    mgr.start("m", 1919, [])
    with pytest.raises(AccountingPrepareError, match="legacy-engine stats fallback failed"):
        mgr.stop()
    assert sent == [] and mgr.status()["running"] is True

    receipt = mgr.stop(force=True)["accounting"]
    assert sent == [signal.SIGTERM]
    assert receipt["reason"] == "legacy-engine-stats-unavailable"
    assert receipt["promptTokensTotal"] is None
    assert receipt["completionTokensTotal"] is None


def test_legacy_retry_after_signal_failure_keeps_newer_totals(tmp_path):
    sp = Spawner()
    totals = {"prompt": 10}
    signals = {"count": 0}

    def read_stats(port):
        return {
            "requests": {
                "promptTokensTotal": totals["prompt"],
                "completionTokensTotal": 1,
            },
            "uptimeS": 5,
            "reachable": True,
        }

    def sig(pid, s):
        signals["count"] += 1
        if signals["count"] == 1:
            raise OSError("signal failed")
        if s == signal.SIGTERM:
            sp.by_pid(pid).die(0)

    mgr, _, _ = make_manager(tmp_path, sp, signal_fn=sig, read_stats=read_stats)
    mgr.start("m", 1919, [])
    with pytest.raises(OSError, match="signal failed"):
        mgr.stop()
    assert mgr.status()["running"] is True

    totals["prompt"] = 20
    second = mgr.stop()["accounting"]
    pending = mgr.pending_accounting()
    assert sorted(receipt["promptTokensTotal"] for receipt in pending) == [10, 20]
    assert len({receipt["receiptId"] for receipt in pending}) == 2
    assert second["promptTokensTotal"] == 20


# --------------------------------------------------------------------------- crash injection


def test_crash_is_reaped_and_reported_without_stop(tmp_path):
    sp = Spawner()
    mgr, store, ring = make_manager(tmp_path, sp)
    res = mgr.start("m", 1919, [])
    # Simulate an unexpected crash (OOM): the child dies with no stop() call.
    sp.by_pid(res["pid"]).die(1, "exited")
    assert wait_until(lambda: mgr.status()["running"] is False)
    st = mgr.status()
    assert st["lastExitCode"] == 1
    assert st["lastExitReason"] == "exited"  # NOT "stopped" — it was a crash
    assert store.load() is None  # dead serve is not left adoptable
    texts = " ".join(r["text"] for r in ring.since(0)[0])
    assert "exited with code 1" in texts


def test_crash_persists_last_observed_degraded_receipt(tmp_path):
    sp = Spawner()
    mgr, store, _ = make_manager(tmp_path, sp, wall_now=lambda: 123.456)
    started = mgr.start("m", 1919, [])
    mgr.observe_accounting(
        {
            "reachable": True,
            "instanceId": "engine-crash-1",
            "model": {"id": "m"},
            "requests": {
                "promptTokensTotal": 17,
                "completionTokensTotal": 4,
            },
            "uptimeS": 9,
        }
    )

    sp.by_pid(started["pid"]).die(1, "exited")
    assert wait_until(lambda: mgr.status()["running"] is False)

    receipts = mgr.pending_accounting()
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["instanceId"] == "engine-crash-1"
    assert receipt["promptTokensTotal"] == 17
    assert receipt["completionTokensTotal"] == 4
    assert receipt["reason"] == "engine-crashed"
    assert receipt["degraded"] is True and receipt["drainComplete"] is False
    assert store.load() is None


def test_crash_during_failed_prepare_still_persists_receipt_and_never_restarts(tmp_path):
    sp = Spawner()

    def prepare(_port):
        sp.children[0].die(1, "exited")
        raise AccountingPrepareError("connection lost")

    mgr, _, _ = make_manager(
        tmp_path, sp, prepare_stop=prepare, auto_restart=True
    )
    mgr.start("m", 1919, [])
    with pytest.raises(AccountingPrepareError, match="connection lost"):
        mgr.stop()

    assert wait_until(lambda: mgr.status()["running"] is False)
    assert len(sp.calls) == 1
    receipts = mgr.pending_accounting()
    assert len(receipts) == 1
    assert receipts[0]["reason"] == "engine-crashed-during-stop"
    assert receipts[0]["receiptId"]


def test_prepare_failure_keeps_stop_latch_for_a_later_crash(tmp_path):
    sp = Spawner()

    def prepare(_port):
        raise AccountingPrepareError("outbox unavailable")

    mgr, _, _ = make_manager(
        tmp_path, sp, prepare_stop=prepare, auto_restart=True
    )
    started = mgr.start("m", 1919, [])
    with pytest.raises(AccountingPrepareError, match="outbox unavailable"):
        mgr.stop()

    # The child was still alive when prepare failed. If it crashes later, the earlier explicit
    # stop intent must still suppress auto-restart; no racy poll() result may clear that latch.
    sp.by_pid(started["pid"]).die(1, "exited")
    assert wait_until(lambda: mgr.status()["running"] is False)
    assert len(sp.calls) == 1


def test_auto_restart_on_crash(tmp_path):
    sp = Spawner()
    mgr, _, _ = make_manager(tmp_path, sp, auto_restart=True)
    r1 = mgr.start("m", 1919, [])
    sp.by_pid(r1["pid"]).die(1, "exited")
    # A new serve should come up automatically.
    assert wait_until(lambda: mgr.status()["running"] and mgr.status()["pid"] != r1["pid"])
    assert len(sp.calls) == 2


# --------------------------------------------------------------------------- switch


def test_stop_during_inflight_start_stops_the_new_serve(tmp_path):
    # A stop() racing a start() must wait out the spawn and then stop the freshly-started serve,
    # never no-op while the serve is left running.
    sp = Spawner()
    sp.gate = threading.Event()

    def sig(pid, s):
        if s == signal.SIGTERM:
            sp.by_pid(pid).die(0)

    mgr, _, _ = make_manager(tmp_path, sp, signal_fn=sig)
    results = {}
    a = threading.Thread(target=lambda: results.__setitem__("start", mgr.start("m", 1919, [])))
    a.start()
    assert wait_until(lambda: len(sp.calls) == 1)  # A is inside the gated spawn
    b = threading.Thread(target=lambda: results.__setitem__("stop", mgr.stop()))
    b.start()
    time.sleep(0.05)  # B reaches `while _starting: wait()` and blocks
    sp.gate.set()
    a.join(3)
    b.join(3)
    assert results["stop"]["stopped"] is True
    assert wait_until(lambda: mgr.status()["running"] is False)  # new serve was stopped, not left up


def test_auto_restart_does_not_resurrect_after_user_stop(tmp_path):
    sp = Spawner()

    def sig(pid, s):
        if s == signal.SIGTERM:
            sp.by_pid(pid).die(0)

    mgr, _, _ = make_manager(tmp_path, sp, signal_fn=sig, auto_restart=True)
    mgr.start("m", 1919, [])
    mgr.stop()  # user stop → latches _stop_requested
    assert wait_until(lambda: mgr.status()["running"] is False)
    # An auto-restart attempt now must abort (the user asked for the engine to be down).
    assert mgr.start("m", 1919, [], _auto=True) == {"pid": None, "aborted": True}
    assert len(sp.calls) == 1  # no respawn
    # An explicit client start clears the stop intent and runs again.
    r = mgr.start("m", 1919, [])
    assert r["idempotent"] is False and len(sp.calls) == 2


def test_switch_stops_old_and_starts_new(tmp_path):
    sp = Spawner()

    def sig(pid, s):
        if s == signal.SIGTERM:
            sp.by_pid(pid).die(0)

    mgr, _, _ = make_manager(tmp_path, sp, signal_fn=sig)
    r1 = mgr.start("m1", 1919, [])
    r2 = mgr.switch("m2", 1919, [])
    assert r2["pid"] != r1["pid"]
    st = mgr.status()
    assert st["running"] and st["model"] == "m2"
    assert len(sp.calls) == 2
    assert r2["accounting"]["reason"] == "legacy-engine"


def test_switch_is_one_serial_transaction_with_no_interleaving_start(tmp_path):
    sp = Spawner()
    prepare_entered = threading.Event()
    prepare_release = threading.Event()

    def prepare(port):
        prepare_entered.set()
        assert prepare_release.wait(2)
        return _sealed("old-engine")

    def sig(pid, s):
        if s == signal.SIGTERM:
            sp.by_pid(pid).die(0)

    mgr, _, _ = make_manager(tmp_path, sp, signal_fn=sig, prepare_stop=prepare)
    mgr.start("m", 1919, [])
    results = {}

    switcher = threading.Thread(
        target=lambda: results.__setitem__("switch", mgr.switch("m2", 1919, []))
    )

    def intruding_start():
        try:
            mgr.start("intruder", 1919, [])
        except Conflict as exc:
            results["intruder"] = exc

    switcher.start()
    assert prepare_entered.wait(2)
    intruder = threading.Thread(target=intruding_start)
    intruder.start()
    time.sleep(0.05)
    assert len(sp.calls) == 1
    prepare_release.set()
    switcher.join(3)
    intruder.join(3)

    assert results["switch"]["accounting"]["instanceId"] == "old-engine"
    assert isinstance(results["intruder"], Conflict)
    assert [call[0] for call in sp.calls] == ["m", "m2"]
    assert mgr.status()["model"] == "m2"


def test_shutdown_latch_rejects_start_queued_behind_final_stop(tmp_path):
    sp = Spawner()
    prepare_entered = threading.Event()
    prepare_release = threading.Event()

    def prepare(port):
        prepare_entered.set()
        assert prepare_release.wait(2)
        return _sealed()

    def sig(pid, s):
        if s == signal.SIGTERM:
            sp.by_pid(pid).die(0)

    mgr, _, _ = make_manager(tmp_path, sp, signal_fn=sig, prepare_stop=prepare)
    mgr.start("m", 1919, [])
    results = {}
    shutdown = threading.Thread(target=lambda: results.__setitem__("shutdown", mgr.shutdown()))

    def late_start():
        try:
            mgr.start("late", 1919, [])
        except Conflict as exc:
            results["start"] = exc

    shutdown.start()
    assert prepare_entered.wait(2)
    starter = threading.Thread(target=late_start)
    starter.start()
    prepare_release.set()
    shutdown.join(3)
    starter.join(3)

    assert results["shutdown"]["stopped"] is True
    assert "shutdown is in progress" in str(results["start"])
    assert len(sp.calls) == 1
    assert mgr.status()["running"] is False


def test_failed_shutdown_reopens_lifecycle_because_daemon_remains_up(tmp_path):
    sp = Spawner()

    def fail_prepare(port):
        raise AccountingPrepareError("cannot seal")

    mgr, _, _ = make_manager(tmp_path, sp, prepare_stop=fail_prepare)
    mgr.start("m", 1919, [])
    with pytest.raises(AccountingPrepareError, match="cannot seal"):
        mgr.shutdown()
    assert mgr.start("m", 1919, [])["idempotent"] is True


def test_switch_outbox_failure_preserves_old_engine_and_does_not_start_new(tmp_path):
    sp = Spawner()
    sent = []
    outbox = AccountingOutbox(
        str(tmp_path / "outbox"), replace_fn=lambda src, dst: (_ for _ in ()).throw(OSError("full"))
    )
    mgr, _, _ = make_manager(
        tmp_path,
        sp,
        signal_fn=lambda pid, sig: sent.append(sig),
        prepare_stop=lambda port: _sealed("old-engine"),
        accounting_outbox=outbox,
    )
    old = mgr.start("m", 1919, [])
    with pytest.raises(AccountingOutboxError, match="full"):
        mgr.switch("m2", 1919, [])
    assert sent == []
    assert [call[0] for call in sp.calls] == ["m"]
    status = mgr.status()
    assert status["running"] is True and status["pid"] == old["pid"] and status["model"] == "m"


# --------------------------------------------------------------------------- re-adoption


def test_readopt_attaches_running_serve(tmp_path):
    sp = Spawner()
    store = ServeStateStore(str(tmp_path / "serve.json"))
    store.save(ServeState(model="m", port=1919, pid=4242, args=["--y"], starttime=7))
    adopted = FakeChild(4242, adopted=True)

    ring = LogRing()
    mgr = ServeManager(
        ring, store, spawn_fn=sp, adopt_fn=lambda state: adopted, tailer_factory=None,
        signal_fn=lambda p, s: None, apply_oom=False,
    )
    assert mgr.readopt() is True
    st = mgr.status()
    assert st["running"] and st["adopted"] is True and st["pid"] == 4242 and st["model"] == "m"


def test_readopt_clears_stale_state(tmp_path):
    store = ServeStateStore(str(tmp_path / "serve.json"))
    store.save(ServeState(model="m", port=1919, pid=999999, args=[], starttime=1))
    ring = LogRing()
    mgr = ServeManager(
        ring, store, spawn_fn=Spawner(), adopt_fn=lambda state: None, tailer_factory=None,
        signal_fn=lambda p, s: None, apply_oom=False,
    )
    assert mgr.readopt() is False
    assert store.load() is None  # a serve that is no longer alive is cleared, not adopted


def test_readopt_no_state_is_noop(tmp_path):
    store = ServeStateStore(str(tmp_path / "serve.json"))
    ring = LogRing()
    mgr = ServeManager(ring, store, spawn_fn=Spawner(), tailer_factory=None,
                       signal_fn=lambda p, s: None, apply_oom=False)
    assert mgr.readopt() is False


# --------------------------------------------------------------------------- API key


def test_api_key_is_serve_config_kept_owner_only_and_readopted(tmp_path):
    sp = Spawner()
    mgr, store, _ = make_manager(tmp_path, sp)
    # an older daemon left a world-readable state file and a stale tmp behind
    for path in (store.path, f"{store.path}.tmp"):
        with open(path, "w") as fh:
            fh.write("{}")
        os.chmod(path, 0o644)
    mgr.start("m", 1919, [], "k1")
    assert sp.api_keys == ["k1"] and mgr.serve_api_key() == "k1"
    assert mgr.start("m", 1919, [], "k1")["idempotent"] is True
    with pytest.raises(Conflict):
        mgr.start("m", 1919, [], "k2")
    assert store.load().api_key == "k1"
    assert os.stat(store.path).st_mode & 0o777 == 0o600

    adopted = FakeChild(4242, adopted=True)
    fresh = ServeManager(
        LogRing(), store, spawn_fn=Spawner(), adopt_fn=lambda state: adopted, tailer_factory=None,
        signal_fn=lambda p, s: None, apply_oom=False,
    )
    assert fresh.readopt() is True and fresh.serve_api_key() == "k1"
    adopted.die()
    sp.children[0].die()


def test_switch_replaces_the_api_key(tmp_path):
    sp = Spawner()

    def sig(pid, s):
        if s == signal.SIGTERM:
            sp.by_pid(pid).die(0)

    mgr, store, _ = make_manager(tmp_path, sp, signal_fn=sig)
    mgr.start("m", 1919, [], "k1")
    mgr.switch("m", 1919, [], False, "k2")
    assert sp.api_keys == ["k1", "k2"]
    assert mgr.serve_api_key() == "k2" and store.load().api_key == "k2"
    sp.children[1].die()


def test_auto_restart_keeps_the_api_key(tmp_path):
    sp = Spawner()
    mgr, _, _ = make_manager(tmp_path, sp, auto_restart=True)
    mgr.start("m", 1919, [], "k1")
    sp.children[0].die(1)
    assert wait_until(lambda: len(sp.children) == 2)
    assert sp.api_keys == ["k1", "k1"]
    sp.children[1].die()


def test_spawned_serve_sees_only_the_requested_key(tmp_path, monkeypatch):
    monkeypatch.setenv("FREETOKEN_API_KEY", "from-daemon-env")
    code = "import os; print(os.environ.get('FREETOKEN_API_KEY'))"
    for api_key in (None, "k1"):
        log = tmp_path / f"{api_key}.log"
        child = spawn_serve([sys.executable, "-c", code], str(log), api_key)
        child.wait()
        child.close()
        assert log.read_text().strip() == str(api_key)


def test_probe_sends_the_running_serves_key(monkeypatch):
    seen = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout):
        seen.append(req.get_header("Authorization"))
        return Response()

    monkeypatch.setattr(proxy.urllib.request, "urlopen", fake_urlopen)
    key = None
    probe = proxy.ServeProbe(api_key=lambda: key)
    probe.fresh_stats(1919)
    key = "k1"
    probe.fresh_stats(1919)
    probe.prepare_stop(1919)
    assert seen == [None, "Bearer k1", "Bearer k1"]


def test_start_and_switch_routes_hand_the_api_key_to_the_manager():
    calls = []

    class Manager:
        def status(self):
            return {"running": False}

        def start(self, *args):
            calls.append(args)
            return {"pid": 1, "idempotent": False}

        def switch(self, *args):
            calls.append(args)
            return {"pid": 1, "idempotent": False, "accounting": None}

    with ThreadPoolExecutor(1) as lifecycle_pool, ThreadPoolExecutor(1) as proxy_pool:
        app = build_app(
            manager=Manager(), ring=LogRing(), probe=None, footprint_fn=lambda pid: {},
            lifecycle_pool=lifecycle_pool, proxy_pool=proxy_pool,
        )
        with TestClient(app) as client:
            client.post("/engine/start", json={"model": "m", "port": 1919, "apiKey": "k1"})
            client.post("/engine/start", json={"model": "m", "port": 1919, "apiKey": ""})
            client.post("/engine/start", json={"model": "m", "port": 1919})
            client.post("/engine/switch", json={"model": "m", "port": 1919, "apiKey": "k2"})
            # argparse also takes an unambiguous prefix of --api-key
            for args in (["--api-key", "k3"], ["--api-key=k3"], ["--api", "k3"], ["--ap=k3"]):
                rejected = client.post("/engine/start", json={"model": "m", "args": args})
                assert rejected.status_code == 400
            allowed = client.post("/engine/start", json={"model": "m", "args": ["--attn", "fa"]})
            assert allowed.status_code == 200
    assert [args[-1] for args in calls] == ["k1", None, None, "k2", None]


# --------------------------------------------------------------------------- resource leak (L2)


def test_no_thread_leak_over_many_cycles(tmp_path):
    sp = Spawner()

    def sig(pid, s):
        if s == signal.SIGTERM:
            sp.by_pid(pid).die(0)

    mgr, _, _ = make_manager(tmp_path, sp, signal_fn=sig)
    baseline = threading.active_count()
    for _ in range(30):
        mgr.start("m", 1919, [])
        mgr.stop()
        assert wait_until(lambda: mgr.status()["running"] is False)
    # monitor threads must all have exited (daemon threads terminate after _reap)
    assert wait_until(lambda: threading.active_count() <= baseline + 2)
