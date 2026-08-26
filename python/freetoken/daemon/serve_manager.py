"""The supervisor heart: own exactly one ``ft serve`` child's lifecycle.

Spawn, kill, and orphan detection all live here, in the daemon's own process, so parent and child
are co-located and ordinary signals suffice. Torch-free: it only ever launches ``ft serve`` in a
child process and talks to it via signals + ``/proc`` — it never imports the runtime.

Threading contract:
  * One ``threading.Condition`` (``self._cond``) guards all in-memory state. NOTHING blocking
    (spawn, signal, disk IO, callbacks) runs while the lock is held — mutate + snapshot under the
    lock, act outside it. Injected callbacks must never re-enter the manager.
  * ``start()`` is a single serialized gate: alive+exact-match → idempotent no-op; alive+differs
    → 409 Conflict; a start already in flight → await it (Condition), never a second spawn.
  * Exactly ONE reaper: the per-child monitor thread blocks in ``child.wait()`` and calls
    ``_reap`` once. ``stop()`` only signals then waits on the child's ``reaped`` Event — it never
    reaps, so there is no double-reap race and ``switch`` can't see a half-cleared ``_child``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from . import osproc
from .accounting import (
    AccountingOutbox,
    AccountingOutboxError,
    AccountingPrepareError,
    PrepareStopUnavailable,
    stable_receipt_id,
)


class Conflict(RuntimeError):
    """A different serve (model/port/args) is already running; the client should switch()."""


@dataclass
class ExitInfo:
    code: int | None  # Popen convention: >=0 exit status, <0 == -signal; None if unknowable
    source: str  # "exited" | "signalled" | "stopped" | "adopted-vanished" | "unknown"


# --------------------------------------------------------------------------- child abstractions


class PopenChild:
    """A serve WE spawned. ``wait()`` is a real ``os.waitpid`` via Popen."""

    def __init__(self, proc: subprocess.Popen, log_path: str | None) -> None:
        self.proc = proc
        self.pid = proc.pid
        self.starttime = osproc.read_starttime(proc.pid)
        self.log_path = log_path
        self.adopted = False
        self.reaped = threading.Event()
        self.tailer = None

    def wait(self) -> ExitInfo:
        code = self.proc.wait()
        return ExitInfo(code, "signalled" if code is not None and code < 0 else "exited")

    def poll(self) -> ExitInfo | None:
        code = self.proc.poll()
        if code is None:
            return None
        return ExitInfo(code, "signalled" if code < 0 else "exited")

    def close(self) -> None:  # Popen owns its child fds; nothing extra to release
        pass


class AdoptedChild:
    """A serve a PREVIOUS daemon spawned, re-adopted from the pidfile. It is NOT our child, so
    ``os.waitpid`` would raise ECHILD — exit is detected by polling identity instead, and the
    exit code is unknowable."""

    def __init__(
        self,
        pid: int,
        port: int,
        starttime: int | None,
        log_path: str | None,
        *,
        alive_check: Callable[[], bool],
        sleep: Callable[[float], None],
        poll_interval: float,
    ) -> None:
        self.pid = pid
        self.port = port
        self.starttime = starttime
        self.log_path = log_path
        self.adopted = True
        self.reaped = threading.Event()
        self.tailer = None
        self._alive = alive_check
        self._sleep = sleep
        self._poll = poll_interval

    def wait(self) -> ExitInfo:
        while self._alive():
            self._sleep(self._poll)
        return ExitInfo(None, "adopted-vanished")

    def poll(self) -> ExitInfo | None:
        return None if self._alive() else ExitInfo(None, "adopted-vanished")

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- default real spawn


def build_serve_command(
    model: str, port: int, args: list[str], *, python: str, log_dir: str
) -> tuple[list[str], str]:
    """The serve invocation + its log path. ``python -m freetoken.cli serve`` (NOT ``-m
    freetoken``, which is a direct-server entrypoint that ignores subcommand argv) so it uses the
    daemon's own interpreter/venv with no PATH dependency."""
    argv = [python, "-m", "freetoken.cli", "serve", "--model", model, "--port", str(port), *args]
    log_path = os.path.join(log_dir, f"serve-{port}.log")
    return argv, log_path


def spawn_serve(argv: list[str], log_path: str) -> PopenChild:
    """Spawn the serve as a session leader with its stdout+stderr going to a real logfile fd (never
    a PIPE): the file survives daemon death, so a re-adopting daemon just resumes tailing it and
    the serve never writes to a dead pipe. ``start_new_session`` makes it a process
    group leader so signals reach the whole worker tree via the group."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logf = open(log_path, "wb", buffering=0)  # truncate: a fresh serve gets a fresh log
    try:
        proc = subprocess.Popen(
            argv,
            stdout=logf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        logf.close()  # the child holds its own dup; the parent must not keep this fd
    return PopenChild(proc, log_path)


# --------------------------------------------------------------------------- the manager


class ServeManager:
    def __init__(
        self,
        ring,
        state_store,
        *,
        spawn_fn: Callable[[str, int, list[str]], object] | None = None,
        adopt_fn: Callable[[object], object] | None = None,
        tailer_factory: Callable[[object], object] | None = None,
        signal_fn: Callable[[int, int], None] | None = None,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        wall_now: Callable[[], float] = time.time,
        grace_s: float = 10.0,
        reap_wait_s: float = 10.0,
        poll_interval_s: float = 1.0,
        oom_child_score: int = 500,
        apply_oom: bool = True,
        auto_restart: bool = False,
        python: str = sys.executable,
        log_dir: str = ".",
        prepare_stop: Callable[[int], dict[str, Any]] | None = None,
        read_stats: Callable[[int], dict[str, Any]] | None = None,
        accounting_outbox: AccountingOutbox | None = None,
    ) -> None:
        self._ring = ring
        self._store = state_store
        self._now = now
        self._sleep = sleep
        self._wall_now = wall_now
        self._grace_s = grace_s
        self._reap_wait_s = reap_wait_s
        self._poll_interval_s = poll_interval_s
        self._oom_child_score = oom_child_score
        self._apply_oom = apply_oom
        self._auto_restart = auto_restart
        self._python = python
        self._log_dir = log_dir
        self._spawn_fn = spawn_fn or self._default_spawn
        self._adopt_fn = adopt_fn or self._default_adopt
        self._tailer_factory = tailer_factory
        self._signal = signal_fn or osproc.signal_group
        self._prepare_stop = prepare_stop
        self._read_stats = read_stats
        if accounting_outbox is None:
            state_path = getattr(state_store, "path", "serve.json")
            outbox_dir = os.path.join(os.path.dirname(state_path) or ".", "accounting-outbox")
            accounting_outbox = AccountingOutbox(outbox_dir)
        self._accounting = accounting_outbox
        # Serialize final prepare receipts against monitor-thread crash receipts. The monitor
        # must not clear/restart a dead generation while stop is still deciding whether its
        # final receipt is durable.
        self._accounting_guard = threading.RLock()
        self._durable_accounting: set[str] = set()
        self._last_accounting: dict[str, dict[str, Any]] = {}

        # Serialize complete lifecycle transactions, including prepare -> durable receipt -> signal.
        # RLock lets switch() compose stop+start without opening an interleaving window.
        self._lifecycle = threading.RLock()
        self._cond = threading.Condition(threading.Lock())
        # state guarded by _cond
        self._child: object | None = None
        self._model: str | None = None
        self._port: int | None = None
        self._args: list[str] = []
        self._started_at: float | None = None
        self._last_exit: ExitInfo | None = None
        self._adopted = False
        self._stopping = False
        self._starting = False
        # Set by stop(); cleared by an explicit client start(). Prevents an auto-restart (or a
        # start racing a stop) from resurrecting a serve the user asked to stop.
        self._stop_requested = False
        # Permanent after a successful daemon shutdown transaction. It closes the final
        # stop->uvicorn-exit race so no queued or late request can spawn a new serve that the
        # daemon would then detach. Cleared only when shutdown itself fails and the daemon stays.
        self._shutdown_requested = False

    # ---- default factories (overridable for tests) ----

    def _default_spawn(self, model: str, port: int, args: list[str]):
        argv, log_path = build_serve_command(
            model, port, args, python=self._python, log_dir=self._log_dir
        )
        return spawn_serve(argv, log_path)

    def _default_adopt(self, state):
        if not osproc.is_ft_serve_on_port(state.pid, state.port, starttime=state.starttime):
            return None
        return AdoptedChild(
            state.pid,
            state.port,
            state.starttime,
            state.log_path,
            alive_check=lambda: osproc.is_ft_serve_on_port(
                state.pid, state.port, starttime=state.starttime
            ),
            sleep=self._sleep,
            poll_interval=self._poll_interval_s,
        )

    # ---- lifecycle ----

    def start(
        self, model: str, port: int, args: list[str] | None = None, *, _auto: bool = False
    ) -> dict:
        with self._lifecycle:
            return self._start(model, port, args, _auto=_auto)

    def _start(
        self, model: str, port: int, args: list[str] | None = None, *, _auto: bool = False
    ) -> dict:
        args = list(args or [])
        with self._cond:
            if self._shutdown_requested:
                if _auto:
                    return {"pid": None, "aborted": True}
                raise Conflict("daemon shutdown is in progress; new serves are disabled")
            if _auto:
                # An auto-restart: honor a stop the user requested during/after the crash.
                if self._stop_requested:
                    return {"pid": None, "aborted": True}
            else:
                self._stop_requested = False  # an explicit client start clears a prior stop intent
            while True:
                if self._child is not None and not self._stopping:
                    if (self._model, self._port, self._args) == (model, port, args):
                        return {"pid": self._child.pid, "idempotent": True}
                    raise Conflict(
                        f"serve already running (model={self._model!r} port={self._port}); "
                        f"use switch to replace it"
                    )
                if self._starting:
                    # A spawn is in flight. Wait for it, then re-evaluate against the result
                    # (idempotent success if it matches us, Conflict if it doesn't).
                    self._cond.wait()
                    continue
                # Re-check the stop latch here too, not only once above: a restart thread that
                # parked in the wait() above could have missed a stop() that latched while it
                # slept. Without this it would resurrect a serve the user stopped.
                if _auto and self._stop_requested:
                    return {"pid": None, "aborted": True}
                self._starting = True
                break

        child = None
        try:
            child = self._spawn_fn(model, port, args)
        finally:
            if child is None:
                with self._cond:
                    self._starting = False
                    self._cond.notify_all()

        with self._cond:
            self._child = child
            self._model = model
            self._port = port
            self._args = args
            self._started_at = self._now()
            self._last_exit = None
            self._adopted = getattr(child, "adopted", False)
            self._stopping = False
            self._starting = False
            self._cond.notify_all()

        self._persist(child, model, port, args)
        self._maybe_apply_oom(child.pid)
        self._begin_watch(child)
        self._emit(f"serve started (pid={child.pid} model={model} port={port})")
        return {"pid": child.pid, "idempotent": False}

    def stop(self, timeout: float | None = None, force: bool = False) -> dict:
        with self._lifecycle:
            return self._stop(timeout, force)

    def shutdown(self, timeout: float | None = None, force: bool = False) -> dict:
        """Permanently close admission to new serves, then stop the current child.

        The latch and stop share one lifecycle transaction. A start already ahead of us is
        included in the stop; every start queued behind us observes the latch and is rejected.
        If accounting/signalling fails, the daemon remains up and normal lifecycle calls reopen.
        """
        with self._lifecycle:
            with self._cond:
                self._shutdown_requested = True
                self._cond.notify_all()
            try:
                return self._stop(timeout, force)
            except Exception:
                with self._cond:
                    self._shutdown_requested = False
                    self._cond.notify_all()
                raise

    def _stop(self, timeout: float | None = None, force: bool = False) -> dict:
        grace = self._grace_s if timeout is None else timeout
        with self._cond:
            self._stop_requested = True  # latch intent so a racing auto-restart can't undo it
            # Wait out any in-flight spawn so `_child` reflects the just-started serve; otherwise a
            # stop racing a start could no-op while the fresh serve is left running.
            while self._starting:
                self._cond.wait()
            child = self._child
            if child is None:
                return {"stopped": True, "already": True, "accounting": None}
            model = self._model
            port = self._port
            self._stopping = True
            self._cond.notify_all()

        try:
            receipt = self._prepare_and_persist_accounting(
                child=child, model=model, port=port, force=force
            )
        except Exception:
            # No signal has been sent, so keep the child visible/retryable. Preserve the stop
            # latch fail-closed: Popen.poll() can transiently report "alive" while the monitor's
            # waitpid owns its internal lock, and clearing the latch in that window would let a
            # stop-time crash auto-restart. An explicit start clears the latch if the user really
            # wants this generation (or its replacement) running again.
            with self._cond:
                if self._child is child:
                    self._stopping = False
                self._cond.notify_all()
            raise

        # Signal the group, then wait for the monitor (the sole reaper) to observe exit and set
        # `reaped`. Never derive completion from poll() — it can transiently lie while the
        # monitor's waitpid holds Popen's internal lock.
        try:
            if not child.reaped.is_set():
                self._signal(child.pid, signal.SIGTERM)
                if not child.reaped.wait(timeout=grace):
                    self._signal(child.pid, signal.SIGKILL)
                    if not child.reaped.wait(timeout=self._reap_wait_s):
                        raise RuntimeError(
                            f"serve pid={child.pid} did not exit after SIGKILL"
                        )
        except Exception:
            # The receipt is durable, but the process may still be alive.  Keep it visible so a
            # retry can take another (possibly newer, for legacy engines) snapshot and signal it.
            with self._cond:
                if self._child is child:
                    self._stopping = False
                self._cond.notify_all()
            raise
        return {"stopped": True, "accounting": receipt}

    def switch(
        self,
        model: str,
        port: int,
        args: list[str] | None = None,
        force: bool = False,
    ) -> dict:
        with self._lifecycle:
            stopped = self._stop(force=force)
            started = self._start(model, port, args)
            return {**started, "accounting": stopped["accounting"]}

    def pending_accounting(self) -> list[dict[str, Any]]:
        return self._accounting.pending()

    def ack_accounting(self, receipt_id: str) -> dict[str, Any]:
        return self._accounting.ack(receipt_id)

    def observe_accounting(self, snapshot: dict[str, Any]) -> None:
        """Remember the latest validated totals for the current child.

        A whole-process crash can no longer be queried, so the daemon's most recent successful
        stats observation is the best exact snapshot available for its degraded crash receipt.
        """
        with self._cond:
            child = self._child
            model = self._model
        if child is None:
            return
        identity = self._child_identity(child)
        try:
            instance_id, model_id, prompt_total, completion_total, uptime_s = (
                self._snapshot_values(snapshot, model=model, require_instance=False)
            )
        except AccountingPrepareError:
            return
        with self._accounting_guard:
            self._last_accounting[identity] = {
                "instanceId": instance_id,
                "modelId": model_id,
                "promptTokensTotal": int(prompt_total),
                "completionTokensTotal": int(completion_total),
                "uptimeS": int(uptime_s),
            }

    def _prepare_and_persist_accounting(
        self,
        *,
        child,
        model: str | None,
        port: int | None,
        force: bool,
    ) -> dict[str, Any]:
        identity = self._child_identity(child)
        with self._accounting_guard:
            receipt = self._prepare_and_persist_accounting_locked(
                child=child, model=model, port=port, force=force
            )
            self._durable_accounting.add(identity)
            self._last_accounting.pop(identity, None)
            return receipt

    def _prepare_and_persist_accounting_locked(
        self,
        *,
        child,
        model: str | None,
        port: int | None,
        force: bool,
    ) -> dict[str, Any]:
        identity = self._child_identity(child)
        try:
            if self._prepare_stop is None:
                raise PrepareStopUnavailable("legacy-engine")
            if port is None:
                raise AccountingPrepareError("running engine has no control port")
            prepared = self._prepare_stop(port)
            receipt = self._sealed_receipt(prepared, model=model)
        except PrepareStopUnavailable as exc:
            receipt = self._best_effort_receipt(
                identity,
                model=model,
                port=port,
                reason="legacy-engine",
                force=force,
                detail=str(exc),
            )
        except AccountingPrepareError as exc:
            if not force:
                raise
            receipt = self._best_effort_receipt(
                identity,
                model=model,
                port=port,
                reason="prepare-stop-failed",
                force=True,
                detail=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 - injected transport must fail closed
            error = AccountingPrepareError(f"prepare-stop failed: {exc}")
            if not force:
                raise error from exc
            receipt = self._best_effort_receipt(
                identity,
                model=model,
                port=port,
                reason="prepare-stop-failed",
                force=True,
                detail=str(exc),
            )
        return self._accounting.persist(receipt)

    def _persist_crash_accounting(
        self,
        child,
        *,
        model: str | None,
        was_stopping: bool,
        info: ExitInfo,
    ) -> dict[str, Any] | None:
        """Persist one identity-stable degraded receipt before a dead child is forgotten."""
        identity = self._child_identity(child)
        with self._accounting_guard:
            if identity in self._durable_accounting:
                return None
            snapshot = self._last_accounting.get(identity)
            receipt = {
                "receiptId": stable_receipt_id(f"crash:{identity}"),
                "createdAt": max(0, int(self._wall_now() * 1000)),
                "instanceId": snapshot.get("instanceId") if snapshot else None,
                "modelId": snapshot.get("modelId") if snapshot else model,
                "promptTokensTotal": snapshot.get("promptTokensTotal") if snapshot else None,
                "completionTokensTotal": (
                    snapshot.get("completionTokensTotal") if snapshot else None
                ),
                "uptimeS": snapshot.get("uptimeS") if snapshot else None,
                "drainComplete": False,
                "degraded": True,
                "bestEffort": True,
                "reason": "engine-crashed-during-stop" if was_stopping else "engine-crashed",
                "detail": f"exit source={info.source} code={info.code}",
            }
            persisted = self._accounting.persist(receipt)
            self._durable_accounting.add(identity)
            self._last_accounting.pop(identity, None)
            return persisted

    def _sealed_receipt(
        self, prepared: dict[str, Any], *, model: str | None
    ) -> dict[str, Any]:
        if not isinstance(prepared, dict):
            raise AccountingPrepareError("prepare-stop returned a non-object response")

        instance_id, model_id, prompt_total, completion_total, uptime_s = self._snapshot_values(
            prepared, model=model, require_instance=True
        )
        drain_complete = prepared.get("drain_complete", prepared.get("drainComplete"))

        if drain_complete is not True:
            raise AccountingPrepareError("prepare-stop response is not fully drained")

        return {
            "receiptId": stable_receipt_id(f"instance:{instance_id}"),
            "createdAt": max(0, int(self._wall_now() * 1000)),
            "instanceId": instance_id,
            "modelId": model_id,
            "promptTokensTotal": int(prompt_total),
            "completionTokensTotal": int(completion_total),
            "uptimeS": int(uptime_s),
            "drainComplete": True,
            "degraded": False,
            "bestEffort": False,
            "reason": None,
        }

    def _best_effort_receipt(
        self,
        identity: str,
        *,
        model: str | None,
        port: int | None,
        reason: str,
        force: bool,
        detail: str | None = None,
    ) -> dict[str, Any]:
        try:
            if self._read_stats is None or port is None:
                raise AccountingPrepareError("stats fallback is unavailable")
            snapshot = self._read_stats(port)
            instance_id, model_id, prompt_total, completion_total, uptime_s = (
                self._snapshot_values(snapshot, model=model, require_instance=False)
            )
            receipt = {
                "receiptId": stable_receipt_id(
                    "snapshot:"
                    f"{instance_id if instance_id is not None else identity}:"
                    f"{model_id}:{prompt_total}:{completion_total}"
                ),
                "createdAt": max(0, int(self._wall_now() * 1000)),
                "instanceId": instance_id,
                "modelId": model_id,
                "promptTokensTotal": int(prompt_total),
                "completionTokensTotal": int(completion_total),
                "uptimeS": int(uptime_s),
                "drainComplete": False,
                "degraded": True,
                "bestEffort": True,
                "reason": reason,
            }
            if detail:
                receipt["detail"] = detail
            return receipt
        except Exception as exc:  # noqa: BLE001 - injected legacy reader must fail closed
            error = AccountingPrepareError(f"{reason} stats fallback failed: {exc}")
            if not force:
                raise error from exc
            combined = f"{detail}; {error}" if detail else str(error)
            return self._degraded_receipt(
                identity, model=model, reason=f"{reason}-stats-unavailable", detail=combined
            )

    def _snapshot_values(
        self,
        snapshot: dict[str, Any],
        *,
        model: str | None,
        require_instance: bool,
    ) -> tuple[str | None, str, int, int, int | float]:
        if not isinstance(snapshot, dict):
            raise AccountingPrepareError("accounting snapshot is not an object")
        if snapshot.get("reachable") is False or snapshot.get("running") is False:
            raise AccountingPrepareError("accounting snapshot is unreachable")

        instance_id = snapshot.get("instance_id", snapshot.get("instanceId"))
        model_id = snapshot.get("model_id", snapshot.get("modelId"))
        model_doc = snapshot.get("model")
        if model_id is None and isinstance(model_doc, dict):
            model_id = model_doc.get("id")
        model_id = model_id or model

        requests = snapshot.get("requests")
        if not isinstance(requests, dict):
            requests = {}
        prompt_total = snapshot.get("prompt_tokens_total", snapshot.get("promptTokensTotal"))
        if prompt_total is None:
            prompt_total = requests.get(
                "prompt_tokens_total", requests.get("promptTokensTotal")
            )
        completion_total = snapshot.get(
            "completion_tokens_total", snapshot.get("completionTokensTotal")
        )
        if completion_total is None:
            completion_total = requests.get(
                "completion_tokens_total", requests.get("completionTokensTotal")
            )
        uptime_s = snapshot.get("uptime_s", snapshot.get("uptimeS"))

        if instance_id is not None and (not isinstance(instance_id, str) or not instance_id):
            raise AccountingPrepareError("accounting snapshot has invalid instance_id")
        if require_instance and instance_id is None:
            raise AccountingPrepareError("accounting snapshot is missing instance_id")
        if not isinstance(model_id, str) or not model_id:
            raise AccountingPrepareError("accounting snapshot is missing model_id")
        if not self._is_token_total(prompt_total) or not self._is_token_total(completion_total):
            raise AccountingPrepareError("accounting snapshot has invalid token totals")
        if not isinstance(uptime_s, (int, float)) or isinstance(uptime_s, bool) or uptime_s < 0:
            raise AccountingPrepareError("accounting snapshot has invalid uptime_s")
        return instance_id, model_id, prompt_total, completion_total, uptime_s

    def _degraded_receipt(
        self,
        identity: str,
        *,
        model: str | None,
        reason: str,
        detail: str | None = None,
    ) -> dict[str, Any]:
        receipt = {
            "receiptId": stable_receipt_id(identity),
            "createdAt": max(0, int(self._wall_now() * 1000)),
            "instanceId": None,
            "modelId": model,
            "promptTokensTotal": None,
            "completionTokensTotal": None,
            "uptimeS": None,
            "drainComplete": False,
            "degraded": True,
            "bestEffort": True,
            "reason": reason,
        }
        if detail:
            receipt["detail"] = detail
        return receipt

    @staticmethod
    def _child_identity(child) -> str:
        return f"legacy:{child.pid}:{getattr(child, 'starttime', None)}"

    @staticmethod
    def _is_token_total(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    def detach(self) -> None:
        """Daemon is shutting down but the engine must OUTLIVE it: drop the
        monitor/tailer without signalling the serve. State stays persisted for re-adoption. The
        watcher threads are daemons and die with the interpreter; here we just stop the tailer's
        follow loop so it doesn't spin during shutdown."""
        with self._cond:
            child = self._child
        if child is not None and getattr(child, "tailer", None) is not None:
            try:
                child.tailer.stop()
            except Exception:  # noqa: BLE001
                pass

    # ---- re-adoption ----

    def readopt(self) -> bool:
        """At boot, re-attach a still-running serve recorded in the pidfile. Returns
        whether a serve was adopted. A stale/mismatched record is cleared, never adopted."""
        state = self._store.load()
        if state is None:
            return False
        child = self._adopt_fn(state)
        if child is None:
            self._store.clear()
            self._emit(f"stale serve state cleared (pid={state.pid} no longer an ft serve)")
            return False
        with self._cond:
            self._child = child
            self._model = state.model
            self._port = state.port
            self._args = list(state.args)
            self._started_at = self._now()  # unknown true start; report uptime since adoption
            self._last_exit = None
            self._adopted = True
            self._stopping = False
        self._maybe_apply_oom(child.pid)
        self._begin_watch(child)
        self._emit(f"re-adopted running serve (pid={child.pid} model={state.model} port={state.port})")
        return True

    # ---- monitor / reap (single reaper) ----

    def _begin_watch(self, child) -> None:
        if self._tailer_factory is not None:
            tailer = self._tailer_factory(child)
            child.tailer = tailer
            if tailer is not None:
                tailer.start()
        threading.Thread(
            target=self._monitor, args=(child,), name=f"ft-daemon-monitor-{child.pid}", daemon=True
        ).start()

    def _monitor(self, child) -> None:
        try:
            info = child.wait()
        except Exception as exc:  # noqa: BLE001 — a wait() failure must still reap, not wedge state
            info = ExitInfo(None, "unknown")
            self._emit(f"monitor wait failed for pid={child.pid}: {exc}")
        self._reap(child, info)

    def _reap(self, child, info: ExitInfo) -> None:
        # Seal crash accounting before clearing the state file or launching a replacement. A
        # concurrent prepare call holds _accounting_guard until its own receipt is durable, so
        # this path either observes that receipt or writes one degraded crash receipt.
        with self._cond:
            was_stopping = self._stopping or self._stop_requested
            is_current = self._child is child
            model = self._model
        if is_current:
            try:
                self._persist_crash_accounting(
                    child, model=model, was_stopping=was_stopping, info=info
                )
            except AccountingOutboxError as exc:
                # The process is already dead and cannot be preserved. Keep the gap visible and
                # suppress auto-restart rather than hiding it behind a fresh generation.
                self._emit(f"failed to persist crash accounting for pid={child.pid}: {exc}")
                with self._cond:
                    self._stop_requested = True

        with self._cond:
            is_current = self._child is child
            if is_current:
                self._child = None
                self._started_at = None
                self._stopping = False
                if was_stopping:
                    info = ExitInfo(info.code, "stopped")
                self._last_exit = info
            self._cond.notify_all()
        # Outside the lock. Clear the persisted state BEFORE waking stop() waiters, so a caller
        # that sees stop() return also sees an empty pidfile — no window where a racing re-adopt
        # could latch onto the just-killed pid.
        if is_current:
            self._store.clear()
        child.reaped.set()
        if getattr(child, "tailer", None) is not None:
            try:
                child.tailer.stop()
                child.tailer.join(timeout=self._reap_wait_s)
            except Exception:  # noqa: BLE001
                pass
        child.close()
        if is_current:
            self._emit(self._exit_line(child, info, was_stopping))
            # Auto-restart only a genuine crash, and never one the user has asked to stop.
            with self._cond:
                do_restart = not was_stopping and self._auto_restart and not self._stop_requested
            if do_restart:
                self._restart_async(child)

    def _exit_line(self, child, info: ExitInfo, was_stopping: bool) -> str:
        if info.source == "adopted-vanished":
            return f"serve (adopted pid={child.pid}) vanished; exit code unknown"
        if was_stopping:
            return f"serve stopped (pid={child.pid})"
        if info.code is not None and info.code < 0:
            return f"serve killed by signal {-info.code} (pid={child.pid})"
        return f"serve exited with code {info.code} (pid={child.pid})"

    def _restart_async(self, dead_child) -> None:
        model, port, args = self._model, self._port, list(self._args)
        if model is None or port is None:
            return

        def _run():
            try:
                self.start(model, port, args, _auto=True)
            except Exception as exc:  # noqa: BLE001
                self._emit(f"auto-restart failed: {exc}")

        threading.Thread(target=_run, name="ft-daemon-autorestart", daemon=True).start()

    # ---- oom ----

    def _maybe_apply_oom(self, pid: int) -> None:
        if self._apply_oom:
            self.reapply_oom()

    def reapply_oom(self) -> int:
        """Write the positive OOM score to the whole serve tree so the 22 GB serve — not the tiny
        daemon — is the kernel's preferred victim. Called at spawn and periodically
        from the daemon loop, because mp-spawn workers fork AFTER the initial write and would
        otherwise keep the inherited score. Returns pids touched."""
        pid = self.current_pid()
        if pid is None:
            return 0
        n = 0
        for p in osproc.tree_pids(pid):
            if osproc.set_oom_score_adj(p, self._oom_child_score):
                n += 1
        return n

    # ---- introspection ----

    def current_pid(self) -> int | None:
        with self._cond:
            return self._child.pid if self._child is not None else None

    def serve_args(self) -> list[str]:
        """Engine args of the running (or last started) serve."""
        with self._cond:
            return list(self._args)

    def status(self) -> dict:
        with self._cond:
            child = self._child
            running = child is not None and not self._stopping
            uptime = (
                int(self._now() - self._started_at)
                if running and self._started_at is not None
                else 0
            )
            last = self._last_exit
            return {
                "running": running,
                "starting": self._starting,
                "stopping": self._stopping,
                "adopted": self._adopted if running else False,
                "pid": child.pid if child is not None else None,
                "model": self._model,
                "port": self._port,
                "uptimeS": uptime,
                "lastExitCode": last.code if last is not None else None,
                "lastExitReason": last.source if last is not None else None,
            }

    # ---- helpers ----

    def _persist(self, child, model: str, port: int, args: list[str]) -> None:
        from .pidfile import ServeState

        self._store.save(
            ServeState(
                model=model,
                port=port,
                pid=child.pid,
                args=list(args),
                starttime=getattr(child, "starttime", None),
                log_path=getattr(child, "log_path", None),
            )
        )

    def _emit(self, text: str) -> None:
        try:
            self._ring.append(text, kind="event", ts=self._wall_now())
        except Exception:  # noqa: BLE001 — logging must never break control flow
            pass
