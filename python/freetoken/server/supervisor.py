"""Backend startup/liveness supervision, decoupled from run_api_server so it is unit
testable without spawning real processes.

The scheduler/tokenizer worker processes are spawned by launch.start_subprocess, which
hands back a BackendHandle. ``drain_ready`` blocks until every worker has acked ready
(applying interleaved ("progress", desc, done, total) messages to a LoadProgress),
after which the caller flips the maintenance gate to "serving". Task 3 adds liveness
watching on top of this.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from queue import Empty
from typing import Any, Callable, List


def phase_slug(desc: str) -> str:
    """Normalize a progress-bar desc into the stable /health phase enum. Order matters:
    the expert bars ("Loading experts …", "Loading expert banks …") both contain 'expert'
    and must win over the generic 'weight' match ("Loading weights (FTW)")."""
    d = (desc or "").lower()
    if "expert" in d:
        return "expert_banks"
    if "weight" in d:
        return "weights"
    if "graph" in d or "warm" in d:
        return "warmup"
    return "other"


@dataclass
class LoadProgress:
    desc: str = ""
    done_bytes: int = 0
    total_bytes: int = 0

    def update(self, desc: str, done: int, total: int) -> None:
        self.desc = desc
        self.done_bytes = done
        self.total_bytes = total

    @property
    def phase(self) -> str:
        return phase_slug(self.desc)


@dataclass
class BackendHandle:
    """What start_subprocess hands back so run_api_server can supervise the workers it
    no longer blocks on."""

    ack_queue: Any
    processes: List[Any] = field(default_factory=list)
    expected_acks: int = 0


class WorkerDied(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _first_dead(processes: List[Any]) -> Any | None:
    for p in processes:
        try:
            if not p.is_alive():
                return p
        except Exception:  # noqa: BLE001 — treat an unqueryable handle as alive
            continue
    return None


def _as_error(msg: Any) -> str | None:
    """The reason a dying worker pushed as ("error", reason), or None for any other ack — lets
    the supervisor report the real cause (e.g. a config ValueError) rather than the generic
    "exited during load"."""
    if isinstance(msg, tuple) and msg and msg[0] == "error":
        return str(msg[1]) if len(msg) > 1 else "backend worker failed during load"
    return None


def _as_cuda_oom(msg: Any) -> str | None:
    """Only the worker's typed torch.cuda.OutOfMemoryError receipt enables recovery.

    A human exception string is deliberately insufficient: ordinary failures which happen to
    contain 'OOM'/'out of memory' retain the existing terminal process-exit behavior.
    """
    if isinstance(msg, tuple) and msg and msg[0] == "cuda_oom":
        return str(msg[1]) if len(msg) > 1 else "CUDA out of memory"
    return None


def _drain_pending_error(get: Callable[[float], Any], attempts: int = 5, timeout: float = 0.02) -> str | None:
    """After a worker death, poll the ack queue briefly for a final ("error", reason) ack still
    in flight (the queue feeder flushes on process exit), so the real reason wins over the
    generic message. Bounded so a genuinely reason-less death still fails fast."""
    for _ in range(attempts):
        try:
            msg = get(timeout)
        except Empty:
            continue
        err = _as_error(msg)
        if err is not None:
            return err
    return None


def _drain_pending_failure(get: Callable[[float], Any]) -> tuple[str | None, bool]:
    """Return the final worker reason plus whether it was a typed CUDA OOM."""
    for _ in range(5):
        try:
            msg = get(0.02)
        except Empty:
            continue
        if (oom := _as_cuda_oom(msg)) is not None:
            return oom, True
        if (err := _as_error(msg)) is not None:
            return err, False
    return None, False


def _retire_oom_workers(processes: List[Any]) -> None:
    """Terminate, join, and if necessary kill every failed worker before replacement CUDA use."""
    survivors = []
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
            if process.is_alive():
                survivors.append(getattr(process, "name", "unknown"))
        except Exception:  # noqa: BLE001
            survivors.append(getattr(process, "name", "unknown"))
    if survivors:
        raise WorkerDied("OOM workers could not be stopped: " + ", ".join(survivors))


def drain_ready(
    handle: BackendHandle,
    progress: LoadProgress,
    get: Callable[[float], Any] | None = None,
    poll: float = 1.0,
    on_meta: Callable[[dict], None] | None = None,
    timeout_s: float | None = None,
    is_shutting_down: Callable[[], bool] | None = None,
) -> None:
    """Block until ``expected_acks`` non-progress acks arrive, applying ("progress", …)
    tuples to ``progress``. Between acks, poll worker liveness; raise WorkerDied if a worker
    exits before readiness (e.g. CUDA OOM / corrupt weights during load).

    ("meta", payload) tuples carry optional backend metadata (per-unit cache byte costs);
    they are forwarded to ``on_meta`` and do NOT count toward readiness. meta is optional --
    an engine build that never emits it must not stall this drain, so nothing waits for it."""
    if get is None:
        def get(timeout: float) -> Any:
            return handle.ack_queue.get(timeout=timeout)

    ready = 0
    deadline = time.monotonic() + timeout_s if timeout_s is not None else None
    while ready < handle.expected_acks:
        if is_shutting_down is not None and is_shutting_down():
            raise WorkerDied("backend startup cancelled by shutdown")
        if deadline is not None and time.monotonic() >= deadline:
            raise WorkerDied("backend readiness timed out")
        try:
            msg = get(poll)
        except Empty:
            dead = _first_dead(handle.processes)
            if dead is not None:
                # The worker may have pushed its real failure reason just before exiting; give
                # the queue a brief window to surface it, else fall back to the generic message.
                reason = _drain_pending_error(get)
                raise WorkerDied(reason or f"backend worker {getattr(dead, 'name', '?')} exited during load")
            continue
        err = _as_error(msg)
        if err is not None:
            raise WorkerDied(err)
        oom = _as_cuda_oom(msg)
        if oom is not None:
            # Recovery is deliberately post-initial-ready only. A load-time CUDA OOM has no
            # healthy generation to retire and follows the ordinary startup-failure path.
            raise WorkerDied(oom)
        if isinstance(msg, tuple) and msg and msg[0] == "progress":
            _tag, desc, done, total = msg
            progress.update(desc, done, total)
        elif isinstance(msg, tuple) and msg and msg[0] == "meta":
            if on_meta is not None:
                on_meta(msg[1] if len(msg) > 1 else {})
        else:
            ready += 1


def run_backend_supervisor(
    handle: BackendHandle,
    progress: LoadProgress,
    on_ready: Callable[[], None],
    on_failure: Callable[[str], None] | None = None,
    poll: float = 1.0,
    on_meta: Callable[[dict], None] | None = None,
    is_shutting_down: Callable[[], bool] | None = None,
    restart_backend: Callable[[int], BackendHandle] | None = None,
    on_restarting: Callable[[str, int], None] | None = None,
    max_restarts: int = 1,
    restart_ready_timeout: float | None = None,
    can_restart: Callable[[], bool] | None = None,
) -> None:
    """Drain to readiness (watching liveness), flip the gate via ``on_ready``, then watch
    forever: a worker that dies AFTER ready (a scheduler crash while the frontend keeps
    answering 200) also fires ``on_failure`` — closing the half-alive-engine gap.

    ``on_meta`` receives the optional ("meta", …) backend metadata during the drain.

    ``is_shutting_down`` distinguishes a crash from an orderly stop: on SIGTERM/SIGINT
    (``ft serve`` ^C, or the desktop's stop) the api server sets a shutting-down flag before
    the workers exit, so their expected deaths are NOT misreported. When the flag is set at
    the moment a death is observed, the supervisor returns silently — no ``on_failure``, no
    ERROR log, no "failed" latch. A death seen while the flag is clear is a real crash and
    fires ``on_failure`` as before."""

    def _shutting_down() -> bool:
        return bool(is_shutting_down is not None and is_shutting_down())

    def _can_restart() -> bool:
        return not _shutting_down() and (can_restart is None or can_restart())

    try:
        drain_ready(handle, progress, poll=poll, on_meta=on_meta)
    except WorkerDied as exc:
        # A worker dying mid-load during an orderly shutdown is expected, not a load failure.
        if not _shutting_down() and on_failure is not None:
            on_failure(exc.message)
        return

    on_ready()

    attempts = 0
    while True:
        dead = _first_dead(handle.processes)
        if dead is not None:
            if _shutting_down():
                # Orderly stop in progress: the worker exit is expected — stay quiet.
                return
            reason, is_cuda_oom = _drain_pending_failure(lambda timeout: handle.ack_queue.get(timeout=timeout))
            message = reason or f"backend worker {getattr(dead, 'name', '?')} exited"
            if not is_cuda_oom or restart_backend is None or attempts >= max_restarts:
                if on_failure is not None:
                    on_failure(message)
                return
            if not _can_restart():
                return
            attempts += 1
            if on_restarting is not None:
                try:
                    on_restarting(message, attempts)
                except Exception as exc:  # noqa: BLE001
                    if _can_restart() and on_failure is not None:
                        on_failure(f"OOM recovery transition failed: {exc}")
                    return
            fresh = None
            try:
                _retire_oom_workers(handle.processes)
                if not _can_restart():
                    return
                fresh = restart_backend(attempts)
                drain_ready(fresh, progress, poll=poll, on_meta=on_meta,
                            timeout_s=restart_ready_timeout, is_shutting_down=lambda: not _can_restart())
            except Exception as exc:  # noqa: BLE001
                failure = f"OOM recovery failed: {exc}"
                if fresh is not None:
                    try:
                        _retire_oom_workers(fresh.processes)
                    except WorkerDied as cleanup:
                        failure += f"; {cleanup.message}"
                if not _shutting_down() and on_failure is not None:
                    on_failure(failure)
                return
            handle = fresh
            on_ready()
            continue
        time.sleep(poll)
