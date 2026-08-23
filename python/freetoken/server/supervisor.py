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

from .process_utils import process_is_alive


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
            if not process_is_alive(p):
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


def drain_ready(
    handle: BackendHandle,
    progress: LoadProgress,
    get: Callable[[float], Any] | None = None,
    poll: float = 1.0,
    on_meta: Callable[[dict], None] | None = None,
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
    while ready < handle.expected_acks:
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

    try:
        drain_ready(handle, progress, poll=poll, on_meta=on_meta)
    except WorkerDied as exc:
        # A worker dying mid-load during an orderly shutdown is expected, not a load failure.
        if not _shutting_down() and on_failure is not None:
            on_failure(exc.message)
        return

    on_ready()

    while True:
        dead = _first_dead(handle.processes)
        if dead is not None:
            if _shutting_down():
                # Orderly stop in progress: the worker exit is expected — stay quiet.
                return
            if on_failure is not None:
                on_failure(f"backend worker {getattr(dead, 'name', '?')} exited")
            return
        time.sleep(poll)
