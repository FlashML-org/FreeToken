from __future__ import annotations

import queue

from queue import Empty as _Empty

from freetoken.utils import progress
from freetoken.server.supervisor import (
    BackendHandle,
    LoadProgress,
    WorkerDied,
    _retire_oom_workers,
    drain_ready,
    phase_slug,
    run_backend_supervisor,
)


def test_drain_ready_counts_ready_acks_and_applies_progress():
    q: "queue.Queue" = queue.Queue()
    q.put(("progress", "Loading weights (FTW)", 5, 10))
    q.put("Scheduler is ready")
    q.put(("progress", "Loading experts (parallel)", 8, 8))
    q.put("tokenizer ready")
    q.put("detokenizer ready")
    handle = BackendHandle(ack_queue=q, processes=[], expected_acks=3)
    progress = LoadProgress()

    drain_ready(handle, progress)

    assert progress.total_bytes == 8
    assert progress.done_bytes == 8
    assert progress.phase == "expert_banks"
    assert q.empty()


def test_drain_ready_forwards_meta_without_counting_it_ready():
    """("meta", payload) is optional backend metadata: forwarded to on_meta, but it must NOT
    count toward expected_acks (else a meta-emitting engine would flip ready one ack early)."""
    q: "queue.Queue" = queue.Queue()
    q.put(("meta", {"kv_bytes_per_token": 42}))
    q.put("Scheduler is ready")
    q.put("tokenizer ready")
    handle = BackendHandle(ack_queue=q, processes=[], expected_acks=2)
    seen: dict = {}

    drain_ready(handle, LoadProgress(), on_meta=lambda m: seen.update(m))

    assert seen == {"kv_bytes_per_token": 42}
    assert q.empty()  # both real acks consumed; meta did not short-count them


def test_drain_ready_ignores_meta_when_no_callback():
    """An engine that emits meta while the caller passes no on_meta must not stall or error."""
    q: "queue.Queue" = queue.Queue()
    q.put(("meta", {"kv_bytes_per_token": 7}))
    q.put("Scheduler is ready")
    handle = BackendHandle(ack_queue=q, processes=[], expected_acks=1)

    drain_ready(handle, LoadProgress())  # no on_meta

    assert q.empty()


def test_drain_ready_detects_worker_death_during_load():
    import queue

    import pytest

    from freetoken.server.supervisor import WorkerDied

    class DeadProc:
        name = "freetoken-TP0-scheduler"

        def is_alive(self) -> bool:
            return False

    q: "queue.Queue" = queue.Queue()  # never receives a ready ack
    handle = BackendHandle(ack_queue=q, processes=[DeadProc()], expected_acks=1)
    with pytest.raises(WorkerDied):
        drain_ready(handle, LoadProgress(), get=lambda _t: (_ for _ in ()).throw(_Empty()))


def test_drain_ready_raises_the_real_reason_from_an_error_ack():
    """A worker that pushes ("error", reason) just before dying surfaces THAT reason (e.g. a
    config ValueError), not the generic "exited during load"."""
    import queue

    import pytest

    from freetoken.server.supervisor import WorkerDied

    q: "queue.Queue" = queue.Queue()
    q.put(("error", "ValueError: --moe-backend 'hybrid' cannot compute q4_0 experts on the CPU"))
    handle = BackendHandle(ack_queue=q, processes=[], expected_acks=1)
    with pytest.raises(WorkerDied) as exc:
        drain_ready(handle, LoadProgress())
    assert "q4_0" in str(exc.value)


def test_supervisor_reports_the_worker_error_reason_via_on_failure():
    """End to end: an ("error", reason) ack from a dying worker reaches on_failure verbatim, so
    the desktop failure modal shows the actionable cause instead of "exited during load"."""
    import queue

    class DeadProc:
        name = "freetoken-TP0-scheduler"

        def is_alive(self) -> bool:
            return False

    q: "queue.Queue" = queue.Queue()
    q.put(("error", "ValueError: bad checkpoint config"))
    handle = BackendHandle(ack_queue=q, processes=[DeadProc()], expected_acks=1)
    seen: dict = {}
    from freetoken.server.supervisor import run_backend_supervisor

    run_backend_supervisor(
        handle,
        LoadProgress(),
        on_ready=lambda: seen.setdefault("ready", True),
        on_failure=lambda m: seen.setdefault("failure", m),
        poll=0.01,
    )
    assert "ready" not in seen
    assert seen["failure"] == "ValueError: bad checkpoint config"


class _Process:
    def __init__(self, name: str, alive: bool = True, survives_kill: bool = False):
        self.name = name
        self.alive = alive
        self.survives_kill = survives_kill
        self.calls: list[str] = []

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.calls.append("terminate")
        self.alive = False

    def join(self, timeout: float) -> None:
        self.calls.append(f"join:{timeout}")

    def kill(self) -> None:
        self.calls.append("kill")
        if not self.survives_kill:
            self.alive = False


def test_oom_worker_retirement_terminates_joins_and_kills_a_stubborn_worker():
    class Stubborn(_Process):
        def terminate(self) -> None:
            self.calls.append("terminate")

    process = Stubborn("scheduler")

    _retire_oom_workers([process])

    assert process.calls == ["terminate", "join:10", "kill", "join:2"]


def test_oom_worker_retirement_denies_replacement_when_a_worker_cannot_stop():
    class Unstoppable(_Process):
        def terminate(self) -> None:
            self.calls.append("terminate")

        def kill(self) -> None:
            self.calls.append("kill")

    process = Unstoppable("tokenizer")

    with __import__("pytest").raises(WorkerDied, match="could not be stopped: tokenizer"):
        _retire_oom_workers([process])


def test_post_ready_typed_cuda_oom_retires_then_replaces_once():
    class StopAfterReplacement(Exception):
        pass

    initial_process = _Process("scheduler", alive=False)
    initial_queue: "queue.Queue" = queue.Queue()
    initial_queue.put("ready")
    initial_queue.put(("cuda_oom", "CUDA out of memory in decode"))
    initial = BackendHandle(initial_queue, [initial_process], expected_acks=1)

    replacement_process = _Process("replacement")
    replacement_queue: "queue.Queue" = queue.Queue()
    replacement_queue.put("ready")
    replacement = BackendHandle(replacement_queue, [replacement_process], expected_acks=1)
    events: list[tuple] = []

    def ready() -> None:
        events.append(("ready",))
        if len(events) == 3:
            raise StopAfterReplacement()

    with __import__("pytest").raises(StopAfterReplacement):
        run_backend_supervisor(
            initial,
            LoadProgress(),
            on_ready=ready,
            on_failure=lambda reason: events.append(("failure", reason)),
            on_restarting=lambda reason, attempt: events.append(("restarting", reason, attempt)),
            restart_backend=lambda attempt: replacement,
            max_restarts=1,
            poll=0,
        )

    assert initial_process.calls == ["join:10"]
    assert events == [
        ("ready",),
        ("restarting", "CUDA out of memory in decode", 1),
        ("ready",),
    ]


def test_non_oom_and_startup_failure_never_start_a_replacement():
    class DeadProc:
        name = "scheduler"

        def is_alive(self) -> bool:
            return False

    started: list[int] = []
    for receipt in (("error", "RuntimeError: bad checkpoint"), ("error", "CUDA out of memory during load")):
        q: "queue.Queue" = queue.Queue()
        q.put(receipt)
        failures: list[str] = []
        run_backend_supervisor(
            BackendHandle(q, [DeadProc()], expected_acks=1),
            LoadProgress(),
            on_ready=lambda: None,
            on_failure=failures.append,
            restart_backend=lambda attempt: started.append(attempt),
            max_restarts=1,
            poll=0,
        )
        assert failures == [receipt[1]]
    assert started == []


def test_typed_cuda_oom_during_startup_is_a_terminal_startup_failure_not_ready():
    """Recovery is deliberately post-ready only: a load-time CUDA OOM has no healthy
    request generation to preserve and must not be mistaken for a readiness ack."""
    q: "queue.Queue" = queue.Queue()
    q.put(("cuda_oom", "CUDA out of memory while loading weights"))
    failures: list[str] = []
    starts: list[int] = []

    run_backend_supervisor(
        BackendHandle(q, [], expected_acks=1),
        LoadProgress(),
        on_ready=lambda: (_ for _ in ()).throw(AssertionError("startup OOM became ready")),
        on_failure=failures.append,
        restart_backend=lambda attempt: starts.append(attempt),
        max_restarts=1,
        poll=0,
    )

    assert failures == ["CUDA out of memory while loading weights"]
    assert starts == []


def test_second_post_ready_oom_exhausts_the_bounded_recovery_budget():
    initial_process = _Process("initial", alive=False)
    initial_queue: "queue.Queue" = queue.Queue()
    initial_queue.put("ready")
    initial_queue.put(("cuda_oom", "first OOM"))

    replacement_process = _Process("replacement", alive=False)
    replacement_queue: "queue.Queue" = queue.Queue()
    replacement_queue.put("ready")
    replacement_queue.put(("cuda_oom", "second OOM"))
    starts: list[int] = []
    failures: list[str] = []
    ready: list[bool] = []

    run_backend_supervisor(
        BackendHandle(initial_queue, [initial_process], expected_acks=1),
        LoadProgress(),
        on_ready=lambda: ready.append(True),
        on_failure=failures.append,
        restart_backend=lambda attempt: starts.append(attempt) or BackendHandle(
            replacement_queue, [replacement_process], expected_acks=1
        ),
        max_restarts=1,
        poll=0,
    )

    assert starts == [1]
    assert ready == [True, True]
    assert failures == ["second OOM"]


def test_replacement_readiness_deadline_fails_and_retires_the_partial_factory_handle():
    initial_process = _Process("initial", alive=False)
    initial_queue: "queue.Queue" = queue.Queue()
    initial_queue.put("ready")
    initial_queue.put(("cuda_oom", "decode OOM"))
    fresh_process = _Process("partial", alive=True)
    fresh_queue: "queue.Queue" = queue.Queue()
    failures: list[str] = []

    run_backend_supervisor(
        BackendHandle(initial_queue, [initial_process], expected_acks=1),
        LoadProgress(),
        on_ready=lambda: None,
        on_failure=failures.append,
        restart_backend=lambda _: BackendHandle(fresh_queue, [fresh_process], expected_acks=1),
        restart_ready_timeout=0,
        max_restarts=1,
        poll=0,
    )

    assert failures == ["OOM recovery failed: backend readiness timed out"]
    assert fresh_process.calls == ["terminate", "join:10"]


def test_stop_barrier_prevents_launch_after_oom_cleanup():
    class StopOnTerminate(_Process):
        def __init__(self):
            super().__init__("initial", alive=False)
            self.stopped = False

        def join(self, timeout: float) -> None:
            super().join(timeout)
            self.stopped = True

    process = StopOnTerminate()
    q: "queue.Queue" = queue.Queue()
    q.put("ready")
    q.put(("cuda_oom", "decode OOM"))
    launches: list[int] = []

    run_backend_supervisor(
        BackendHandle(q, [process], expected_acks=1),
        LoadProgress(),
        on_ready=lambda: None,
        restart_backend=lambda attempt: launches.append(attempt),
        can_restart=lambda: not process.stopped,
        max_restarts=1,
        poll=0,
    )

    assert process.calls == ["join:10"]
    assert launches == []


def test_preparation_stop_barrier_wins_before_recovery_starts():
    q: "queue.Queue" = queue.Queue()
    q.put("ready")
    q.put(("cuda_oom", "decode OOM"))
    process = _Process("initial", alive=False)
    launches: list[int] = []
    restarts: list[tuple[str, int]] = []

    run_backend_supervisor(
        BackendHandle(q, [process], expected_acks=1),
        LoadProgress(),
        on_ready=lambda: None,
        restart_backend=lambda attempt: launches.append(attempt),
        on_restarting=lambda message, attempt: restarts.append((message, attempt)),
        can_restart=lambda: False,
        max_restarts=1,
        poll=0,
    )

    assert launches == []
    assert restarts == []


def test_stop_during_replacement_readiness_cleans_the_current_handle_and_never_reopens():
    initial_q: "queue.Queue" = queue.Queue()
    initial_q.put("ready")
    initial_q.put(("cuda_oom", "decode OOM"))
    initial = _Process("initial", alive=False)
    fresh = _Process("fresh")
    stopped = {"value": False}

    class StopWhileWaiting:
        def get(self, timeout: float):
            stopped["value"] = True
            raise _Empty()

    failures: list[str] = []
    ready: list[bool] = []
    run_backend_supervisor(
        BackendHandle(initial_q, [initial], expected_acks=1),
        LoadProgress(),
        on_ready=lambda: ready.append(True),
        on_failure=failures.append,
        restart_backend=lambda _: BackendHandle(StopWhileWaiting(), [fresh], expected_acks=1),
        can_restart=lambda: not stopped["value"],
        max_restarts=1,
        poll=0,
    )

    assert ready == [True]
    assert failures == ["OOM recovery failed: backend startup cancelled by shutdown"]
    assert fresh.calls == ["terminate", "join:10"]


def test_supervisor_reports_failure_on_startup_death():
    import queue

    class DeadProc:
        name = "freetoken-detokenizer-0"

        def is_alive(self) -> bool:
            return False

    q: "queue.Queue" = queue.Queue()
    handle = BackendHandle(ack_queue=q, processes=[DeadProc()], expected_acks=1)
    seen: dict = {}
    from freetoken.server.supervisor import run_backend_supervisor

    run_backend_supervisor(
        handle,
        LoadProgress(),
        on_ready=lambda: seen.setdefault("ready", True),
        on_failure=lambda m: seen.setdefault("failure", m),
        poll=0.01,
    )
    assert "ready" not in seen
    assert "detokenizer" in seen["failure"]


def test_supervisor_detects_post_ready_death():
    import queue

    class Proc:
        name = "freetoken-TP0-scheduler"

        def __init__(self) -> None:
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

    proc = Proc()
    q: "queue.Queue" = queue.Queue()
    q.put("scheduler ready")
    handle = BackendHandle(ack_queue=q, processes=[proc], expected_acks=1)
    seen: dict = {}
    from freetoken.server.supervisor import run_backend_supervisor

    def on_ready() -> None:
        seen["ready"] = True
        proc._alive = False  # die right after readiness

    run_backend_supervisor(
        handle, LoadProgress(), on_ready=on_ready,
        on_failure=lambda m: seen.setdefault("failure", m), poll=0.01,
    )
    assert seen.get("ready") is True
    assert "scheduler" in seen["failure"]


def test_supervisor_silent_on_post_ready_death_during_shutdown():
    """An orderly stop (SIGTERM/^C) sets a shutting-down flag before the workers exit. A
    post-ready death observed while that flag is set is EXPECTED — the watchdog must return
    silently: no on_failure, so no ERROR log and no "failed" latch during a clean stop."""
    import queue

    class Proc:
        name = "freetoken-TP0-scheduler"

        def __init__(self) -> None:
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

    proc = Proc()
    q: "queue.Queue" = queue.Queue()
    q.put("scheduler ready")
    handle = BackendHandle(ack_queue=q, processes=[proc], expected_acks=1)
    seen: dict = {}
    shutting_down = {"v": False}
    from freetoken.server.supervisor import run_backend_supervisor

    def on_ready() -> None:
        seen["ready"] = True
        shutting_down["v"] = True  # stop requested…
        proc._alive = False        # …and the worker exits as part of that stop

    run_backend_supervisor(
        handle, LoadProgress(), on_ready=on_ready,
        on_failure=lambda m: seen.setdefault("failure", m), poll=0.01,
        is_shutting_down=lambda: shutting_down["v"],
    )
    assert seen.get("ready") is True
    assert "failure" not in seen  # graceful stop: the death was not reported


def test_supervisor_silent_on_startup_death_during_shutdown():
    """A worker dying mid-load while an orderly stop is already in progress must not be
    reported as a load failure either."""
    import queue

    class DeadProc:
        name = "freetoken-detokenizer-0"

        def is_alive(self) -> bool:
            return False

    q: "queue.Queue" = queue.Queue()  # never receives a ready ack
    handle = BackendHandle(ack_queue=q, processes=[DeadProc()], expected_acks=1)
    seen: dict = {}
    from freetoken.server.supervisor import run_backend_supervisor

    run_backend_supervisor(
        handle,
        LoadProgress(),
        on_ready=lambda: seen.setdefault("ready", True),
        on_failure=lambda m: seen.setdefault("failure", m),
        poll=0.01,
        is_shutting_down=lambda: True,  # stop already requested before load finished
    )
    assert "ready" not in seen
    assert "failure" not in seen  # silenced: expected exit during shutdown


# ---------------------------------------------------------------------------
# progress sink: the byte_bar -> set_progress_sink pipe drain_ready consumes, and the
# phase_slug normalization that labels the three serve bars.
# ---------------------------------------------------------------------------


def test_phase_slug_normalizes_the_three_serve_bars():
    assert phase_slug("Loading weights (FTW)") == "weights"
    assert phase_slug("Loading experts (parallel)") == "expert_banks"
    assert phase_slug("Loading expert banks (FTW)") == "expert_banks"
    assert phase_slug("something else") == "other"
    assert phase_slug("") == "other"


def test_byte_bar_emits_to_installed_sink_then_stops_after_clear():
    seen: list[tuple[str, int, int]] = []
    progress.set_progress_sink(lambda desc, done, total: seen.append((desc, done, total)))
    try:
        bar = progress.byte_bar(total=100, desc="Loading weights (FTW)")
        bar.update(100)  # full jump always emits
        bar.close()
    finally:
        progress.set_progress_sink(None)
    assert seen and seen[-1] == ("Loading weights (FTW)", 100, 100)

    # After clearing, a new bar must not emit.
    seen.clear()
    bar = progress.byte_bar(total=100, desc="Loading weights (FTW)")
    bar.update(100)
    bar.close()
    assert seen == []
