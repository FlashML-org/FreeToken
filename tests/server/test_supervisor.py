from __future__ import annotations

import queue

from queue import Empty as _Empty
import pytest  # Assert that a failed queue-close operation remains visible and retryable.

from freetoken.utils import progress
from freetoken.server.supervisor import BackendHandle, LoadProgress, drain_ready, phase_slug


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


def test_supervisor_closes_the_startup_queue_after_orderly_shutdown():  # Prevent multiprocessing semaphore warnings at interpreter exit.
    """The supervisor owns the parent queue handle and must release it on every return path."""  # State the lifecycle contract under test.
    class ClosableQueue(queue.Queue):  # Model the lifecycle methods exposed by multiprocessing.Queue.
        def __init__(self) -> None:  # Track cleanup calls while retaining ordinary queue behavior.
            super().__init__()  # Initialize the in-memory acknowledgement queue.
            self.closed = False  # Record whether the parent queue handle was closed.
            self.joined = False  # Record whether its feeder thread was joined.

        def close(self) -> None:  # Mirror multiprocessing.Queue.close without destroying test data.
            self.closed = True  # Make the ownership release observable.

        def join_thread(self) -> None:  # Mirror multiprocessing.Queue.join_thread.
            self.joined = True  # Make feeder-thread cleanup observable.

    class Proc:  # Provide one worker that exits as part of an orderly stop.
        name = "freetoken-TP0-scheduler"  # Preserve a realistic worker identity.

        def __init__(self) -> None:  # Begin alive so readiness can complete first.
            self.alive = True  # Keep the supervisor in its post-ready watch loop.

        def is_alive(self) -> bool:  # Expose the process-liveness protocol used by the supervisor.
            return self.alive  # Return the state changed by the ready callback.

    proc = Proc()  # Create the watched worker.
    ack_queue = ClosableQueue()  # Use a queue whose cleanup is directly verifiable.
    ack_queue.put("scheduler ready")  # Satisfy the startup readiness handshake.
    handle = BackendHandle(ack_queue=ack_queue, processes=[proc], expected_acks=1)  # Bind queue ownership to the backend handle.
    shutting_down = {"value": False}  # Share the orderly-stop state with the supervisor callback.
    from freetoken.server.supervisor import run_backend_supervisor  # Import the lifecycle under test.

    def on_ready() -> None:  # Transition from ready serving to an expected worker exit.
        shutting_down["value"] = True  # Mark the death as part of normal shutdown.
        proc.alive = False  # Make the supervisor observe worker termination.

    run_backend_supervisor(  # Exercise the complete ready-to-shutdown lifecycle.
        handle,  # Supply the queue and process handles owned by this serve instance.
        LoadProgress(),  # Collect startup progress without external state.
        on_ready=on_ready,  # Trigger the orderly worker exit after readiness.
        poll=0.01,  # Keep the unit test bounded.
        is_shutting_down=lambda: shutting_down["value"],  # Distinguish the expected exit from a crash.
    )  # The parent queue should be released before this call returns.
    assert ack_queue.closed is True  # Require the semaphore-owning handle to close.
    assert ack_queue.joined is True  # Require its feeder thread to finish as well.


def test_backend_handle_closes_the_startup_queue_only_once():  # Protect concurrent supervisor and lifespan cleanup.
    class ClosableQueue:  # Expose only the lifecycle surface owned by BackendHandle.
        def __init__(self) -> None:  # Count every cleanup call for idempotence assertions.
            self.close_calls = 0  # Record parent endpoint releases.
            self.join_calls = 0  # Record feeder-thread joins.

        def close(self) -> None:  # Mirror multiprocessing.Queue.close.
            self.close_calls += 1  # Make duplicate closure observable.

        def join_thread(self) -> None:  # Mirror multiprocessing.Queue.join_thread.
            self.join_calls += 1  # Make duplicate joins observable.

    ack_queue = ClosableQueue()  # Create a queue with directly observable ownership release.
    handle = BackendHandle(ack_queue=ack_queue)  # Bind cleanup state to the launch handle.
    handle.close_startup_queue()  # Model the supervisor reaching its finally block first.
    handle.close_startup_queue()  # Model lifespan shutdown requesting the same release afterward.
    assert ack_queue.close_calls == 1  # Require exactly one endpoint close.
    assert ack_queue.join_calls == 1  # Require exactly one feeder-thread join.


def test_backend_handle_retries_cleanup_after_close_failure():  # Preserve retryability when queue cleanup raises partway through.
    class FlakyQueue:  # Model a multiprocessing queue whose first close attempt fails.
        def __init__(self) -> None:  # Track each cleanup phase independently.
            self.close_calls = 0  # Count attempts to close parent endpoints.
            self.join_calls = 0  # Count completed feeder-thread joins.

        def close(self) -> None:  # Mirror the queue close surface.
            self.close_calls += 1  # Make retries observable.
            if self.close_calls == 1:  # Fail only the first attempt.
                raise OSError("synthetic close failure")  # Reproduce the partial-cleanup path.

        def join_thread(self) -> None:  # Mirror the feeder cleanup surface.
            self.join_calls += 1  # Prove the successful retry reaches the final phase.

    ack_queue = FlakyQueue()  # Create the failure-injecting queue.
    handle = BackendHandle(ack_queue=ack_queue)  # Bind lifecycle state to the queue.
    with pytest.raises(OSError, match="synthetic close failure"):  # Observe the first cleanup failure.
        handle.close_startup_queue()  # Exercise the failure before completion is published.
    assert handle._queue_closed is False  # Require a later owner to retain retry authority.
    handle.close_startup_queue()  # Retry the complete cleanup sequence.
    assert ack_queue.close_calls == 2  # Require the failed close operation to be retried.
    assert ack_queue.join_calls == 1  # Require the successful attempt to join exactly once.
    assert handle._queue_closed is True  # Publish completion only after both operations succeed.


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
