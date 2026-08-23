from __future__ import annotations

import queue

from queue import Empty as _Empty

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


def test_drain_ready_detects_popen_death_during_load():
    import pytest

    from freetoken.server.supervisor import WorkerDied

    class DeadPopen:
        name = "llama-server"

        def poll(self):
            return 1

    handle = BackendHandle(
        ack_queue=queue.Queue(), processes=[DeadPopen()], expected_acks=1
    )

    with pytest.raises(WorkerDied, match="llama-server"):
        drain_ready(
            handle,
            LoadProgress(),
            get=lambda _t: (_ for _ in ()).throw(_Empty()),
        )


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
