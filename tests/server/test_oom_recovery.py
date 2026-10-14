from __future__ import annotations

import asyncio
import queue
import threading
from types import SimpleNamespace

import pytest

from freetoken.message import UserReply
from freetoken.server import api_server
from freetoken.server.accounting import AdmissionClosedError
from freetoken.server.api_server import FrontendManager


class _AsyncQueue:
    def __init__(self) -> None:
        self.q: asyncio.Queue = asyncio.Queue()

    async def get(self):
        return await self.q.get()

    async def put(self, item) -> None:
        await self.q.put(item)


def _state(*, maintenance_state: str = "serving") -> FrontendManager:
    recv, send = _AsyncQueue(), _AsyncQueue()
    return FrontendManager(
        config=SimpleNamespace(
            served_model_name="test",
            model_path="/models/test",
            tool_call_parser="llama3",
            reasoning_parser=None,
            mm=SimpleNamespace(text_model_only=False, disabled_encoders=frozenset()),
            max_output_tokens=None,
            oom_recovery_wait_queue=2,
            oom_recovery_ready_timeout=0.02,
        ),
        send_tokenizer=send,
        recv_tokenizer=recv,
        maintenance_state=maintenance_state,
    )


def test_restarting_admission_is_bounded_cancellable_and_stamped_after_ready(monkeypatch):
    async def scenario() -> None:
        state = _state(maintenance_state="restarting")
        state._loop = asyncio.get_running_loop()
        state._ready_event = asyncio.Event()

        with pytest.raises(AdmissionClosedError, match="timed out"):
            await state.new_user()
        assert state._restart_waiters == 0

        state._restart_waiters = state.config.oom_recovery_wait_queue
        with pytest.raises(AdmissionClosedError, match="queue is full"):
            await state.new_user()
        state._restart_waiters = 0

        cancelled = asyncio.create_task(state.new_user())
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert state._restart_waiters == 0

        admitted = asyncio.create_task(state.new_user())
        await asyncio.sleep(0)
        state.backend_epoch = 4
        state.maintenance_state = "serving"
        state._ready_event.set()
        uid = await admitted
        assert state.request_epoch(uid) == 4
        assert state._restart_waiters == 0

        state.maintenance_state = "restarting"
        state._ready_event = asyncio.Event()
        failed = asyncio.create_task(state.new_user())
        await asyncio.sleep(0)
        state.maintenance_state = "failed"
        state.wake_failed_restart_waiters()
        with pytest.raises(AdmissionClosedError, match="failed"):
            await failed

    asyncio.run(scenario())


def test_oom_terminalizes_every_dispatched_stream_once_and_never_replays_it():
    async def scenario() -> None:
        state = _state()
        state._loop = asyncio.get_running_loop()
        first, second = await state.new_user(), await state.new_user()

        state.fail_dispatched_for_oom("CUDA out of memory")
        state.fail_dispatched_for_oom("second notification")
        await asyncio.sleep(0)

        for uid in (first, second):
            terminal = state.ack_map[uid]
            assert len(terminal) == 1
            assert terminal[0].finished is True
            assert terminal[0].error_code == "insufficient_workspace"
            assert terminal[0].epoch == state.request_epoch(uid)

    asyncio.run(scenario())


def test_listener_drops_stale_generation_before_delivery_or_global_stats():
    class _Stats:
        def __init__(self) -> None:
            self.observed: list[UserReply] = []

        def on_new_user(self, uid: int) -> None:
            pass

        def observe(self, reply: UserReply) -> None:
            self.observed.append(reply)

    async def scenario() -> None:
        state = _state()
        state.stats = _Stats()
        state._create_listener_once()
        state.backend_epoch = 7
        uid = await state.new_user()
        await state.recv_tokenizer.put(UserReply(uid=uid, incremental_output="old", finished=True, epoch=6))
        await asyncio.sleep(0)
        assert state.ack_map[uid] == []
        assert state.stats.observed == []

        fresh = UserReply(uid=uid, incremental_output="new", finished=True, epoch=7)
        await state.recv_tokenizer.put(fresh)
        await asyncio.sleep(0)
        assert state.ack_map[uid] == [fresh]
        assert state.stats.observed == [fresh]

        # An abort removes delivery state but cannot erase work already performed by the
        # current backend; count it. A retired backend's unknown-uid reply is different and
        # must remain invisible to global stats.
        disconnected = UserReply(uid=999, incremental_output="late", finished=True, epoch=7)
        stale_unknown = UserReply(uid=998, incremental_output="old late", finished=True, epoch=6)
        await state.recv_tokenizer.put(disconnected)
        await state.recv_tokenizer.put(stale_unknown)
        await asyncio.sleep(0)
        assert state.stats.observed == [fresh, disconnected]

        for task in asyncio.all_tasks():
            if task is not asyncio.current_task() and task.get_coro().__name__ == "listen":
                task.cancel()

    asyncio.run(scenario())


def test_recovery_defaults_remain_disabled_and_limits_are_validated(monkeypatch):
    from freetoken.server.args import parse_args

    monkeypatch.delenv("FREETOKEN_API_KEY", raising=False)
    defaults, _ = parse_args(["--model", "/models/test", "--dtype", "bfloat16"])
    assert defaults.oom_recovery is False
    assert defaults.oom_recovery_max_restarts == 1
    assert defaults.oom_recovery_ready_timeout == 900.0
    assert defaults.oom_recovery_wait_queue == 128

    configured, _ = parse_args([
        "--model", "/models/test", "--dtype", "bfloat16", "--oom-recovery", "--oom-recovery-max-restarts", "2",
        "--oom-recovery-ready-timeout", "1.5", "--oom-recovery-wait-queue", "3",
    ])
    assert (configured.oom_recovery, configured.oom_recovery_max_restarts,
            configured.oom_recovery_ready_timeout, configured.oom_recovery_wait_queue) == (True, 2, 1.5, 3)

    with pytest.raises(SystemExit):
        parse_args(["--model", "/models/test", "--dtype", "bfloat16", "--oom-recovery-max-restarts", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--model", "/models/test", "--dtype", "bfloat16", "--oom-recovery-wait-queue", "0"])
    for invalid in ("0", "-1", "inf", "nan"):
        with pytest.raises(SystemExit):
            parse_args(["--model", "/models/test", "--dtype", "bfloat16", "--oom-recovery-ready-timeout", invalid])


def test_restart_gauge_reset_preserves_lifetime_request_and_token_accounting():
    from freetoken.server.stats import StatsTracker

    stats = StatsTracker()
    stats.on_new_user(7)
    stats.observe(UserReply(
        uid=7, incremental_output="x", finished=True, prompt_tokens_delta=3,
        completion_tokens_delta=5, kv_used_pages=2, kv_total_pages=8,
        mamba_used_slots=1, mamba_total_slots=4, swa_used_tokens=10,
        swa_total_tokens=20, gpu_mem_bytes=123,
    ))

    stats.clear_runtime_gauges()

    assert stats.completed == 1
    assert stats.prompt_tokens_total == 3
    assert stats.completion_tokens_total == 5
    assert (stats.kv_used_pages, stats.kv_total_pages, stats.mamba_used_slots,
            stats.mamba_total_slots, stats.swa_used_tokens, stats.swa_total_tokens,
            stats.vram_bytes) == (0, 0, 0, 0, 0, 0, 0)


def test_rebuild_yield_racing_epoch_transition_cannot_register_or_reopen_after_restart():
    from freetoken.server.api_server import dispatch_rebuild

    async def scenario() -> None:
        entered, release = asyncio.Event(), asyncio.Event()
        sent = []

        async def send_one(message):
            sent.append(message)
            entered.set()
            await release.wait()

        state = SimpleNamespace(
            backend_epoch=0, maintenance_state="serving", rebuild_futures={}, send_one=send_one
        )
        task = asyncio.create_task(dispatch_rebuild(state, moe_cache_size=8, num_pages=None))
        await entered.wait()
        assert len(state.rebuild_futures) == 1
        assert sent[0].epoch == 0

        # This is the loop-serialized OOM transition while dispatch_rebuild is suspended in
        # its transport await: it owns the gate and resolves/removes existing waiters first.
        state.backend_epoch = 1
        state.maintenance_state = "restarting"
        state.rebuild_futures.clear()
        release.set()
        result = await task

        assert result == {"status": "failed", "error": "cache rebuild superseded by engine lifecycle"}
        assert state.rebuild_futures == {}
        assert state.maintenance_state == "restarting"
        assert [message.epoch for message in sent] == [0]

    asyncio.run(scenario())


def test_post_restart_tokenize_error_is_terminal_in_replacement_epoch():
    from freetoken.core import SamplingParams
    from freetoken.message import TokenizeMsg
    from freetoken.tokenizer.server import _tokenize_requests

    class BrokenTokenizer:
        def tokenize(self, messages):
            raise ValueError("template rejected request")

    class Logger:
        def warning(self, message):
            pass

    _, errors = _tokenize_requests(
        BrokenTokenizer(),
        [TokenizeMsg(uid=12, text="prompt", sampling_params=SamplingParams(), epoch=1)],
        Logger(),
    )

    assert len(errors) == 1
    assert errors[0].finished is True
    assert errors[0].epoch == 1
    assert "could not encode request" in errors[0].error


def test_prepare_stop_while_restart_loading_keeps_admission_closed():
    async def scenario() -> None:
        state = _state(maintenance_state="restarting")
        state._loop = asyncio.get_running_loop()
        state._ready_event = asyncio.Event()
        waiter = asyncio.create_task(state.new_user())
        await asyncio.sleep(0)

        # prepare-stop owns the lifecycle state before replacement readiness races in.
        state.maintenance_state = "stopping"
        state._ready_event.set()
        with pytest.raises(AdmissionClosedError, match="stopping"):
            await waiter
        with pytest.raises(AdmissionClosedError, match="stopping"):
            await state.new_user()
        assert state.uid_counter == 0

    asyncio.run(scenario())


def test_timed_out_queued_lifecycle_callback_cannot_mutate_after_loop_releases():
    """A supervisor timeout may cancel only an unclaimed loop callback. When that
    callback finally runs it must observe cancellation and leave lifecycle state intact."""
    import concurrent.futures

    from freetoken.server.api_server import _run_on_loop

    class DelayedLoop:
        callback = None

        def call_soon_threadsafe(self, callback):
            self.callback = callback

    loop = DelayedLoop()
    state = {"maintenance": "restarting"}
    outcome: list[BaseException] = []

    def caller() -> None:
        try:
            _run_on_loop(loop, lambda: state.__setitem__("maintenance", "serving"), timeout=0.01)
        except BaseException as exc:
            outcome.append(exc)

    thread = threading.Thread(target=caller)
    thread.start()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert isinstance(outcome[0], concurrent.futures.TimeoutError)
    assert state["maintenance"] == "restarting"

    loop.callback()
    assert state["maintenance"] == "restarting"


class _ThreadQueue:
    """CPU-only bridge between a TestClient request thread and FrontendManager's loop."""
    def __init__(self) -> None:
        self.q: queue.Queue = queue.Queue()

    async def get(self):
        # Never delegate a permanent Queue.get() to the default executor: pytest's
        # asyncio.run teardown waits for executor workers, which would hang this CPU test.
        while True:
            try:
                return self.q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.001)

    async def put(self, item) -> None:
        self.q.put(item)


def test_four_protocol_routes_dispatch_after_restart_at_replacement_epoch():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from freetoken.server.anthropic_api import register_anthropic_routes
    from freetoken.server.openai_api import register_openai_routes
    from freetoken.server.responses_api import register_responses_routes

    requests = [
        ("/v1/chat/completions", {"model": "test", "messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/completions", {"model": "test", "prompt": "hi"}),
        ("/v1/messages", {"model": "test", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/responses", {"model": "test", "input": "hi"}),
    ]
    for path, payload in requests:
        recv, send = _ThreadQueue(), _ThreadQueue()
        state = _state(maintenance_state="restarting")
        state.recv_tokenizer = recv
        state.send_tokenizer = send
        app = FastAPI()
        register_openai_routes(app, lambda: state, lambda: {})
        register_anthropic_routes(app, lambda: state, lambda: {})
        register_responses_routes(app, lambda: state, lambda: {})
        with TestClient(app) as client:
            result: list = []
            thread = threading.Thread(target=lambda: result.append(client.post(path, json=payload)))
            thread.start()
            for _ in range(100):
                if state._ready_event is not None:
                    break
                threading.Event().wait(0.01)
            assert state._ready_event is not None, path
            assert send.q.empty(), path
            state.backend_epoch = 1
            state.maintenance_state = "serving"
            # The waiting admission belongs to TestClient's event-loop thread.
            # Cross-thread Event.set is not a valid asyncio synchronization primitive.
            state._ready_event._loop.call_soon_threadsafe(state._ready_event.set)
            tokenized = send.q.get(timeout=2)
            assert tokenized.epoch == 1
            recv.q.put(UserReply(uid=tokenized.uid, incremental_output="ok", finished=True, epoch=1))
            thread.join(timeout=2)
            assert not thread.is_alive(), path
            assert result[0].status_code == 200, (path, result[0].text)
