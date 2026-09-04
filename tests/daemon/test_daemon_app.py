"""Tests for daemon app request models and /bench/run child-process ownership."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pydantic
import pytest

from freetoken.daemon import app as app_mod
from freetoken.daemon.app import BenchBody, CheckpointBody, StartBody, build_app
from freetoken.daemon.logring import LogRing


# --------------------------------------------------------------------------- fakes


class _Mgr:
    def status(self):
        return {"running": False}

    def start(self, model, port, args):
        return {"running": True}

    def stop(self, *a):
        return {}

    def current_pid(self):
        return None


def _build_app(monkeypatch, spawn_fn):
    """Build app with `asyncio.create_subprocess_exec` swapped for ``spawn_fn``; return (app, endpoint)."""
    monkeypatch.setattr(app_mod.asyncio, "create_subprocess_exec", spawn_fn)
    application = build_app(
        manager=_Mgr(),
        ring=LogRing(),
        probe=None,
        footprint_fn=lambda pid: {},
        lifecycle_pool=ThreadPoolExecutor(1),
        proxy_pool=ThreadPoolExecutor(1),
    )
    route = next(r for r in application.routes if getattr(r, "path", None) == "/bench/run")
    return application, route.endpoint


def _spawn_recorder(script: str):
    """Return (spawn_fn, pids). The app module's create_subprocess_exec is swapped for spawn_fn,
    which asserts group ownership, records child pid, and launches ``script``."""
    pids: list[int] = []
    real_create = asyncio.create_subprocess_exec

    def spawn(*argv, **kwargs):
        assert kwargs.get("start_new_session") is True, "child must own its process group"
        assert kwargs.get("stdout") == asyncio.subprocess.PIPE

        class _Spawn:
            def __await__(self):
                proc = yield from real_create(
                    sys.executable,
                    "-c",
                    script,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                ).__await__()
                pids.append(proc.pid)
                return proc

        return _Spawn()

    return spawn, pids


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


async def _wait_gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        await asyncio.sleep(0.05)
    return not _alive(pid)


async def _collect(body, bench_run) -> list[tuple[str, dict]]:
    resp = await bench_run(body=body)
    out = []
    async for chunk in resp.body_iterator:
        event, data = None, None
        for raw_line in chunk.splitlines():
            line = raw_line.decode(errors="replace") if isinstance(raw_line, (bytes, bytearray)) else raw_line
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        out.append((event, data))
    return out


# --------------------------------------------------------------------------- request models


def test_start_body_port_bounds():
    with pytest.raises(pydantic.ValidationError):
        StartBody(model="m", port=0)
    with pytest.raises(pydantic.ValidationError):
        StartBody(model="m", port=-1)
    with pytest.raises(pydantic.ValidationError):
        StartBody(model="m", port=65536)
    assert StartBody.model_validate({"model": "m"}).port is None
    assert StartBody(model="m", port=1).port == 1
    assert StartBody(model="m", port=65535).port == 65535


def test_invalid_port_fails_before_manager_start():
    with pytest.raises(pydantic.ValidationError):
        StartBody.model_validate({"model": "m", "port": 70000})
    assert _Mgr().status().get("running") is False


@pytest.mark.parametrize("cls", [StartBody, CheckpointBody, BenchBody])
def test_list_defaults_are_independent(cls):
    kwargs = {"model": "m"} if cls is StartBody else {"id": "x"} if cls is CheckpointBody else {}
    a = cls.model_validate(kwargs)
    b = cls.model_validate(kwargs)
    a.args.append("--dtype")
    assert b.args == []
    assert cls.model_validate(kwargs).args == []


# --------------------------------------------------------------------------- bench SSE frames


def test_bench_spawn_error_frame(monkeypatch):
    async def boom(*argv, **kwargs):
        raise RuntimeError("no such binary")

    _, bench_run = _build_app(monkeypatch, boom)
    events = asyncio.run(_collect(BenchBody(), bench_run))
    assert len(events) == 1
    event, data = events[0]
    assert event == "error"
    assert "failed to spawn bench" in data["message"]


def test_bench_progress_frames(monkeypatch):
    script = 'print("FTBENCH 1 2 dtype", flush=True)\nprint("FTBENCH 2 2 dtype", flush=True)\n'
    spawn, pids = _spawn_recorder(script)
    monkeypatch.setattr(app_mod, "_read_bench_profile", lambda *_a, **_k: None)
    _, bench_run = _build_app(monkeypatch, spawn)
    events = asyncio.run(_collect(BenchBody(), bench_run))
    kinds = [e for e, _ in events]
    assert kinds == ["progress", "progress", "error"]
    assert events[-1][1]["message"] == "bench finished but no profile was written"
    for pid in pids:
        assert not _alive(pid)


def test_bench_result_frame(monkeypatch):
    script = 'print("FTBENCH 1 1 dtype", flush=True)\n'
    spawn, _ = _spawn_recorder(script)
    monkeypatch.setattr(app_mod, "_read_bench_profile", lambda *_a, **_k: {"result": {"ok": True}})
    _, bench_run = _build_app(monkeypatch, spawn)
    events = asyncio.run(_collect(BenchBody(), bench_run))
    kinds = [e for e, _ in events]
    assert kinds == ["progress", "result"]
    assert events[-1][1] == {"result": {"ok": True}}


# --------------------------------------------------------------------------- orphaning on close


TERM_DIES = (
    'import sys, time\n'
    'print("FTBENCH 1 99 dtype", flush=True)\n'
    "time.sleep(60)\n"
)
TERM_TRAPS = (
    "import signal, time\n"
    "signal.signal(signal.SIGTERM, lambda *a: None)\n"
    'print("FTBENCH 1 99 dtype", flush=True)\n'
    "time.sleep(60)\n"
)


async def _run_close_case(monkeypatch, script):
    """Park the bench generator at its first SSE yield while the child still runs,
    then aclose() it and assert the child was terminated and reaped."""
    spawn, pids = _spawn_recorder(script)
    _, bench_run = _build_app(monkeypatch, spawn)
    resp = await bench_run(body=BenchBody())
    agen = resp.body_iterator
    frame = await agen.__anext__()
    if isinstance(frame, (bytes, bytearray)):
        frame = frame.decode(errors="replace")
    assert frame.startswith("event: progress"), frame
    while not pids or not _alive(pids[0]):
        await asyncio.sleep(0.02)
    pid = pids[0]
    return agen, pid


def test_generator_close_terminates_and_reaps_child(monkeypatch):
    """Reproduces the pre-fix bug: closing the response generator mid-stream used to
    abandon the sleeping bench child. After the fix, aclose() terminates and reaps."""

    async def run():
        agen, pid = await _run_close_case(monkeypatch, TERM_DIES)
        t0 = time.monotonic()
        await agen.aclose()
        elapsed = time.monotonic() - t0
        assert elapsed < 4.5, "well-behaved child should die on SIGTERM, not wait out grace"
        assert await _wait_gone(pid), "bench child orphaned after generator close"

    asyncio.run(run())


@pytest.mark.slow
def test_kill_only_on_ignored_terminate(monkeypatch):
    """A child that traps SIGTERM survives the grace period and dies by SIGKILL."""

    async def run():
        agen, pid = await _run_close_case(monkeypatch, TERM_TRAPS)
        t0 = time.monotonic()
        await agen.aclose()
        elapsed = time.monotonic() - t0
        assert elapsed >= 4.5, "SIGKILL must wait out the terminate grace"
        assert await _wait_gone(pid), "term-trapping child killed after grace"

    asyncio.run(run())
