from __future__ import annotations

import asyncio
import inspect
import os
import subprocess
import sys
import textwrap
import uuid

import pytest
import uvicorn
import zmq
import zmq.asyncio


async def _zmq_asyncio_roundtrip() -> None:
    context = zmq.asyncio.Context()
    receiver = context.socket(zmq.PAIR)
    sender = context.socket(zmq.PAIR)
    endpoint = f"inproc://freetoken-windows-selector-{uuid.uuid4()}"
    receiver.bind(endpoint)
    sender.connect(endpoint)
    try:
        receive_task = asyncio.ensure_future(receiver.recv())
        await asyncio.sleep(0)
        await sender.send(b"selector-ready")
        assert await asyncio.wait_for(receive_task, timeout=2) == b"selector-ready"
    finally:
        sender.close(linger=0)
        receiver.close(linger=0)
        context.term()


@pytest.mark.skipif(os.name != "nt", reason="native Windows asyncio/ZMQ regression")
def test_raw_windows_proactor_reproduces_pyzmq_failure_without_tornado() -> None:
    code = textwrap.dedent(
        """
        import asyncio
        import importlib.abc
        import sys
        import uuid

        class BlockTornado(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "tornado" or fullname.startswith("tornado."):
                    raise ModuleNotFoundError("No module named 'tornado'", name=fullname)
                return None

        sys.meta_path.insert(0, BlockTornado())

        import zmq
        import zmq.asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        async def probe():
            context = zmq.asyncio.Context()
            receiver = context.socket(zmq.PAIR)
            endpoint = f"inproc://raw-proactor-{uuid.uuid4()}"
            receiver.bind(endpoint)
            try:
                await asyncio.wait_for(receiver.recv(), timeout=0.1)
            finally:
                receiver.close(linger=0)
                context.term()

        asyncio.run(probe())
        """
    )

    completed = subprocess.run(
        [sys.executable, "-B", "-s", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode != 0
    assert "No module named 'tornado'" in completed.stderr
    assert "Proactor event loop does not implement add_reader" in completed.stderr


@pytest.mark.skipif(os.name != "nt", reason="native Windows asyncio/ZMQ regression")
def test_server_bootstrap_selects_compatible_loop_before_zmq_asyncio() -> None:
    from freetoken.server.launch import _configure_windows_asyncio_policy

    original_policy = asyncio.get_event_loop_policy()
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        assert _configure_windows_asyncio_policy() is True
        assert isinstance(
            asyncio.get_event_loop_policy(),
            asyncio.WindowsSelectorEventLoopPolicy,
        )
        asyncio.run(_zmq_asyncio_roundtrip())
    finally:
        asyncio.set_event_loop_policy(original_policy)


@pytest.mark.skipif(os.name != "nt", reason="native Windows uvicorn loop regression")
def test_uvicorn_config_and_server_use_explicit_selector_loop_factory() -> None:
    from freetoken.server import api_server

    observed: dict[str, asyncio.AbstractEventLoop] = {}

    class LoopProbeServer(uvicorn.Server):
        async def serve(self, sockets=None) -> None:
            observed["running_loop"] = asyncio.get_running_loop()

    async def noop_app(scope, receive, send) -> None:
        return None

    original_policy = asyncio.get_event_loop_policy()
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        get_loop_factory = getattr(api_server, "_get_uvicorn_loop_factory", None)
        loop_option = get_loop_factory() if get_loop_factory is not None else "auto"
        config = uvicorn.Config(noop_app, loop=loop_option, log_config=None)

        resolved_factory = config.get_loop_factory()
        if callable(loop_option):
            assert resolved_factory is loop_option
        resolved_loop = resolved_factory()
        try:
            assert isinstance(resolved_loop, asyncio.SelectorEventLoop)
            assert not isinstance(resolved_loop, asyncio.ProactorEventLoop)
        finally:
            resolved_loop.close()

        LoopProbeServer(config).run()
        assert isinstance(observed["running_loop"], asyncio.SelectorEventLoop)
        assert not isinstance(observed["running_loop"], asyncio.ProactorEventLoop)
    finally:
        asyncio.set_event_loop_policy(original_policy)


def test_non_windows_uvicorn_loop_selection_remains_auto() -> None:
    from freetoken.server.api_server import _get_uvicorn_loop_factory

    assert _get_uvicorn_loop_factory(platform_name="posix") == "auto"


def test_both_uvicorn_entrypoints_pass_the_explicit_loop_factory() -> None:
    from freetoken.server.api_server import _serve_and_run_shell, run_api_server

    assert "loop=_get_uvicorn_loop_factory()" in inspect.getsource(_serve_and_run_shell)
    assert "loop=_get_uvicorn_loop_factory()" in inspect.getsource(run_api_server)


def test_non_windows_bootstrap_leaves_event_loop_policy_unchanged() -> None:
    from freetoken.server.launch import _configure_windows_asyncio_policy

    original_policy = asyncio.get_event_loop_policy()
    assert _configure_windows_asyncio_policy(platform_name="linux") is False
    assert asyncio.get_event_loop_policy() is original_policy


def test_launch_configures_windows_policy_before_api_server_import() -> None:
    from freetoken.server.launch import launch_server

    source = inspect.getsource(launch_server)
    configure_at = source.index("_configure_windows_asyncio_policy()")
    api_import_at = source.index("from .api_server import run_api_server")
    assert configure_at < api_import_at
