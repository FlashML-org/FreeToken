"""The HTTP idle timeout must reach uvicorn in both server entry paths."""

from __future__ import annotations

import queue
from unittest.mock import patch

import pytest
import uvicorn
from fastapi import FastAPI

from freetoken.server import api_server
from freetoken.server.args import parse_args
from freetoken.server.supervisor import BackendHandle


@pytest.mark.parametrize("run_shell", [False, True])
@pytest.mark.parametrize("argv, expected", [([], 360), (["--http-keep-alive-s", "420"], 420)])
def test_http_keep_alive_reaches_uvicorn(argv, expected, run_shell, monkeypatch):
    config, _ = parse_args([
        "--model", "/models/anon", "--dtype", "bfloat16",
        "--tool-call-parser", "llama3", "--reasoning-parser", "off",
        "--sampling-defaults", "none", *argv,
    ], run_shell=run_shell)

    def check_config(self, config):
        assert config.timeout_keep_alive == expected
        assert config.limit_concurrency is None
        raise RuntimeError("stop before binding HTTP")

    monkeypatch.setattr(api_server, "app", FastAPI())
    monkeypatch.setattr(api_server, "_GLOBAL_STATE", None)
    monkeypatch.setattr(api_server, "_MODEL_SAMPLING", {})
    monkeypatch.setattr(api_server, "init_request_logging", lambda: None)
    monkeypatch.setattr(api_server, "install_polling_access_log_filter", lambda: None)
    monkeypatch.setattr(api_server, "ZmqAsyncPullQueue", lambda *a, **kw: None)
    monkeypatch.setattr(api_server, "ZmqAsyncPushQueue", lambda *a, **kw: None)
    backend = BackendHandle(ack_queue=queue.Queue(), processes=[], expected_acks=0)
    # Stop at the real HTTP config, before any socket, worker or terminal starts.
    with patch.object(api_server.threading.Thread, "start"), patch.object(uvicorn.Server, "__init__", check_config):
        with pytest.raises(RuntimeError, match="stop before binding HTTP"):
            api_server.run_api_server(config, lambda: backend, run_shell=run_shell)
