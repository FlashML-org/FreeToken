"""--api-key: the gate in front of every route but /health, and where the key comes from."""

from __future__ import annotations

import json
import os
import stat
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from freetoken import launch
from freetoken.server.api_server import install_api_key, install_cors
from freetoken.server.args import parse_args

KEY = "sk-test"
ORIGIN = "http://localhost:1420"


def _client(api_key: str | None) -> TestClient:
    app = FastAPI()
    app.get("/health")(lambda: {"status": "ok"})
    app.get("/v1/models")(lambda: {"data": []})
    install_api_key(app, api_key)
    install_cors(app, ORIGIN)
    return TestClient(app)


def _parse(*argv: str):
    hf = SimpleNamespace(to_dict=lambda: {"architectures": ["Qwen3ForCausalLM"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: hf):
        return parse_args(["--model", "/models/anon", *argv])[0]


@pytest.mark.parametrize(
    "headers, status",
    [
        ({}, 401),
        ({"Authorization": "Bearer wrong"}, 401),
        ({"Authorization": f"Basic {KEY}"}, 401),
        ({"Authorization": KEY}, 401),
        ({"Authorization": b"Bearer caf\xe9"}, 401),
        ({"x-api-key": "wrong"}, 401),
        ({"Authorization": f"Bearer {KEY}"}, 200),
        ({"Authorization": f"bearer {KEY}"}, 200),
        ({"x-api-key": KEY}, 200),
    ],
)
def test_key_gates_every_route_but_health(headers, status):
    client = _client(KEY)
    response = client.get("/v1/models", headers=headers)
    assert response.status_code == status
    if status == 401:
        assert response.json() == {"error": "Unauthorized"}
    assert client.get("/health").status_code == 200


def test_no_key_serves_open():
    assert _client(None).get("/v1/models").status_code == 200


def test_cors_preflight_passes_and_401_carries_cors_headers():
    client = _client(KEY)
    preflight = client.options(
        "/v1/models",
        headers={"Origin": ORIGIN, "Access-Control-Request-Method": "GET",
                 "Access-Control-Request-Headers": "authorization"},
    )
    assert preflight.status_code == 200
    denied = client.get("/v1/models", headers={"Origin": ORIGIN})
    assert denied.status_code == 401
    assert denied.headers["access-control-allow-origin"] == ORIGIN


def test_options_outside_a_cors_preflight_needs_the_key():
    assert _client(KEY).options("/v1/models").status_code == 401


def test_flag_wins_over_env_and_empty_values_do_not_enable_auth(monkeypatch):
    monkeypatch.delenv("FREETOKEN_API_KEY", raising=False)
    assert _parse().api_key is None
    monkeypatch.setenv("FREETOKEN_API_KEY", "from-env")
    assert _parse().api_key == "from-env"
    assert _parse("--api-key", KEY).api_key == KEY
    monkeypatch.setenv("FREETOKEN_API_KEY", "")
    assert _parse().api_key is None
    with pytest.raises(SystemExit):
        _parse("--api-key", "")


def test_key_stays_out_of_the_logged_config(monkeypatch):
    monkeypatch.delenv("FREETOKEN_API_KEY", raising=False)
    assert KEY not in str(_parse("--api-key", KEY))


@pytest.mark.parametrize("agent", sorted(launch.PREPARERS))
def test_dry_run_prints_a_placeholder_instead_of_the_key(agent, monkeypatch, capsys):
    key = 'p"w\\d\xe9'
    model = launch.ServedModel(model_id="m", models=["m"], context_length=8192)
    monkeypatch.setattr(launch, "discover_server_model", lambda _server, _key: model)
    assert launch.main([agent, "--dry-run", "--api-key", key]) == 0
    out = capsys.readouterr().out
    assert key not in out and json.dumps(key)[1:-1] not in out
    if agent in ("claude", "codex", "dsh", "opencode"):
        assert "<api-key>" in out


def test_dsh_keeps_the_key_out_of_the_variable_its_web_search_reads(monkeypatch, tmp_path):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    monkeypatch.setattr(launch, "_warn_dsh_node_version", lambda: None)
    ctx = launch.LaunchContext(
        server=launch.resolve_server_url("http://127.0.0.1:1919"),
        model=launch.ServedModel(model_id="m", models=["m"], context_length=8192),
        extra_args=[],
        dry_run=False,
        api_key=KEY,
    )
    env = launch.prepare_dsh(ctx).env
    settings = yaml.safe_load((tmp_path / launch.DSH_LAUNCH_SETTINGS_NAME).read_text())
    assert env["DEEPSEEK_API_KEY"] != KEY
    assert env[settings["llm-deepseek"]["apiKeyEnv"]] == KEY


@pytest.mark.parametrize("agent", ["hermes", "openclaw"])
def test_config_files_holding_the_key_are_owner_only(agent, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    old_umask = os.umask(0o022)
    try:
        for key in ("sk-first", "sk-second"):
            ctx = launch.LaunchContext(
                server=launch.resolve_server_url("http://127.0.0.1:1919"),
                model=launch.ServedModel(model_id="m", models=["m"], context_length=65536),
                extra_args=[],
                dry_run=False,
                assume_yes=True,
                api_key=key,
            )
            launch.PREPARERS[agent](ctx)
    finally:
        os.umask(old_umask)
    written = [p for p in tmp_path.rglob("*") if p.is_file() and "sk-" in p.read_text()]
    assert len(written) == 2
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in written)
