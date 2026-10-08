"""--api-key: the gate in front of every route but /health, and where the key comes from."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
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

