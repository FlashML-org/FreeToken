"""--api-key: the bearer gate on the API server, its argument parsing, and the shell client.

The gate is a middleware registered at import time and armed by ``install_api_key`` (what
``run_api_server`` calls with ``ServerArgs.api_key``). With no key it is inert, so every
existing test keeps its behaviour. These tests drive the real ``api.app`` through a TestClient
and flip ``_API_KEY`` directly, the way ``test_rebuild_maintenance`` flips ``_GLOBAL_STATE``.
"""

from __future__ import annotations

import urllib.request
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from freetoken.server.args import parse_args


class _Config:
    def to_dict(self) -> dict:
        return {"architectures": ["DeepseekV4ForCausalLM"], "torch_dtype": "bfloat16"}


def _parse(extra: list[str]):
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: _Config()):
        return parse_args(["--model", "/models/anon", *extra])


# ------------------------------------------------------------------ argument parsing


def test_api_key_defaults_to_none(monkeypatch):
    monkeypatch.delenv("FREETOKEN_API_KEY", raising=False)
    args, _ = _parse([])
    assert args.api_key is None


def test_api_key_flag_is_parsed(monkeypatch):
    monkeypatch.delenv("FREETOKEN_API_KEY", raising=False)
    args, _ = _parse(["--api-key", "s3cret"])
    assert args.api_key == "s3cret"


def test_api_key_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("FREETOKEN_API_KEY", "from-env")
    args, _ = _parse([])
    assert args.api_key == "from-env"


def test_api_key_flag_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("FREETOKEN_API_KEY", "from-env")
    args, _ = _parse(["--api-key", "from-flag"])
    assert args.api_key == "from-flag"


def test_empty_environment_key_means_unset(monkeypatch):
    monkeypatch.setenv("FREETOKEN_API_KEY", "")
    args, _ = _parse([])
    assert args.api_key is None


def test_empty_api_key_flag_is_rejected(monkeypatch):
    monkeypatch.delenv("FREETOKEN_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="2"):
        _parse(["--api-key", "  "])


def test_shell_mode_accepts_a_key(monkeypatch):
    # Unlike TLS, a key is fine in shell mode: the attached client carries it (see below).
    monkeypatch.delenv("FREETOKEN_API_KEY", raising=False)
    args, run_shell = _parse(["--shell-mode", "--api-key", "s3cret"])
    assert run_shell is True
    assert args.api_key == "s3cret"


# ------------------------------------------------------------------ the middleware


def _serving_state():
    return SimpleNamespace(
        maintenance_state="serving",
        config=SimpleNamespace(served_model_name="anon"),
        fatal_error=None,
        ready_at=None,
        instance_id=None,
        rebuild_futures={},
        last_rebuild=None,
    )


@pytest.fixture
def keyed_client(monkeypatch):
    import freetoken.server.api_server as api

    monkeypatch.setattr(api, "_API_KEY", "s3cret")
    monkeypatch.setattr(api, "_GLOBAL_STATE", _serving_state())
    return TestClient(api.app)


def test_no_key_configured_leaves_every_route_open(monkeypatch):
    import freetoken.server.api_server as api

    monkeypatch.setattr(api, "_API_KEY", None)
    monkeypatch.setattr(api, "_GLOBAL_STATE", _serving_state())
    client = TestClient(api.app)
    assert client.get("/v1").status_code == 200
    assert client.get("/health").status_code == 200


def test_missing_header_is_401_with_a_bearer_challenge(keyed_client):
    r = keyed_client.get("/v1")
    assert r.status_code == 401
    assert r.headers["WWW-Authenticate"] == "Bearer"
    assert r.json()["error"]["type"] == "authentication_error"


@pytest.mark.parametrize(
    "authorization",
    ["Bearer wrong", "Bearer s3cre", "Bearer s3cret-and-more", "Basic s3cret", "s3cret"],
)
def test_wrong_or_malformed_credentials_are_401(keyed_client, authorization):
    r = keyed_client.get("/v1", headers={"Authorization": authorization})
    assert r.status_code == 401


def test_matching_bearer_passes(keyed_client):
    r = keyed_client.get("/v1", headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    # scheme is case-insensitive, surrounding whitespace on the token is not the token
    r = keyed_client.get("/v1", headers={"Authorization": "bearer  s3cret "})
    assert r.status_code == 200


def test_health_stays_open_for_liveness_probes(keyed_client):
    r = keyed_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.parametrize(
    "method, path",
    [
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/messages"),
        ("POST", "/v1/responses"),
        ("GET", "/v1/models"),
        ("GET", "/v1/stats"),
        ("GET", "/v1/requests"),
        ("POST", "/v1/cache/rebuild"),
        ("POST", "/v1/admin/prepare-stop"),
        ("POST", "/generate"),
    ],
)
def test_every_other_route_is_gated_before_its_handler(keyed_client, method, path):
    # No body and no engine behind these: a 401 proves the gate answered first, since the
    # handler would have produced a 422/503 or needed the state.
    r = keyed_client.request(method, path)
    assert r.status_code == 401


def test_cors_preflight_is_not_challenged(keyed_client):
    # A preflight carries no credentials by design; the CORS middleware (installed at startup,
    # outermost) answers it. Without CORS configured FastAPI's own answer is what we see --
    # anything but a 401 is the assertion.
    r = keyed_client.options("/v1/chat/completions")
    assert r.status_code != 401


def test_install_api_key_arms_and_disarms():
    import freetoken.server.api_server as api

    prev = api._API_KEY
    try:
        api.install_api_key("k")
        assert api._API_KEY == "k"
        api.install_api_key("")
        assert api._API_KEY is None
        api.install_api_key(None)
        assert api._API_KEY is None
    finally:
        api._API_KEY = prev


# ------------------------------------------------------------------ the shell client


def test_shell_client_sends_the_key_on_the_control_plane(monkeypatch):
    from freetoken.shell.client import ShellClient

    seen: dict = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"status": "ok"}'

    def _urlopen(request, timeout):
        seen["authorization"] = request.get_header("Authorization")
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    client = ShellClient("http://127.0.0.1:1", api_key="s3cret")
    assert client._request_json_blocking("GET", "/health", None, 1.0) == {"status": "ok"}
    assert seen["authorization"] == "Bearer s3cret"
