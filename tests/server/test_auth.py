"""Exercise auth at the ASGI boundary, including streaming and admin exemptions."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from freetoken.server.auth import install_auth
from starlette.responses import StreamingResponse


def client(keys="", *, root_path=""):
    app = FastAPI()

    @app.get("/{path:path}")
    def route(path: str):
        return {"path": path}

    install_auth(app, keys)
    return TestClient(app, root_path=root_path)


@pytest.mark.parametrize("key", ["first", "second"])
def test_each_rotation_key_is_accepted(key):
    with client(" first, second ") as http:
        assert http.get("/v1/models", headers={"Authorization": f"Bearer {key}"}).status_code == 200


@pytest.mark.parametrize("header", [None, "Bearer wrong", "Basic first", "Bearer", "Bearer "])
def test_invalid_or_missing_key_uses_openai_error_shape(header):
    with client("first") as http:
        response = http.get("/v1/models", headers={} if header is None else {"Authorization": header})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"] == {
        "message": "Invalid or missing API key",
        "type": "authentication_error",
        "param": None,
        "code": "invalid_api_key",
    }


@pytest.mark.parametrize("path", ["/health", "/healthz", "/readyz", "/v1/admin/prepare-stop"])
def test_only_explicit_exemptions_reach_their_handler(path):
    with client("first") as http:
        assert http.get(path).status_code == 200


@pytest.mark.parametrize("path", ["/metrics", "/generate", "/docs", "/v1/administer", "/health/extra"])
def test_non_v1_routes_and_similar_prefixes_are_protected(path):
    with client("first") as http:
        assert http.get(path).status_code == 401


@pytest.mark.parametrize("keys", ["", " , "])
def test_disabled_auth_preserves_keyless_and_arbitrary_header_clients(keys):
    with client(keys) as http:
        assert http.get("/v1/models").status_code == 200
        assert http.get("/generate", headers={"Authorization": "Bearer arbitrary"}).status_code == 200


def test_root_path_does_not_hide_probes_or_unprotect_api():
    with client("first", root_path="/proxy") as http:
        assert http.get("/proxy/health").status_code == 200
        assert http.get("/proxy/v1/models").status_code == 401


def test_duplicate_credentials_are_rejected():
    with client("first") as http:
        assert http.get("/v1/models", headers=[
            ("authorization", "Bearer first"), ("authorization", "Bearer wrong")
        ]).status_code == 401


def test_stream_is_forwarded_without_reframing():
    app = FastAPI()

    @app.get("/stream")
    def stream():
        return StreamingResponse(iter([b"data: hello\n\n", b"data: [DONE]\n\n"]))

    install_auth(app, "first")
    with TestClient(app) as http:
        response = http.get("/stream", headers={"Authorization": "Bearer first"})
    assert response.content == b"data: hello\n\ndata: [DONE]\n\n"


def test_exempt_admin_still_checks_the_socket_peer():
    from types import SimpleNamespace

    from freetoken.server.accounting import register_accounting_routes

    app = FastAPI()
    register_accounting_routes(app, lambda: SimpleNamespace())
    install_auth(app, "first")
    with TestClient(app, client=("203.0.113.8", 1234)) as http:
        response = http.post("/v1/admin/prepare-stop", json={})
    assert response.status_code == 403


def test_rotation_checks_every_key_even_after_a_match(monkeypatch):
    from freetoken.server import auth

    compare = auth.hmac.compare_digest
    lengths = []

    def observe(left, right):
        lengths.append((len(left), len(right)))
        return compare(left, right)

    monkeypatch.setattr(auth.hmac, "compare_digest", observe)
    with client("first,second,third") as http:
        assert http.get("/v1/models", headers={"Authorization": "bearer first"}).status_code == 200
    assert lengths == [(32, 32)] * 3


def test_cors_preflight_can_negotiate_but_actual_request_requires_auth():
    from starlette.middleware.cors import CORSMiddleware

    app = FastAPI()

    @app.get("/v1/models")
    def models():
        return {"data": []}

    install_auth(app, "first")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:1420"],
                       allow_methods=["*"], allow_headers=["*"])
    with TestClient(app) as http:
        preflight = http.options("/v1/models", headers={
            "Origin": "http://localhost:1420",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Authorization",
        })
        assert preflight.status_code == 200
        response = http.get("/v1/models", headers={"Origin": "http://localhost:1420"})
        assert response.status_code == 401
        assert response.headers["access-control-allow-origin"] == "http://localhost:1420"
