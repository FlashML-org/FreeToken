"""Unit tests for the POST /v1/tokenize endpoint."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from freetoken.server.api_models import TokenizeRequest
from freetoken.server.openai_api import register_openai_routes


class FakeTokenizer:
    """Stands in for the frontend TokenizeManager: raw encode only."""

    class _Tok:
        @staticmethod
        def encode(text, add_special_tokens=True, **_):
            # Deterministic stand-in: one token per 4 chars, ids are positions.
            n = max(1, len(text) // 4)
            return list(range(100, 100 + n))

    tokenizer = _Tok()


class FakeState:
    def __init__(self) -> None:
        from types import SimpleNamespace

        self.config = SimpleNamespace(
            model_path="/models/unit-model",
            served_model_name="unit-model",
            default_thinking_mode="auto",
            maintenance_state="serving",
        )

    def frontend_tokenizer(self):
        return FakeTokenizer()


@pytest.fixture()
def client():
    app = FastAPI()
    state = FakeState()
    register_openai_routes(app, get_state=lambda: state, get_model_sampling=lambda: {})
    return TestClient(app)


def test_tokenize_raw_text(client):
    resp = client.post("/v1/tokenize", json={"input": "hello world, this is a test"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["tokens"] > 0
    assert isinstance(body["token_ids"], list)
    assert len(body["token_ids"]) == body["tokens"]


def test_tokenize_requires_input_or_messages(client):
    resp = client.post("/v1/tokenize", json={})
    assert resp.status_code == 400
    assert "required" in resp.json()["error"]["message"]


def test_tokenize_response_model_defaults():
    req = TokenizeRequest(input="hi")
    assert req.input == "hi"
    assert req.messages is None
    assert req.add_special_tokens is True
    req2 = TokenizeRequest(messages=[{"role": "user", "content": "hi"}])
    assert req2.input is None