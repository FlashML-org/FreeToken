"""Unsupported image prompts must never be silently changed into text prompts."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from freetoken.server.anthropic_api import register_anthropic_routes
from freetoken.server.generation import GenerationError, _flatten_text_parts, render_messages
from freetoken.server.openai_api import register_openai_routes
from freetoken.server.responses_api import register_responses_routes


ERROR = "image content not supported by this text-only server"


class State:
    config = SimpleNamespace(reasoning_parser=None)

    def new_user(self):
        raise AssertionError("image requests must be rejected before admission")

    def frontend_tokenizer(self):
        raise AssertionError("image prompts must not reach token counting")


def client():
    app = FastAPI()
    state = State()
    register_openai_routes(app, lambda: state, lambda: {})
    register_anthropic_routes(app, lambda: state, lambda: {})
    register_responses_routes(app, lambda: state, lambda: {})
    return TestClient(app)


@pytest.mark.parametrize("kind", ["image", "image_url", "input_image"])
def test_shared_normalization_raises_exact_generation_error(kind):
    with pytest.raises(GenerationError) as exc:
        _flatten_text_parts([{"type": "text", "text": "describe"}, {"type": kind}])
    assert str(exc.value) == ERROR


@pytest.mark.parametrize("path", [
    "/v1/chat/completions", "/v1/messages", "/v1/messages/count_tokens", "/v1/responses",
])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_every_protocol_rejects_images_before_generation(path, stream, mixed):
    payload = {"model": "test", "max_tokens": 8, "stream": stream}
    parts = [{"type": "text", "text": "describe this"}] if mixed else []
    if path == "/v1/chat/completions":
        parts.append({"type": "image_url", "image_url": {"url": "https://example.invalid/image.png"}})
        payload["messages"] = [{"role": "user", "content": parts}]
    elif path.startswith("/v1/messages"):
        parts.append({"type": "image", "source": {"type": "url", "url": "https://example.invalid/image.png"}})
        payload["messages"] = [{"role": "user", "content": parts}]
    else:
        parts.append({"type": "input_image", "image_url": "https://example.invalid/image.png"})
        payload["input"] = [{"role": "user", "content": parts}]
    with client() as http:
        response = http.post(path, json=payload)
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["message"] == ERROR


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/messages/count_tokens"])
@pytest.mark.parametrize("location", ["system", "system_message", "tool_result"])
def test_nested_anthropic_images_cannot_bypass_the_shared_error(path, location):
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "unused"}}
    payload = {"model": "test", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    if location == "system":
        payload["system"] = [image]
    elif location == "system_message":
        payload["messages"].insert(0, {"role": "system", "content": [image]})
    else:
        payload["messages"][0]["content"] = [
            {"type": "tool_result", "tool_use_id": "call_1", "content": [
                {"type": "text", "text": "some text"}, image,
            ]}
        ]
    with client() as http:
        response = http.post(path, json=payload)
    assert response.status_code == 400
    assert response.json()["error"]["message"] == ERROR


def test_plain_text_normalization_is_unchanged():
    assert render_messages([{"role": "user", "content": [
        {"type": "text", "text": "hello"}, {"type": "text", "text": " world"},
    ]}]) == [{"role": "user", "content": "hello world"}]
    with pytest.raises(ValueError, match="Unsupported content part"):
        _flatten_text_parts([{"type": "audio_url"}])
