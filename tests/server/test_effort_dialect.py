"""Effort/thinking dialect handling at the OpenAI API layer.

Covers the wire-level half of the reasoning-effort pipeline: the superset
validation and DeepSeek ``thinking`` toggle in ``handle_chat_completion``, the
pre-stream render validation, and the probed vocabulary on ``/v1/models``.
The quantization itself is covered in tests/tokenizer/test_effort.py.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from freetoken.message import TokenizeMsg, UserReply
from freetoken.server.model_meta import effort_toggle_kwargs
from freetoken.server.openai_api import (
    ChatCompletionRequest,
    handle_chat_completion,
    register_openai_routes,
)
from freetoken.tokenizer.effort import EffortProfile


def run(coro):
    return asyncio.run(coro)


class FakeState:
    def __init__(self, reasoning_parser: str | None = None) -> None:
        self.config = SimpleNamespace(
            model_path="/models/unit-model",
            served_model_name="unit-model",
            tool_call_parser="llama3",
            reasoning_parser=reasoning_parser,
        )
        self.sent: TokenizeMsg | None = None

    def new_user(self) -> int:
        return 42

    async def send_one(self, msg):
        self.sent = msg

    async def wait_for_ack(self, uid: int):
        yield UserReply(uid=uid, incremental_output="ok", finished=True, finish_reason="stop")


class FakeManager:
    def __init__(self, profile: EffortProfile | None = None, render_error: Exception | None = None):
        self._profile = profile
        self._render_error = render_error

    def effort_profile(self) -> EffortProfile:
        assert self._profile is not None
        return self._profile

    def render_prompt(self, msg) -> str:
        if self._render_error is not None:
            raise self._render_error
        return "rendered"


def chat_request(**overrides) -> ChatCompletionRequest:
    payload = {
        "model": "unit-model",
        "messages": [{"role": "user", "content": "hi"}],
        **overrides,
    }
    return ChatCompletionRequest(**payload)


# --------------------------------------------------------------------------- #
# effort_toggle_kwargs: the DeepSeek thinking toggle folds into template kwargs.
# --------------------------------------------------------------------------- #
OFF = {"enable_thinking": False, "thinking_mode": "disabled"}
ON = {"enable_thinking": True, "thinking_mode": "enabled"}


def test_thinking_disabled_wins_over_an_effort():
    ctk = effort_toggle_kwargs("high", {}, thinking_type="disabled")
    assert ctk == OFF


def test_thinking_enabled_forwards_the_effort():
    ctk = effort_toggle_kwargs("high", {}, thinking_type="enabled")
    assert ctk == {**ON, "reasoning_effort": "high"}


def test_thinking_enabled_alone_turns_thinking_on():
    ctk = effort_toggle_kwargs(None, {}, thinking_type="enabled")
    assert ctk == ON


def test_explicit_template_kwargs_still_win_wholesale():
    ctk = effort_toggle_kwargs("high", {"enable_thinking": False}, thinking_type="enabled")
    assert ctk == {"enable_thinking": False}


# --------------------------------------------------------------------------- #
# handle_chat_completion: superset validation and the pre-stream render check.
# --------------------------------------------------------------------------- #
def test_unknown_reasoning_effort_is_a_400():
    response = run(
        handle_chat_completion(chat_request(reasoning_effort="banana"), None, FakeState(), {})
    )
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert "reasoning_effort" in json.loads(response.body)["error"]["message"]


def test_unknown_thinking_type_is_a_400():
    response = run(
        handle_chat_completion(
            chat_request(thinking={"type": "sideways"}), None, FakeState(), {}
        )
    )
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400


def test_thinking_disabled_reaches_the_tokenizer_as_enable_thinking_false():
    state = FakeState(reasoning_parser="qwen3")
    response = run(
        handle_chat_completion(
            chat_request(thinking={"type": "disabled"}), None, state, {}
        )
    )
    assert not isinstance(response, JSONResponse)  # plain successful completion
    assert state.sent is not None
    assert state.sent.chat_template_kwargs == OFF


def test_off_and_mixed_case_efforts_stay_accepted():
    # effort_toggle_kwargs has always normalized case/whitespace and honored
    # "off" as a disable synonym; the superset gate must not reject them.
    for effort, expected in (
        ("off", OFF),
        ("High", {**ON, "reasoning_effort": "high"}),
        (" high ", {**ON, "reasoning_effort": "high"}),
    ):
        state = FakeState(reasoning_parser="qwen3")
        response = run(
            handle_chat_completion(chat_request(reasoning_effort=effort), None, state, {})
        )
        assert not isinstance(response, JSONResponse), effort
        assert state.sent.chat_template_kwargs == expected, effort


def test_empty_effort_is_treated_as_absent():
    state = FakeState(reasoning_parser="qwen3")
    response = run(
        handle_chat_completion(chat_request(reasoning_effort=""), None, state, {})
    )
    assert not isinstance(response, JSONResponse)
    assert state.sent.chat_template_kwargs == {}


def test_foreign_thinking_shapes_stay_ignored():
    # extra="allow" swallowed any thinking shape before the field existed;
    # a bare string, a bool, or a typeless dict must keep working unchanged.
    for shape in ("enabled", True, {}, {"budget_tokens": 1024}):
        state = FakeState(reasoning_parser="qwen3")
        response = run(
            handle_chat_completion(chat_request(thinking=shape), None, state, {})
        )
        assert not isinstance(response, JSONResponse), shape
        assert state.sent.chat_template_kwargs == {}, shape


def test_anthropic_style_thinking_dict_works():
    state = FakeState(reasoning_parser="qwen3")
    response = run(
        handle_chat_completion(
            chat_request(thinking={"type": "enabled", "budget_tokens": 1024}), None, state, {}
        )
    )
    assert not isinstance(response, JSONResponse)
    assert state.sent.chat_template_kwargs == ON


def test_stream_returns_400_when_the_template_rejects_the_render():
    state = FakeState()
    state.frontend_tokenizer = lambda: FakeManager(
        render_error=ValueError("Unexpected reasoning effort high.")
    )
    response = run(handle_chat_completion(chat_request(stream=True), None, state, {}))
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    message = json.loads(response.body)["error"]["message"]
    assert message.startswith("could not encode request")
    assert state.sent is None  # rejected before submission


def test_stream_proceeds_without_a_frontend_tokenizer():
    # Minimal embeddings (and the unit FakeState) have no frontend tokenizer;
    # validation degrades to the old worker-side path instead of blocking.
    response = run(handle_chat_completion(chat_request(stream=True), None, FakeState(), {}))
    assert isinstance(response, StreamingResponse)


# --------------------------------------------------------------------------- #
# /v1/models: the probed vocabulary is published; absence stays None.
# --------------------------------------------------------------------------- #
def _models_payload(state) -> dict:
    app = FastAPI()
    register_openai_routes(app, lambda: state, dict)
    with TestClient(app) as client:
        response = client.get("/v1/models")
    assert response.status_code == 200
    return response.json()["data"][0]


def test_v1_models_publishes_the_probed_efforts():
    state = FakeState()
    state.frontend_tokenizer = lambda: FakeManager(
        profile=EffortProfile(
            supported=frozenset({"xhigh", "medium", "low"}),
            default="xhigh",
            consumes_effort=True,
        )
    )
    card = _models_payload(state)
    assert card["supported_reasoning_efforts"] == ["xhigh", "medium", "low"]
    assert card["default_reasoning_effort"] == "xhigh"


def test_v1_models_omits_efforts_without_a_frontend_tokenizer():
    card = _models_payload(FakeState())
    assert card["supported_reasoning_efforts"] is None
    assert card["default_reasoning_effort"] is None


def test_v1_models_omits_efforts_for_models_without_the_knob():
    state = FakeState()
    state.frontend_tokenizer = lambda: FakeManager(
        profile=EffortProfile(supported=frozenset(), default=None, consumes_effort=False)
    )
    card = _models_payload(state)
    assert card["supported_reasoning_efforts"] is None
    assert card["default_reasoning_effort"] is None
