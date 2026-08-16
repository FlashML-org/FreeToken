"""Tests for the Anthropic /v1/messages adapter.

Two levels:
  * unit tests — feed the protocol-neutral primitive's GenResult / GenEvents straight
    into the Anthropic formatters (no engine, no model).
  * route smoke tests — drive the real FastAPI app + the shared generation primitive
    with a fake FrontendManager that scripts the engine ack stream (no GPU, no model).

Run:  PYTHONPATH=python <venv>/bin/python -m pytest tests/server/test_anthropic_api.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from types import SimpleNamespace

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PY = os.path.join(_ROOT, "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from freetoken.message.frontend import UserReply  # noqa: E402
from freetoken.server import anthropic_api as A  # noqa: E402
from freetoken.server.anthropic_models import AnthropicMessagesRequest  # noqa: E402
from freetoken.server.function_call_parser import ToolCallItem  # noqa: E402
from freetoken.server.generation import (  # noqa: E402
    ContentDelta,
    GenDone,
    GenResult,
    ReasoningDelta,
    ToolCallsDelta,
)


async def _aiter(items):
    for it in items:
        yield it


def _collect_events(events, model="claude-x", uid=1, cache_report=False):
    """Run the Anthropic event stream over neutral GenEvents; return [(type, data), ...]."""

    async def run():
        out = []
        async for frame in A.anthropic_event_stream(_aiter(events), model, uid, cache_report=cache_report):
            etype = None
            data = None
            for line in frame.split("\n"):
                if line.startswith("event:"):
                    etype = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    raw = line[len("data:"):].strip()
                    data = raw if raw == "[DONE]" else json.loads(raw)
            out.append((etype, data))
        return out

    return asyncio.run(run())


# --------------------------------------------------------------------------- #
# Request conversion
# --------------------------------------------------------------------------- #
def test_convert_request_system_text_tools():
    req = AnthropicMessagesRequest.model_validate(
        {
            "model": "claude-x",
            "max_tokens": 64,
            "system": "you are helpful",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "get weather",
                    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                }
            ],
            "tool_choice": {"type": "auto"},
            "temperature": 0.7,
        }
    )
    spec = A.convert_anthropic_to_genspec(req, {})
    assert spec.messages[0]["role"] == "system"
    assert spec.messages[0]["content"] == "you are helpful"
    assert spec.messages[1]["role"] == "user"
    assert spec.messages[1]["content"] == "hi"
    assert spec.sampling_params.max_tokens == 64
    assert spec.sampling_params.temperature == 0.7
    assert spec.parse_tools  # tools present + tool_choice auto
    assert spec.template_tools[0]["function"]["name"] == "get_weather"
    assert spec.template_tools[0]["function"]["parameters"]["type"] == "object"


def test_convert_request_tool_use_and_result_roundtrip():
    req = AnthropicMessagesRequest.model_validate(
        {
            "model": "claude-x",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "weather in SF?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "let me check"},
                        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "id": "toolu_1", "content": "72F"},
                    ],
                },
            ],
        }
    )
    spec = A.convert_anthropic_to_genspec(req, {})
    asst = spec.messages[1]
    assert asst["role"] == "assistant"
    assert asst["tool_calls"][0]["id"] == "toolu_1"
    assert asst["tool_calls"][0]["function"]["name"] == "get_weather"
    # render_messages decodes the JSON arguments string into a dict.
    assert asst["tool_calls"][0]["function"]["arguments"] == {"city": "SF"}
    tool_msg = spec.messages[2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "toolu_1"
    assert tool_msg["content"] == "72F"


def test_convert_system_role_message_and_unknown_block():
    # In-array system messages and unknown block types must not 422: accept system,
    # replay thinking as reasoning_content, skip redacted_thinking.
    req = AnthropicMessagesRequest.model_validate(
        {
            "model": "claude-x",
            "max_tokens": 64,
            "messages": [
                {"role": "system", "content": "be brief"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hmm"},
                        {"type": "redacted_thinking", "data": "opaque"},
                        {"type": "text", "text": "ok"},
                    ],
                },
                {"role": "user", "content": "hi"},
            ],
        }
    )
    spec = A.convert_anthropic_to_genspec(req, {})
    assert [m["role"] for m in spec.messages] == ["system", "assistant", "user"]
    assert spec.messages[0]["content"] == "be brief"
    assert spec.messages[1]["content"] == "ok"
    assert spec.messages[1]["reasoning_content"] == "hmm"  # replayed, not dropped
    assert spec.messages[2]["content"] == "hi"


def test_convert_thinking_toggle_broadcasts_every_spelling():
    """The toggle is broadcast in every spelling templates read (enable_thinking
    bool + M3's thinking_mode); each template picks the knob it knows and Jinja
    ignores the rest, so the kwargs are family-independent."""
    def _req(ttype):
        return AnthropicMessagesRequest.model_validate(
            {
                "model": "m", "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "thinking": {"type": ttype, "budget_tokens": 1024},
            }
        )

    on = {"enable_thinking": True, "thinking_mode": "enabled"}
    off = {"enable_thinking": False, "thinking_mode": "disabled"}
    for parser in ("minimax_m3", "qwen3", "minimax", None):
        spec = A.convert_anthropic_to_genspec(_req("enabled"), {}, reasoning_parser=parser)
        assert spec.chat_template_kwargs == on, parser
        spec = A.convert_anthropic_to_genspec(_req("disabled"), {}, reasoning_parser=parser)
        assert spec.chat_template_kwargs == off, parser


def test_convert_thinking_only_assistant_message_keeps_empty_content():
    # Truncated-then-resent turn: survives with content="" for templates that
    # concatenate message.content unconditionally.
    req = AnthropicMessagesRequest.model_validate(
        {
            "model": "claude-x",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "partial"}]},
                {"role": "user", "content": "continue"},
            ],
        }
    )
    spec = A.convert_anthropic_to_genspec(req, {})
    asst = spec.messages[1]
    assert asst["reasoning_content"] == "partial"
    assert asst["content"] == ""


def test_convert_thinking_replay_in_tool_loop():
    # thinking + tool_use replayed with the tool_result land on one assistant turn.
    req = AnthropicMessagesRequest.model_validate(
        {
            "model": "claude-x",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "need the tool", "signature": ""},
                        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}},
                    ],
                },
                {"role": "user", "content": [{"type": "tool_result", "id": "toolu_1", "content": "72F"}]},
            ],
        }
    )
    spec = A.convert_anthropic_to_genspec(req, {})
    asst = spec.messages[1]
    assert asst["role"] == "assistant"
    assert asst["reasoning_content"] == "need the tool"
    assert asst["thinking"] == "need the tool"  # gpt-oss templates read this alias
    assert asst["tool_calls"][0]["id"] == "toolu_1"
    assert spec.messages[2]["role"] == "tool"


def test_convert_hoists_and_merges_system_messages():
    # Claude Code interleaves system messages mid-array; strict chat templates
    # (e.g. Qwen3.5: "System message must be at the beginning") require ONE system
    # message at the front. Merge top-level system + in-array system, hoist to front.
    req = AnthropicMessagesRequest.model_validate(
        {
            "model": "claude-x",
            "max_tokens": 64,
            "system": "top-level sys",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "system", "content": "mid-stream sys"},
                {"role": "assistant", "content": "hi"},
            ],
        }
    )
    spec = A.convert_anthropic_to_genspec(req, {})
    assert [m["role"] for m in spec.messages] == ["system", "user", "assistant"]
    assert sum(1 for m in spec.messages if m["role"] == "system") == 1
    assert "top-level sys" in spec.messages[0]["content"]
    assert "mid-stream sys" in spec.messages[0]["content"]


# --------------------------------------------------------------------------- #
# Non-streaming response formatting (GenResult -> Anthropic response)
# --------------------------------------------------------------------------- #
def test_full_response_text_and_tool():
    result = GenResult(
        reasoning="",
        content="calling tool",
        tool_calls=[ToolCallItem(tool_index=0, name="get_weather", parameters='{"city": "SF"}')],
        finish_reason="tool_calls",
        prompt_tokens=11,
        completion_tokens=5,
    )
    resp = A.anthropic_full_response(result, "claude-x", uid=9)
    assert resp.type == "message"
    assert resp.role == "assistant"
    assert resp.model == "claude-x"
    assert resp.id == "msg_9"
    assert resp.stop_reason == "tool_use"
    assert resp.content[0].type == "text" and resp.content[0].text == "calling tool"
    assert resp.content[1].type == "tool_use"
    assert resp.content[1].name == "get_weather"
    assert resp.content[1].input == {"city": "SF"}
    assert resp.usage.input_tokens == 11 and resp.usage.output_tokens == 5


def test_full_response_length_truncation_is_max_tokens():
    result = GenResult(
        reasoning="", content="partial", tool_calls=[],
        finish_reason="length", prompt_tokens=11, completion_tokens=4096,
    )
    resp = A.anthropic_full_response(result, "claude-x", uid=9)
    assert resp.stop_reason == "max_tokens"


def test_full_response_cache_report_flips_input_tokens():
    result = GenResult(
        reasoning="", content="hi", tool_calls=[], finish_reason="stop",
        prompt_tokens=11, completion_tokens=5, cached_tokens=8,
    )
    # Flag on: Anthropic billing semantics — input_tokens excludes the cached prefix.
    resp = A.anthropic_full_response(result, "claude-x", uid=9, cache_report=True)
    assert resp.usage.input_tokens == 3
    assert resp.usage.cache_read_input_tokens == 8
    # Flag off: full prompt length, cache field absent from the wire.
    resp = A.anthropic_full_response(result, "claude-x", uid=9)
    assert resp.usage.input_tokens == 11
    assert "cache_read_input_tokens" not in resp.model_dump(exclude_none=True)["usage"]


def test_full_response_cache_report_zero_hit_keeps_field_absent():
    result = GenResult(
        reasoning="", content="hi", tool_calls=[], finish_reason="stop",
        prompt_tokens=11, completion_tokens=5,
    )
    resp = A.anthropic_full_response(result, "claude-x", uid=9, cache_report=True)
    assert resp.usage.input_tokens == 11
    assert "cache_read_input_tokens" not in resp.model_dump(exclude_none=True)["usage"]


# --------------------------------------------------------------------------- #
# Streaming event formatting (GenEvent -> Anthropic events)
# --------------------------------------------------------------------------- #
def test_stream_text():
    events = [ContentDelta("Hel"), ContentDelta("lo"), GenDone("stop", 4, 2)]
    collected = _collect_events(events)
    types = [e[0] for e in collected if e[0]]
    assert types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    # Anthropic streams must NOT emit an OpenAI-style `data: [DONE]` sentinel.
    assert all(data != "[DONE]" for _, data in collected)
    text = "".join(
        e[1]["delta"]["text"]
        for e in collected
        if e[0] == "content_block_delta" and e[1]["delta"].get("type") == "text_delta"
    )
    assert text == "Hello"
    md = next(e[1] for e in collected if e[0] == "message_delta")
    assert md["delta"]["stop_reason"] == "end_turn"
    assert md["usage"]["input_tokens"] == 4 and md["usage"]["output_tokens"] == 2


def test_stream_message_delta_usage_cache_report():
    events = [ContentDelta("hi"), GenDone("stop", 4, 2, cached_tokens=3)]
    collected = _collect_events(events, cache_report=True)
    md = next(e[1] for e in collected if e[0] == "message_delta")
    assert md["usage"]["input_tokens"] == 1
    assert md["usage"]["cache_read_input_tokens"] == 3

    # Flag off: unchanged wire shape (full prompt, no cache key).
    collected = _collect_events(events)
    md = next(e[1] for e in collected if e[0] == "message_delta")
    assert md["usage"]["input_tokens"] == 4
    assert "cache_read_input_tokens" not in md["usage"]


def test_stream_reasoning_as_thinking_block():
    events = [ReasoningDelta("thinking..."), ContentDelta("hi"), GenDone("stop", 1, 1)]
    collected = _collect_events(events)
    types = [e[0] for e in collected if e[0]]
    assert types == [
        "message_start",
        "content_block_start",   # thinking block
        "content_block_delta",   # thinking_delta
        "content_block_delta",   # signature_delta (empty, shape compliance)
        "content_block_stop",
        "content_block_start",   # text block
        "content_block_delta",   # text_delta
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    blocks = [e[1] for e in collected if e[0] == "content_block_start"]
    assert blocks[0]["content_block"]["type"] == "thinking"
    assert blocks[0]["index"] == 0
    assert blocks[1]["content_block"]["type"] == "text"
    assert blocks[1]["index"] == 1
    deltas = [e[1] for e in collected if e[0] == "content_block_delta"]
    assert deltas[0]["delta"] == {"type": "thinking_delta", "thinking": "thinking..."}
    assert deltas[1]["delta"]["type"] == "signature_delta"
    assert deltas[2]["delta"] == {"type": "text_delta", "text": "hi"}


def test_stream_tool_call():
    args = '{"city": "SF"}'
    events = [
        ToolCallsDelta([ToolCallItem(tool_index=0, name="get_weather", parameters=args)]),
        GenDone("tool_calls", 7, 3),
    ]
    collected = _collect_events(events)
    types = [e[0] for e in collected if e[0]]
    assert types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    start = next(e[1] for e in collected if e[0] == "content_block_start")
    assert start["content_block"]["type"] == "tool_use"
    assert start["content_block"]["name"] == "get_weather"
    delta = next(e[1] for e in collected if e[0] == "content_block_delta")
    assert delta["delta"]["type"] == "input_json_delta"
    assert delta["delta"]["partial_json"] == args
    md = next(e[1] for e in collected if e[0] == "message_delta")
    assert md["delta"]["stop_reason"] == "tool_use"


def test_stream_empty_turn_terminates():
    # GenDone always drives the terminal sequence — no usage-chunk dependency.
    collected = _collect_events([GenDone("stop", 3, 0)])
    types = [e[0] for e in collected if e[0]]
    assert types == ["message_start", "message_delta", "message_stop"]


# --------------------------------------------------------------------------- #
# Route smoke tests (real app + shared primitive + fake engine)
# --------------------------------------------------------------------------- #
class FakeState:
    def __init__(self, outputs):
        self._outputs = outputs
        self.maintenance_state = "serving"
        self.config = SimpleNamespace(
            reasoning_parser=None,
            tool_call_parser="llama3",
            served_model_name="test-model",
            model_path="/test",
        )
        self._uid = 0
        self._count_manager = None

    def frontend_tokenizer(self):
        return self._count_manager

    def new_user(self):
        self._uid += 1
        return self._uid

    async def send_one(self, msg):
        return None

    async def wait_for_ack(self, uid):
        for text, finished, pt, ct in self._outputs:
            yield UserReply(
                uid=uid,
                incremental_output=text,
                finished=finished,
                prompt_tokens_delta=pt,
                completion_tokens_delta=ct,
            )

    async def stream_with_cancellation(self, gen, request, uid):
        async for chunk in gen:
            yield chunk


def _client(fake):
    from fastapi.testclient import TestClient
    from freetoken.server import api_server

    api_server._GLOBAL_STATE = fake
    return TestClient(api_server.app)


def test_route_nonstream_text():
    fake = FakeState([("Hello world", True, 5, 2)])
    client = _client(fake)
    r = client.post(
        "/v1/messages",
        json={"model": "claude-x", "max_tokens": 32, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "claude-x"
    assert body["content"][0]["type"] == "text"
    assert body["content"][0]["text"] == "Hello world"
    assert body["stop_reason"] == "end_turn"
    assert body["usage"]["input_tokens"] == 5
    assert body["usage"]["output_tokens"] == 2


def test_route_stream_text():
    fake = FakeState([("Hello world", True, 5, 2)])
    client = _client(fake)
    r = client.post(
        "/v1/messages",
        json={
            "model": "claude-x",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200, r.text
    etypes = []
    text = ""
    for block in r.text.split("\n\n"):
        etype = None
        data = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                etype = line[len("event:"):].strip()
            elif line.startswith("data:"):
                raw = line[len("data:"):].strip()
                if raw and raw != "[DONE]":
                    data = json.loads(raw)
        if etype:
            etypes.append(etype)
        if etype == "content_block_delta" and data and data["delta"].get("type") == "text_delta":
            text += data["delta"]["text"]
    assert etypes[0] == "message_start"
    assert "content_block_start" in etypes
    assert etypes[-1] == "message_stop"
    assert text == "Hello world"


def test_stream_request_error_is_invalid_request_not_internal():
    # A request-side failure raised mid-stream (template rejection / over-length prompt) must be
    # classified as invalid_request_error — matching the non-streaming path — not internal_error,
    # so Claude Code treats it as a client error rather than a server fault to retry. (codex P3)
    from freetoken.server.generation import GenerationError

    async def boom():
        raise GenerationError("prompt is too long: 8181 tokens > 7223 maximum",
                              "context_length_exceeded")
        yield  # noqa: unreachable — makes this an async generator

    async def run():
        out = []
        async for frame in A.anthropic_event_stream(boom(), "claude-x", 1):
            etype = data = None
            for line in frame.split("\n"):
                if line.startswith("event:"):
                    etype = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data = json.loads(line[len("data:"):].strip())
            out.append((etype, data))
        return out

    events = asyncio.run(run())
    types = [e[0] for e in events]
    assert types[0] == "message_start"
    assert types[-1] == "error", types
    err = events[-1][1]["error"]
    assert err["type"] == "invalid_request_error", err
    assert "7223" in err["message"]
    # No error `code` on this wire, so the text is the whole signal — must survive verbatim.
    assert err["message"].startswith("prompt is too long: 8181 tokens > 7223")


def test_stream_tool_block_closes_before_following_text():
    args = '{"city": "SF"}'
    events = [
        ToolCallsDelta([ToolCallItem(tool_index=0, name="get_weather", parameters=args)]),
        ContentDelta("done"),
        GenDone("tool_calls", 7, 3),
    ]
    collected = _collect_events(events)
    types = [e[0] for e in collected if e[0]]
    assert types == [
        "message_start",
        "content_block_start",   # tool_use (index 0)
        "content_block_delta",   # input_json_delta (complete args)
        "content_block_stop",
        "content_block_start",   # text (index 1)
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    starts = [e[1] for e in collected if e[0] == "content_block_start"]
    assert starts[0]["content_block"]["type"] == "tool_use"
    assert starts[0]["index"] == 0
    assert starts[1]["content_block"]["type"] == "text"
    assert starts[1]["index"] == 1


def test_full_response_includes_thinking_block():
    result = GenResult(
        reasoning="Pondered.", content="Answer.", tool_calls=[],
        finish_reason="stop", prompt_tokens=5, completion_tokens=3,
    )
    response = A.anthropic_full_response(result, "claude-x", 1)
    assert [b.type for b in response.content] == ["thinking", "text"]
    assert response.content[0].thinking == "Pondered."
    assert response.content[1].text == "Answer."


def test_convert_native_thinking_toggle():
    base = {"model": "claude-x", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}
    on = A.convert_anthropic_to_genspec(
        AnthropicMessagesRequest.model_validate({**base, "thinking": {"type": "enabled", "budget_tokens": 1024}}), {}
    )
    assert on.chat_template_kwargs == {"enable_thinking": True, "thinking_mode": "enabled"}
    off = A.convert_anthropic_to_genspec(
        AnthropicMessagesRequest.model_validate({**base, "thinking": {"type": "disabled"}}), {}
    )
    assert off.chat_template_kwargs == {"enable_thinking": False, "thinking_mode": "disabled"}
    absent = A.convert_anthropic_to_genspec(AnthropicMessagesRequest.model_validate(base), {})
    assert absent.chat_template_kwargs == {}


# --------------------------------------------------------------------------- #
# /v1/messages/count_tokens (route + neutral count_prompt_tokens primitive)
# --------------------------------------------------------------------------- #
class _FakeIds:
    def __init__(self, n):
        self._n = n

    def numel(self):
        return self._n


class _FakeTokenizeManager:
    """Counts tokens as len(str(prompt)) so tests are deterministic without a model."""

    def __init__(self):
        self.msgs = []

    def tokenize(self, msgs):
        self.msgs.extend(msgs)
        return [_FakeIds(len(str(m.text))) for m in msgs]


def _count_client(manager=None, maintenance_state="serving"):
    fake = FakeState([])
    fake.maintenance_state = maintenance_state
    fake._count_manager = manager
    return _client(fake), fake


_COUNT_BODY = {
    "model": "claude-x",
    "system": "you are helpful",
    "messages": [{"role": "user", "content": "hi"}],
    "tools": [
        {
            "name": "get_weather",
            "description": "get weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ],
    "thinking": {"type": "enabled", "budget_tokens": 1024},
}


def test_count_tokens_route():
    manager = _FakeTokenizeManager()
    client, _ = _count_client(manager)
    r = client.post("/v1/messages/count_tokens", json=_COUNT_BODY)
    assert r.status_code == 200, r.text
    assert set(r.json().keys()) == {"input_tokens"}
    assert r.json()["input_tokens"] > 0

    # The counted TokenizeMsg must match what a generation would send the engine.
    [msg] = manager.msgs
    spec = A.convert_anthropic_to_genspec(
        AnthropicMessagesRequest.model_validate({**_COUNT_BODY, "max_tokens": 64}), {}
    )
    assert msg.text == spec.messages
    assert msg.tools == spec.template_tools
    assert msg.chat_template_kwargs == spec.chat_template_kwargs


def test_count_tokens_empty_messages_400():
    client, _ = _count_client(_FakeTokenizeManager())
    r = client.post("/v1/messages/count_tokens", json={"model": "claude-x", "messages": []})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_count_tokens_works_while_not_serving():
    # Counting never touches the engine, so it must not 503 during load/rebuild.
    client, _ = _count_client(_FakeTokenizeManager(), maintenance_state="rebuilding")
    r = client.post(
        "/v1/messages/count_tokens",
        json={"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200, r.text


class _RaisingManager:
    def __init__(self, exc):
        self._exc = exc

    def tokenize(self, msgs):
        raise self._exc


def test_count_tokens_tokenizer_init_failure_500():
    # A tokenizer initialization failure is a server fault. It must be 500 even when it surfaces
    # as ValueError (e.g. a checkpoint with no chat template) — not the convert/empty 400 branch.
    for exc in (RuntimeError("load failed"), ValueError("no chat template is set")):
        client, _ = _count_client(_RaisingManager(exc))
        r = client.post(
            "/v1/messages/count_tokens",
            json={"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 500, f"{type(exc).__name__}: {r.text}"
        assert r.json()["error"]["type"] == "api_error"


def test_count_tokens_template_render_error_400():
    # A chat template that rejects the specific conversation (jinja TemplateError: bad role
    # ordering, unmatched tool_result, raise_exception) is an input-driven client error — 400,
    # matching how /v1/messages surfaces the same failure. Not a 500.
    from jinja2 import TemplateError

    client, _ = _count_client(_RaisingManager(TemplateError("Unexpected role 'tool'")))
    r = client.post(
        "/v1/messages/count_tokens",
        json={"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 400, r.text
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_count_tokens_excluded_from_request_ring():
    # count_tokens never enters generation accounting; its latency (incl. first-touch tokenizer
    # load) must not land in the /v1/requests ring or the /v1/stats p95.
    from freetoken.server import request_ring

    request_ring.reset()
    client, _ = _count_client(_FakeTokenizeManager())
    r = client.post(
        "/v1/messages/count_tokens",
        json={"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200, r.text
    assert request_ring.requests_count() == 0

    # A real /v1/messages request IS tracked, proving the exclusion is scoped to the subpath.
    client2 = _client(FakeState([("hi", True, 3, 1)]))
    client2.post("/v1/messages", json={"model": "claude-x", "max_tokens": 8,
                                       "messages": [{"role": "user", "content": "hi"}]})
    assert request_ring.requests_count() == 1
    request_ring.reset()


def test_count_tokens_image_only_message_400():
    # Message list is non-empty on the wire but empty after block filtering: the neutral
    # count_prompt_tokens raises ValueError -> 400, not a 500 from an empty chat template.
    client, _ = _count_client(_FakeTokenizeManager())
    r = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "claude-x",
            "messages": [
                {"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "x"}}]}
            ],
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_tool_choice_none_hides_tools():
    body = {**_COUNT_BODY, "tool_choice": {"type": "none"}}
    _, template_tools, parser_tools, _ = A.convert_anthropic_prompt(
        A.AnthropicCountTokensRequest.model_validate(body)
    )
    assert template_tools is None
    assert parser_tools is None
    spec = A.convert_anthropic_to_genspec(
        AnthropicMessagesRequest.model_validate({**body, "max_tokens": 64}), {}
    )
    assert spec.template_tools is None
    assert not spec.parse_tools


def test_validation_error_uses_anthropic_envelope():
    client = _client(FakeState([]))
    # count_tokens: missing required `messages`
    r = client.post("/v1/messages/count_tokens", json={"model": "claude-x"})
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "messages" in body["error"]["message"]
    # /v1/messages inherits the same envelope
    r = client.post("/v1/messages", json={"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
    # non-Anthropic routes keep FastAPI's default 422 shape
    r = client.post("/v1/chat/completions", json={})
    assert r.status_code == 422
    assert "detail" in r.json()


def test_frontend_tokenizer_concurrent_first_build_dedupes():
    # FrontendManager.frontend_tokenizer() must build the tokenizer exactly once under a
    # concurrent first-touch burst (the lock's job). Patches the lazy imports the method does.
    import threading as _threading
    import time as _time

    import freetoken.tokenizer.tokenize as _tok
    import freetoken.utils as _utils
    from freetoken.server.api_server import FrontendManager

    calls = []

    class _SlowTok:
        pass

    def _slow_load(path):
        calls.append(path)
        _time.sleep(0.05)
        return _SlowTok()

    orig_load, orig_tm = _utils.load_tokenizer, _tok.TokenizeManager
    _utils.load_tokenizer = _slow_load
    _tok.TokenizeManager = lambda tok: tok
    try:
        fm = FrontendManager(
            config=SimpleNamespace(model_path="dedupe-test"),
            send_tokenizer=None,
            recv_tokenizer=None,
        )
        results = []
        threads = [
            _threading.Thread(target=lambda: results.append(fm.frontend_tokenizer()))
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        _utils.load_tokenizer = orig_load
        _tok.TokenizeManager = orig_tm
    assert len(calls) == 1
    assert len(set(id(r) for r in results)) == 1


def test_tool_result_carries_the_wire_tool_use_id():
    """Real Anthropic clients name the answered call with ``tool_use_id``; ``id`` is the
    tool_use side. Losing it leaves every tool result unattributed, which the DSV4 encoder
    needs both to emit the id and to re-sort parallel results into call order."""
    req = AnthropicMessagesRequest.model_validate(
        {
            "model": "claude-x",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "weather in SF and NY?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_a", "name": "w", "input": {"c": "SF"}},
                        {"type": "tool_use", "id": "toolu_b", "name": "w", "input": {"c": "NY"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [  # answered out of call order, as real clients may send them
                        {"type": "tool_result", "tool_use_id": "toolu_b", "content": "60F"},
                        {"type": "tool_result", "tool_use_id": "toolu_a", "content": "72F"},
                    ],
                },
            ],
        }
    )
    spec = A.convert_anthropic_to_genspec(req, {})
    results = [m for m in spec.messages if m["role"] == "tool"]
    assert [(m["tool_call_id"], m["content"]) for m in results] == [
        ("toolu_b", "60F"), ("toolu_a", "72F")]
