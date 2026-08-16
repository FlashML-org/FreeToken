"""Tests for the OpenAI Responses /v1/responses adapter (codex backend).

Converter + stream-event unit tests feed the protocol-neutral primitive's
GenResult / GenEvents into the Responses formatters (no engine). Route smoke
tests drive the real app + shared primitive with a fake engine state.

Run:  PYTHONPATH=python <venv>/bin/python -m pytest tests/server/test_responses_api.py -v
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
from freetoken.server import responses_api as RP  # noqa: E402
from freetoken.server.responses_api import ResponsesRequest  # noqa: E402
from freetoken.server.function_call_parser import ToolCallItem  # noqa: E402
from freetoken.server.generation import ContentDelta, GenDone, GenResult, ToolCallsDelta  # noqa: E402


async def _aiter(items):
    for it in items:
        yield it


def _collect(events, req):
    async def run():
        out = []
        async for frame in RP.responses_stream_generator(_aiter(events), req, "resp_1", 0):
            etype = None
            data = None
            for line in frame.split("\n"):
                if line.startswith("event:"):
                    etype = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data = json.loads(line[len("data:"):].strip())
            out.append((etype, data))
        return out

    return asyncio.run(run())


# --------------------------------------------------------------------------- #
# Request conversion
# --------------------------------------------------------------------------- #
def test_convert_string_input_and_instructions():
    req = ResponsesRequest.model_validate(
        {"model": "gpt-x", "instructions": "be terse", "input": "hello", "max_output_tokens": 50}
    )
    spec = RP.convert_responses_to_genspec(req, {})
    assert spec.messages[0]["role"] == "system" and spec.messages[0]["content"] == "be terse"
    assert spec.messages[1]["role"] == "user" and spec.messages[1]["content"] == "hello"
    assert spec.sampling_params.max_tokens == 50


def test_convert_defaults_max_output_tokens_when_omitted():
    # codex omits max_output_tokens; must NOT fall to the 16-token floor (bug b1).
    req = ResponsesRequest.model_validate({"model": "gpt-x", "input": "hi"})
    assert RP.convert_responses_to_genspec(req, {}).sampling_params.max_tokens == RP.DEFAULT_MAX_OUTPUT_TOKENS
    assert RP.convert_responses_to_genspec(req, {}).sampling_params.max_tokens >= 1024
    req2 = ResponsesRequest.model_validate({"model": "gpt-x", "input": "hi", "max_output_tokens": 123})
    assert RP.convert_responses_to_genspec(req2, {}).sampling_params.max_tokens == 123
    assert RP.convert_responses_to_genspec(req, {}, default_max_tokens=2048).sampling_params.max_tokens == 2048


def test_convert_list_input_with_tool_roundtrip_and_tools():
    req = ResponsesRequest.model_validate(
        {
            "model": "gpt-x",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
                {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": '{"city": "SF"}'},
                {"type": "function_call_output", "call_id": "call_1", "output": "72F"},
            ],
            "tools": [
                {"type": "function", "name": "get_weather", "description": "d", "parameters": {"type": "object"}}
            ],
            "tool_choice": "auto",
        }
    )
    spec = RP.convert_responses_to_genspec(req, {})
    assert spec.messages[0]["role"] == "user" and spec.messages[0]["content"] == "weather?"
    asst = spec.messages[1]
    assert asst["role"] == "assistant" and asst["tool_calls"][0]["id"] == "call_1"
    assert asst["tool_calls"][0]["function"]["name"] == "get_weather"
    tool_msg = spec.messages[2]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "call_1" and tool_msg["content"] == "72F"
    assert spec.template_tools[0]["function"]["name"] == "get_weather"
    assert spec.parse_tools


def test_convert_reasoning_item_merges_into_assistant_turn():
    # One turn's reasoning/message/function_call items fold into ONE assistant message.
    req = ResponsesRequest.model_validate(
        {
            "model": "gpt-x",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "need the tool"}],
                },
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "checking"}]},
                {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "72F"},
            ],
        }
    )
    spec = RP.convert_responses_to_genspec(req, {})
    assert [m["role"] for m in spec.messages] == ["user", "assistant", "tool"]
    asst = spec.messages[1]
    assert asst["reasoning_content"] == "need the tool"
    # No `thinking` alias here: gpt-oss's template raises on a tool-call turn carrying both
    # content and thinking, and the visible text is what the user saw.
    assert "thinking" not in asst
    assert asst["content"] == "checking"
    assert asst["tool_calls"][0]["id"] == "call_1"
    assert spec.messages[2]["role"] == "tool" and spec.messages[2]["content"] == "72F"


def test_convert_parallel_function_calls_merge_into_one_turn():
    req = ResponsesRequest.model_validate(
        {
            "model": "gpt-x",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "both?"}]},
                {"type": "function_call", "call_id": "call_1", "name": "a", "arguments": "{}"},
                {"type": "function_call", "call_id": "call_2", "name": "b", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "1"},
                {"type": "function_call_output", "call_id": "call_2", "output": "2"},
            ],
        }
    )
    spec = RP.convert_responses_to_genspec(req, {})
    asst = spec.messages[1]
    assert [tc["id"] for tc in asst["tool_calls"]] == ["call_1", "call_2"]
    # tool outputs stay separate messages, after the merged assistant turn
    assert [m["role"] for m in spec.messages] == ["user", "assistant", "tool", "tool"]


def test_convert_distinct_assistant_messages_not_merged():
    # Only complementary items of one turn coalesce; filled slots keep boundaries.
    req = ResponsesRequest.model_validate(
        {
            "model": "gpt-x",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "A"}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "B"}]},
                {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "r1"}]},
                {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "r2"}]},
            ],
        }
    )
    spec = RP.convert_responses_to_genspec(req, {})
    assert [m.get("content") for m in spec.messages[:3]] == ["hi", "A", "B"]
    assert [m["reasoning_content"] for m in spec.messages[3:]] == ["r1", "r2"]


def test_convert_reasoning_only_assistant_turn_gets_empty_content():
    # Truncated-then-resent turn: survives with content="" for templates that
    # concatenate message.content unconditionally.
    req = ResponsesRequest.model_validate(
        {
            "model": "gpt-x",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "partial"}]},
            ],
        }
    )
    spec = RP.convert_responses_to_genspec(req, {})
    asst = spec.messages[1]
    assert asst["reasoning_content"] == "partial"
    assert asst["content"] == ""


def test_convert_encrypted_only_reasoning_item_is_dropped():
    # OpenAI-style items carry only encrypted_content/summary — nothing recoverable.
    req = ResponsesRequest.model_validate(
        {
            "model": "gpt-x",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "s"}], "encrypted_content": "opaque"},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]},
            ],
        }
    )
    spec = RP.convert_responses_to_genspec(req, {})
    asst = spec.messages[1]
    assert asst["content"] == "hello"
    assert "reasoning_content" not in asst


def test_convert_developer_role_maps_to_system():
    # codex sends a developer-role message (Responses instructions); fold to system
    # so the chat template (which only knows system/user/assistant/tool) accepts it.
    req = ResponsesRequest.model_validate(
        {
            "model": "gpt-x",
            "input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "be a calculator"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "2+2"}]},
            ],
        }
    )
    spec = RP.convert_responses_to_genspec(req, {})
    assert spec.messages[0]["role"] == "system"
    assert spec.messages[0]["content"] == "be a calculator"
    assert spec.messages[1]["role"] == "user"


# --------------------------------------------------------------------------- #
# Non-streaming response assembly (GenResult -> Response)
# --------------------------------------------------------------------------- #
def test_build_response_text_and_tool():
    result = GenResult(
        reasoning="",
        content="let me check",
        tool_calls=[ToolCallItem(tool_index=0, name="get_weather", parameters='{"city": "SF"}')],
        finish_reason="tool_calls",
        prompt_tokens=8,
        completion_tokens=4,
    )
    req = ResponsesRequest.model_validate({"model": "gpt-x", "input": "weather?"})
    resp = RP.build_responses_response(result, req, "resp_42", 123)
    body = resp.model_dump(mode="json")
    assert body["object"] == "response"
    assert body["id"] == "resp_42"
    assert body["status"] == "completed"
    assert body["output"][0]["type"] == "message"
    assert body["output"][0]["content"][0]["type"] == "output_text"
    assert body["output"][0]["content"][0]["text"] == "let me check"
    assert body["output"][1]["type"] == "function_call"
    assert body["output"][1]["call_id"].startswith("call_")
    assert body["output"][1]["name"] == "get_weather"
    assert body["output"][1]["arguments"] == '{"city": "SF"}'
    assert body["usage"]["input_tokens"] == 8 and body["usage"]["output_tokens"] == 4


# --------------------------------------------------------------------------- #
# Streaming event taxonomy (GenEvent -> Responses events)
# --------------------------------------------------------------------------- #
def test_stream_text_events():
    req = ResponsesRequest.model_validate({"model": "gpt-x", "input": "hi", "stream": True})
    events = [ContentDelta("Hel"), ContentDelta("lo"), GenDone("stop", 4, 2)]
    collected = _collect(events, req)
    types = [e[0] for e in collected]
    assert types == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    text = "".join(e[1]["delta"] for e in collected if e[0] == "response.output_text.delta")
    assert text == "Hello"
    seqs = [e[1]["sequence_number"] for e in collected]
    assert seqs == list(range(len(seqs)))
    completed = collected[-1][1]["response"]
    assert completed["status"] == "completed"
    assert completed["output"][0]["content"][0]["text"] == "Hello"
    assert completed["usage"]["input_tokens"] == 4


def test_stream_tool_call_events():
    req = ResponsesRequest.model_validate({"model": "gpt-x", "input": "weather?", "stream": True})
    args = '{"city": "SF"}'
    events = [
        ToolCallsDelta([ToolCallItem(tool_index=0, name="get_weather", parameters=args)]),
        GenDone("tool_calls", 7, 3),
    ]
    collected = _collect(events, req)
    types = [e[0] for e in collected]
    assert types == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    added = next(e[1] for e in collected if e[0] == "response.output_item.added")
    assert added["item"]["type"] == "function_call"
    assert added["item"]["name"] == "get_weather"
    done = next(e[1] for e in collected if e[0] == "response.function_call_arguments.done")
    assert done["arguments"] == args
    completed = collected[-1][1]["response"]
    assert completed["output"][0]["type"] == "function_call"
    assert completed["output"][0]["arguments"] == args


# --------------------------------------------------------------------------- #
# Route smoke tests
# --------------------------------------------------------------------------- #
class FakeState:
    def __init__(self, outputs, finish_reason=None, cached_tokens=0):
        self._outputs = outputs
        self._finish_reason = finish_reason  # stamped on the terminal ack
        self._cached_tokens = cached_tokens  # stamped on the first ack (admission reply)
        self.maintenance_state = "serving"
        self.config = SimpleNamespace(
            reasoning_parser=None, tool_call_parser="llama3",
            served_model_name="test-model", model_path="/test",
        )
        self._uid = 0
        self.last_sent = None

    def new_user(self):
        self._uid += 1
        return self._uid

    async def send_one(self, msg):
        self.last_sent = msg  # capture the TokenizeMsg so tests can assert sampling_params
        return None

    async def wait_for_ack(self, uid):
        for i, (text, finished, pt, ct) in enumerate(self._outputs):
            yield UserReply(uid=uid, incremental_output=text, finished=finished,
                            finish_reason=self._finish_reason if finished else None,
                            prompt_tokens_delta=pt, completion_tokens_delta=ct,
                            cached_tokens=self._cached_tokens if i == 0 else 0)

    async def stream_with_cancellation(self, gen, request, uid):
        async for chunk in gen:
            yield chunk


def _client(fake):
    from fastapi.testclient import TestClient
    from freetoken.server import api_server

    api_server._GLOBAL_STATE = fake
    return TestClient(api_server.app)


def test_route_nonstream_text():
    client = _client(FakeState([("Hello world", True, 5, 2)]))
    r = client.post("/v1/responses", json={"model": "gpt-x", "input": "hi"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["output"][0]["content"][0]["text"] == "Hello world"
    assert body["usage"]["input_tokens"] == 5 and body["usage"]["output_tokens"] == 2


def test_route_stream_text():
    client = _client(FakeState([("Hello world", True, 5, 2)]))
    r = client.post("/v1/responses", json={"model": "gpt-x", "input": "hi", "stream": True})
    assert r.status_code == 200, r.text
    types = []
    text = ""
    for block in r.text.split("\n\n"):
        etype = None
        data = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                etype = line[len("event:"):].strip()
            elif line.startswith("data:"):
                raw = line[len("data:"):].strip()
                if raw:
                    data = json.loads(raw)
        if etype:
            types.append(etype)
        if etype == "response.output_text.delta":
            text += data["delta"]
    assert types[0] == "response.created"
    assert types[-1] == "response.completed"
    assert text == "Hello world"


def test_route_nonstream_usage_cached_tokens_with_cache_report():
    fake = FakeState([("Hello world", True, 5, 2)], cached_tokens=3)
    fake.config.enable_cache_report = True
    client = _client(fake)
    r = client.post("/v1/responses", json={"model": "gpt-x", "input": "hi"})
    assert r.status_code == 200, r.text
    usage = r.json()["usage"]
    # input_tokens stays inclusive; the details carry the cached split.
    assert usage["input_tokens"] == 5
    assert usage["input_tokens_details"]["cached_tokens"] == 3


def test_route_nonstream_usage_cached_tokens_zero_without_flag():
    client = _client(FakeState([("Hello world", True, 5, 2)], cached_tokens=3))
    r = client.post("/v1/responses", json={"model": "gpt-x", "input": "hi"})
    assert r.status_code == 200, r.text
    assert r.json()["usage"]["input_tokens_details"]["cached_tokens"] == 0


def test_route_stream_completed_usage_carries_cached_tokens():
    fake = FakeState([("Hello world", True, 5, 2)], cached_tokens=3)
    fake.config.enable_cache_report = True
    client = _client(fake)
    r = client.post("/v1/responses", json={"model": "gpt-x", "input": "hi", "stream": True})
    assert r.status_code == 200, r.text
    completed = None
    for block in r.text.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data:"):
                data = json.loads(line[len("data:"):].strip())
                if data.get("type") == "response.completed":
                    completed = data
    assert completed is not None
    assert completed["response"]["usage"]["input_tokens_details"]["cached_tokens"] == 3


def test_route_stream_gemma4_tools_do_not_force_reasoning():
    fake = FakeState([("visible answer", True, 5, 2)])
    fake.config.reasoning_parser = "gemma4"
    fake.config.tool_call_parser = "gemma4"
    client = _client(fake)

    r = client.post(
        "/v1/responses",
        json={
            "model": "gpt-x",
            "input": "hi",
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "name": "shell",
                    "description": "run a command",
                    "parameters": {"type": "object"},
                }
            ],
        },
    )
    assert r.status_code == 200, r.text
    text = ""
    completed = None
    for block in r.text.split("\n\n"):
        etype = None
        data = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                etype = line[len("event:"):].strip()
            elif line.startswith("data:"):
                raw = line[len("data:"):].strip()
                if raw:
                    data = json.loads(raw)
        if etype == "response.output_text.delta":
            text += data["delta"]
        elif etype == "response.completed":
            completed = data["response"]

    assert text == "visible answer"
    assert completed["output"][0]["content"][0]["text"] == "visible answer"


def test_route_stream_gemma4_parses_codex_namespaced_tool_call():
    output = "<|tool_call>call:superpowers:using_superpowers{}<tool_call|>"
    fake = FakeState([(output, True, 5, 4)])
    fake.config.tool_call_parser = "gemma4"
    client = _client(fake)

    r = client.post(
        "/v1/responses",
        json={
            "model": "gpt-x",
            "input": "use a skill",
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "name": "superpowers:using_superpowers",
                    "description": "load the superpowers skill",
                    "parameters": {"type": "object"},
                }
            ],
        },
    )
    assert r.status_code == 200, r.text

    types = []
    text = ""
    added = None
    done = None
    completed = None
    for block in r.text.split("\n\n"):
        etype = None
        data = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                etype = line[len("event:"):].strip()
            elif line.startswith("data:"):
                raw = line[len("data:"):].strip()
                if raw:
                    data = json.loads(raw)
        if etype:
            types.append(etype)
        if etype == "response.output_text.delta":
            text += data["delta"]
        elif etype == "response.output_item.added":
            added = data
        elif etype == "response.function_call_arguments.done":
            done = data
        elif etype == "response.completed":
            completed = data["response"]

    assert "response.output_text.delta" not in types
    assert text == ""
    assert added["item"]["type"] == "function_call"
    assert added["item"]["name"] == "superpowers:using_superpowers"
    assert done["name"] == "superpowers:using_superpowers"
    assert done["arguments"] == "{}"
    assert completed["output"][0]["type"] == "function_call"
    assert completed["output"][0]["name"] == "superpowers:using_superpowers"
    assert completed["output"][0]["arguments"] == "{}"


def test_route_stream_gemma4_forwards_namespaced_skill_call_without_declared_tool():
    output = "<|tool_call>call:superpowers:using_superpowers{}<tool_call|>"
    fake = FakeState([(output, True, 5, 4)])
    fake.config.tool_call_parser = "gemma4"
    client = _client(fake)

    r = client.post(
        "/v1/responses",
        json={
            "model": "gpt-x",
            "input": "use a skill",
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "run a command",
                    "parameters": {"type": "object"},
                }
            ],
        },
    )
    assert r.status_code == 200, r.text

    types = []
    text = ""
    added = None
    for block in r.text.split("\n\n"):
        etype = None
        data = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                etype = line[len("event:"):].strip()
            elif line.startswith("data:"):
                raw = line[len("data:"):].strip()
                if raw:
                    data = json.loads(raw)
        if etype:
            types.append(etype)
        if etype == "response.output_text.delta":
            text += data["delta"]
        elif etype == "response.output_item.added":
            added = data

    assert "response.output_text.delta" not in types
    assert text == ""
    assert added["item"]["type"] == "function_call"
    assert added["item"]["name"] == "superpowers:using_superpowers"


def test_route_stateful_stubs():
    client = _client(FakeState([("x", True, 1, 1)]))
    assert client.get("/v1/responses/resp_abc").status_code == 404
    assert client.post("/v1/responses/resp_abc/cancel").status_code == 404
    r = client.post("/v1/responses", json={"model": "gpt-x", "input": "hi", "previous_response_id": "resp_1"})
    assert r.status_code == 400


def test_convert_merges_instructions_and_developer_into_one_system():
    # codex sends a top-level `instructions` AND a leading `developer` message. Both must
    # collapse into exactly ONE leading system message — Qwen3.5's template rejects a second
    # system ("System message must be at the beginning") and the old 2-system output crashed
    # the tokenizer worker, hanging codex for 300s.
    req = ResponsesRequest.model_validate(
        {
            "model": "gpt-x",
            "instructions": "agent system prompt",
            "input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "permissions"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "env ctx"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "the task"}]},
            ],
        }
    )
    spec = RP.convert_responses_to_genspec(req, {})
    roles = [m["role"] for m in spec.messages]
    assert roles == ["system", "user", "user"], roles  # exactly one system, at the front
    assert spec.messages[0]["content"] == "agent system prompt\n\npermissions"


def test_stream_surfaces_generation_error_as_failed():
    # A request that fails mid-generation (template/over-length) must terminate the stream with
    # response.failed, not stall — otherwise codex hits its idle timeout and "Reconnecting".
    from freetoken.server.generation import GenerationError

    async def boom():
        raise GenerationError("chat template rejected the conversation")
        yield  # noqa: unreachable — makes this an async generator

    req = ResponsesRequest.model_validate({"model": "gpt-x", "input": "hi", "stream": True})

    async def run():
        out = []
        async for frame in RP.responses_stream_generator(boom(), req, "resp_1", 0):
            for line in frame.split("\n"):
                if line.startswith("event:"):
                    out.append(line[len("event:"):].strip())
                elif line.startswith("data:"):
                    out[-1] = (out[-1], json.loads(line[len("data:"):].strip()))
        return out

    events = asyncio.run(run())
    types = [e[0] if isinstance(e, tuple) else e for e in events]
    assert types[0] == "response.created"
    assert types[-1] == "response.failed"
    failed = events[-1][1]
    assert failed["response"]["status"] == "failed"
    assert "chat template rejected" in failed["response"]["error"]["message"]
    # No specific class on this failure, so the generic one stands.
    assert failed["response"]["error"]["code"] == "server_error"


def test_stream_failure_keeps_the_error_code_codex_matches_on():
    """codex reads only `error.code` here to tell a blown context window from a generic failure."""
    from freetoken.server.generation import GenerationError

    async def boom():
        raise GenerationError("prompt is too long: 8178 tokens > 7223 maximum",
                              "context_length_exceeded")
        yield  # noqa: unreachable — makes this an async generator

    req = ResponsesRequest.model_validate({"model": "gpt-x", "input": "hi", "stream": True})

    async def run():
        frames = []
        async for frame in RP.responses_stream_generator(boom(), req, "resp_1", 0):
            frames.append(frame)
        return frames

    failed = [f for f in asyncio.run(run()) if "response.failed" in f][-1]
    payload = json.loads(failed.split("data:", 1)[1].strip())
    assert payload["response"]["error"]["code"] == "context_length_exceeded"


class _ErrState(FakeState):
    """Engine that reports a per-request error (tokenizer/scheduler rejection) via UserReply."""

    async def wait_for_ack(self, uid):
        yield UserReply(uid=uid, incremental_output="", finished=True, error="input too long for KV budget")


def test_route_nonstream_error_returns_400():
    client = _client(_ErrState([]))
    r = client.post("/v1/responses", json={"model": "gpt-x", "input": "hi"})
    assert r.status_code == 400, r.text
    assert "too long" in r.json()["error"]["message"]


def test_route_stream_error_emits_failed_event():
    client = _client(_ErrState([]))
    r = client.post("/v1/responses", json={"model": "gpt-x", "input": "hi", "stream": True})
    assert r.status_code == 200, r.text  # stream already started; failure is in-band
    etypes = [
        line[len("event:"):].strip()
        for block in r.text.split("\n\n")
        for line in block.split("\n")
        if line.startswith("event:")
    ]
    assert etypes[-1] == "response.failed", etypes


# --------------------------------------------------------------------------- #
# Truncation: finish_reason "length" -> incomplete
# --------------------------------------------------------------------------- #
def test_default_max_output_tokens_is_large():
    assert RP.DEFAULT_MAX_OUTPUT_TOKENS >= 32768


def test_build_response_length_truncation_is_incomplete():
    result = GenResult(
        reasoning="", content="partial ans", tool_calls=[],
        finish_reason="length", prompt_tokens=5, completion_tokens=8192,
    )
    req = ResponsesRequest.model_validate({"model": "gpt-x", "input": "hi"})
    body = RP.build_responses_response(result, req, "resp_1", 0).model_dump(mode="json")
    assert body["status"] == "incomplete"
    assert body["incomplete_details"]["reason"] == "max_output_tokens"
    assert body["output"][0]["status"] == "incomplete"
    # a clean stop must stay completed with no incomplete_details
    result.finish_reason = "stop"
    body2 = RP.build_responses_response(result, req, "resp_1", 0).model_dump(mode="json")
    assert body2["status"] == "completed"
    assert body2.get("incomplete_details") is None
    assert body2["output"][0]["status"] == "completed"


def test_max_output_tokens_must_be_positive():
    client = _client(FakeState([("hi", True, 1, 1)]))
    assert client.post("/v1/responses", json={"model": "gpt-x", "input": "hi", "max_output_tokens": 0}).status_code == 400
    assert client.post("/v1/responses", json={"model": "gpt-x", "input": "hi", "max_output_tokens": -5}).status_code == 400
    assert client.post("/v1/responses", json={"model": "gpt-x", "input": "hi", "max_output_tokens": 100}).status_code == 200


def test_stream_unexpected_error_emits_failed():
    async def boom():
        raise ValueError("kaboom")
        yield  # noqa: unreachable — makes this an async generator

    req = ResponsesRequest.model_validate({"model": "gpt-x", "input": "hi", "stream": True})

    async def run():
        out = []
        async for frame in RP.responses_stream_generator(boom(), req, "resp_1", 0):
            for line in frame.split("\n"):
                if line.startswith("event:"):
                    out.append(line[len("event:"):].strip())
        return out

    types = asyncio.run(run())
    assert types[-1] == "response.failed"


def test_stream_length_truncation_emits_incomplete_event():
    req = ResponsesRequest.model_validate({"model": "gpt-x", "input": "hi", "stream": True})
    events = [ContentDelta("partial"), GenDone("length", 4, 8192)]
    collected = _collect(events, req)
    assert collected[-1][0] == "response.incomplete"  # NOT response.completed
    final = collected[-1][1]["response"]
    assert final["status"] == "incomplete"
    assert final["incomplete_details"]["reason"] == "max_output_tokens"


def test_route_length_truncation_reports_incomplete():
    client = _client(FakeState([("partial", True, 5, 8192)], finish_reason="length"))
    r = client.post("/v1/responses", json={"model": "gpt-x", "input": "hi"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "incomplete"
    assert body["incomplete_details"]["reason"] == "max_output_tokens"

    client = _client(FakeState([("partial", True, 5, 8192)], finish_reason="length"))
    r = client.post("/v1/responses", json={"model": "gpt-x", "input": "hi", "stream": True})
    etypes = [
        line[len("event:"):].strip()
        for block in r.text.split("\n\n")
        for line in block.split("\n")
        if line.startswith("event:")
    ]
    assert etypes[-1] == "response.incomplete", etypes


def test_default_output_tokens_honors_server_config():
    fake = FakeState([("hi", True, 1, 1)])
    fake.config.max_output_tokens = 5000
    client = _client(fake)
    client.post("/v1/responses", json={"model": "gpt-x", "input": "hi"})
    assert fake.last_sent.sampling_params.max_tokens == 5000
    client.post("/v1/responses", json={"model": "gpt-x", "input": "hi", "max_output_tokens": 77})
    assert fake.last_sent.sampling_params.max_tokens == 77
    fake.config.max_output_tokens = None
    client.post("/v1/responses", json={"model": "gpt-x", "input": "hi"})
    assert fake.last_sent.sampling_params.max_tokens == RP.DEFAULT_MAX_OUTPUT_TOKENS


# --------------------------------------------------------------------------- #
# Reasoning streaming (codex renders response.reasoning_text.delta natively)
# --------------------------------------------------------------------------- #
def test_stream_reasoning_events():
    from freetoken.server.generation import ReasoningDelta

    req = ResponsesRequest.model_validate({"model": "gpt-x", "input": "hi", "stream": True})
    events = [
        ReasoningDelta("Think"),
        ReasoningDelta("ing."),
        ContentDelta("Answer."),
        GenDone("stop", 4, 2),
    ]
    collected = _collect(events, req)
    types = [e[0] for e in collected]
    assert types == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",       # reasoning item
        "response.reasoning_text.delta",
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
        "response.output_item.done",        # reasoning item closed
        "response.output_item.added",       # message item
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    added = next(e[1] for e in collected if e[0] == "response.output_item.added")
    assert added["item"]["type"] == "reasoning"
    deltas = "".join(e[1]["delta"] for e in collected if e[0] == "response.reasoning_text.delta")
    assert deltas == "Thinking."
    rs_done = next(e[1] for e in collected if e[0] == "response.output_item.done")
    assert rs_done["item"]["type"] == "reasoning"
    assert rs_done["item"]["content"][0]["text"] == "Thinking."
    # completed carries both items, reasoning first, and message output_index is 1
    completed = collected[-1][1]["response"]
    assert [item["type"] for item in completed["output"]] == ["reasoning", "message"]
    seqs = [e[1]["sequence_number"] for e in collected]
    assert seqs == list(range(len(seqs)))


def test_stream_tool_call_item_closes_before_following_text():
    # Each completed call closes its item immediately, so codex persists it even if
    # the stream dies before response.completed.
    req = ResponsesRequest.model_validate({"model": "gpt-x", "input": "hi", "stream": True})
    args = '{"city": "SF"}'
    events = [
        ToolCallsDelta([ToolCallItem(tool_index=0, name="get_weather", parameters=args)]),
        ContentDelta("done"),
        GenDone("tool_calls", 7, 3),
    ]
    collected = _collect(events, req)
    types = [e[0] for e in collected]
    fc_done = types.index("response.output_item.done")
    msg_added = types.index("response.output_item.added", types.index("response.function_call_arguments.done"))
    assert fc_done < msg_added, types
    completed = collected[-1][1]["response"]
    assert [item["type"] for item in completed["output"]] == ["function_call", "message"]


def test_build_response_includes_reasoning_item():
    result = GenResult(
        reasoning="Thought about it.", content="Answer.", tool_calls=[],
        finish_reason="stop", prompt_tokens=5, completion_tokens=3,
    )
    req = ResponsesRequest.model_validate({"model": "gpt-x", "input": "hi"})
    response = RP.build_responses_response(result, req, "resp_1", 0)
    types = [item.type for item in response.output]
    assert types == ["reasoning", "message"]
    assert response.output[0].content[0].text == "Thought about it."


def test_convert_reasoning_field_enables_thinking():
    # codex sends reasoning={"effort": ...}; protocol-native thinking toggle maps
    # to template kwargs so thinking-off-by-default templates (gemma4) turn it on.
    req = ResponsesRequest.model_validate(
        {"model": "m", "input": "hi", "reasoning": {"effort": "high"}}
    )
    spec = RP.convert_responses_to_genspec(req, {})
    assert spec.chat_template_kwargs == {
        "enable_thinking": True, "thinking_mode": "enabled", "reasoning_effort": "high"
    }

    # an explicit thinking-related chat_template_kwargs key wins over the mapping
    req2 = ResponsesRequest.model_validate(
        {"model": "m", "input": "hi", "reasoning": {"effort": "low"},
         "chat_template_kwargs": {"thinking_mode": "chat"}}
    )
    spec2 = RP.convert_responses_to_genspec(req2, {})
    assert spec2.chat_template_kwargs == {"thinking_mode": "chat"}

    # unrelated extra kwargs ride along without discarding the reasoning mapping
    req3 = ResponsesRequest.model_validate(
        {"model": "m", "input": "hi", "reasoning": {"effort": "none"},
         "chat_template_kwargs": {"custom_var": 1}}
    )
    assert RP.convert_responses_to_genspec(req3, {}).chat_template_kwargs == {
        "enable_thinking": False, "thinking_mode": "disabled", "custom_var": 1,
    }

    # absent reasoning -> no kwargs
    req4 = ResponsesRequest.model_validate({"model": "m", "input": "hi"})
    assert RP.convert_responses_to_genspec(req4, {}).chat_template_kwargs == {}


def test_convert_reasoning_effort_none_disables_thinking():
    # vLLM-compatible semantics: an explicit effort "none" DISABLES thinking.
    req = ResponsesRequest.model_validate(
        {"model": "m", "input": "hi", "reasoning": {"effort": "none"}}
    )
    assert RP.convert_responses_to_genspec(req, {}).chat_template_kwargs == {
        "enable_thinking": False, "thinking_mode": "disabled"
    }


def test_convert_reasoning_toggle_broadcasts_every_spelling():
    """The toggle is broadcast in every spelling templates read; a template
    picks the knob it knows (M3: thinking_mode) and ignores the rest, so the
    kwargs are family-independent."""
    req = ResponsesRequest.model_validate(
        {"model": "m", "input": "hi", "reasoning": {"effort": "high"}}
    )
    on = {"enable_thinking": True, "thinking_mode": "enabled", "reasoning_effort": "high"}
    off = ResponsesRequest.model_validate(
        {"model": "m", "input": "hi", "reasoning": {"effort": "none"}}
    )
    for parser in ("minimax_m3", "gpt_oss", None):
        spec = RP.convert_responses_to_genspec(req, {}, reasoning_parser=parser)
        assert spec.chat_template_kwargs == on, parser
        spec = RP.convert_responses_to_genspec(off, {}, reasoning_parser=parser)
        assert spec.chat_template_kwargs == {
            "enable_thinking": False, "thinking_mode": "disabled"
        }, parser
