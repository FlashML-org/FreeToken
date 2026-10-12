from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from freetoken.message import TokenizeMsg, UserReply
from freetoken.server.openai_api import (
    ChatCompletionRequest,
    CompletionRequest,
    chat_request_to_genspec,
    handle_chat_completion,
    handle_completion,
    register_openai_routes,
    stream_chat_completion_chunks,
)


def run(coro):
    return asyncio.run(coro)


class FakeState:
    def __init__(
        self,
        replies: list[UserReply],
        tool_call_parser: str = "llama3",
        reasoning_parser: str | None = None,
    ) -> None:
        self.config = SimpleNamespace(
            mm=SimpleNamespace(text_model_only=False, disabled_encoders=frozenset()),
            model_path="/models/unit-model",
            served_model_name="unit-model",
            tool_call_parser=tool_call_parser,
            reasoning_parser=reasoning_parser,
        )
        self.replies = replies
        self.sent: TokenizeMsg | None = None

    def new_user(self) -> int:
        return 42

    async def send_one(self, msg):
        self.sent = msg

    async def wait_for_ack(self, uid: int):
        assert uid == 42
        for reply in self.replies:
            yield reply


def tool_schema():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Return weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]


def opencode_tool_schema():
    return [
        {
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": {"filePath": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "glob",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                    },
                },
            },
        },
    ]


def chat_request(**kwargs) -> ChatCompletionRequest:
    payload = {
        "model": "client-model",
        "messages": [{"role": "user", "content": "weather in Paris?"}],
        "tools": tool_schema(),
        "max_tokens": 8,
    }
    payload.update(kwargs)
    return ChatCompletionRequest(**payload)


def parse_sse(chunks: list[bytes]) -> list[dict | str]:
    events: list[dict | str] = []
    for chunk in chunks:
        for line in chunk.decode().splitlines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            events.append(data if data == "[DONE]" else json.loads(data))
    return events


def test_chat_request_accepts_tool_messages_and_assistant_tool_calls():
    req = ChatCompletionRequest(
        model="client-model",
        messages=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ],
        tools=tool_schema(),
        max_completion_tokens=11,
        stream_options={"include_usage": True},
    )

    assert req.max_tokens == 11
    assert req.stream_options is not None
    assert req.stream_options.include_usage is True
    assert req.messages[0].tool_calls[0].function.arguments == '{"city":"Paris"}'


def test_chat_request_reasoning_replay_field_aliases():
    # Any replay field name in -> both template-read field names out.
    for field in ("reasoning_content", "reasoning", "thinking"):
        req = ChatCompletionRequest(
            model="client-model",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok", field: "prior thought"},
                {"role": "user", "content": "next"},
            ],
        )
        asst = chat_request_to_genspec(req, {}).messages[1]
        assert asst["reasoning_content"] == "prior thought", field
        assert asst["thinking"] == "prior thought", field


def test_chat_reasoning_effort_enables_thinking():
    spec = chat_request_to_genspec(chat_request(reasoning_effort="high"), {})
    assert spec.chat_template_kwargs == {
        "enable_thinking": True, "thinking_mode": "enabled", "reasoning_effort": "high"
    }

    # an explicit thinking-related chat_template_kwargs key wins over the mapping
    spec = chat_request_to_genspec(
        chat_request(reasoning_effort="none", chat_template_kwargs={"enable_thinking": True}), {}
    )
    assert spec.chat_template_kwargs == {"enable_thinking": True}

    # unrelated extra kwargs ride along without discarding the effort mapping
    spec = chat_request_to_genspec(
        chat_request(reasoning_effort="none", chat_template_kwargs={"custom_var": 1}), {}
    )
    assert spec.chat_template_kwargs == {
        "enable_thinking": False, "thinking_mode": "disabled", "custom_var": 1
    }

    # absent effort -> kwargs pass through untouched
    assert chat_request_to_genspec(chat_request(), {}).chat_template_kwargs == {}


def test_chat_reasoning_effort_none_disables_thinking():
    # vLLM-compatible semantics: an explicit effort "none" DISABLES thinking.
    spec = chat_request_to_genspec(chat_request(reasoning_effort="none"), {})
    assert spec.chat_template_kwargs == {"enable_thinking": False, "thinking_mode": "disabled"}


def test_chat_reasoning_effort_broadcasts_every_toggle_spelling():
    """The toggle is broadcast in every spelling templates read (enable_thinking
    bool + M3's thinking_mode); each template picks the knob it knows and Jinja
    ignores the rest, so no per-family routing exists."""
    on = chat_request(reasoning_effort="high")
    spec = chat_request_to_genspec(on, {})
    assert spec.chat_template_kwargs == {
        "enable_thinking": True, "thinking_mode": "enabled", "reasoning_effort": "high"
    }

    off = chat_request(reasoning_effort="none")
    spec = chat_request_to_genspec(off, {})
    assert spec.chat_template_kwargs == {"enable_thinking": False, "thinking_mode": "disabled"}



def test_glm_reasoning_parser_honors_disabled_thinking_with_tools():
    # The parse side must match the encode side: thinking off + tools present
    # must not start the parser inside a think block.
    from freetoken.server.generation import _make_reasoning_parser

    state = FakeState([], reasoning_parser="glm")
    off = chat_request_to_genspec(chat_request(reasoning_effort="none"), {})
    parser = _make_reasoning_parser(off, state)
    assert parser is not None and parser.detector.force_reasoning is False

    on = chat_request_to_genspec(chat_request(), {})
    parser = _make_reasoning_parser(on, state)
    assert parser is not None and parser.detector.force_reasoning is True


def test_mimo_reasoning_parser_never_forces_implicit_thinking():
    from freetoken.server.generation import _make_reasoning_parser

    state = FakeState([], tool_call_parser="mimo_v2", reasoning_parser="mimo_v2")
    for kwargs in (
        {},
        {"reasoning_effort": "high"},
        {"reasoning_effort": "none"},
        {"chat_template_kwargs": {"enable_thinking": True}},
    ):
        spec = chat_request_to_genspec(chat_request(**kwargs), {})
        parser = _make_reasoning_parser(spec, state)
        assert parser is not None and parser.detector.force_reasoning is False

    no_tools = chat_request_to_genspec(chat_request(tools=None, reasoning_effort="high"), {})
    parser = _make_reasoning_parser(no_tools, state)
    assert parser is not None and parser.detector.force_reasoning is False


def test_mimo_tool_response_roundtrips_through_openai_history_and_tool_result():
    output = (
        "<think>Check the forecast for the requested city.</think>"
        "Looking it up. "
        "<tool_call><function=get_weather>"
        "<parameter=city>Paris</parameter></function></tool_call>"
    )
    state = FakeState(
        [UserReply(uid=42, incremental_output=output, finished=True)],
        tool_call_parser="mimo_v2",
        reasoning_parser="mimo_v2",
    )
    first = run(handle_chat_completion(chat_request(), request=None, state=state, model_sampling={}))
    assistant = first["choices"][0]["message"]
    assert first["choices"][0]["finish_reason"] == "tool_calls"
    assert assistant["reasoning_content"] == "Check the forecast for the requested city."
    assert assistant["content"] == "Looking it up. "
    call = assistant["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}

    follow_up = ChatCompletionRequest(
        model="client-model",
        messages=[
            {"role": "user", "content": "weather in Paris?"},
            assistant,
            {"role": "tool", "tool_call_id": call["id"], "content": "sunny"},
        ],
        tools=tool_schema(),
        max_tokens=8,
    )
    replay = chat_request_to_genspec(follow_up, {}).messages
    assert replay[-2]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
    assert replay[-1] == {
        "role": "tool",
        "tool_call_id": call["id"],
        "content": "sunny",
    }


def test_mimo_openai_stream_cutoffs_keep_argument_deltas_valid_json():
    schema = [{
        "type": "function",
        "function": {
            "name": "demo",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                    "enabled": {"type": "boolean"},
                    "items": {"type": "array"},
                    "text": {"type": "string"},
                },
            },
        },
    }]
    cases = (
        (
            "<tool_call><function=demo><parameter=limit>3</parameter>",
            {"limit": 3},
        ),
        (
            '<tool_call><function=demo><parameter=text>quote " slash\\ snow 雪',
            {"text": 'quote " slash\\ snow 雪'},
        ),
        (
            "<tool_call><function=demo><parameter=limit>42",
            {"limit": 42},
        ),
        (
            "<tool_call><function=demo><parameter=enabled>true",
            {"enabled": True},
        ),
        (
            "<tool_call><function=demo><parameter=items>[1,2]",
            {"items": [1, 2]},
        ),
        (
            "<tool_call><function=demo><parameter=enabled>tru",
            {"enabled": "tru"},
        ),
        (
            "<tool_call><function=demo><parameter=items>[1,",
            {"items": "[1,"},
        ),
    )
    for tool_parser in ("qwen3_coder", "mimo_v2"):
        for output, expected in cases:
            pieces = [output[i : i + 5] for i in range(0, len(output), 5)]
            state = FakeState(
                [
                    UserReply(
                        uid=42,
                        incremental_output=piece,
                        finished=(i == len(pieces) - 1),
                        finish_reason="length" if i == len(pieces) - 1 else None,
                    )
                    for i, piece in enumerate(pieces)
                ],
                tool_call_parser=tool_parser,
                reasoning_parser="mimo_v2" if tool_parser == "mimo_v2" else None,
            )
            req = ChatCompletionRequest(
                model="client-model",
                messages=[{"role": "user", "content": "run the tool"}],
                tools=schema,
                stream=True,
                max_tokens=8,
            )
            events = parse_sse(run(_collect(stream_chat_completion_chunks(42, req, state))))
            deltas = [
                call["function"].get("arguments", "")
                for event in events
                if isinstance(event, dict)
                for choice in event.get("choices", [])
                for call in choice.get("delta", {}).get("tool_calls", [])
            ]
            assert json.loads("".join(deltas)) == expected
            assert events[-2]["choices"][0]["finish_reason"] == "length"


def test_mimo_openai_stream_recovers_missing_parameter_close_before_next_call():
    schema = [
        {
            "type": "function",
            "function": {
                "name": "first",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "typed",
                "parameters": {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "second",
                "parameters": {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "outer",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            },
        },
    ]
    output = (
        "<tool_call><function=first><parameter=text>first value"
        "</function></tool_call>"
        "<tool_call><function=typed><parameter=count>5</function></tool_call>"
        "<tool_call><function=outer><parameter=text>outer only</tool_call>"
        "<tool_call><function=second><parameter=count>2</parameter>"
        "</function></tool_call>"
    )
    pieces = [output[i : i + 3] for i in range(0, len(output), 3)]
    state = FakeState(
        [
            UserReply(
                uid=42,
                incremental_output=piece,
                finished=(i == len(pieces) - 1),
            )
            for i, piece in enumerate(pieces)
        ],
        tool_call_parser="mimo_v2",
        reasoning_parser="mimo_v2",
    )
    req = ChatCompletionRequest(
        model="client-model",
        messages=[{"role": "user", "content": "run both tools"}],
        tools=schema,
        stream=True,
        max_tokens=16,
    )
    events = parse_sse(run(_collect(stream_chat_completion_chunks(42, req, state))))
    calls = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        for choice in event.get("choices", []):
            for delta in choice.get("delta", {}).get("tool_calls", []):
                current = calls.setdefault(delta["index"], {"name": None, "arguments": ""})
                current["name"] = delta.get("function", {}).get("name") or current["name"]
                current["arguments"] += delta.get("function", {}).get("arguments", "")

    assert [calls[index]["name"] for index in sorted(calls)] == [
        "first", "typed", "outer", "second"
    ]
    assert json.loads(calls[0]["arguments"]) == {"text": "first value"}
    assert json.loads(calls[1]["arguments"]) == {"count": 5}
    assert json.loads(calls[2]["arguments"]) == {"text": "outer only"}
    assert json.loads(calls[3]["arguments"]) == {"count": 2}
    assert events[-2]["choices"][0]["finish_reason"] == "tool_calls"


def test_rejected_typed_argument_keeps_following_call_valid_in_shared_stream_parsers():
    schema = [
        {
            "type": "function",
            "function": {
                "name": "demo",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "count": {"type": "integer"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "next",
                "parameters": {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                },
            },
        },
    ]
    cases = (
        (
            "qwen3_coder",
            None,
            "<tool_call><function=demo><parameter=enabled>perhaps</parameter>"
            "<parameter=count>2</parameter></function>"
            "<function=next><parameter=count>3</parameter></function></tool_call>",
        ),
        (
            "mimo_v2",
            "mimo_v2",
            "<tool_call><function=demo><parameter=enabled>perhaps</parameter>"
            "<parameter=count>2</parameter></function>"
            "<function=next><parameter=count>3</parameter></function></tool_call>",
        ),
        (
            "minimax",
            None,
            '<minimax:tool_call><invoke name="demo">'
            '<parameter name="enabled">perhaps</parameter>'
            '<parameter name="count">2</parameter></invoke>'
            '<invoke name="next"><parameter name="count">3</parameter></invoke>'
            '</minimax:tool_call>',
        ),
        (
            "muse_glimmer",
            "muse_glimmer",
            '<|start|>assistant to=demo.demo<|message|><atem:function_calls>'
            '<atem:invoke name="demo"><atem:parameter name="enabled">perhaps</atem:parameter>'
            '<atem:parameter name="count">2</atem:parameter></atem:invoke>'
            '<atem:invoke name="next"><atem:parameter name="count">3</atem:parameter>'
            '</atem:invoke>'
            '</atem:function_calls><|eot|>',
        ),
    )
    for tool_parser, reasoning_parser, output in cases:
        pieces = [output[i : i + 3] for i in range(0, len(output), 3)]
        state = FakeState(
            [
                UserReply(
                    uid=42,
                    incremental_output=piece,
                    finished=(i == len(pieces) - 1),
                )
                for i, piece in enumerate(pieces)
            ],
            tool_call_parser=tool_parser,
            reasoning_parser=reasoning_parser,
        )
        req = ChatCompletionRequest(
            model="client-model",
            messages=[{"role": "user", "content": "run demo"}],
            tools=schema,
            stream=True,
            max_tokens=8,
        )
        events = parse_sse(run(_collect(stream_chat_completion_chunks(42, req, state))))
        calls = {}
        for event in events:
            if not isinstance(event, dict):
                continue
            for choice in event.get("choices", []):
                for delta in choice.get("delta", {}).get("tool_calls", []):
                    current = calls.setdefault(delta["index"], {"name": None, "arguments": ""})
                    current["name"] = delta.get("function", {}).get("name") or current["name"]
                    current["arguments"] += delta.get("function", {}).get("arguments", "")
        assert [calls[index]["name"] for index in sorted(calls)] == ["demo", "next"], tool_parser
        # DR-03 preserve-raw policy: the rejected boolean streams through as its
        # raw value, exactly like the non-streaming path.
        assert json.loads(calls[0]["arguments"]) == {"enabled": "perhaps", "count": 2}, tool_parser
        assert json.loads(calls[1]["arguments"]) == {"count": 3}, tool_parser


def _dr03_typed_tools():
    return [{
        "type": "function",
        "function": {
            "name": "demo",
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "count": {"type": "integer"},
                },
            },
        },
    }]


_DR03_REJECTED_WIRE = (
    "<tool_call><function=demo><parameter=enabled>perhaps</parameter>"
    "<parameter=count>2</parameter></function></tool_call>"
)


def test_dr03_openai_adapter_non_stream_preserves_raw_rejected_value():
    state = FakeState(
        [UserReply(uid=42, incremental_output=_DR03_REJECTED_WIRE, finished=True)],
        tool_call_parser="qwen3_coder",
    )
    request = ChatCompletionRequest(
        model="client-model",
        messages=[{"role": "user", "content": "run demo"}],
        tools=_dr03_typed_tools(),
        max_tokens=8,
    )
    response = run(handle_chat_completion(request, request=None, state=state, model_sampling={}))
    tool_calls = response["choices"][0]["message"]["tool_calls"]
    assert [
        (call["function"]["name"], json.loads(call["function"]["arguments"]))
        for call in tool_calls
    ] == [("demo", {"enabled": "perhaps", "count": 2})]


def test_dr03_openai_adapter_stream_preserves_raw_rejected_value():
    pieces = [
        _DR03_REJECTED_WIRE[i : i + 3] for i in range(0, len(_DR03_REJECTED_WIRE), 3)
    ]
    state = FakeState(
        [
            UserReply(
                uid=42,
                incremental_output=piece,
                finished=(i == len(pieces) - 1),
            )
            for i, piece in enumerate(pieces)
        ],
        tool_call_parser="qwen3_coder",
    )
    request = ChatCompletionRequest(
        model="client-model",
        messages=[{"role": "user", "content": "run demo"}],
        tools=_dr03_typed_tools(),
        stream=True,
        max_tokens=8,
    )
    events = parse_sse(run(_collect(stream_chat_completion_chunks(42, request, state))))
    calls = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        for choice in event.get("choices", []):
            for delta in choice.get("delta", {}).get("tool_calls", []) or []:
                current = calls.setdefault(delta["index"], {"name": None, "arguments": ""})
                current["name"] = delta.get("function", {}).get("name") or current["name"]
                current["arguments"] += delta.get("function", {}).get("arguments", "")
    assert [
        (calls[index]["name"], json.loads(calls[index]["arguments"]))
        for index in sorted(calls)
    ] == [("demo", {"enabled": "perhaps", "count": 2})]


def test_dr03_openai_adapter_delivery_modes_agree():
    state = FakeState(
        [UserReply(uid=42, incremental_output=_DR03_REJECTED_WIRE, finished=True)],
        tool_call_parser="qwen3_coder",
    )
    nonstream_request = ChatCompletionRequest(
        model="client-model",
        messages=[{"role": "user", "content": "run demo"}],
        tools=_dr03_typed_tools(),
        max_tokens=8,
    )
    response = run(
        handle_chat_completion(nonstream_request, request=None, state=state, model_sampling={})
    )
    nonstream_calls = [
        (call["function"]["name"], json.loads(call["function"]["arguments"]))
        for call in response["choices"][0]["message"]["tool_calls"]
    ]
    pieces = [
        _DR03_REJECTED_WIRE[i : i + 3] for i in range(0, len(_DR03_REJECTED_WIRE), 3)
    ]
    stream_state = FakeState(
        [
            UserReply(
                uid=42,
                incremental_output=piece,
                finished=(i == len(pieces) - 1),
            )
            for i, piece in enumerate(pieces)
        ],
        tool_call_parser="qwen3_coder",
    )
    stream_request = ChatCompletionRequest(
        model="client-model",
        messages=[{"role": "user", "content": "run demo"}],
        tools=_dr03_typed_tools(),
        stream=True,
        max_tokens=8,
    )
    events = parse_sse(
        run(_collect(stream_chat_completion_chunks(42, stream_request, stream_state)))
    )
    stream_calls = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        for choice in event.get("choices", []):
            for delta in choice.get("delta", {}).get("tool_calls", []) or []:
                current = stream_calls.setdefault(
                    delta["index"], {"name": None, "arguments": ""}
                )
                current["name"] = delta.get("function", {}).get("name") or current["name"]
                current["arguments"] += delta.get("function", {}).get("arguments", "")
    assert [
        (stream_calls[index]["name"], json.loads(stream_calls[index]["arguments"]))
        for index in sorted(stream_calls)
    ] == nonstream_calls


def test_mimo_nonstream_and_stream_preserve_text_between_tool_calls():
    schema = [{
        "type": "function",
        "function": {
            "name": "demo",
            "parameters": {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
            },
        },
    }]
    request = ChatCompletionRequest(
        model="client-model",
        messages=[{"role": "user", "content": "run twice"}],
        tools=schema,
        max_tokens=16,
    )

    def check_roundtrip(output, expected):
        nonstream_state = FakeState(
            [UserReply(uid=42, incremental_output=output, finished=True)],
            tool_call_parser="mimo_v2",
            reasoning_parser="mimo_v2",
        )
        nonstream = run(
            handle_chat_completion(request, request=None, state=nonstream_state, model_sampling={})
        )
        nonstream_content = nonstream["choices"][0]["message"]["content"]
        assert nonstream_content == expected

        pieces = [output[i : i + 4] for i in range(0, len(output), 4)]
        stream_state = FakeState(
            [
                UserReply(
                    uid=42,
                    incremental_output=piece,
                    finished=(i == len(pieces) - 1),
                )
                for i, piece in enumerate(pieces)
            ],
            tool_call_parser="mimo_v2",
            reasoning_parser="mimo_v2",
        )
        stream_request = ChatCompletionRequest(
            model="client-model",
            messages=[{"role": "user", "content": "run twice"}],
            tools=schema,
            stream=True,
            max_tokens=16,
        )
        events = parse_sse(
            run(_collect(stream_chat_completion_chunks(42, stream_request, stream_state)))
        )
        streamed_content = "".join(
            choice.get("delta", {}).get("content", "")
            for event in events
            if isinstance(event, dict)
            for choice in event.get("choices", [])
        )
        assert streamed_content == nonstream_content

    first = "<tool_call><function=demo><parameter=count>1</parameter></function></tool_call>"
    second = "<tool_call><function=demo><parameter=count>2</parameter></function></tool_call>"
    for middle, expected in (("middle", "BeforemiddleAfter"), ("After", "BeforeAfterAfter")):
        check_roundtrip("Before" + first + middle + second + "After", expected)


def test_mimo_history_renders_verbatim_arguments_in_reference_template():
    import os
    from pathlib import Path

    from jinja2 import Environment

    expected = {
        "leading": "\nline\n\n",
        "empty": "",
        "special": 'quote " slash\\ snow 雪',
    }
    output = (
        "<tool_call><function=echo>"
        "<parameter=leading>\nline\n\n</parameter>"
        "<parameter=empty></parameter>"
        '<parameter=special>quote " slash\\ snow 雪</parameter>'
        "</function></tool_call>"
    )
    schema = [{
        "type": "function",
        "function": {
            "name": "echo",
            "parameters": {
                "type": "object",
                "properties": {key: {"type": "string"} for key in expected},
            },
        },
    }]
    state = FakeState(
        [UserReply(uid=42, incremental_output=output, finished=True)],
        tool_call_parser="mimo_v2",
        reasoning_parser="mimo_v2",
    )
    request = ChatCompletionRequest(
        model="client-model",
        messages=[{"role": "user", "content": "echo these values"}],
        tools=schema,
        max_tokens=8,
    )
    response = run(handle_chat_completion(request, request=None, state=state, model_sampling={}))
    assistant = response["choices"][0]["message"]
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == expected

    follow_up = ChatCompletionRequest(
        model="client-model",
        messages=[
            {"role": "user", "content": "echo these values"},
            assistant,
            {
                "role": "tool",
                "tool_call_id": assistant["tool_calls"][0]["id"],
                "content": "received",
            },
        ],
        tools=schema,
        max_tokens=8,
    )
    spec = chat_request_to_genspec(follow_up, {})
    root = Path(__file__).resolve().parents[2]
    template_path = Path(
        os.environ.get(
            "FREETOKEN_MIMO_CHAT_TEMPLATE",
            root / "validation" / "real_ckpt_ref" / "chat_template.jinja",
        )
    )
    if not template_path.is_file():
        pytest.skip("MiMo validation chat template is not available")
    template = template_path.read_text()
    environment = Environment()
    environment.filters["tojson"] = lambda value, ensure_ascii=True: json.dumps(
        value, ensure_ascii=ensure_ascii
    )
    rendered = environment.from_string(template).render(
        messages=spec.messages,
        tools=spec.template_tools,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    expected_envelope = (
        "<tool_call><function=echo>"
        "<parameter=leading>\nline\n\n</parameter>"
        "<parameter=empty></parameter>"
        '<parameter=special>quote " slash\\ snow 雪</parameter>'
        "</function></tool_call>"
    )
    assert expected_envelope in rendered


# --------------------------------------------------------------------------- #
# DR-02: think markers inside verbatim tool arguments (MiMo)                   #
# --------------------------------------------------------------------------- #
from freetoken.server.reasoning_parser import (  # noqa: E402
    THINK_END_TOKEN as _DR02_CLOSE,
    THINK_START_TOKEN as _DR02_OPEN,
)

_DR02_ARG = f"a{_DR02_CLOSE}b{_DR02_OPEN}c"
_DR02_BLOCK = (
    "<tool_call><function=get_weather><parameter=city>" + _DR02_ARG + "</parameter></function>ing"
)


def test_mimo_dr02_tool_arg_markers_non_stream_reasoning_channel_clean():
    output = f"{_DR02_OPEN}Checking the weather{_DR02_CLOSE}" + _DR02_BLOCK
    state = FakeState(
        [UserReply(uid=42, incremental_output=output, finished=True)],
        tool_call_parser="mimo_v2",
        reasoning_parser="mimo_v2",
    )

    response = run(handle_chat_completion(chat_request(), request=None, state=state, model_sampling={}))

    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["reasoning_content"] == "Checking the weather"
    assert choice["message"]["content"] == ""
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"city": _DR02_ARG}


def test_mimo_dr02_zero_reasoning_tool_arg_markers_non_stream():
    state = FakeState(
        [UserReply(uid=42, incremental_output=_DR02_BLOCK, finished=True)],
        tool_call_parser="mimo_v2",
        reasoning_parser="mimo_v2",
    )

    response = run(handle_chat_completion(chat_request(), request=None, state=state, model_sampling={}))

    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"].get("reasoning_content") is None
    assert choice["message"]["content"] == ""
    tool_call = choice["message"]["tool_calls"][0]
    assert json.loads(tool_call["function"]["arguments"]) == {"city": _DR02_ARG}


def test_mimo_dr02_zero_reasoning_tool_arg_markers_stream():
    pieces = [_DR02_BLOCK[i : i + 5] for i in range(0, len(_DR02_BLOCK), 5)]
    state = FakeState(
        [
            UserReply(uid=42, incremental_output=piece, finished=(i == len(pieces) - 1))
            for i, piece in enumerate(pieces)
        ],
        tool_call_parser="mimo_v2",
        reasoning_parser="mimo_v2",
    )
    events = parse_sse(run(_collect(stream_chat_completion_chunks(42, chat_request(stream=True), state))))

    reasoning = "".join(
        choice["delta"]["reasoning_content"]
        for event in events
        if isinstance(event, dict)
        for choice in event.get("choices", [])
        if "reasoning_content" in choice.get("delta", {})
    )
    assert reasoning == ""
    content = "".join(
        choice["delta"]["content"]
        for event in events
        if isinstance(event, dict)
        for choice in event.get("choices", [])
        if choice["delta"].get("content")
    )
    assert content == ""
    args = "".join(
        tc["function"].get("arguments", "")
        for event in events
        if isinstance(event, dict)
        for choice in event.get("choices", [])
        for tc in choice.get("delta", {}).get("tool_calls", [])
    )
    assert json.loads(args) == {"city": _DR02_ARG}


def test_non_stream_chat_completion_returns_openai_tool_calls_and_sends_tools():
    output = '[TOOL_CALLS] [{"name":"get_weather","arguments":{"city":"Paris"}}]'
    state = FakeState(
        [
            UserReply(uid=42, incremental_output=output, finished=True, prompt_tokens_delta=5, completion_tokens_delta=7),
        ],
        tool_call_parser="mistral",
    )

    response = run(handle_chat_completion(chat_request(), request=None, state=state, model_sampling={}))

    assert state.sent is not None
    assert state.sent.text == [{"role": "user", "content": "weather in Paris?"}]
    assert state.sent.tools == tool_schema()
    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == ""
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"city": "Paris"}
    assert response["usage"] == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}


def test_non_stream_chat_completion_length_truncation_overrides_tool_calls():
    output = '[TOOL_CALLS] [{"name":"get_weather","arguments":{"city":"Paris"}}]'
    state = FakeState(
        [UserReply(uid=42, incremental_output=output, finished=True, finish_reason="length")],
        tool_call_parser="mistral",
    )
    response = run(handle_chat_completion(chat_request(), request=None, state=state, model_sampling={}))
    assert response["choices"][0]["finish_reason"] == "length"


def test_non_stream_chat_completion_parses_configured_family_tool_shape():
    output = (
        "<|channel|>analysis<|message|>Need files.<|end|><|start|>assistant"
        "<|channel|>commentary to=functions.glob <|constrain|>json<|message|>"
        '{"pattern":"**/*.py","path":"/tmp/ws"}'
    )
    state = FakeState([UserReply(uid=42, incremental_output=output, finished=True)], tool_call_parser="gpt_oss")
    req = ChatCompletionRequest(
        model="client-model",
        messages=[{"role": "user", "content": "inspect"}],
        tools=opencode_tool_schema(),
        max_tokens=8,
    )

    response = run(handle_chat_completion(req, request=None, state=state, model_sampling={}))

    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "glob"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "pattern": "**/*.py",
        "path": "/tmp/ws",
    }


def test_stream_chat_completion_emits_chat_chunks_tool_delta_and_done():
    output = '[TOOL_CALLS] [{"name":"get_weather","arguments":{"city":"Paris"}}]'
    state = FakeState(
        [
            UserReply(uid=42, incremental_output=output, finished=True, prompt_tokens_delta=5, completion_tokens_delta=7),
        ],
        tool_call_parser="mistral",
    )

    chunks = run(_collect(stream_chat_completion_chunks(42, chat_request(stream=True), state)))
    events = parse_sse(chunks)

    assert events[-1] == "[DONE]"
    assert events[0]["object"] == "chat.completion.chunk"
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    tool_deltas = [
        tool_call
        for event in events
        if isinstance(event, dict)
        for choice in event.get("choices", [])
        for tool_call in choice.get("delta", {}).get("tool_calls", [])
    ]
    assert tool_deltas[0]["function"]["name"] == "get_weather"
    assert json.loads("".join(delta["function"].get("arguments", "") for delta in tool_deltas)) == {
        "city": "Paris"
    }
    finish_reasons = [
        choice["finish_reason"]
        for event in events
        if isinstance(event, dict)
        for choice in event.get("choices", [])
        if choice.get("finish_reason")
    ]
    assert finish_reasons == ["tool_calls"]


def test_tool_choice_none_keeps_tool_tags_as_content():
    output = '[TOOL_CALLS] [{"name":"get_weather","arguments":{"city":"Paris"}}]'
    state = FakeState([UserReply(uid=42, incremental_output=output, finished=True)])

    response = run(
        handle_chat_completion(
            chat_request(tool_choice="none"),
            request=None,
            state=state,
            model_sampling={},
        )
    )

    assert state.sent is not None
    assert state.sent.tools is None
    choice = response["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"] == {"role": "assistant", "content": output}


def test_completion_rejects_token_id_prompts():
    state = FakeState([])
    response = run(
        handle_completion(
            CompletionRequest(model="client-model", prompt=[1, 2, 3]),
            request=None,
            state=state,
            model_sampling={},
        )
    )

    assert response.status_code == 400
    body = json.loads(response.body)
    assert body["error"]["type"] == "invalid_request_error"
    assert "token-id prompt" in body["error"]["message"]


def test_completion_accepts_text_prompt():
    state = FakeState(
        [UserReply(uid=42, incremental_output="hello", finished=True, prompt_tokens_delta=2, completion_tokens_delta=1)]
    )

    response = run(
        handle_completion(
            CompletionRequest(model="client-model", prompt="say hi"),
            request=None,
            state=state,
            model_sampling={},
        )
    )

    assert state.sent is not None
    assert state.sent.text == "say hi"
    assert response["object"] == "text_completion"
    assert response["choices"][0]["text"] == "hello"
    assert response["usage"] == {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}


def test_completion_forwards_length_finish_reason():
    state = FakeState(
        [UserReply(uid=42, incremental_output="hello", finished=True, finish_reason="length")]
    )
    response = run(
        handle_completion(
            CompletionRequest(model="client-model", prompt="say hi"),
            request=None,
            state=state,
            model_sampling={},
        )
    )
    assert response["choices"][0]["finish_reason"] == "length"


def test_omitted_max_tokens_honors_server_default():
    from freetoken.server.generation import DEFAULT_MAX_OUTPUT_TOKENS

    chat_state = FakeState([UserReply(uid=42, incremental_output="hi", finished=True)])
    chat_state.config.max_output_tokens = 4096
    run(handle_chat_completion(
        ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}]),
        request=None, state=chat_state, model_sampling={},
    ))
    assert chat_state.sent.sampling_params.max_tokens == 4096

    cmpl_state = FakeState([UserReply(uid=42, incremental_output="hi", finished=True)])
    cmpl_state.config.max_output_tokens = 4096
    run(handle_completion(
        CompletionRequest(model="m", prompt="hi"),
        request=None, state=cmpl_state, model_sampling={},
    ))
    assert cmpl_state.sent.sampling_params.max_tokens == 4096

    # explicit value wins
    exp_state = FakeState([UserReply(uid=42, incremental_output="hi", finished=True)])
    exp_state.config.max_output_tokens = 4096
    run(handle_completion(
        CompletionRequest(model="m", prompt="hi", max_tokens=50),
        request=None, state=exp_state, model_sampling={},
    ))
    assert exp_state.sent.sampling_params.max_tokens == 50

    fallback_state = FakeState([UserReply(uid=42, incremental_output="hi", finished=True)])
    run(handle_chat_completion(
        ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}]),
        request=None, state=fallback_state, model_sampling={},
    ))
    assert fallback_state.sent.sampling_params.max_tokens == DEFAULT_MAX_OUTPUT_TOKENS


def test_models_route_returns_served_model_name():
    state = FakeState([])
    app = FastAPI()
    register_openai_routes(app, lambda: state, lambda: {})

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    card = response.json()["data"][0]
    assert card["id"] == "unit-model"
    # No max_seq_len on this config: null rather than a 500.
    assert card["max_model_len"] is None and card["context_length"] is None


def test_models_route_publishes_the_model_context_length():
    """`ft launch` reads this to size each agent's context window."""
    state = FakeState([])
    state.config.max_seq_len = 262144
    app = FastAPI()
    register_openai_routes(app, lambda: state, lambda: {})

    card = TestClient(app).get("/v1/models").json()["data"][0]

    assert card["max_model_len"] == 262144
    assert card["context_length"] == 262144


async def _collect(generator):
    return [chunk async for chunk in generator]


# --------------------------------------------------------------- dsv4 reasoning
from freetoken.server.reasoning_parser import DSML_TOKEN  # noqa: E402

_TC_OPEN = f"<{DSML_TOKEN}tool_calls>"
_DSV4_TOOL_BLOCK = (
    f"{_TC_OPEN}\n"
    f'<{DSML_TOKEN}invoke name="get_weather">\n'
    f'<{DSML_TOKEN}parameter name="city" string="true">Paris</{DSML_TOKEN}parameter>\n'
    f"</{DSML_TOKEN}invoke>\n"
    f"</{DSML_TOKEN}tool_calls>"
)


def _dsv4_state(replies):
    return FakeState(replies, tool_call_parser="deepseekv32", reasoning_parser="deepseekv32")


def test_dsv4_non_stream_splits_reasoning_and_tool_call():
    # tools present -> thinking mode -> output starts inside the reasoning block.
    output = f"I should look up the weather.</think>Let me check.\n\n{_DSV4_TOOL_BLOCK}"
    state = _dsv4_state([UserReply(uid=42, incremental_output=output, finished=True)])

    response = run(handle_chat_completion(chat_request(), request=None, state=state, model_sampling={}))

    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["reasoning_content"] == "I should look up the weather."
    assert choice["message"]["content"] == "Let me check."
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"city": "Paris"}


def test_dsv4_non_stream_missing_end_token_before_tool_block():
    # dsv4 sometimes skips </think> and jumps straight to the tool block.
    output = f"Looking it up now.\n\n{_DSV4_TOOL_BLOCK}"
    state = _dsv4_state([UserReply(uid=42, incremental_output=output, finished=True)])

    response = run(handle_chat_completion(chat_request(), request=None, state=state, model_sampling={}))

    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["reasoning_content"] == "Looking it up now."
    assert choice["message"]["content"] == ""
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_dsv4_non_stream_reasoning_without_tools():
    # No tools, but thinking explicitly requested.
    output = "Let me think about it.</think>The answer is 42."
    state = _dsv4_state([UserReply(uid=42, incremental_output=output, finished=True)])
    req = ChatCompletionRequest(
        model="client-model",
        messages=[{"role": "user", "content": "hi"}],
        chat_template_kwargs={"thinking": True},
        max_tokens=8,
    )

    response = run(handle_chat_completion(req, request=None, state=state, model_sampling={}))

    choice = response["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["reasoning_content"] == "Let me think about it."
    assert choice["message"]["content"] == "The answer is 42."
    assert "tool_calls" not in choice["message"]


def test_dsv4_non_stream_strips_leaked_special_tokens():
    bos = "<｜begin▁of▁sentence｜>"
    eos = "<｜end▁of▁sentence｜>"
    output = f"reasoning here</think>Hello{eos} world{bos}"
    state = _dsv4_state([UserReply(uid=42, incremental_output=output, finished=True)])
    req = ChatCompletionRequest(
        model="client-model",
        messages=[{"role": "user", "content": "hi"}],
        chat_template_kwargs={"thinking": True},
        max_tokens=8,
    )

    response = run(handle_chat_completion(req, request=None, state=state, model_sampling={}))

    assert response["choices"][0]["message"]["content"] == "Hello world"


def test_dsv4_stream_emits_reasoning_then_tool_calls():
    # Token-aligned deltas (the detokenizer emits one token's text per message,
    # so markers like </think> never arrive glued to preceding text).
    chunks = ["Thinking ", "hard.", "</think>", "One ", "sec.", "\n\n", _DSV4_TOOL_BLOCK]
    replies = [
        UserReply(uid=42, incremental_output=c, finished=(i == len(chunks) - 1))
        for i, c in enumerate(chunks)
    ]
    state = _dsv4_state(replies)

    events = parse_sse(run(_collect(stream_chat_completion_chunks(42, chat_request(stream=True), state))))

    reasoning = "".join(
        choice["delta"]["reasoning_content"]
        for event in events
        if isinstance(event, dict)
        for choice in event.get("choices", [])
        if "reasoning_content" in choice.get("delta", {})
    )
    assert reasoning == "Thinking hard."
    content = "".join(
        choice["delta"]["content"]
        for event in events
        if isinstance(event, dict)
        for choice in event.get("choices", [])
        if "content" in choice.get("delta", {}) and choice["delta"]["content"]
    )
    # Streaming releases the pre-tag separator whitespace as content (it is emitted
    # before the tool tag is seen); only trailing whitespace may differ from the
    # old buffer-then-strip behavior.
    assert content.rstrip() == "One sec."
    tool_names = [
        tc["function"]["name"]
        for event in events
        if isinstance(event, dict)
        for choice in event.get("choices", [])
        for tc in choice.get("delta", {}).get("tool_calls", [])
        if tc.get("function", {}).get("name")
    ]
    assert "get_weather" in tool_names
    finish_reasons = [
        choice["finish_reason"]
        for event in events
        if isinstance(event, dict)
        for choice in event.get("choices", [])
        if choice.get("finish_reason")
    ]
    assert finish_reasons == ["tool_calls"]


# --------------------------------------------------------------- gpt-oss harmony
def test_gptoss_non_stream_splits_reasoning_and_clean_content():
    output = (
        "<|channel|>analysis<|message|>The user says hi.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Hello there!"
    )
    state = FakeState(
        [UserReply(uid=42, incremental_output=output, finished=True)],
        tool_call_parser="gpt_oss",
        reasoning_parser="gpt_oss",
    )
    req = chat_request(tools=None)
    response = run(handle_chat_completion(req, request=None, state=state, model_sampling={}))
    message = response["choices"][0]["message"]
    assert message["reasoning_content"] == "The user says hi."
    assert message["content"] == "Hello there!"
    assert "<|channel|>" not in message["content"]


def test_gptoss_non_stream_still_extracts_tool_call():
    output = (
        "<|channel|>analysis<|message|>need weather<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.get_weather "
        '<|message|>{"city":"Paris"}<|call|>'
    )
    state = FakeState(
        [UserReply(uid=42, incremental_output=output, finished=True)],
        tool_call_parser="gpt_oss",
        reasoning_parser="gpt_oss",
    )
    response = run(handle_chat_completion(chat_request(), request=None, state=state, model_sampling={}))
    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_calls = choice["message"]["tool_calls"]
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert choice["message"]["reasoning_content"] == "need weather"


# ----------------------------------------------------------- cache report
def _cache_hit_replies() -> list[UserReply]:
    return [
        UserReply(uid=42, incremental_output="", finished=False, prompt_tokens_delta=5, cached_tokens=3),
        UserReply(uid=42, incremental_output="hi", finished=True, completion_tokens_delta=1),
    ]


def test_non_stream_chat_usage_reports_cached_tokens_only_with_flag():
    state = FakeState(_cache_hit_replies())
    state.config.enable_cache_report = True
    response = run(handle_chat_completion(chat_request(tools=None), request=None, state=state, model_sampling={}))
    # prompt_tokens stays inclusive of the cached prefix; the details carry the split.
    assert response["usage"]["prompt_tokens"] == 5
    assert response["usage"]["prompt_tokens_details"] == {"cached_tokens": 3}

    response = run(handle_chat_completion(chat_request(tools=None), request=None, state=FakeState(_cache_hit_replies()), model_sampling={}))
    assert "prompt_tokens_details" not in response["usage"]


def test_non_stream_chat_usage_omits_details_on_zero_hit():
    state = FakeState(
        [UserReply(uid=42, incremental_output="hi", finished=True, prompt_tokens_delta=5, completion_tokens_delta=1)]
    )
    state.config.enable_cache_report = True
    response = run(handle_chat_completion(chat_request(tools=None), request=None, state=state, model_sampling={}))
    assert "prompt_tokens_details" not in response["usage"]


def test_stream_chat_usage_chunk_carries_cached_tokens():
    state = FakeState(_cache_hit_replies())
    state.config.enable_cache_report = True
    req = chat_request(tools=None, stream_options={"include_usage": True})

    async def collect():
        return [chunk async for chunk in stream_chat_completion_chunks(42, req, state)]

    events = parse_sse(run(collect()))
    usage = next(e["usage"] for e in reversed(events) if isinstance(e, dict) and e.get("usage"))
    assert usage["prompt_tokens"] == 5
    assert usage["prompt_tokens_details"] == {"cached_tokens": 3}


# --------------------------------------------------------------- minimax think
def test_minimax_http_non_stream_forces_implicit_reasoning_without_request_knob():
    state = FakeState(
        [UserReply(uid=42, incremental_output="private thought</think>visible answer", finished=True)],
        reasoning_parser="minimax",
    )
    req = chat_request(tools=None)

    response = run(handle_chat_completion(req, request=None, state=state, model_sampling={}))

    message = response["choices"][0]["message"]
    assert message["reasoning_content"] == "private thought"
    assert message["content"] == "visible answer"
