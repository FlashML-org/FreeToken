"""Correctness matrix across every supported model family.

Two axes, per the parser combinations `args.py` auto-infers for supported models:

* TOOL-CALL STREAMING: for every tool_call_parser, the streamed events under
  randomized chunking must agree with the non-streaming path on the same text —
  same content, same calls, right ordering, no markup leakage. Detectors without
  incremental support (gpt_oss) satisfy this via the buffered fallback.
* REASONING DELIVERY: for every reasoning-capable family, thinking text must
  arrive intact and separated from content at each API entry point
  (/v1/responses, /v1/messages, /v1/chat/completions), streaming and not.

Run:  PYTHONPATH=python <venv>/bin/python -m pytest tests/server/test_streaming_model_matrix.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from types import SimpleNamespace

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PY = os.path.join(_ROOT, "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from freetoken.message.frontend import UserReply  # noqa: E402
from freetoken.server import anthropic_api as A  # noqa: E402
from freetoken.server import responses_api as RP  # noqa: E402
from freetoken.server.api_models import ChatCompletionRequest  # noqa: E402
from freetoken.server.generation import (  # noqa: E402
    ContentDelta,
    GenDone,
    GenSpec,
    ReasoningDelta,
    ToolCallArgsDelta,
    ToolCallsDelta,
    ToolCallStart,
    generate_events,
    generate_full,
)
from freetoken.server.openai_api import stream_chat_completion_chunks  # noqa: E402
from freetoken.server.responses_api import ResponsesRequest  # noqa: E402

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}}},
        },
    },
]

READ_ARGS = {"filePath": "/tmp/test_calc.py"}

# One realistic single-call block per supported tool_call_parser (shapes match the
# family fixtures in test_function_call_parser.py).
CALL_BLOCKS = {
    "deepseekv32": (
        '<｜DSML｜function_calls><｜DSML｜invoke name="read">'
        '<｜DSML｜parameter name="filePath" string="true">/tmp/test_calc.py</｜DSML｜parameter>'
        "</｜DSML｜invoke></｜DSML｜function_calls>"
    ),
    "qwen25": '<tool_call>\n{"name": "read", "arguments": {"filePath": "/tmp/test_calc.py"}}\n</tool_call>',
    "qwen3_coder": (
        "<tool_call><function=read><parameter=filePath>/tmp/test_calc.py</parameter>"
        "</function></tool_call>"
    ),
    "glm47": (
        "<tool_call>read<arg_key>filePath</arg_key><arg_value>/tmp/test_calc.py</arg_value></tool_call>"
    ),
    "poolside_v1": (
        "<tool_call>read<arg_key>filePath</arg_key><arg_value>/tmp/test_calc.py</arg_value></tool_call>"
    ),
    "gemma4": '<|tool_call>call:read{filePath:<|"|>/tmp/test_calc.py<|"|>}<tool_call|>',
    "minimax": (
        '<minimax:tool_call><invoke name="read"><parameter name="filePath">'
        "/tmp/test_calc.py</parameter></invoke></minimax:tool_call>"
    ),
    "minimax_m3": (
        "]<]minimax[>[<tool_call>\n"
        ']<]minimax[>[<invoke name="read">'
        "]<]minimax[>[<filePath>/tmp/test_calc.py]<]minimax[>[</filePath>"
        "]<]minimax[>[</invoke>\n"
        "]<]minimax[>[</tool_call>"
    ),
    "mistral": '[TOOL_CALLS] [{"name": "read", "arguments": {"filePath": "/tmp/test_calc.py"}}]',
    "llama3": '<|python_tag|>{"name": "read", "arguments": {"filePath": "/tmp/test_calc.py"}}',
    "gpt_oss": (
        "<|start|>assistant<|channel|>commentary to=functions.read "
        '<|constrain|>json<|message|>{"filePath": "/tmp/test_calc.py"}<|end|>'
    ),
    "muse_glimmer": (
        "<|start|>assistant to=read<|message|><atem:function_calls>\n"
        '<atem:invoke name="read">\n'
        "<atem:parameter name=\"filePath\">/tmp/test_calc.py</atem:parameter>\n"
        "</atem:invoke>\n</atem:function_calls><|eot|>"
    ),
}

# Substrings that must never leak into user-visible content.
MARKUP_MARKERS = {
    "deepseekv32": ["｜DSML｜"],
    "qwen25": ["<tool_call>", "</tool_call>"],
    "qwen3_coder": ["<tool_call>", "<function="],
    "glm47": ["<arg_key>", "<arg_value>", "</tool_call>"],
    "poolside_v1": ["<arg_key>", "<arg_value>", "</tool_call>"],
    "gemma4": ["<|tool_call>", "<tool_call|>"],
    "minimax": ["<minimax:tool_call>", "<invoke"],
    "minimax_m3": ["]<]minimax[>[", "<invoke"],
    "mistral": ["[TOOL_CALLS]"],
    "llama3": ["<|python_tag|>"],
    "gpt_oss": ["<|channel|>", "to=functions."],
    "muse_glimmer": ["<atem:", "<|message|>", "<|eot|>"],
}

# (tool_call_parser, reasoning_parser, think_open, think_close) per reasoning-capable
# family, mirroring args.py auto-inference. think_open == "" when the chat template
# opens the block implicitly (model emits only the closing marker).
REASONING_FAMILIES = {
    "dsv4": ("deepseekv32", "deepseekv32", "", "</think>"),
    "qwen3.5": ("qwen3_coder", "qwen3", "", "</think>"),
    "qwen": ("qwen25", "qwen3", "", "</think>"),
    "glm4.7": ("glm47", "glm", "", "</think>"),
    "laguna": ("poolside_v1", "poolside_v1", "", "</think>"),
    "minimax-m2": ("minimax", "minimax", "", "</think>"),
    # M3 adaptive mode: the model opens <mm:think> itself (enabled mode pre-opens it
    # in the template; the parser then runs with force_reasoning=True instead).
    "minimax-m3": ("minimax_m3", "minimax_m3", "<mm:think>", "</mm:think>"),
    "gemma4": ("gemma4", "gemma4", "<|channel>thought\n", "<channel|>"),
    "gpt-oss": ("gpt_oss", "gpt_oss", None, None),  # harmony channels; custom fixture
    "muse-glimmer": ("muse_glimmer", "muse_glimmer", None, None),  # ATEM channels; custom fixture
}


class FakeState:
    def __init__(self, chunks, tool_call_parser, reasoning_parser=None, finish_reason=None):
        self.config = SimpleNamespace(
            tool_call_parser=tool_call_parser,
            reasoning_parser=reasoning_parser,
        )
        self._chunks = chunks
        self._finish_reason = finish_reason

    async def wait_for_ack(self, uid):
        last = len(self._chunks) - 1
        for i, chunk in enumerate(self._chunks):
            yield UserReply(
                uid=uid,
                incremental_output=chunk,
                finished=(i == last),
                finish_reason=self._finish_reason if i == last else None,
                prompt_tokens_delta=1 if i == 0 else 0,
                completion_tokens_delta=1,
            )


def _spec(tools=TOOLS):
    return GenSpec(messages=[], sampling_params=None, parser_tools=tools, template_tools=tools)


def _random_chunks(text: str, seed: int) -> list[str]:
    rng = random.Random(seed)
    chunks, i = [], 0
    while i < len(text):
        step = rng.randint(1, 9)
        chunks.append(text[i : i + step])
        i += step
    return chunks


def _stream_events(chunks, tool, reasoning=None, finish_reason=None):
    state = FakeState(chunks, tool_call_parser=tool, reasoning_parser=reasoning,
                      finish_reason=finish_reason)

    async def run():
        return [ev async for ev in generate_events(42, _spec(), state)]

    return asyncio.run(run())


def _full_result(text, tool, reasoning=None):
    state = FakeState([text], tool_call_parser=tool, reasoning_parser=reasoning)
    return asyncio.run(generate_full(42, _spec(), state))


def _content(events) -> str:
    return "".join(ev.text for ev in events if isinstance(ev, ContentDelta))


def _reasoning(events) -> str:
    return "".join(ev.text for ev in events if isinstance(ev, ReasoningDelta))


def _calls(events):
    return [c for ev in events if isinstance(ev, ToolCallsDelta) for c in ev.calls]


def _assert_stream_invariants(events, family):
    kinds = [type(ev).__name__ for ev in events]
    assert kinds[-1] == "GenDone" and kinds.count("GenDone") == 1
    # Every Start is closed, ordinals are sequential, args deltas sit between
    # their Start and close, and fragments prefix the final arguments.
    open_ord = None
    seen = 0
    frags: dict[int, str] = {}
    for ev in events:
        if isinstance(ev, ToolCallStart):
            assert open_ord is None, f"{family}: Start while call {open_ord} still open"
            assert ev.tool_index == seen
            open_ord = ev.tool_index
        elif isinstance(ev, ToolCallArgsDelta):
            assert open_ord == ev.tool_index, f"{family}: args delta outside its call"
            frags[ev.tool_index] = frags.get(ev.tool_index, "") + ev.fragment
        elif isinstance(ev, ToolCallsDelta):
            for call in ev.calls:
                if open_ord is None:
                    # standalone complete call: legal on the buffered fallback path
                    # (detectors with supports_streaming=False emit no Start events)
                    seen += 1
                    continue
                assert open_ord == call.tool_index, f"{family}: close does not match open call"
                if call.tool_index in frags:
                    assert call.parameters.startswith(frags[call.tool_index])
                open_ord = None
                seen += 1
    assert open_ord is None, f"{family}: call {open_ord} never closed"
    content = _content(events)
    for marker in MARKUP_MARKERS[family]:
        assert marker not in content, f"{family}: markup {marker!r} leaked into content"


# --------------------------------------------------------------------------- #
# Axis 1: tool-call streaming parity for every supported tool_call_parser
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", sorted(CALL_BLOCKS))
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_tool_stream_matches_non_stream(family, seed):
    text = "Let me check the files. " + CALL_BLOCKS[family]
    events = _stream_events(_random_chunks(text, seed), family)
    _assert_stream_invariants(events, family)

    calls = _calls(events)
    assert len(calls) == 1, f"{family}: expected 1 call, got {[c.name for c in calls]}"
    assert calls[0].name == "read"
    assert json.loads(calls[0].parameters) == READ_ARGS
    assert events[-1].finish_reason == "tool_calls"

    full = _full_result(text, family)
    assert [c.name for c in full.tool_calls] == ["read"]
    assert json.loads(full.tool_calls[0].parameters) == READ_ARGS
    assert _content(events).strip() == full.content.strip()


@pytest.mark.parametrize("family", sorted(CALL_BLOCKS))
def test_pure_text_streams_and_matches_non_stream(family):
    text = "Just a plain answer, no tools needed today."
    events = _stream_events(_random_chunks(text, 7), family)
    assert _calls(events) == []
    assert _content(events) == text
    assert events[-1].finish_reason == "stop"
    full = _full_result(text, family)
    assert full.content.strip() == text
    assert full.tool_calls == []


# --------------------------------------------------------------------------- #
# Axis 2: reasoning delivery at every API entry point
# --------------------------------------------------------------------------- #
THINKING = "I should read the file first."
ANSWER = "Reading it now."


def _reasoning_fixture(name):
    tool, reasoning, think_open, think_close = REASONING_FAMILIES[name]
    if name == "gpt-oss":
        text = (
            f"<|channel|>analysis<|message|>{THINKING}<|end|>"
            f"<|start|>assistant<|channel|>final<|message|>{ANSWER}<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.read <|constrain|>json"
            f"<|message|>{json.dumps(READ_ARGS)}<|call|>"
        )
        return tool, reasoning, text
    if name == "muse-glimmer":
        # Generation resumes after the template's <|start|>assistant, so the first
        # segment is a bare header continuation.
        text = (
            f" to=self<|message|>{THINKING}<|eom|>"
            f"<|start|>assistant to=user<|message|>{ANSWER}<|eom|>"
            + CALL_BLOCKS[tool]
        )
        return tool, reasoning, text
    text = f"{think_open}{THINKING}{think_close}{ANSWER} {CALL_BLOCKS[tool]}"
    return tool, reasoning, text


@pytest.mark.parametrize("name", sorted(REASONING_FAMILIES))
def test_reasoning_generation_level(name):
    tool, reasoning, text = _reasoning_fixture(name)
    events = _stream_events(_random_chunks(text, 11), tool, reasoning)
    _assert_stream_invariants(events, tool)
    reasoning_deltas = [ev for ev in events if isinstance(ev, ReasoningDelta)]
    assert len(reasoning_deltas) >= 2  # thinking streams per chunk, not one blob
    assert _reasoning(events).strip() == THINKING
    assert _content(events).strip() == ANSWER
    calls = _calls(events)
    assert len(calls) == 1 and calls[0].name == "read"
    assert json.loads(calls[0].parameters) == READ_ARGS

    full = _full_result(text, tool, reasoning)
    assert full.reasoning.strip() == THINKING
    assert full.content.strip() == ANSWER
    assert [c.name for c in full.tool_calls] == ["read"]


async def _aiter(items):
    for it in items:
        yield it


def _sse_frames(agen):
    async def run():
        out = []
        async for frame in agen:
            etype, data = None, None
            raw = frame.decode() if isinstance(frame, bytes) else frame
            for line in raw.split("\n"):
                if line.startswith("event:"):
                    etype = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    payload = line[len("data:"):].strip()
                    if payload != "[DONE]":
                        data = json.loads(payload)
            out.append((etype, data))
        return out

    return asyncio.run(run())


@pytest.mark.parametrize("name", sorted(REASONING_FAMILIES))
def test_reasoning_responses_entrypoint(name):
    tool, reasoning, text = _reasoning_fixture(name)
    gen_events = _stream_events(_random_chunks(text, 13), tool, reasoning)
    req = ResponsesRequest.model_validate({"model": "m", "input": "hi", "stream": True})
    frames = _sse_frames(RP.responses_stream_generator(_aiter(gen_events), req, "resp_1", 0))

    thinking = "".join(d["delta"] for t, d in frames if t == "response.reasoning_text.delta")
    assert thinking.strip() == THINKING
    content = "".join(d["delta"] for t, d in frames if t == "response.output_text.delta")
    assert content.strip() == ANSWER
    args_done = next(d for t, d in frames if t == "response.function_call_arguments.done")
    assert json.loads(args_done["arguments"]) == READ_ARGS

    completed = frames[-1][1]["response"]
    types = [item["type"] for item in completed["output"]]
    assert types[0] == "reasoning"
    assert set(types) == {"reasoning", "message", "function_call"}
    reasoning_item = completed["output"][0]
    assert reasoning_item["content"][0]["text"].strip() == THINKING


@pytest.mark.parametrize("name", sorted(REASONING_FAMILIES))
def test_reasoning_anthropic_entrypoint(name):
    tool, reasoning, text = _reasoning_fixture(name)
    gen_events = _stream_events(_random_chunks(text, 17), tool, reasoning)
    frames = _sse_frames(A.anthropic_event_stream(_aiter(gen_events), "m", 1))

    thinking = "".join(
        d["delta"]["thinking"] for t, d in frames
        if t == "content_block_delta" and d["delta"].get("type") == "thinking_delta"
    )
    assert thinking.strip() == THINKING
    content = "".join(
        d["delta"]["text"] for t, d in frames
        if t == "content_block_delta" and d["delta"].get("type") == "text_delta"
    )
    assert content.strip() == ANSWER
    starts = [d for t, d in frames if t == "content_block_start"]
    assert [s["content_block"]["type"] for s in starts] == ["thinking", "text", "tool_use"]
    assert [s["index"] for s in starts] == [0, 1, 2]
    tool_json = "".join(
        d["delta"]["partial_json"] for t, d in frames
        if t == "content_block_delta" and d["delta"].get("type") == "input_json_delta"
    )
    assert json.loads(tool_json) == READ_ARGS
    assert frames[-1][0] == "message_stop"


@pytest.mark.parametrize("name", sorted(REASONING_FAMILIES))
def test_reasoning_chat_entrypoint(name):
    tool, reasoning, text = _reasoning_fixture(name)
    state = FakeState(_random_chunks(text, 19), tool_call_parser=tool, reasoning_parser=reasoning)
    req = ChatCompletionRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    )
    frames = _sse_frames(stream_chat_completion_chunks(42, req, state, _spec()))

    deltas = [c["delta"] for _, d in frames if d for c in d.get("choices", [])]
    thinking = "".join(d.get("reasoning_content", "") for d in deltas)
    assert thinking.strip() == THINKING
    content = "".join(d.get("content", "") or "" for d in deltas)
    assert content.strip() == ANSWER
    args = "".join(
        tc["function"].get("arguments", "")
        for d in deltas for tc in d.get("tool_calls", [])
    )
    assert json.loads(args) == READ_ARGS
    finish = [c.get("finish_reason") for _, d in frames if d for c in d.get("choices", [])]
    assert "tool_calls" in finish


# --------------------------------------------------------------------------- #
# Argument streaming: long string values must flow as multiple fragments
# --------------------------------------------------------------------------- #
LONG_VALUE = "line one of a large file...\n" * 40

ARGS_STREAMING_BLOCKS = {
    "deepseekv32": (
        '<｜DSML｜function_calls><｜DSML｜invoke name="read">'
        f'<｜DSML｜parameter name="filePath" string="true">{LONG_VALUE}</｜DSML｜parameter>'
        "</｜DSML｜invoke></｜DSML｜function_calls>"
    ),
    "qwen3_coder": (
        f"<tool_call><function=read><parameter=filePath>{LONG_VALUE}</parameter>"
        "</function></tool_call>"
    ),
    "glm47": (
        f"<tool_call>read<arg_key>filePath</arg_key><arg_value>{LONG_VALUE}</arg_value></tool_call>"
    ),
    "poolside_v1": (
        f"<tool_call>read<arg_key>filePath</arg_key><arg_value>{LONG_VALUE}</arg_value></tool_call>"
    ),
    "gemma4": f'<|tool_call>call:read{{filePath:<|"|>{LONG_VALUE}<|"|>}}<tool_call|>',
    "minimax": (
        f'<minimax:tool_call><invoke name="read"><parameter name="filePath">{LONG_VALUE}'
        "</parameter></invoke></minimax:tool_call>"
    ),
    "gpt_oss": (
        "<|start|>assistant<|channel|>commentary to=functions.read <|constrain|>json"
        f"<|message|>{json.dumps({'filePath': LONG_VALUE})}<|call|>"
    ),
    "muse_glimmer": (
        "<|start|>assistant to=read<|message|><atem:function_calls>\n"
        '<atem:invoke name="read">\n'
        f"<atem:parameter name=\"filePath\">{LONG_VALUE}</atem:parameter>\n"
        "</atem:invoke>\n</atem:function_calls><|eot|>"
    ),
}


@pytest.mark.parametrize("family", sorted(ARGS_STREAMING_BLOCKS))
def test_long_string_value_streams_as_many_fragments(family):
    text = "Writing the file. " + ARGS_STREAMING_BLOCKS[family]
    events = _stream_events(_random_chunks(text, 23), family)
    _assert_stream_invariants(events, family)

    frags = [ev.fragment for ev in events if isinstance(ev, ToolCallArgsDelta)]
    # the value must NOT arrive as one blob at close — it streams while decoding
    assert len(frags) >= 10, f"{family}: only {len(frags)} argument fragment(s)"

    calls = _calls(events)
    assert len(calls) == 1 and calls[0].name == "read"
    args = json.loads(calls[0].parameters)
    assert args["filePath"].strip() == LONG_VALUE.strip()
    # fragments concatenate to the final arguments (prefix-stable all the way)
    assert calls[0].parameters.startswith("".join(frags)[: len(calls[0].parameters)])


@pytest.mark.parametrize(
    ("family", "block", "expected"),
    [
        (
            "qwen3_coder",
            "<tool_call><function=glob><parameter=pattern>*.py</parameter>"
            "<parameter=path>/src</parameter></function></tool_call>",
            {"pattern": "*.py", "path": "/src"},
        ),
        (
            "glm47",
            "<tool_call>glob<arg_key>pattern</arg_key><arg_value>*.py</arg_value>"
            "<arg_key>path</arg_key><arg_value>/src</arg_value></tool_call>",
            {"pattern": "*.py", "path": "/src"},
        ),
        (
            "minimax",
            '<minimax:tool_call><invoke name="glob"><parameter name="pattern">*.py</parameter>'
            '<parameter name="path">/src</parameter></invoke></minimax:tool_call>',
            {"pattern": "*.py", "path": "/src"},
        ),
        (
            "gemma4",
            '<|tool_call>call:glob{pattern:<|"|>*.py<|"|>, path:<|"|>/src<|"|>}<tool_call|>',
            {"pattern": "*.py", "path": "/src"},
        ),
    ],
)
def test_multi_param_string_args_stream_correctly(family, block, expected):
    events = _stream_events(_random_chunks(block, 29), family)
    _assert_stream_invariants(events, family)
    calls = _calls(events)
    assert len(calls) == 1 and calls[0].name == "glob"
    assert json.loads(calls[0].parameters) == expected


# --------------------------------------------------------------------------- #
# Generation-level streaming behavior (family-generic)
# --------------------------------------------------------------------------- #
def test_content_streams_live_with_tools_configured():
    # Regression for the old "pseudo-streaming": with tools present, plain text
    # used to buffer until generation finished and arrive as one delta.
    events = _stream_events(["Hello ", "wor", "ld."], "qwen25")
    contents = [ev.text for ev in events if isinstance(ev, ContentDelta)]
    assert contents == ["Hello ", "wor", "ld."]
    assert isinstance(events[-1], GenDone) and events[-1].finish_reason == "stop"


def test_tool_call_events_shape_and_order():
    block = '<tool_call>\n{"name": "read", "arguments": {"filePath": "/a"}}\n</tool_call>'
    events = _stream_events(["Checking. ", block + "\n", "Done"], "qwen25")
    kinds = [type(ev).__name__ for ev in events]
    # text, call open + streamed args + close, trailing text, done — in wire order
    assert kinds[0] == "ContentDelta" and kinds[1] == "ToolCallStart"
    assert kinds[-3:] == ["ToolCallsDelta", "ContentDelta", "GenDone"]
    assert all(k == "ToolCallArgsDelta" for k in kinds[2:-3])
    close = events[-3].calls[0]
    frags = "".join(ev.fragment for ev in events if isinstance(ev, ToolCallArgsDelta))
    assert close.parameters.startswith(frags)
    assert json.loads(close.parameters) == {"filePath": "/a"}
    assert events[-2].text.strip() == "Done"
    assert events[-1].finish_reason == "tool_calls"


def test_two_calls_get_output_ordinals():
    block1 = '<tool_call>\n{"name": "read", "arguments": {"filePath": "/a"}}\n</tool_call>'
    block2 = '<tool_call>\n{"name": "read", "arguments": {"filePath": "/b"}}\n</tool_call>'
    events = _stream_events([block1 + "\n", block2], "qwen25")
    calls = _calls(events)
    assert [c.tool_index for c in calls] == [0, 1]
    assert [json.loads(c.parameters)["filePath"] for c in calls] == ["/a", "/b"]


@pytest.mark.parametrize(
    ("tool", "chunks", "expected_args"),
    [
        (  # base-family: argument stream cut short, recovered from parse state
            "deepseekv32",
            [
                "<｜DSML｜function_calls>\n",
                '<｜DSML｜invoke name="read">\n',
                '<｜DSML｜parameter name="filePath" string="true">/a</｜DSML｜parameter>\n',
            ],
            {"filePath": "/a"},
        ),
        (  # tag-block family: unterminated block recovered by closing it
            "glm47",
            ["Checking. ", "<tool_call>read\n", "<arg_key>filePath</arg_key><arg_value>/a</arg_value>\n"],
            {"filePath": "/a"},
        ),
    ],
)
def test_truncated_call_recovers_arguments(tool, chunks, expected_args):
    events = _stream_events(chunks, tool, finish_reason="length")
    calls = _calls(events)
    assert len(calls) == 1 and calls[0].name == "read"
    assert json.loads(calls[0].parameters) == expected_args
    assert events[-1].finish_reason == "length"  # truncation wins over tool_calls


def test_dsv4_reasoning_then_tool_block_streams_and_orders_correctly():
    # Bounded-hold regression: with reasoning_parser=deepseekv32 a tool block used
    # to be swallowed into the reasoning parser's buffer until flush, then arrive
    # AFTER the trailing text. Past TOOL_HOLD_MAX it must commit and stream live,
    # and wire order must be reasoning -> call -> trailing text.
    filler = "x" * 600
    block = (
        "<｜DSML｜function_calls>\n"
        '<｜DSML｜invoke name="read">\n'
        f'<｜DSML｜parameter name="filePath" string="true">{filler}</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜function_calls>"
    )
    chunks = ["Let me think. ", *[block[i : i + 40] for i in range(0, len(block), 40)], "\n\nAll checked."]
    events = _stream_events(chunks, "deepseekv32", "deepseekv32")
    kinds = [type(ev).__name__ for ev in events]
    start_idx = kinds.index("ToolCallStart")
    close_idx = kinds.index("ToolCallsDelta")
    trailing_idx = next(
        i for i, ev in enumerate(events)
        if isinstance(ev, ContentDelta) and "All checked." in ev.text
    )
    assert start_idx < close_idx < trailing_idx
    assert json.loads(events[close_idx].calls[0].parameters) == {"filePath": filler}


def test_llama3_pre_tag_text_in_same_chunk_not_dropped():
    chunk = 'Checking. <|python_tag|>{"name": "read", "arguments": {"filePath": "/a"}}'
    events = _stream_events([chunk], "llama3")
    assert _content(events) == "Checking. "
    assert [c.name for c in _calls(events)] == ["read"]


def test_mistral_two_parallel_calls_survive_separator_chunk():
    # A bare separator chunk between two calls used to flush the parser state and
    # destroy the second call (and leak the separator as content).
    events = _stream_events(
        [
            '[TOOL_CALLS] [{"name": "read", "arguments": {"filePath": "/a"}}',
            ", \n",
            '{"name": "read", "arguments": {"filePath": "/b"}}]',
        ],
        "mistral",
    )
    calls = _calls(events)
    assert [json.loads(c.parameters)["filePath"] for c in calls] == ["/a", "/b"]
    assert _content(events).strip() == ""


def test_non_streaming_detector_falls_back_to_buffered_parse(monkeypatch):
    # supports_streaming=False is the escape hatch for future formats without an
    # incremental implementation: content buffers and arrives once at the end.
    from freetoken.server.function_call_parser import Qwen25Detector

    monkeypatch.setattr(Qwen25Detector, "supports_streaming", False)
    events = _stream_events(["Hello ", "world."], "qwen25")
    contents = [ev.text for ev in events if isinstance(ev, ContentDelta)]
    assert contents == ["Hello world."]


def test_with_keepalive_emits_sentinel_during_silence():
    from freetoken.server.generation import KEEPALIVE, with_keepalive

    async def slow_gen():
        await asyncio.sleep(0.08)
        yield ContentDelta("hi")

    async def run():
        return [ev async for ev in with_keepalive(slow_gen(), interval=0.02)]

    out = asyncio.run(run())
    assert out[-1] == ContentDelta("hi")
    assert out[:-1] and all(ev is KEEPALIVE for ev in out[:-1])


# --------------------------------------------------------------------------- #
# Review regressions: empty args, same-chunk trailing text, non-stream parity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("tool", "block"),
    [
        ("qwen25", '<tool_call>\n{"name": "read", "arguments": {}}\n</tool_call>'),
        ("mistral", '[TOOL_CALLS] [{"name": "read", "arguments": {}}]'),
    ],
)
def test_empty_arguments_call_emitted_exactly_once(tool, block):
    # {} is falsy: the completion path must still run (buffer consumed, call
    # closed once) — previously the finish-time recovery re-parsed the residue
    # and emitted the same call twice.
    for seed in (3, 5):
        events = _stream_events(_random_chunks("Check. " + block, seed), tool)
        _assert_stream_invariants(events, tool)
        calls = _calls(events)
        assert len(calls) == 1, [c.name for c in calls]
        assert calls[0].name == "read"
        assert json.loads(calls[0].parameters) == {}


@pytest.mark.parametrize(
    "family",
    ["qwen25", "qwen3_coder", "glm47", "poolside_v1", "gemma4", "minimax", "minimax_m3",
     "deepseekv32", "gpt_oss", "muse_glimmer"],
)
def test_call_then_trailing_text_in_one_chunk_keeps_order(family):
    # Worst case: the whole generation arrives as ONE chunk. Text after the call
    # must reach the wire AFTER the call's close, and must not spawn a phantom
    # nameless call.
    text = "Pre. " + CALL_BLOCKS[family] + " Post text."
    events = _stream_events([text], family)
    _assert_stream_invariants(events, family)
    calls = _calls(events)
    assert len(calls) == 1 and calls[0].name == "read"
    assert json.loads(calls[0].parameters) == READ_ARGS
    kinds = [type(ev).__name__ for ev in events]
    close_i = kinds.index("ToolCallsDelta")
    pre_i = next(i for i, ev in enumerate(events)
                 if isinstance(ev, ContentDelta) and "Pre." in ev.text)
    post_i = next(i for i, ev in enumerate(events)
                  if isinstance(ev, ContentDelta) and "Post text." in ev.text)
    assert pre_i < close_i < post_i


@pytest.mark.parametrize("family", sorted(set(CALL_BLOCKS) - {"llama3"}))
def test_trailing_text_parity_with_non_stream(family):
    # llama3 excluded: its format has no closing marker, so one-shot parsing
    # cannot locate text after the call.
    text = "Before. " + CALL_BLOCKS[family] + "\nAfter."
    events = _stream_events(_random_chunks(text, 31), family)
    full = _full_result(text, family)
    assert "After." in _content(events)
    assert "After." in full.content  # non-streaming must not drop post-tool text

    def norm(s):
        return " ".join(s.split())

    assert norm(_content(events)) == norm(full.content)
    assert [c.name for c in _calls(events)] == [c.name for c in full.tool_calls] == ["read"]
