"""Qwen protocol markers are private to semantic chat-style API responses."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PY = os.path.join(_ROOT, "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from freetoken.llm.llm import LLM, RequestStatus  # noqa: E402
from freetoken.message import DetokenizeMsg, UserReply  # noqa: E402
from freetoken.server.api_server import FrontendManager  # noqa: E402
from freetoken.server.api_models import CompletionRequest  # noqa: E402
from freetoken.server.generation import (  # noqa: E402
    _QWEN_SEMANTIC_HIDDEN_TOKENS,
    ContentDelta,
    GenDone,
    GenSpec,
    ReasoningDelta,
    ToolCallsDelta,
    generate_events,
    generate_full,
)
from freetoken.server.openai_api import handle_completion, stream_completion_chunks  # noqa: E402


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "echo",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
        },
    }
]


class FakeState:
    def __init__(
        self,
        chunks: list[str],
        *,
        completion_tokens: list[int] | None = None,
        reasoning_parser: str | None = "qwen3",
        tool_call_parser: str = "qwen3_coder",
    ) -> None:
        self.config = SimpleNamespace(
            reasoning_parser=reasoning_parser,
            tool_call_parser=tool_call_parser,
            served_model_name="qwen-unit",
        )
        self._chunks = chunks
        self._completion_tokens = completion_tokens or [1] * len(chunks)
        self.sent = None

    def new_user(self) -> int:
        return 42

    async def send_one(self, msg) -> None:
        self.sent = msg

    async def wait_for_ack(self, uid: int):
        assert uid == 42
        for index, (chunk, token_delta) in enumerate(zip(self._chunks, self._completion_tokens)):
            last = index == len(self._chunks) - 1
            yield UserReply(
                uid=uid,
                incremental_output=chunk,
                finished=last,
                finish_reason="stop" if last else None,
                prompt_tokens_delta=5 if index == 0 else 0,
                completion_tokens_delta=token_delta,
            )


def _spec(*, tools=None) -> GenSpec:
    return GenSpec(
        messages=[],
        sampling_params=SimpleNamespace(),
        chat_template_kwargs={"enable_thinking": False},
        template_tools=tools,
        parser_tools=tools,
    )


async def _events(chunks: list[str], *, tools=None):
    return [event async for event in generate_events(42, _spec(tools=tools), FakeState(chunks))]


def _visible(events) -> str:
    return "".join(event.text for event in events if isinstance(event, ContentDelta))


@pytest.mark.parametrize("marker", _QWEN_SEMANTIC_HIDDEN_TOKENS)
def test_every_hidden_marker_is_chunk_boundary_safe(marker: str):
    text = f"before{marker}after"
    events = asyncio.run(_events(list(text)))

    assert _visible(events) == "beforeafter"
    assert isinstance(events[-1], GenDone)
    assert events[-1].completion_tokens == len(text)


def test_stream_and_nonstream_match_while_preserving_semantic_markers_and_literals():
    grounding = (
        "<|object_ref_start|>cat<|object_ref_end|>"
        "<|box_start|>(1,2),(3,4)<|box_end|>"
        "<|quad_start|>(1,2),(3,4),(5,6),(7,8)<|quad_end|>"
    )
    text = (
        "<|im_start|><think>plan</think>answer"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{grounding} literal \"<|im_start|>\" and `<|image_pad|>`<|im_end|>"
    )
    streamed = asyncio.run(_events(list(text)))
    full = asyncio.run(generate_full(42, _spec(), FakeState([text], completion_tokens=[len(text)])))

    reasoning = "".join(event.text for event in streamed if isinstance(event, ReasoningDelta))
    expected_content = f"answer{grounding} literal \"<|im_start|>\" and `<|image_pad|>`"
    assert reasoning == full.reasoning == "plan"
    assert _visible(streamed) == full.content == expected_content
    assert full.completion_tokens == streamed[-1].completion_tokens == len(text)


def test_tool_and_think_markers_reach_their_parsers_before_filtering():
    block = (
        "<tool_call><function=echo><parameter=value>"
        'literal <|im_start|>'
        "</parameter></function></tool_call>"
    )
    text = "<think>choose tool</think><|im_start|>" + block + "<|im_end|>"
    streamed = asyncio.run(_events(list(text), tools=TOOLS))
    full = asyncio.run(generate_full(42, _spec(tools=TOOLS), FakeState([text])))

    stream_calls = [
        call for event in streamed if isinstance(event, ToolCallsDelta) for call in event.calls
    ]
    assert "".join(
        event.text for event in streamed if isinstance(event, ReasoningDelta)
    ) == full.reasoning == "choose tool"
    assert _visible(streamed) == full.content == ""
    assert len(stream_calls) == len(full.tool_calls) == 1
    assert stream_calls[0].name == full.tool_calls[0].name == "echo"
    assert json.loads(stream_calls[0].parameters) == json.loads(full.tool_calls[0].parameters) == {
        "value": "literal <|im_start|>"
    }


def test_filter_state_is_isolated_per_concurrent_request():
    async def run_pair():
        return await asyncio.gather(
            _events(["left<|im_", "start|>right"]),
            _events(['quoted "<|im_start|>"']),
        )

    first, second = asyncio.run(run_pair())
    assert _visible(first) == "leftright"
    assert _visible(second) == 'quoted "<|im_start|>"'


@pytest.mark.parametrize("quote", ['"', "`", "'"])
def test_unterminated_literal_cannot_disable_filtering(quote: str):
    text = f"answer {quote}oops <|im_end|>"
    streamed = asyncio.run(_events(list(text)))
    full = asyncio.run(generate_full(42, _spec(), FakeState([text])))

    assert _visible(streamed) == full.content == f"answer {quote}oops "


@pytest.mark.parametrize("slashes", ["\\", "\\\\"])
def test_escaped_marker_opener_does_not_bypass_unterminated_quote_filter(slashes: str):
    text = f'answer "oops {slashes}<|im_end|>'
    expected = f'answer "oops {slashes}'
    streamed = asyncio.run(_events(list(text)))
    full = asyncio.run(generate_full(42, _spec(), FakeState([text])))

    assert _visible(streamed) == full.content == expected


@pytest.mark.parametrize(("opening", "closing"), [("“", "”"), ("‘", "’")])
def test_typographic_quoted_marker_is_preserved(opening: str, closing: str):
    text = f"literal {opening}<|im_start|>{closing}"
    streamed = asyncio.run(_events(list(text)))
    full = asyncio.run(generate_full(42, _spec(), FakeState([text])))

    assert _visible(streamed) == full.content == text


def test_suppressed_marker_does_not_turn_a_possessive_into_an_open_quote():
    text = "foo<|im_end|>'s <|vision_pad|>done"
    streamed = asyncio.run(_events(list(text)))
    full = asyncio.run(generate_full(42, _spec(), FakeState([text])))

    assert _visible(streamed) == full.content == "foo's done"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<<|im_end|>", "<"),
        ("<|im_<|im_end|>", "<|im_"),
        ("x<<|vision_pad|>y", "x<y"),
    ],
)
def test_overlapping_candidate_start_reprocesses_the_mismatch(text: str, expected: str):
    streamed = asyncio.run(_events(list(text)))
    full = asyncio.run(generate_full(42, _spec(), FakeState([text])))

    assert _visible(streamed) == full.content == expected


def test_partial_candidate_before_quote_closer_does_not_extend_literal_scope():
    text = 'literal "<|im_" after <|im_end|> "done"'
    expected = 'literal "<|im_" after  "done"'
    streamed = asyncio.run(_events(list(text)))
    full = asyncio.run(generate_full(42, _spec(), FakeState([text])))

    assert _visible(streamed) == full.content == expected


def test_tool_event_inside_marker_fragments_keeps_stream_nonstream_parity():
    block = (
        "<tool_call><function=echo><parameter=value>ok</parameter>"
        "</function></tool_call>"
    )
    chunks = ["before<|im_", block, "start|>after"]
    streamed = asyncio.run(_events(chunks, tools=TOOLS))
    full = asyncio.run(generate_full(42, _spec(tools=TOOLS), FakeState(["".join(chunks)])))

    assert _visible(streamed) == full.content == "beforeafter"
    assert len([event for event in streamed if isinstance(event, ToolCallsDelta)]) == 1
    assert len(full.tool_calls) == 1


def test_disproved_marker_prefix_stays_before_intervening_tool_event():
    block = (
        "<tool_call><function=echo><parameter=value>ok</parameter>"
        "</function></tool_call>"
    )
    streamed = asyncio.run(_events(["pre<|im_", block, "Xpost"], tools=TOOLS))

    kinds = [type(event).__name__ for event in streamed]
    call_start_index = kinds.index("ToolCallStart")
    call_end_index = kinds.index("ToolCallsDelta")
    before = "".join(
        event.text
        for event in streamed[:call_start_index]
        if isinstance(event, ContentDelta)
    )
    after = "".join(
        event.text
        for event in streamed[call_end_index + 1 :]
        if isinstance(event, ContentDelta)
    )

    assert before == "pre<|im_"
    assert after == "Xpost"


def test_tool_event_flushes_prior_quoted_prose_before_call():
    block = (
        "<tool_call><function=echo><parameter=value>ok</parameter>"
        "</function></tool_call>"
    )
    chunks = [
        'before "<|im_start|> quoted',
        block,
        ' tail <|vision_pad|>" after<|im_end|>',
    ]
    streamed = asyncio.run(_events(chunks, tools=TOOLS))
    full = asyncio.run(generate_full(42, _spec(tools=TOOLS), FakeState(["".join(chunks)])))

    kinds = [type(event).__name__ for event in streamed]
    call_start_index = kinds.index("ToolCallStart")
    call_end_index = kinds.index("ToolCallsDelta")
    prefix = "".join(
        event.text
        for event in streamed[:call_start_index]
        if isinstance(event, ContentDelta)
    )
    suffix = "".join(
        event.text
        for event in streamed[call_end_index + 1 :]
        if isinstance(event, ContentDelta)
    )
    assert prefix == 'before "<|im_start|> quoted'
    assert suffix == ' tail <|vision_pad|>" after'
    assert call_start_index < call_end_index
    assert _visible(streamed) == full.content == (
        'before "<|im_start|> quoted tail <|vision_pad|>" after'
    )


def test_reasoning_filter_hides_transport_marker_and_preserves_fim_tokens():
    text = (
        "<think>plan<|vision_pad|></think>"
        "<|fim_prefix|>left<|fim_suffix|>right<|fim_middle|><|im_end|>"
    )
    streamed = asyncio.run(_events(list(text)))
    full = asyncio.run(generate_full(42, _spec(), FakeState([text])))

    reasoning = "".join(
        event.text for event in streamed if isinstance(event, ReasoningDelta)
    )
    expected_content = "<|fim_prefix|>left<|fim_suffix|>right<|fim_middle|>"
    assert reasoning == full.reasoning == "plan"
    assert _visible(streamed) == full.content == expected_content


def test_non_qwen_semantic_dialect_is_unchanged():
    text = '<|im_start|>literal "<|vision_pad|>"<|im_end|>'
    state = FakeState(
        [text],
        reasoning_parser=None,
        tool_call_parser="llama3",
    )

    result = asyncio.run(generate_full(42, _spec(), state))

    assert result.content == text


def test_legacy_qwen_parser_alias_enables_semantic_filter_without_reasoning():
    text = "<|im_start|>visible<|vision_pad|><|im_end|>"
    state = FakeState([text], reasoning_parser=None, tool_call_parser="qwen")
    result = asyncio.run(generate_full(42, _spec(), state))

    assert result.content == "visible"


def test_legacy_completions_paths_remain_raw_for_qwen():
    raw = '<|im_start|>raw "<|vision_pad|>"<|im_end|>'

    full = asyncio.run(
        handle_completion(
            CompletionRequest(model="qwen-unit", prompt="continue", max_tokens=8),
            request=None,
            state=FakeState([raw]),
            model_sampling={},
        )
    )
    assert full["choices"][0]["text"] == raw

    async def stream_text() -> str:
        chunks = [
            chunk async for chunk in stream_completion_chunks(
                42,
                CompletionRequest(model="qwen-unit", prompt="continue", max_tokens=8, stream=True),
                FakeState([raw]),
            )
        ]
        frames = []
        for chunk in chunks:
            for line in chunk.decode().splitlines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    frames.append(json.loads(line[6:]))
        return "".join(frame["choices"][0]["text"] for frame in frames)

    assert asyncio.run(stream_text()) == raw


def test_raw_generate_stream_remains_unfiltered_for_qwen():
    raw = "<|im_start|>raw<|vision_pad|><|im_end|>"

    async def stream_text() -> str:
        chunks = [
            chunk async for chunk in FrontendManager.stream_generate(FakeState([raw]), 42)
        ]
        frames = []
        for chunk in chunks:
            for line in chunk.decode().splitlines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    frames.append(json.loads(line[6:]))
        return "".join(frame["text"] for frame in frames)

    assert asyncio.run(stream_text()) == raw


def test_offline_generation_keeps_exact_output_token_ids():
    offline = SimpleNamespace(
        status_map={7: RequestStatus(uid=7, input_ids=[], output_ids=[])},
        eos_token_ids=set(),
    )
    marker_token_ids = [248045, 248044, 248046]

    LLM.offline_send_result(
        offline,
        [
            DetokenizeMsg(uid=7, next_token=token_id, finished=False)
            for token_id in marker_token_ids
        ],
    )

    assert offline.status_map[7].output_ids == marker_token_ids
