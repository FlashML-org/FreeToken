from __future__ import annotations

from freetoken.server.reasoning_parser import ReasoningParser


def _stream(parser: ReasoningParser, chunks):
    reasoning, content = "", ""
    for chunk in chunks:
        r, c = parser.parse_stream_chunk(chunk)
        reasoning += r
        content += c
    fr, fc = parser.flush()
    reasoning += fr
    content += fc
    return reasoning, content


# --------------------------------------------------------------- gpt-oss harmony
ANALYSIS_FINAL = (
    "<|channel|>analysis<|message|>The user says hi. Greet back.<|end|>"
    "<|start|>assistant<|channel|>final<|message|>Hello! How can I help you today?"
)


def test_harmony_non_stream_splits_analysis_and_final():
    parser = ReasoningParser("gpt_oss")
    reasoning, content = parser.parse_non_stream(ANALYSIS_FINAL)
    assert reasoning == "The user says hi. Greet back."
    assert content == "Hello! How can I help you today?"
    assert "<|channel|>" not in content


def test_harmony_non_stream_analysis_only():
    parser = ReasoningParser("gpt_oss")
    reasoning, content = parser.parse_non_stream(
        "<|channel|>analysis<|message|>still thinking"
    )
    assert reasoning == "still thinking"
    assert content == ""


def test_harmony_non_stream_final_only():
    parser = ReasoningParser("gpt_oss")
    reasoning, content = parser.parse_non_stream(
        "<|channel|>final<|message|>Just the answer.<|return|>"
    )
    assert reasoning == ""
    assert content == "Just the answer."


def test_harmony_non_stream_preserves_commentary_tool_block_verbatim():
    text = (
        "<|channel|>analysis<|message|>need weather<|end|>"
        '<|start|>assistant<|channel|>commentary to=functions.get_weather '
        '<|message|>{"city":"Paris"}<|call|>'
    )
    parser = ReasoningParser("gpt_oss")
    reasoning, content = parser.parse_non_stream(text)
    assert reasoning == "need weather"
    # The whole commentary block (markers included) survives for the tool parser.
    assert content.startswith("<|channel|>commentary to=functions.get_weather")
    assert '{"city":"Paris"}' in content
    # The closing marker must be present verbatim so the tool parser sees it.
    assert content.endswith("<|call|>")


def test_harmony_non_stream_passthrough_when_no_channels():
    parser = ReasoningParser("gpt_oss")
    reasoning, content = parser.parse_non_stream("plain text, no channels")
    assert reasoning == ""
    assert content == "plain text, no channels"


def test_harmony_streaming_splits_marker_across_chunks():
    parser = ReasoningParser("gpt_oss")
    # Deliberately split inside the markers and bodies.
    chunks = [
        "<|chan",
        "nel|>analysis<|mess",
        "age|>think a<|e",
        "nd|><|start|>assistant<|channel|>fin",
        "al<|message|>Ans",
        "wer.",
    ]
    reasoning, content = _stream(parser, chunks)
    assert reasoning == "think a"
    assert content == "Answer."


# ---------------------------------------------------------------- think family
import pytest


@pytest.mark.parametrize("name", ["qwen3", "glm", "minimax", "mimo_v2"])
def test_think_non_stream_splits_reasoning(name):
    parser = ReasoningParser(name, force_reasoning=False)
    reasoning, content = parser.parse_non_stream("<think>weigh options</think>final answer")
    assert reasoning == "weigh options"
    assert content == "final answer"


@pytest.mark.parametrize("name", ["qwen3", "glm", "minimax", "mimo_v2"])
def test_think_implicit_only_closing_tag(name):
    # Implicit-think: template injected the opening <think>; output starts inside.
    parser = ReasoningParser(name, force_reasoning=True)
    reasoning, content = parser.parse_non_stream("reasoning here</think>the answer")
    assert reasoning == "reasoning here"
    assert content == "the answer"


@pytest.mark.parametrize("name", ["qwen3", "glm", "minimax", "mimo_v2"])
def test_think_no_reasoning_passthrough(name):
    parser = ReasoningParser(name, force_reasoning=False)
    reasoning, content = parser.parse_non_stream("just an answer")
    assert reasoning == ""
    assert content == "just an answer"


def test_mimo_missing_think_closer_keeps_tool_envelope_in_content():
    text = "<think>planning the call<tool_call><function=read></function></tool_call>"
    parser = ReasoningParser("mimo_v2", force_reasoning=False)
    assert parser.parse_non_stream(text) == (
        "planning the call",
        "<tool_call><function=read></function></tool_call>",
    )

    parser = ReasoningParser("mimo_v2", force_reasoning=False)
    assert _stream(
        parser,
        ["<think>planning", " the call<tool_", "call>", "<function=read></function></tool_call>"],
    ) == (
        "planning the call",
        "<tool_call><function=read></function></tool_call>",
    )


def test_mimo_truncated_reasoning_flushes_to_reasoning():
    parser = ReasoningParser("mimo_v2", force_reasoning=False)
    assert _stream(parser, ["<think>unfinished thought</th"]) == (
        "unfinished thought</th",
        "",
    )


# --------------------------------------------- DR-02: tool args are opaque
from freetoken.server.reasoning_parser import (  # noqa: E402
    THINK_END_TOKEN as _T_CLOSE,
    THINK_START_TOKEN as _T_OPEN,
)

_DR02_ARG = f"a{_T_CLOSE}b{_T_OPEN}c"
_DR02_ENVELOPE = (
    "<tool_call><function=echo><parameter=text>" + _DR02_ARG + "</parameter></function></tool_call>"
)


@pytest.mark.parametrize("stream_reasoning", [True, False])
@pytest.mark.parametrize("tail_len", [0, 1024])
@pytest.mark.parametrize("force_reasoning", [True, False])
def test_vfa01_explicit_reasoning_closer_keeps_quoted_tool_envelope(stream_reasoning, tail_len, force_reasoning):
    quoted = "Example <tool_call><function=echo><parameter=text>x</parameter></function></tool_call> is not a call" + "a" * tail_len
    answer = "answer <think>ordinary marker</think>"
    text = ("" if force_reasoning else "<think>") + quoted + "</think>" + answer
    assert ReasoningParser("mimo_v2", force_reasoning=force_reasoning).parse_non_stream(text) == (quoted, answer)
    for cut in range(len(text) + 1):
        parser = ReasoningParser("mimo_v2", force_reasoning=force_reasoning, stream_reasoning=stream_reasoning)
        assert _stream(parser, [text[:cut], text[cut:]]) == (quoted, answer)


def test_dr02_literal_tool_closer_does_not_end_argument_opacity():
    envelope = "<tool_call><function=echo><parameter=text></function></tool_call><think>x</think></parameter></function></tool_call>"
    text = envelope + " ordinary <think>quoted</think>"
    assert ReasoningParser("mimo_v2", force_reasoning=False).parse_non_stream(text) == ("", text)
    for cut in range(len(text) + 1):
        assert _stream(ReasoningParser("mimo_v2", force_reasoning=False), [text[:cut], text[cut:]]) == ("", text)


def test_mimo_tool_arg_think_markers_stay_verbatim():
    # DR-02: think markers inside a verbatim string argument are data, not
    # protocol -- the envelope must reach the tool parser byte-identical.
    parser = ReasoningParser("mimo_v2", force_reasoning=False)
    assert parser.parse_non_stream(_DR02_ENVELOPE) == ("", _DR02_ENVELOPE)


def test_mimo_tool_arg_think_markers_streaming():
    parser = ReasoningParser("mimo_v2", force_reasoning=False)
    chunks = [_DR02_ENVELOPE[i : i + 7] for i in range(0, len(_DR02_ENVELOPE), 7)]
    assert _stream(parser, chunks) == ("", _DR02_ENVELOPE)


def test_mimo_reasoning_then_tool_arg_think_markers():
    text = f"{_T_OPEN}I'll echo it verbatim{_T_CLOSE}" + _DR02_ENVELOPE
    parser = ReasoningParser("mimo_v2", force_reasoning=False)
    assert parser.parse_non_stream(text) == ("I'll echo it verbatim", _DR02_ENVELOPE)
    parser = ReasoningParser("mimo_v2", force_reasoning=False)
    chunks = [text[i : i + 9] for i in range(0, len(text), 9)]
    assert _stream(parser, chunks) == ("I'll echo it verbatim", _DR02_ENVELOPE)


def test_mimo_explicit_closer_reclaims_quoted_envelope():
    # Inside reasoning, a real closer wins over the missing-closer fallback.
    text = f"{_T_OPEN}the tag " + _DR02_ENVELOPE
    parser = ReasoningParser("mimo_v2", force_reasoning=False)
    expected = ("the tag " + _DR02_ENVELOPE.split(_T_CLOSE, 1)[0], _DR02_ENVELOPE.split(_T_CLOSE, 1)[1])
    assert parser.parse_non_stream(text) == expected
    parser = ReasoningParser("mimo_v2", force_reasoning=False)
    chunks = [text[i : i + 9] for i in range(0, len(text), 9)]
    assert _stream(parser, chunks) == expected


def test_mimo_tool_arg_chunk_cuts_inside_think_markers():
    # Chunk cuts land before/inside the think markers INSIDE the argument value.
    cuts = sorted(
        [
            _DR02_ENVELOPE.index(_T_OPEN) + 3,  # inside the opener
            _DR02_ENVELOPE.index(_T_CLOSE) + len(_T_CLOSE) // 2,  # inside closer
        ]
    )
    bounds = [0] + cuts + [len(_DR02_ENVELOPE)]
    chunks = [_DR02_ENVELOPE[a:b] for a, b in zip(bounds, bounds[1:])]
    parser = ReasoningParser("mimo_v2", force_reasoning=False)
    assert _stream(parser, chunks) == ("", _DR02_ENVELOPE)


# ----------------------------------------------------------------------- gemma
def test_gemma_thought_split():
    parser = ReasoningParser("gemma4", force_reasoning=True)
    reasoning, content = parser.parse_non_stream("my private thought<channel|>visible answer")
    assert reasoning == "my private thought"
    assert content == "visible answer"


# ------------------------------------------------------- build_reasoning_parser
from types import SimpleNamespace

from freetoken.server.reasoning_parser import build_reasoning_parser


def test_build_reasoning_parser_returns_none_when_unset():
    assert build_reasoning_parser(SimpleNamespace(reasoning_parser=None), True) is None


def test_build_reasoning_parser_builds_named_parser():
    parser = build_reasoning_parser(SimpleNamespace(reasoning_parser="gpt_oss"), False)
    assert parser is not None
    reasoning, content = parser.parse_non_stream(ANALYSIS_FINAL)
    assert reasoning == "The user says hi. Greet back."
