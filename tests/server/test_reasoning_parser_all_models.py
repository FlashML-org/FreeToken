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


@pytest.mark.parametrize("name", ["qwen3", "glm", "minimax", "lfm2"])
def test_think_non_stream_splits_reasoning(name):
    parser = ReasoningParser(name, force_reasoning=False)
    reasoning, content = parser.parse_non_stream("<think>weigh options</think>final answer")
    assert reasoning == "weigh options"
    assert content == "final answer"


@pytest.mark.parametrize("name", ["qwen3", "glm", "minimax", "lfm2"])
def test_think_implicit_only_closing_tag(name):
    # Implicit-think: template injected the opening <think>; output starts inside.
    parser = ReasoningParser(name, force_reasoning=True)
    reasoning, content = parser.parse_non_stream("reasoning here</think>the answer")
    assert reasoning == "reasoning here"
    assert content == "the answer"


@pytest.mark.parametrize("name", ["qwen3", "glm", "minimax", "lfm2"])
def test_think_no_reasoning_passthrough(name):
    parser = ReasoningParser(name, force_reasoning=False)
    reasoning, content = parser.parse_non_stream("just an answer")
    assert reasoning == ""
    assert content == "just an answer"


def test_lfm2_builds_forced_because_its_template_pre_opens_think():
    """LFM2.5's generation prompt always ends with <think> (no thinking-mode variable), so the
    frontend must force the parser -- otherwise the chain of thought leaks into content."""
    from freetoken.server.generation import _make_reasoning_parser

    spec = SimpleNamespace(chat_template_kwargs={}, template_tools=None)
    parser = _make_reasoning_parser(spec, SimpleNamespace(config=SimpleNamespace(reasoning_parser="lfm2")))
    assert parser is not None
    reasoning, content = parser.parse_non_stream("weigh options</think>final answer")
    assert reasoning == "weigh options"
    assert content == "final answer"


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
