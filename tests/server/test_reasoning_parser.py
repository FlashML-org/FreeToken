"""Character-boundary contracts for every reasoning detector and template mode."""
from types import SimpleNamespace

import pytest

from freetoken.server.generation import GenSpec, _make_reasoning_parser
from freetoken.server.reasoning_parser import BaseReasoningParser, ReasoningParser


class _ReasoningContract:
    name: str
    opener: str
    closer: str
    content_opener = ""

    def parser(self):
        return ReasoningParser(self.name, force_reasoning=False)

    def stream(self, text):
        parser = self.parser()
        parts = [parser.parse_stream_chunk(char) for char in text]
        parts.append(parser.flush())
        assert parser.flush() == ("", "")
        return tuple("".join(pair[i] for pair in parts) for i in (0, 1))

    def test_complete_reasoning(self):
        text = self.opener + "consider" + self.closer + self.content_opener + "answer"
        assert self.parser().parse_non_stream(text) == ("consider", "answer")

    def test_every_character_boundary(self):
        text = self.opener + "consider" + self.closer + self.content_opener + "answer"
        assert self.stream(text) == ("consider", "answer")

    def test_truncated_reasoning_flush(self):
        # Reasoning detectors expose flush(), not recover_truncated_call(). EOS
        # inside an open reasoning body must retain it in the reasoning channel.
        assert self.stream(self.opener + "partial thought") == ("partial thought", "")


class TestDeepSeekV32ReasoningParser(_ReasoningContract):
    name = "deepseekv32"
    opener = "<think>"
    closer = "</think>"


class TestThinkReasoningParser(_ReasoningContract):
    name = "qwen3"
    opener = "<think>"
    closer = "</think>"


class TestMiniMaxM3ReasoningParser(_ReasoningContract):
    name = "minimax_m3"
    opener = "<mm:think>"
    closer = "</mm:think>"


class TestGemmaThoughtReasoningParser(_ReasoningContract):
    name = "gemma4"
    opener = "<|channel>thought\n"
    closer = "<channel|>"


class TestGptOssHarmonyReasoningParser(_ReasoningContract):
    name = "gpt_oss"
    opener = "<|channel|>analysis<|message|>"
    closer = "<|end|>"
    content_opener = "<|start|>assistant<|channel|>final<|message|>"


class TestMuseGlimmerReasoningParser(_ReasoningContract):
    name = "muse_glimmer"
    opener = "<|start|>assistant to=self<|message|>"
    closer = "<|eom|>"
    content_opener = "<|start|>assistant to=user<|message|>"


def test_contract_classes_cover_every_reasoning_detector():
    covered = {type(cls().parser().detector) for cls in _ReasoningContract.__subclasses__()}
    assert covered == set(BaseReasoningParser.__subclasses__())
    assert covered == set(ReasoningParser.ReasoningParserEnum.values())


@pytest.mark.parametrize("tools", [None, [{"type": "function", "function": {"name": "read"}}]])
@pytest.mark.parametrize(("name", "kwargs", "forced"), [
    ("qwen3", {}, True),
    ("qwen3", {"enable_thinking": True}, True),
    ("qwen3", {"enable_thinking": False}, False),
    ("glm", {}, True),
    ("glm", {"enable_thinking": True}, True),
    ("glm", {"enable_thinking": False}, False),
    ("gemma4", {}, False),
    ("gemma4", {"thinking_mode": "thinking"}, True),
    ("gemma4", {"enable_thinking": True}, True),
    ("gemma4", {"thinking": True}, True),
    ("gemma4", {"thinking_mode": "chat", "enable_thinking": False}, False),
    ("minimax_m3", {}, False),
    ("minimax_m3", {"thinking_mode": "adaptive"}, False),
    ("minimax_m3", {"thinking_mode": "disabled"}, False),
    ("minimax_m3", {"thinking_mode": "enabled"}, True),
])
def test_generation_force_reasoning_matches_template(name, kwargs, forced, tools):
    spec = GenSpec(messages=[], sampling_params=None, chat_template_kwargs=kwargs,
                   template_tools=tools, parser_tools=tools)
    parser = _make_reasoning_parser(spec, SimpleNamespace(config=SimpleNamespace(reasoning_parser=name)))
    # Observe routing, not an internal flag: tools must not turn a disabled
    # template's ordinary response into invisible reasoning.
    assert parser.parse_non_stream("probe") == (("probe", "") if forced else ("", "probe"))
