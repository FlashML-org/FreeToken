"""`--tool-call-parser auto` / `--reasoning-parser auto`: architecture -> parser name.

Three tables are maintained independently -- the model registry, the substring cascade in
``server/args.py``, and the two parser factories -- and nothing links them. Add a model family and
forget the cascade, and the model serves perfectly: it just stops emitting tool calls and stops
separating out its thinking, because it silently fell through to the generic default. Every other
parser test still passes, since they all start from a parser name that is already correct.

So this drives the *live* registry rather than a list: a newly registered architecture is covered
the moment it is added, and has to be dispositioned here to stay green.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from freetoken.models.register import _MODEL_REGISTRY
from freetoken.server.args import parse_args
from freetoken.server.function_call_parser import FunctionCallParser
from freetoken.server.reasoning_parser import ReasoningParser

ARCHITECTURES = sorted(_MODEL_REGISTRY)

# The cascade also reads the model path, so the path here is deliberately anonymous: it must
# resolve off the checkpoint's own architecture, not off a directory somebody happened to name.
ANON_PATH = "/models/anon"

# Families with no thinking format of their own. Everything else must resolve to a real reasoning
# parser, and a new architecture landing here is the bug this file exists to catch.
NO_REASONING_FORMAT = {
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "Mistral3ForConditionalGeneration",
    "Qwen2ForCausalLM",
}

# `llama3` is the end of the cascade -- the answer when nothing matched.
GENERIC_TOOL_CALL_FALLBACK = "llama3"
NO_DEDICATED_TOOL_FORMAT = {"LlamaForCausalLM"}


class _Config:
    def __init__(self, data: dict) -> None:
        self._data = data

    def to_dict(self) -> dict:
        return self._data


def _inferred(architecture: str) -> tuple[str, str | None]:
    """(tool_call_parser, reasoning_parser) that `auto` picks for this architecture."""
    config = _Config({"architectures": [architecture], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: config):
        args, _run_shell = parse_args(["--model", ANON_PATH])
    return args.tool_call_parser, args.reasoning_parser


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_inferred_parser_names_are_names_the_factories_know(architecture):
    """A name the cascade invents but no factory can build fails at request time, not at boot."""
    tool_call, reasoning = _inferred(architecture)

    assert tool_call in FunctionCallParser.ToolCallParserEnum, tool_call
    if reasoning is not None:
        assert reasoning in ReasoningParser.ReasoningParserEnum, reasoning


def test_only_the_families_without_a_thinking_format_get_no_reasoning_parser():
    fell_through = {a for a in ARCHITECTURES if _inferred(a)[1] is None}
    assert fell_through == NO_REASONING_FORMAT


def test_only_the_families_without_a_tool_format_get_the_generic_fallback():
    fell_through = {
        a for a in ARCHITECTURES if _inferred(a)[0] == GENERIC_TOOL_CALL_FALLBACK
    }
    assert fell_through == NO_DEDICATED_TOOL_FORMAT


def test_qwen3_5_is_not_shadowed_by_the_generic_qwen_branch():
    """The cascade matches substrings in order, so the specific arm has to come first: a bare
    ``"qwen" -> qwen25`` reached earlier would swallow every later Qwen and lose its tool format."""
    assert _inferred("Qwen3_5MoeForConditionalGeneration")[0] == "qwen3_coder"
    assert _inferred("Qwen3_5ForConditionalGeneration")[0] == "qwen3_coder"
    assert _inferred("Qwen3MoeForCausalLM")[0] == "qwen25"


def test_an_explicit_choice_beats_inference():
    config = _Config({"architectures": ["DeepseekV4ForCausalLM"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: config):
        off, _ = parse_args(["--model", ANON_PATH, "--reasoning-parser", "off"])
        pinned, _ = parse_args(["--model", ANON_PATH, "--reasoning-parser", "qwen3"])
    assert off.reasoning_parser is None
    assert pinned.reasoning_parser == "qwen3"


def _qwen3_moe(model_path: str, *extra: str):
    """Parse ``model_path`` as ``Qwen3MoeForCausalLM``. The architecture is shared by
    the Instruct-2507 and Thinking-2507 checkpoints, so the path is the only signal."""
    config = _Config({"architectures": ["Qwen3MoeForCausalLM"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: config):
        return parse_args(["--model", model_path, *extra])


def test_qwen3_instruct_2507_does_not_get_a_reasoning_parser():
    """Instruct-2507 emits plain answers. Auto must not select qwen3, or the whole
    answer is labelled reasoning and message.content comes back empty."""
    for model_path in (
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4",
    ):
        args, _ = _qwen3_moe(model_path)
        assert args.reasoning_parser is None, model_path
        assert args.tool_call_parser == "qwen25", model_path


def test_qwen3_thinking_and_generic_checkpoints_still_select_qwen3():
    for model_path in (
        "Qwen/Qwen3-235B-A22B-Thinking-2507",
        "Qwen/Qwen3-32B",
    ):
        args, _ = _qwen3_moe(model_path)
        assert args.reasoning_parser == "qwen3", model_path
        assert args.tool_call_parser == "qwen25", model_path


def test_explicit_qwen3_overrides_instruct_2507():
    args, _ = _qwen3_moe(
        "Qwen/Qwen3-235B-A22B-Instruct-2507", "--reasoning-parser", "qwen3"
    )
    assert args.reasoning_parser == "qwen3"
    assert args.tool_call_parser == "qwen25"


def test_instruct_2507_text_stays_content_and_thinking_still_splits():
    """None skips the parser, so a plain completion is content. The Thinking
    checkpoint still opens inside a think block and splits on the closer."""
    from types import SimpleNamespace

    from freetoken.server.generation import _split_reasoning

    spec = SimpleNamespace(chat_template_kwargs={}, template_tools=None)
    plain = "Introduce yourself in three sentences."
    no_parser = SimpleNamespace(config=SimpleNamespace(reasoning_parser=None))
    assert _split_reasoning(plain, spec, no_parser) == ("", plain)

    thinking = SimpleNamespace(config=SimpleNamespace(reasoning_parser="qwen3"))
    assert _split_reasoning("reason</think>answer", spec, thinking) == ("reason", "answer")
