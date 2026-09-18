from __future__ import annotations

import json

import pytest

from freetoken.server.function_call_parser import FunctionCallParser, SUPPORTED_TOOL_CALL_PARSERS


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]

OPENCODE_TOOLS = [
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


@pytest.mark.parametrize("parser_name", SUPPORTED_TOOL_CALL_PARSERS)
def test_supported_parser_names_instantiate(parser_name):
    parser = FunctionCallParser(TOOLS, tool_call_parser=parser_name)

    result = parser.parse_non_stream("plain text")

    assert result.normal_text == "plain text"
    assert result.calls == []


def test_mistral_parser_accepts_tool_calls_tag():
    parser = FunctionCallParser(TOOLS, tool_call_parser="mistral")

    result = parser.parse_non_stream(
        '[TOOL_CALLS] [{"name":"get_weather","arguments":{"city":"Paris"}}]'
    )

    assert result.normal_text == ""
    assert len(result.calls) == 1
    assert result.calls[0].name == "get_weather"
    assert json.loads(result.calls[0].parameters) == {"city": "Paris"}


@pytest.mark.parametrize(
    ("parser_name", "text"),
    [
        (
            "mistral",
            '[TOOL_CALLS] [{"name":"get_weather","arguments":{"city":"Paris"}}]',
        ),
        (
            "qwen25",
            '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
        ),
        (
            "llama3",
            '<|python_tag|>{"name":"get_weather","arguments":{"city":"Paris"}}',
        ),
    ],
)
def test_base_json_parsers_use_declared_tool_index(parser_name, text):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "other_tool",
                "parameters": {"type": "object"},
            },
        },
        TOOLS[0],
    ]
    parser = FunctionCallParser(tools, tool_call_parser=parser_name)

    result = parser.parse_non_stream(text)

    assert len(result.calls) == 1
    assert result.calls[0].name == "get_weather"
    assert result.calls[0].tool_index == 1


def test_base_json_parsers_forward_unknown_tools_by_default():
    parser = FunctionCallParser(TOOLS, tool_call_parser="mistral")

    result = parser.parse_non_stream(
        '[TOOL_CALLS] [{"name":"not_declared","arguments":{"city":"Paris"}}]'
    )

    assert len(result.calls) == 1
    assert result.calls[0].tool_index == -1
    assert result.calls[0].name == "not_declared"
    assert json.loads(result.calls[0].parameters) == {"city": "Paris"}


@pytest.mark.parametrize(
    "text",
    [
        '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
    ],
)
def test_parser_accepts_common_tagged_tool_call_shapes(text):
    parser = FunctionCallParser(TOOLS, tool_call_parser="qwen25")

    result = parser.parse_non_stream(text)

    assert len(result.calls) == 1
    assert result.calls[0].name == "get_weather"
    assert json.loads(result.calls[0].parameters) == {"city": "Paris"}


def test_gemma4_parser_accepts_compact_tool_call_shape():
    parser = FunctionCallParser(TOOLS, tool_call_parser="gemma4")

    result = parser.parse_non_stream('<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>')

    assert len(result.calls) == 1
    assert result.calls[0].name == "get_weather"
    assert json.loads(result.calls[0].parameters) == {"city": "Paris"}


def test_gemma4_parser_accepts_namespaced_tool_name():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "superpowers:using_superpowers",
                "parameters": {"type": "object"},
            },
        }
    ]
    parser = FunctionCallParser(tools, tool_call_parser="gemma4")

    result = parser.parse_non_stream(
        "<|tool_call>call:superpowers:using_superpowers{}<tool_call|>"
    )

    assert result.normal_text == ""
    assert len(result.calls) == 1
    assert result.calls[0].name == "superpowers:using_superpowers"
    assert json.loads(result.calls[0].parameters) == {}


def test_gemma4_parser_forwards_namespaced_skill_without_declared_tool():
    parser = FunctionCallParser(TOOLS, tool_call_parser="gemma4")

    result = parser.parse_non_stream(
        "<|tool_call>call:superpowers:using_superpowers{}<tool_call|>"
    )

    assert result.normal_text == ""
    assert len(result.calls) == 1
    assert result.calls[0].name == "superpowers:using_superpowers"
    assert json.loads(result.calls[0].parameters) == {}


def test_gpt_oss_parser_accepts_namespaced_tool_name():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "superpowers:using_superpowers",
                "parameters": {"type": "object"},
            },
        }
    ]
    parser = FunctionCallParser(tools, tool_call_parser="gpt_oss")

    result = parser.parse_non_stream(
        "<|start|>assistant<|channel|>commentary "
        "to=functions.superpowers:using_superpowers <|constrain|>json<|message|>{}<|end|>"
    )

    assert result.normal_text == ""
    assert len(result.calls) == 1
    assert result.calls[0].name == "superpowers:using_superpowers"
    assert json.loads(result.calls[0].parameters) == {}


@pytest.mark.parametrize(
    ("parser_name", "text", "expected_name", "expected_args"),
    [
        (
            "qwen3_coder",
            "<tool_call><function=read><parameter=filePath>/tmp/test_calc.py</parameter></function></tool_call>",
            "read",
            {"filePath": "/tmp/test_calc.py"},
        ),
        (
            "glm47",
            "<tool_call>read<arg_key>filePath</arg_key><arg_value>/tmp/test_calc.py</arg_value></tool_call>",
            "read",
            {"filePath": "/tmp/test_calc.py"},
        ),
        (
            "minimax",
            "<minimax:tool_call><invoke name=\"read\"><parameter name=\"filePath\">"
            "/tmp/test_calc.py</parameter></invoke></minimax:tool_call>",
            "read",
            {"filePath": "/tmp/test_calc.py"},
        ),
        (
            "gpt_oss",
            "<|channel|>analysis<|message|>Need files.<|end|><|start|>assistant"
            "<|channel|>commentary to=functions.glob <|constrain|>json<|message|>"
            "{\"pattern\":\"**/*.py\",\"path\":\"/tmp/ws\"}",
            "glob",
            {"pattern": "**/*.py", "path": "/tmp/ws"},
        ),
        (
            "deepseekv32",
            "<｜DSML｜function_calls><｜DSML｜invoke name=\"read\">"
            "<｜DSML｜parameter name=\"filePath\" string=\"true\">/tmp/test_calc.py</｜DSML｜parameter>"
            "</｜DSML｜invoke></｜DSML｜function_calls>",
            "read",
            {"filePath": "/tmp/test_calc.py"},
        ),
        (
            "deepseekv32",
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"read\">"
            "<｜DSML｜parameter name=\"filePath\">/tmp/test_calc.py</｜DSML｜parameter>"
            "</｜DSML｜invoke></｜DSML｜tool_calls>",
            "read",
            {"filePath": "/tmp/test_calc.py"},
        ),
    ],
)
def test_parser_accepts_family_specific_tool_call_shapes(
    parser_name, text, expected_name, expected_args
):
    parser = FunctionCallParser(OPENCODE_TOOLS, tool_call_parser=parser_name)

    result = parser.parse_non_stream(text)

    assert result.normal_text in ("", "Need files.")
    assert len(result.calls) == 1
    assert result.calls[0].name == expected_name
    assert json.loads(result.calls[0].parameters) == expected_args


# --------------------------------------------------------------------------- #
# Streaming incremental parsing (parse_stream_chunk)
# --------------------------------------------------------------------------- #
def _feed(parser, chunks):
    """Feed chunks through the streaming parser; return (per-chunk normal texts, calls)."""
    texts, calls = [], []
    for chunk in chunks:
        normal, chunk_calls = parser.parse_stream_chunk(chunk)
        texts.append(normal)
        calls.extend(chunk_calls)
    return texts, calls


@pytest.mark.parametrize("parser_name", ["qwen25", "glm47", "gemma4", "minimax", "deepseekv32", "qwen3_coder"])
def test_streaming_plain_text_releases_per_chunk(parser_name):
    # A pure-text response must stream out chunk by chunk, not buffer to the end.
    parser = FunctionCallParser(TOOLS, tool_call_parser=parser_name)
    texts, calls = _feed(parser, ["Hello ", "world."])
    assert texts == ["Hello ", "world."]
    assert calls == []
    assert parser.finish_stream() == ""


def test_streaming_partial_tag_holdback_then_release():
    # Text held back as a suspected tag prefix must be released once disambiguated.
    parser = FunctionCallParser(TOOLS, tool_call_parser="qwen25")
    texts, _ = _feed(parser, ["Hi <", "there"])
    assert "".join(texts) + parser.finish_stream() == "Hi <there"


def test_dsv32_streaming_partial_prefix_not_dropped():
    # Regression: the non-tool branch used to clear the whole buffer but return only
    # the newest chunk, dropping text held back as a suspected partial bot_token.
    parser = FunctionCallParser(TOOLS, tool_call_parser="deepseekv32")
    texts, calls = _feed(parser, ["a<", "b"])
    assert "".join(texts) + parser.finish_stream() == "a<b"
    assert calls == []


def test_streaming_finish_stream_releases_held_tail():
    parser = FunctionCallParser(TOOLS, tool_call_parser="qwen25")
    texts, calls = _feed(parser, ["text <tool"])
    assert calls == []
    assert "".join(texts) + parser.finish_stream() == "text <tool"


def test_dsv32_streaming_multi_param_args_prefix_stable():
    # The DSML streaming state machine emits prefix-stable fragments (vLLM-style):
    # '{"key":"' at parameter open, escaped value chars, '"' at close, '}' at
    # invoke close — concatenation IS the final arguments JSON.
    parser = FunctionCallParser(OPENCODE_TOOLS, tool_call_parser="deepseekv32")
    assert parser.args_fragments_prefix_stable() is True
    block = (
        "<｜DSML｜function_calls>\n"
        '<｜DSML｜invoke name="glob">\n'
        '<｜DSML｜parameter name="pattern" string="true">*.py</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="path" string="true">/src</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜function_calls>"
    )
    chunks = [block[i : i + 7] for i in range(0, len(block), 7)]
    _, calls = _feed(parser, chunks)
    named = [c for c in calls if c.name]
    assert len(named) == 1 and named[0].name == "glob"
    joined = "".join(c.parameters for c in calls if c.name is None)
    assert json.loads(joined) == {"pattern": "*.py", "path": "/src"}
    # multiple argument fragments streamed mid-call, not one blob at close
    assert sum(1 for c in calls if c.name is None) >= 4
    # detector parse state agrees (used as the truncation fallback)
    assert json.loads(parser.unstreamed_arguments(named[0].tool_index)) == {
        "pattern": "*.py",
        "path": "/src",
    }


def test_streaming_support_flags():
    # Every registered detector is incremental-safe (buffered fallback remains as
    # the escape hatch for future formats, exercised via monkeypatch in
    # test_streaming_model_matrix.py::test_non_streaming_detector_falls_back_to_buffered_parse).
    for name in SUPPORTED_TOOL_CALL_PARSERS:
        assert FunctionCallParser(TOOLS, tool_call_parser=name).supports_streaming() is True


# One class per concrete detector makes format coverage visible in pytest output.
# A shared contract avoids accidentally giving a newly added format weaker checks.
class _DetectorContract:
    parser_name: str
    block: str
    truncate_before: str
    buffered_recovery = False

    def parser(self):
        return FunctionCallParser(OPENCODE_TOOLS, self.parser_name)

    def test_complete_call(self):
        result = self.parser().parse_non_stream(self.block)
        assert result.normal_text == ""
        assert len(result.calls) == 1
        assert result.calls[0].name == "read"
        assert json.loads(result.calls[0].parameters) == {"filePath": "/tmp/test_calc.py"}

    def test_every_character_boundary(self):
        parser = self.parser()
        texts, calls = _feed(parser, self.block)
        assert "".join(texts) + parser.finish_stream() == ""
        assert [call.name for call in calls if call.name] == ["read"]
        assert json.loads("".join(call.parameters for call in calls)) == {
            "filePath": "/tmp/test_calc.py"
        }
        # A completed call must never be duplicated by the end-of-stream recovery.
        assert parser.recover_truncated_call() == []

    def test_truncated_mid_call_recovery(self):
        parser = self.parser()
        cut = self.block.rindex(self.truncate_before)
        texts, calls = _feed(parser, self.block[:cut])
        recovered = parser.recover_truncated_call()
        assert bool(recovered) is self.buffered_recovery
        # Incremental detectors have already consumed the opener and emitted the
        # call identity. Their recovery method MUST NOT invent a duplicate call:
        # generation.py recovers missing arguments from their partial-parse ledger.
        # Buffered M3 instead recovers a complete call from the retained markup.
        all_calls = calls + recovered
        assert [call.name for call in all_calls if call.name] == ["read"]
        arguments = "".join(call.parameters for call in all_calls)
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            parsed = json.loads(parser.unstreamed_arguments(0))
        assert parsed == {"filePath": "/tmp/test_calc.py"}
        assert parser.recover_truncated_call() == []
        assert "".join(texts) + parser.finish_stream() == ""


class TestQwen25Detector(_DetectorContract):
    parser_name = 'qwen25'
    block = '<tool_call>{"name": "read", "arguments": {"filePath": "/tmp/test_calc.py"}}</tool_call>'
    truncate_before = '</tool_call>'


class TestMistralDetector(_DetectorContract):
    parser_name = 'mistral'
    block = '[TOOL_CALLS] [{"name": "read", "arguments": {"filePath": "/tmp/test_calc.py"}}]'
    truncate_before = '}]'


class TestLlama32Detector(_DetectorContract):
    parser_name = 'llama3'
    block = '<|python_tag|>{"name": "read", "arguments": {"filePath": "/tmp/test_calc.py"}}'
    truncate_before = '}'


class TestGlm47Detector(_DetectorContract):
    parser_name = 'glm47'
    block = '<tool_call>read<arg_key>filePath</arg_key><arg_value>/tmp/test_calc.py</arg_value></tool_call>'
    truncate_before = '</tool_call>'


class TestDeepSeekV32Detector(_DetectorContract):
    parser_name = 'deepseekv32'
    block = '<｜DSML｜function_calls><｜DSML｜invoke name="read"><｜DSML｜parameter name="filePath" string="true">/tmp/test_calc.py</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜function_calls>'
    truncate_before = '</｜DSML｜invoke>'


class TestQwen3CoderDetector(_DetectorContract):
    parser_name = 'qwen3_coder'
    block = '<tool_call><function=read><parameter=filePath>/tmp/test_calc.py</parameter></function></tool_call>'
    truncate_before = '</function>'


class TestGemma4Detector(_DetectorContract):
    parser_name = 'gemma4'
    block = '<|tool_call>call:read{filePath:<|"|>/tmp/test_calc.py<|"|>}<tool_call|>'
    truncate_before = '<tool_call|>'


class TestMiniMaxDetector(_DetectorContract):
    parser_name = 'minimax'
    block = '<minimax:tool_call><invoke name="read"><parameter name="filePath">/tmp/test_calc.py</parameter></invoke></minimax:tool_call>'
    truncate_before = '</invoke>'


class TestMiniMaxM3Detector(_DetectorContract):
    parser_name = 'minimax_m3'
    block = ']<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name="read">]<]minimax[>[<filePath>/tmp/test_calc.py]<]minimax[>[</filePath>]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>'
    truncate_before = ']<]minimax[>[</invoke>'
    buffered_recovery = True


class TestGptOssDetector(_DetectorContract):
    parser_name = 'gpt_oss'
    block = '<|start|>assistant<|channel|>commentary to=functions.read <|constrain|>json<|message|>{"filePath": "/tmp/test_calc.py"}<|end|>'
    truncate_before = '<|end|>'


class TestMuseGlimmerDetector(_DetectorContract):
    parser_name = 'muse_glimmer'
    block = '<|start|>assistant to=read<|message|><atem:function_calls>\n<atem:invoke name="read">\n<atem:parameter name="filePath">/tmp/test_calc.py</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eot|>'
    truncate_before = '</atem:invoke>'


def test_contract_classes_cover_every_concrete_detector():
    from freetoken.server.function_call_parser import BaseFormatDetector

    covered = {
        type(cls().parser().detector)
        for cls in _DetectorContract.__subclasses__()
    }
    assert covered == set(BaseFormatDetector.__subclasses__())
    assert covered == set(FunctionCallParser.ToolCallParserEnum.values())
