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


def test_mimo_tool_parser_resolves_schema_unions_and_string_null():
    tools = [{
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "score": {"type": "number"},
                    "cached": {"type": "boolean"},
                    "filters": {"type": "array"},
                    "metadata": {"type": "object"},
                    "nullable_int": {"type": ["integer", "null"]},
                    "nullable_bool": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
                    "nullable_array": {"oneOf": [{"type": "array"}, {"type": "null"}]},
                    "nullable_object": {"anyOf": [{"type": "object"}, {"type": "null"}]},
                    "nullable_number": {"type": ["number", "null"]},
                    "literal_null": {"type": "string"},
                    "actual_null": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                },
            },
        },
    }]
    block = (
        "<tool_call><function=lookup>"
        "<parameter=query>MiMo docs</parameter>"
        "<parameter=limit>3</parameter>"
        "<parameter=score>1.5</parameter>"
        "<parameter=cached>true</parameter>"
        '<parameter=filters>["recent", "public"]</parameter>'
        '<parameter=metadata>{"source":"api"}</parameter>'
        "<parameter=nullable_int>7</parameter>"
        "<parameter=nullable_bool>false</parameter>"
        "<parameter=nullable_array>[1,2]</parameter>"
        '<parameter=nullable_object>{"ok":true}</parameter>'
        "<parameter=nullable_number>1.25</parameter>"
        "<parameter=literal_null>null</parameter>"
        "<parameter=actual_null>null</parameter>"
        "</function></tool_call>"
    )
    result = FunctionCallParser(tools, "mimo_v2").parse_non_stream(
        "Before call. " + block + " After call."
    )

    assert result.normal_text == "Before call.  After call."
    assert len(result.calls) == 1
    assert result.calls[0].name == "lookup"
    assert json.loads(result.calls[0].parameters) == {
        "query": "MiMo docs",
        "limit": 3,
        "score": 1.5,
        "cached": True,
        "filters": ["recent", "public"],
        "metadata": {"source": "api"},
        "nullable_int": 7,
        "nullable_bool": False,
        "nullable_array": [1, 2],
        "nullable_object": {"ok": True},
        "nullable_number": 1.25,
        "literal_null": "null",
        "actual_null": None,
    }

    parser = FunctionCallParser(tools, "mimo_v2")
    emitted = []
    for i in range(0, len(block), 5):
        _, calls = parser.parse_stream_chunk(block[i : i + 5])
        emitted.extend(calls)
    streamed = json.loads("".join(call.parameters for call in emitted if call.name is None))
    assert streamed == json.loads(result.calls[0].parameters)


def test_mimo_tool_parser_streaming_schema_union_matches_one_shot():
    tools = [{
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
            },
        },
    }]
    block = "<tool_call><function=lookup><parameter=limit>3</parameter></function></tool_call>"
    parser = FunctionCallParser(tools, "mimo_v2")
    calls = []
    text = []
    for i in range(0, len(block), 3):
        normal, emitted = parser.parse_stream_chunk(block[i : i + 3])
        text.append(normal)
        calls.extend(emitted)
    assert "".join(text) == ""
    named = next(call for call in calls if call.name is not None)
    assert named.name == "lookup"
    assert json.loads("".join(call.parameters for call in calls if call.name is None)) == {
        "limit": 3
    }
    assert parser.finish_stream() == ""

    truncated = FunctionCallParser(tools, "qwen3_coder").parse_non_stream(
        "Checking. <tool_call><function=lookup><parameter=limit>3</parameter>"
    )
    assert truncated.normal_text == "Checking."
    assert len(truncated.calls) == 1
    assert truncated.calls[0].name == "lookup"
    assert json.loads(truncated.calls[0].parameters) == {"limit": 3}


def test_mimo_preserves_exact_string_values_while_qwen_keeps_legacy_trim():
    tools = [{
        "type": "function",
        "function": {
            "name": "echo",
            "parameters": {
                "type": "object",
                "properties": {
                    "leading": {"type": "string"},
                    "empty": {"type": "string"},
                    "special": {"type": "string"},
                    "json": {"type": "object"},
                },
            },
        },
    }]
    expected = {
        "leading": "\nline\n\n",
        "empty": "",
        "special": 'quote " slash\\ snow 雪 </function></tool_call>',
        "json": {"function": "</function>", "outer": "</tool_call>"},
    }
    block = (
        "<tool_call><function=echo>"
        "<parameter=leading>\nline\n\n</parameter>"
        "<parameter=empty></parameter>"
        '<parameter=special>quote " slash\\ snow 雪 </function></tool_call></parameter>'
        '<parameter=json>{"function":"</function>","outer":"</tool_call>"}</parameter>'
        "</function></tool_call>"
    )
    one_shot = FunctionCallParser(tools, "mimo_v2").parse_non_stream(block)
    assert json.loads(one_shot.calls[0].parameters) == expected

    streamed = FunctionCallParser(tools, "mimo_v2")
    chunks = []
    calls = []
    for i in range(0, len(block), 2):
        normal, emitted = streamed.parse_stream_chunk(block[i : i + 2])
        chunks.append(normal)
        calls.extend(emitted)
    streamed_args = json.loads(
        "".join(call.parameters for call in calls if call.name is None)
    )
    assert streamed_args == expected
    assert "".join(chunks) == ""

    qwen_block = (
        "<tool_call><function=echo><parameter=leading>\nline\n\n</parameter>"
        "</function></tool_call>"
    )
    qwen = FunctionCallParser(tools, "qwen3_coder").parse_non_stream(qwen_block)
    assert json.loads(qwen.calls[0].parameters) == {"leading": "line\n"}


def test_mimo_outer_close_only_at_end_of_stream_recovers_string_call():
    tools = [{
        "type": "function",
        "function": {
            "name": "echo",
            "parameters": {"properties": {"text": {"type": "string"}}},
        },
    }]
    wire = "<tool_call><function=echo><parameter=text>partial</tool_call>"
    parser = FunctionCallParser(tools, "mimo_v2")
    fragments = []
    for i in range(0, len(wire), 2):
        _, calls = parser.parse_stream_chunk(wire[i : i + 2])
        fragments.extend(call.parameters for call in calls if call.name is None)
    fragments.extend(call.parameters for call in parser.finalize_stream())
    assert json.loads("".join(fragments)) == {"text": "partial"}
    assert parser.finish_stream() == ""


def test_mimo_unknown_tool_with_missing_parameter_close_does_not_swallow_next_call(monkeypatch):
    monkeypatch.setattr("freetoken.server.function_call_parser.FORWARD_UNKNOWN_TOOLS", False)
    tools = [{
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {"properties": {"limit": {"type": "integer"}}},
        },
    }]
    wire = (
        "<tool_call><function=unknown><parameter=unused>ignored"
        "</function></tool_call>"
        "<tool_call><function=lookup><parameter=limit>3</parameter>"
        "</function></tool_call>"
    )
    parser = FunctionCallParser(tools, "mimo_v2")
    calls = []
    for i in range(0, len(wire), 2):
        _, emitted = parser.parse_stream_chunk(wire[i : i + 2])
        calls.extend(emitted)
    named = [call for call in calls if call.name is not None]
    assert [call.name for call in named] == ["lookup"]
    args = "".join(call.parameters for call in calls if call.name is None)
    assert json.loads(args) == {"limit": 3}


@pytest.mark.parametrize(
    ("parameter", "value", "expected"),
    [
        ("limit", "3</parameter>", {"limit": 3}),
        ("limit", "42", {"limit": 42}),
        ("enabled", "true", {"enabled": True}),
        ("nullable", "null", {"nullable": None}),
        # DR-03 preserve-raw: an invalid partial value keeps its raw text at EOS
        # (streaming finalize matches the one-shot $-anchor path).
        ("enabled", "tru", {"enabled": "tru"}),
        ("items", "[1,", {"items": "[1,"}),
        ("text", 'quote " slash\\ 雪', {"text": 'quote " slash\\ 雪'}),
    ],
)
def test_mimo_truncated_stream_finalizes_prefix_stable_json(parameter, value, expected):
    properties = {
        "limit": {"type": "integer"},
        "enabled": {"type": "boolean"},
        "nullable": {"type": ["integer", "null"]},
        "items": {"type": "array"},
        "text": {"type": "string"},
    }
    tools = [{
        "type": "function",
        "function": {"name": "demo", "parameters": {"properties": properties}},
    }]
    prefix = f"<tool_call><function=demo><parameter={parameter}>"
    parser = FunctionCallParser(tools, "mimo_v2")
    fragments = []
    for piece in (prefix, value):
        _, calls = parser.parse_stream_chunk(piece)
        fragments.extend(call.parameters for call in calls if call.name is None)
    fragments.extend(call.parameters for call in parser.finalize_stream())
    joined = "".join(fragments)
    assert joined.startswith("{")
    assert json.loads(joined) == expected


def test_minimax_m2_string_cutoff_finalizes_valid_json():
    tools = [{
        "type": "function",
        "function": {
            "name": "search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    }]
    wire = (
        "<minimax:tool_call><invoke name=\"search\"><parameter name=\"query\">"
        "partial query"
    )
    parser = FunctionCallParser(tools, "minimax")
    fragments = []
    for i in range(0, len(wire), 4):
        _, calls = parser.parse_stream_chunk(wire[i : i + 4])
        fragments.extend(call.parameters for call in calls if call.name is None)
    fragments.extend(call.parameters for call in parser.finalize_stream())
    assert json.loads("".join(fragments)) == {"query": "partial query"}


def test_minimax_m2_literal_invoke_closer_is_preserved_in_parameter():
    tools = [{
        "type": "function",
        "function": {
            "name": "search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    }]
    wire = (
        '<minimax:tool_call><invoke name="search"><parameter name="query">'
        "value </invoke> still value</parameter></invoke></minimax:tool_call>"
    )
    parser = FunctionCallParser(tools, "minimax")
    calls = []
    for i in range(0, len(wire), 3):
        _, emitted = parser.parse_stream_chunk(wire[i : i + 3])
        calls.extend(emitted)
    args = "".join(call.parameters for call in calls if call.name is None)
    assert json.loads(args) == {"query": "value </invoke> still value"}


# --------------------------------------------------------------------------- #
# DR-01: literal closers inside a string value must survive chunk boundaries
# --------------------------------------------------------------------------- #
_DR01_OPENER = "<tool_call>"
_DR01_INVOKE_CLOSE = "</function>"
_DR01_OUTER_CLOSE = "</tool_call>"
_DR01_PARAM_CLOSE = "</parameter>"
_DR01_CLOSER = _DR01_INVOKE_CLOSE + _DR01_OUTER_CLOSE
_DR01_PREFIX = _DR01_OPENER + "<function=echo><parameter=text>"
_DR01_SUFFIX = _DR01_PARAM_CLOSE + _DR01_INVOKE_CLOSE + _DR01_OUTER_CLOSE


def _dr01_echo_tools():
    return [{
        "type": "function",
        "function": {
            "name": "echo",
            "parameters": {"properties": {"text": {"type": "string"}}},
        },
    }]


def _dr01_wire(tail_len):
    return _DR01_PREFIX + _DR01_CLOSER + "a" * tail_len + _DR01_SUFFIX


def _dr01_stream(parser, chunks):
    fragments, normal = [], []
    for chunk in chunks:
        text, emitted = parser.parse_stream_chunk(chunk)
        normal.append(text)
        fragments.extend(call.parameters for call in emitted if call.name is None)
    return "".join(fragments), "".join(normal)


@pytest.mark.parametrize("parser_name", ["mimo_v2", "qwen3_coder"])
def test_dr01_literal_closer_pair_inside_value_survives_chunking(parser_name):
    # Chunk 3 buffers >128 chars after a literal closer pair with no parameter
    # closer yet: the value must keep the pair, never close at it.
    chunks = [_DR01_PREFIX, _DR01_CLOSER, "a" * 129, _DR01_SUFFIX]
    parser = FunctionCallParser(_dr01_echo_tools(), parser_name)
    args, normal = _dr01_stream(parser, chunks)
    assert json.loads(args) == {"text": _DR01_CLOSER + "a" * 129}
    assert normal == ""  # value bytes must not leak out as ordinary content
    assert parser.finish_stream() == ""


@pytest.mark.parametrize("parser_name", ["mimo_v2", "qwen3_coder"])
def test_dr01_literal_closer_chunked_matches_whole_input(parser_name):
    wire = _dr01_wire(129)
    tools = _dr01_echo_tools()
    whole_args, _ = _dr01_stream(FunctionCallParser(tools, parser_name), [wire])
    chunked_args, _ = _dr01_stream(
        FunctionCallParser(tools, parser_name),
        [wire[i : i + 40] for i in range(0, len(wire), 40)],
    )
    assert json.loads(chunked_args) == json.loads(whole_args)
    if parser_name == "mimo_v2":  # its one-shot IS a whole-input streaming replay
        one_shot = FunctionCallParser(tools, parser_name).parse_non_stream(wire)
        assert json.loads(one_shot.calls[0].parameters) == json.loads(chunked_args)


@pytest.mark.parametrize("tail_len", [127, 128, 129, 200])
def test_dr01_literal_closer_value_survives_tail_lengths(tail_len):
    chunks = [_DR01_PREFIX, _DR01_CLOSER, "a" * tail_len, _DR01_SUFFIX]
    args, _ = _dr01_stream(FunctionCallParser(_dr01_echo_tools(), "mimo_v2"), chunks)
    assert json.loads(args) == {"text": _DR01_CLOSER + "a" * tail_len}


def _dr01_marker_cuts(tail_len):
    """Mid-marker split points: the literal invoke/outer closers in the value,
    then the real parameter closer, final invoke closer, final outer closer."""
    prefix_end = len(_DR01_PREFIX)
    invoke_len = len(_DR01_INVOKE_CLOSE)
    outer_len = len(_DR01_OUTER_CLOSE)
    param_len = len(_DR01_PARAM_CLOSE)
    pair_end = prefix_end + len(_DR01_CLOSER)
    pc_start = pair_end + tail_len
    pc_end = pc_start + param_len
    ic_end = pc_end + invoke_len
    return {
        "literal-invoke-closer": prefix_end + invoke_len // 2,
        "literal-outer-closer": pair_end - outer_len // 2,
        "parameter-closer": pc_start + param_len // 2,
        "final-invoke-closer": pc_end + invoke_len // 2,
        "final-outer-closer": ic_end + outer_len // 2,
    }


@pytest.mark.parametrize(
    "label",
    [
        "literal-invoke-closer",
        "literal-outer-closer",
        "parameter-closer",
        "final-invoke-closer",
        "final-outer-closer",
    ],
)
def test_dr01_cut_inside_markers_preserves_value(label):
    tail = 129
    wire = _dr01_wire(tail)
    cut = _dr01_marker_cuts(tail)[label]
    # A later chunk must leave the buffer holding the complete literal pair
    # plus >128 tail chars before the parameter closer completes: the trigger.
    pair_tail_end = len(_DR01_PREFIX) + len(_DR01_CLOSER) + tail
    pc_end = pair_tail_end + len(_DR01_PARAM_CLOSE)
    ic_end = pc_end + len(_DR01_INVOKE_CLOSE)
    chunks = [wire[:cut]] + [
        piece
        for piece in (
            wire[cut:pair_tail_end],
            wire[max(cut, pair_tail_end):pc_end],
            wire[max(cut, pc_end):ic_end],
            wire[max(cut, ic_end):],
        )
        if piece
    ]
    args, _ = _dr01_stream(FunctionCallParser(_dr01_echo_tools(), "mimo_v2"), chunks)
    assert json.loads(args) == {"text": _DR01_CLOSER + "a" * tail}


def test_dr01_literal_closer_in_value_then_real_second_call():
    second = (
        _DR01_OPENER + "<function=echo><parameter=text>second</parameter>"
        + _DR01_INVOKE_CLOSE + _DR01_OUTER_CLOSE
    )
    parser = FunctionCallParser(_dr01_echo_tools(), "mimo_v2")
    calls, args_by_tool = [], {}
    for chunk in [_DR01_PREFIX, _DR01_CLOSER, "a" * 129, _DR01_SUFFIX, second]:
        _, emitted = parser.parse_stream_chunk(chunk)
        calls.extend(emitted)
        for call in emitted:
            if call.name is None:
                args_by_tool[call.tool_index] = args_by_tool.get(call.tool_index, "") + call.parameters
    named = [call for call in calls if call.name is not None]
    assert [call.name for call in named] == ["echo", "echo"]
    assert [call.tool_index for call in named] == [0, 1]
    assert json.loads(args_by_tool[0]) == {"text": _DR01_CLOSER + "a" * 129}
    assert json.loads(args_by_tool[1]) == {"text": "second"}


def test_dr01_unterminated_value_end_of_stream_recovery_unchanged():
    # Pins current EOS policy: finalize_stream closes an unterminated string
    # value so the client's concatenated fragments stay valid JSON.
    parser = FunctionCallParser(_dr01_echo_tools(), "mimo_v2")
    wire = _DR01_PREFIX + "partial value"
    fragments = []
    for i in range(0, len(wire), 3):
        _, emitted = parser.parse_stream_chunk(wire[i : i + 3])
        fragments.extend(call.parameters for call in emitted if call.name is None)
    fragments.extend(call.parameters for call in parser.finalize_stream())
    assert json.loads("".join(fragments)) == {"text": "partial value"}
    assert parser.finish_stream() == ""


def test_dr01_real_closer_pair_without_parameter_close_eos_recovery_unchanged():
    # Pins current EOS policy: a missing parameter closer (malformed output)
    # still ends the value at the real invoke closer at end of stream.
    parser = FunctionCallParser(_dr01_echo_tools(), "mimo_v2")
    wire = _DR01_PREFIX + "value" + _DR01_CLOSER
    fragments = []
    for i in range(0, len(wire), 3):
        _, emitted = parser.parse_stream_chunk(wire[i : i + 3])
        fragments.extend(call.parameters for call in emitted if call.name is None)
    fragments.extend(call.parameters for call in parser.finalize_stream())
    assert json.loads("".join(fragments)) == {"text": "value"}
    assert parser.finish_stream() == ""


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


@pytest.mark.parametrize("parser_name", ["qwen25", "glm47", "gemma4", "minimax", "deepseekv32", "qwen3_coder", "mimo_v2"])
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


# --------------------------------------------------------------------------- #
# DR-03: typed rejection is preserved raw in BOTH delivery modes
#
# Streaming used to silently omit a parameter whose typed conversion failed,
# while the one-shot regex paths kept the raw value (_convert_param_value).
# Approved policy (Option A): the streaming emit sites always emit - the
# converted value on success, the raw text on failure - so both modes agree
# for every InvokeParamStreamMixin family (qwen3_coder, minimax), and the
# replay one-shot detectors (mimo_v2, muse_glimmer) inherit it automatically.
# --------------------------------------------------------------------------- #
DR03_FAMILIES = ("qwen3_coder", "minimax", "mimo_v2", "muse_glimmer")


def _dr03_tools(properties, required=None):
    schema = {"type": "object", "properties": properties}
    if required is not None:
        schema["required"] = required
    return [{"type": "function", "function": {"name": "demo", "parameters": schema}}]


def _dr03_wire(family, calls):
    """Build a wire input; ``calls`` is a list of tool calls, each a list of
    ``(parameter_name, raw_value)`` pairs."""
    if family in ("qwen3_coder", "mimo_v2"):
        return "".join(
            "<tool_call><function=demo>"
            + "".join(f"<parameter={key}>{value}</parameter>" for key, value in call)
            + "</function></tool_call>"
            for call in calls
        )
    if family == "minimax":
        return "".join(
            '<minimax:tool_call><invoke name="demo">'
            + "".join(f'<parameter name="{key}">{value}</parameter>' for key, value in call)
            + "</invoke></minimax:tool_call>"
            for call in calls
        )
    if family == "muse_glimmer":
        invokes = "".join(
            '<atem:invoke name="demo">'
            + "".join(
                f'<atem:parameter name="{key}">{value}</atem:parameter>'
                for key, value in call
            )
            + "</atem:invoke>"
            for call in calls
        )
        return (
            "<|start|>assistant to=demo.demo<|message|><atem:function_calls>"
            + invokes
            + "</atem:function_calls><|eot|>"
        )
    raise AssertionError(f"unknown family {family}")


def _dr03_parse_chunks(tools, family, chunks):
    """Feed ``chunks`` sequentially; return [(name, parsed_arguments), ...]."""
    parser = FunctionCallParser(tools, family)
    named, fragments = [], []
    for chunk in chunks:
        _, emitted = parser.parse_stream_chunk(chunk)
        for item in emitted:
            (named if item.name is not None else fragments).append(item)
    for item in parser.finalize_stream():
        (named if item.name is not None else fragments).append(item)
    order = [item.tool_index for item in named]
    args = {}
    for item in fragments:
        args[item.tool_index] = args.get(item.tool_index, "") + item.parameters
    return [
        (item.name, json.loads(args.get(index) or "{}"))
        for index, item in zip(order, named)
    ]


def _dr03_stream(tools, family, wire, chunk_size=3):
    """Stream ``wire``; return [(name, parsed_arguments), ...] per emitted call."""
    return _dr03_parse_chunks(
        tools, family, [wire[i : i + chunk_size] for i in range(0, len(wire), chunk_size)]
    )


def _dr03_oneshot(tools, family, wire):
    result = FunctionCallParser(tools, family).parse_non_stream(wire)
    return [(item.name, json.loads(item.parameters)) for item in result.calls]


_DR03_BOOL_INT_TOOLS = _dr03_tools(
    {"enabled": {"type": "boolean"}, "count": {"type": "integer"}}
)
_DR03_REQUIRED_TOOLS = _dr03_tools(
    {"enabled": {"type": "boolean"}, "count": {"type": "integer"}},
    required=["enabled"],
)


@pytest.mark.parametrize("family", ["qwen3_coder", "minimax"])
def test_dr03_rejected_typed_value_preserved_raw_in_both_modes(family):
    # Exact finding example: demo(enabled="perhaps", count=2). Regression pin
    # of the preserve-raw policy: streaming delivers the raw value exactly
    # like the one-shot path.
    wire = _dr03_wire(family, [[("enabled", "perhaps"), ("count", "2")]])
    expected = [("demo", {"enabled": "perhaps", "count": 2})]
    assert _dr03_stream(_DR03_BOOL_INT_TOOLS, family, wire) == expected
    assert _dr03_oneshot(_DR03_BOOL_INT_TOOLS, family, wire) == expected


@pytest.mark.parametrize("family", ["qwen3_coder", "minimax"])
def test_dr03_streaming_matches_oneshot_for_rejected_typed_value(family):
    wire = _dr03_wire(family, [[("enabled", "perhaps"), ("count", "2")]])
    assert _dr03_stream(_DR03_BOOL_INT_TOOLS, family, wire) == _dr03_oneshot(
        _DR03_BOOL_INT_TOOLS, family, wire
    )


@pytest.mark.parametrize("family", ["mimo_v2", "muse_glimmer"])
def test_dr03_replay_families_preserve_raw_in_both_modes(family):
    # One-shot IS a streaming replay for these families: both modes must carry
    # the raw value (previously both silently omitted it).
    wire = _dr03_wire(family, [[("enabled", "perhaps"), ("count", "2")]])
    expected = [("demo", {"enabled": "perhaps", "count": 2})]
    assert _dr03_stream(_DR03_BOOL_INT_TOOLS, family, wire) == expected
    assert _dr03_oneshot(_DR03_BOOL_INT_TOOLS, family, wire) == expected


@pytest.mark.parametrize("family", ["qwen3_coder", "minimax"])
def test_dr03_required_field_rejection_preserved_raw_requiredness_ignored(family):
    # ``required`` is never consulted by the parser (only the properties map
    # reaches _get_param_config), so a rejected REQUIRED field is preserved
    # raw exactly like an optional one; validation stays client-side.
    wire = _dr03_wire(family, [[("enabled", "perhaps"), ("count", "2")]])
    expected = [("demo", {"enabled": "perhaps", "count": 2})]
    assert _dr03_stream(_DR03_REQUIRED_TOOLS, family, wire) == expected
    assert _dr03_oneshot(_DR03_REQUIRED_TOOLS, family, wire) == expected


@pytest.mark.parametrize("family", ["qwen3_coder", "minimax"])
def test_dr03_optional_field_rejection_preserved_raw(family):
    wire = _dr03_wire(family, [[("enabled", "perhaps"), ("count", "2")]])
    expected = [("demo", {"enabled": "perhaps", "count": 2})]
    assert _dr03_stream(_DR03_BOOL_INT_TOOLS, family, wire) == expected
    assert _dr03_oneshot(_DR03_BOOL_INT_TOOLS, family, wire) == expected


@pytest.mark.parametrize("family", DR03_FAMILIES)
def test_dr03_rejected_first_field_then_valid_comma_bookkeeping(family):
    # Rejected FIRST field: streaming must re-open the object with "{" for the
    # later valid field (the rejected key is emitted raw, then a comma joins
    # the next key) and stay equivalent to one-shot declaration order.
    wire = _dr03_wire(family, [[("count", "abc"), ("enabled", "true")]])
    expected = [("demo", {"count": "abc", "enabled": True})]
    assert _dr03_stream(_DR03_BOOL_INT_TOOLS, family, wire) == expected
    assert _dr03_oneshot(_DR03_BOOL_INT_TOOLS, family, wire) == expected


@pytest.mark.parametrize("family", DR03_FAMILIES)
def test_dr03_valid_call_after_rejected_call_preserves_raw(family):
    wire = _dr03_wire(family, [[("enabled", "perhaps")], [("enabled", "true")]])
    expected = [("demo", {"enabled": "perhaps"}), ("demo", {"enabled": True})]
    assert _dr03_stream(_DR03_BOOL_INT_TOOLS, family, wire) == expected
    assert _dr03_oneshot(_DR03_BOOL_INT_TOOLS, family, wire) == expected


DR03_MISMATCH_CASES = [
    ("boolean", {"type": "boolean"}, "perhaps"),
    ("integer", {"type": "integer"}, "abc"),
    ("integer_float_text", {"type": "integer"}, "3.5"),
    ("number", {"type": "number"}, "abc"),
    ("object", {"type": "object"}, "not json"),
    ("array_comma_text", {"type": "array"}, "1,2"),
    ("nullable_int", {"type": "integer", "nullable": True}, "perhaps"),
    ("anyof_int_null", {"anyOf": [{"type": "integer"}, {"type": "null"}]}, "perhaps"),
]


@pytest.mark.parametrize(("label", "prop", "value"), DR03_MISMATCH_CASES)
def test_dr03_mismatched_typed_value_preserved_raw_both_modes(label, prop, value):
    # Dossier matrix: every mismatched shape (plain type, nullable, anyOf) is
    # preserved raw in both modes.
    tools = _dr03_tools({"value": prop})
    wire = _dr03_wire("qwen3_coder", [[("value", value)]])
    expected = [("demo", {"value": value})]
    assert _dr03_stream(tools, "qwen3_coder", wire) == expected
    assert _dr03_oneshot(tools, "qwen3_coder", wire) == expected


# DR-07: adjacent invoke openers confirm missing-parameter-close recovery.
def _dr07_tools(typed=False):
    return [{"type": "function", "function": {
        "name": name, "parameters": {"properties": {
            "count" if typed and name == "first" else "text": {
                "type": "integer" if typed and name == "first" else "string"
            }
        }}
    }} for name in ("first", "second", "third")]


def _dr07_opener(detector, prefix_attr, name):
    prefix = getattr(detector, prefix_attr)
    return prefix + (name + ">" if prefix.endswith("=") else ' name="' + name + '">')


def _dr07_wire(family="qwen3_coder", variant="same", typed=False):
    detector = FunctionCallParser(_dr07_tools(typed), family).detector
    outer, outer_close = detector._ps_outer_open, detector._ps_outer_close
    invoke_close, param_close = detector._ps_invoke_close, detector._ps_param_close

    def invoke(name):
        return _dr07_opener(detector, "_ps_invoke_open_prefix", name)

    def param(name):
        return _dr07_opener(detector, "_ps_param_open_prefix", name)

    first = outer + invoke("first") + param("count" if typed else "text") + "x"
    second = invoke("second") + param("text") + "y" + param_close + invoke_close
    if variant == "missing_invoke":
        wire = first
    elif variant == "missing_outer":
        wire = first + param_close + invoke_close
    else:
        if variant == "wellformed":
            first += param_close
        elif variant == "literal":
            first = first[:-1] + "use " + outer
        separator = invoke_close + "\n"
        if variant == "separate":
            separator += outer_close + outer
        elif variant == "missing_outer_next":
            separator += outer
        wire = first + separator + second + outer_close
        if variant == "third":
            wire += outer + invoke("third") + param("text") + "z" + param_close + invoke_close + outer_close
    if family == "muse_glimmer":
        from freetoken.server.reasoning_parser import ATEM_START, ATEM_MESSAGE
        wire = ATEM_START + "assistant to=first.first" + ATEM_MESSAGE + wire
    return wire


def _dr07_stream(family="qwen3_coder", variant="same", typed=False, chunk_size=3):
    tools = _dr07_tools(typed)
    wire = _dr07_wire(family, variant, typed)
    return _dr03_stream(tools, family, wire, chunk_size)


_DR07_CLEAN = [("first", {"text": "x"}), ("second", {"text": "y"})]
_DR07_TYPED = [("first", {"count": "x"}), ("second", {"text": "y"})]


def test_dr07_pin_qwen3_coder_streaming_recovers_second_call():
    assert _dr07_stream(chunk_size=len(_dr07_wire())) == _DR07_CLEAN


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 40])
def test_dr07_pin_qwen3_coder_recovery_is_chunk_stable(chunk_size):
    assert _dr07_stream(chunk_size=chunk_size) == _DR07_CLEAN


def test_dr07_pin_qwen3_coder_recovered_calls_complete_before_eos():
    parser = FunctionCallParser(_dr07_tools(), "qwen3_coder")
    _normal, emitted = parser.parse_stream_chunk(_dr07_wire())
    named = [item for item in emitted if item.name is not None]
    args = {}
    for item in emitted:
        if item.name is None:
            args[item.tool_index] = args.get(item.tool_index, "") + item.parameters
    assert [
        (item.name, json.loads(args.get(item.tool_index) or "{}")) for item in named
    ] == _DR07_CLEAN
    assert parser.finalize_stream() == []
    assert parser.finish_stream() == ""


def test_dr07_pin_qwen3_coder_oneshot_regex_emits_both_calls_cleanly():
    assert _dr03_oneshot(_dr07_tools(), "qwen3_coder", _dr07_wire()) == _DR07_CLEAN


def test_dr07_parity_qwen3_coder_streaming_matches_oneshot():
    assert _dr07_stream() == _dr03_oneshot(_dr07_tools(), "qwen3_coder", _dr07_wire())


def test_dr07_pin_mimo_v2_replay_one_shot_recovers_second_call():
    assert _dr07_stream("mimo_v2") == _DR07_CLEAN
    assert _dr03_oneshot(_dr07_tools(), "mimo_v2", _dr07_wire("mimo_v2")) == _DR07_CLEAN


def test_dr07_pin_qwen3_coder_typed_first_value_preserved_raw():
    assert _dr07_stream(typed=True) == _DR07_TYPED


def test_dr07_parity_qwen3_coder_typed_variant_matches_oneshot():
    assert _dr07_stream(typed=True) == _dr03_oneshot(
        _dr07_tools(True), "qwen3_coder", _dr07_wire(typed=True)
    )


def test_dr07_pin_mimo_v2_typed_variant_preserves_raw_in_both_modes():
    assert _dr07_stream("mimo_v2", typed=True) == _DR07_TYPED
    assert _dr03_oneshot(_dr07_tools(True), "mimo_v2", _dr07_wire("mimo_v2", typed=True)) == _DR07_TYPED


def test_dr07_pin_same_wrapper_consecutive_calls_stream_normally_when_wellformed():
    for family in DR03_FAMILIES:
        assert _dr07_stream(family, "wellformed") == _DR07_CLEAN
        assert _dr03_oneshot(_dr07_tools(), family, _dr07_wire(family, "wellformed")) == _DR07_CLEAN


def test_dr07_pin_minimax_streaming_recovers_second_call():
    assert _dr07_stream("minimax") == _DR07_CLEAN


def test_dr07_pin_minimax_oneshot_emits_first_with_empty_args_and_second():
    assert _dr03_oneshot(_dr07_tools(), "minimax", _dr07_wire("minimax")) == [
        ("first", {}), ("second", {"text": "y"})
    ]


def test_dr07_pin_minimax_cross_mode_empty_args_gap():
    # Full MiniMax parity remains a separate pre-existing one-shot regex gap.
    streamed = _dr07_stream("minimax")
    oneshot = _dr03_oneshot(_dr07_tools(), "minimax", _dr07_wire("minimax"))
    assert streamed == _DR07_CLEAN
    assert oneshot == [("first", {}), ("second", {"text": "y"})]
    assert streamed != oneshot


def test_dr07_pin_muse_glimmer_replay_one_shot_recovers_second_call():
    assert _dr07_stream("muse_glimmer") == _DR07_CLEAN
    assert _dr03_oneshot(_dr07_tools(), "muse_glimmer", _dr07_wire("muse_glimmer")) == _DR07_CLEAN


@pytest.mark.parametrize("family", ["qwen3_coder", "mimo_v2"])
def test_dr07_pin_separate_wrapper_recovery_finds_next_call(family):
    assert _dr07_stream(family, "separate") == _DR07_CLEAN
    assert _dr03_oneshot(_dr07_tools(), family, _dr07_wire(family, "separate")) == _DR07_CLEAN


@pytest.mark.parametrize("family", ["qwen3_coder", "mimo_v2"])
def test_dr07_pin_later_envelope_preserves_all_three_calls(family):
    assert _dr07_stream(family, "third") == _DR07_CLEAN + [("third", {"text": "z"})]


def test_dr07_pin_missing_invoke_closer_eos_completes_call():
    for family in ("qwen3_coder", "mimo_v2"):
        assert _dr07_stream(family, "missing_invoke") == [("first", {"text": "x"})]
        assert _dr03_oneshot(_dr07_tools(), family, _dr07_wire(family, "missing_invoke")) == [("first", {"text": "x"})]


def test_dr07_pin_missing_outer_closer_eos_completes_call():
    for family in ("qwen3_coder", "mimo_v2"):
        assert _dr07_stream(family, "missing_outer") == [("first", {"text": "x"})]
        assert _dr03_oneshot(_dr07_tools(), family, _dr07_wire(family, "missing_outer")) == [("first", {"text": "x"})]


def test_dr07_pin_missing_outer_closer_next_envelope_recovers():
    assert _dr07_stream(variant="missing_outer_next") == _DR07_CLEAN


@pytest.mark.parametrize("family", ["qwen3_coder", "mimo_v2"])
def test_dr07_pin_literal_function_looking_string_stays_data(family):
    detector = FunctionCallParser(_dr07_tools(), family).detector
    expected = [("first", {"text": "use " + detector._ps_outer_open}), ("second", {"text": "y"})]
    assert _dr07_stream(family, "literal") == expected
    assert _dr03_oneshot(_dr07_tools(), family, _dr07_wire(family, "literal")) == expected


@pytest.mark.parametrize("family", DR03_FAMILIES)
def test_dr07_literal_invoke_looking_prose_does_not_confirm_recovery(family):
    detector = FunctionCallParser(_dr07_tools(), family).detector
    wire = _dr07_wire(family, "same")
    boundary = detector._ps_invoke_close
    opener_pos = wire.index(boundary) + len(boundary) + 1
    wire = wire[:opener_pos] + "quoted example: " + wire[opener_pos:]
    value = "x" + wire[wire.index(boundary):wire.index(detector._ps_param_close)]
    assert _dr03_stream(_dr07_tools(), family, wire, 1) == [("first", {"text": value})]


@pytest.mark.parametrize("family", DR03_FAMILIES)
def test_dr07_wellformed_value_keeps_literal_invoke_opener(family):
    detector = FunctionCallParser(_dr07_tools(), family).detector
    wire = _dr07_wire(family, "wellformed")
    boundary = detector._ps_invoke_close
    following = wire[wire.index(boundary) + len(boundary) + 1:]
    literal = following[:following.index(">") + 1]
    wire = wire.replace("x" + detector._ps_param_close, "x " + literal + detector._ps_param_close, 1)
    expected = [("first", {"text": "x " + literal}), ("second", {"text": "y"})]
    assert _dr03_stream(_dr07_tools(), family, wire, 1) == expected


def _dr07_modified_wire(family, replacement):
    """Canonical malformed pair with the inter-call separator replaced."""
    detector = FunctionCallParser(_dr07_tools(), family).detector
    return _dr07_wire(family).replace(detector._ps_invoke_close + "\n", replacement, 1)


def _dr07_two_chunk_invariant(family, tools, wire, expected):
    """Assert ``expected`` for whole delivery and for every two-chunk split."""
    whole = _dr03_parse_chunks(tools, family, [wire])
    assert whole == expected
    for split in range(len(wire) + 1):
        actual = _dr03_parse_chunks(tools, family, [wire[:split], wire[split:]])
        assert actual == whole, (family, split)


@pytest.mark.parametrize("family", DR03_FAMILIES)
@pytest.mark.parametrize("kind", ["close_then_bare", "outer_boundary_bare"])
def test_dr07_unreachable_adjacent_invoke_stays_value_data(family, kind):
    # The block scanner cannot consume an invoke opener straight after an
    # outer closer (it would end idle and leak the call), so confirmation is
    # withheld: the resolution becomes value data and nothing is truncated.
    detector = FunctionCallParser(_dr07_tools(), family).detector
    replacement = detector._ps_outer_close
    if kind == "close_then_bare":
        replacement = detector._ps_invoke_close + replacement
    wire = _dr07_modified_wire(family, replacement)
    token = detector._ps_invoke_close if kind == "close_then_bare" else detector._ps_outer_close
    absorbed = wire[wire.index(token) : wire.index(detector._ps_param_close)]
    _dr07_two_chunk_invariant(
        family, _dr07_tools(), wire, [("first", {"text": "x" + absorbed})]
    )


@pytest.mark.parametrize("family", DR03_FAMILIES)
def test_dr07_wrapper_repetition_follows_reachability(family):
    detector = FunctionCallParser(_dr07_tools(), family).detector
    closer, opener = detector._ps_outer_close, detector._ps_outer_open
    # A second outer closer before the reopen leaks in idle state, so the
    # arrangement is not reachable and stays value data.
    wire = _dr07_modified_wire(family, detector._ps_invoke_close + closer * 2 + opener)
    absorbed = wire[wire.index(detector._ps_invoke_close) : wire.index(detector._ps_param_close)]
    _dr07_two_chunk_invariant(
        family, _dr07_tools(), wire, [("first", {"text": "x" + absorbed})]
    )
    # Repeated outer openers are debris the block scanner drops silently, so
    # recovery still confirms and both calls survive.
    wire = _dr07_modified_wire(family, detector._ps_invoke_close + opener * 2)
    _dr07_two_chunk_invariant(family, _dr07_tools(), wire, _DR07_CLEAN)


@pytest.mark.parametrize("family", ["minimax", "muse_glimmer"])
@pytest.mark.parametrize("registered", [True, False])
def test_dr07_embedded_param_close_in_header_is_chunk_invariant(family, registered):
    # A quoted tool name may embed a parameter closer. Whole delivery sees a
    # complete header; a split inside the header must hold pending instead of
    # resolving at the embedded closer, so both modes agree on the boundary.
    detector = FunctionCallParser(_dr07_tools(), family).detector
    name = "sec" + detector._ps_param_close + "ond"
    tools = _dr07_tools()
    if registered:
        tools[1]["function"]["name"] = name
    wire = _dr07_wire(family).replace("second", name, 1)
    expected = [("first", {"text": "x"}), (name, {"text": "y"})]
    _dr07_two_chunk_invariant(family, tools, wire, expected)


def _dr07_phantom_wire(family, wellformed):
    """Canonical pair whose first value holds an uncompleting opener prefix."""
    detector = FunctionCallParser(_dr07_tools(), family).detector
    wire = _dr07_wire(family, "wellformed" if wellformed else "same")
    param = _dr07_opener(detector, "_ps_param_open_prefix", "text")
    return wire.replace(
        param + "x", param + "x call " + detector._ps_invoke_open_prefix + " to see", 1
    )


@pytest.mark.parametrize("family", DR03_FAMILIES)
@pytest.mark.parametrize("wellformed", [False, True])
def test_dr07_phantom_opener_prefix_in_value_is_chunk_invariant(family, wellformed):
    # An opener prefix inside the value is data: its span must never suppress
    # the first structural closer, in whole or any split delivery.
    detector = FunctionCallParser(_dr07_tools(), family).detector
    value = "x call " + detector._ps_invoke_open_prefix + " to see"
    _dr07_two_chunk_invariant(
        family,
        _dr07_tools(),
        _dr07_phantom_wire(family, wellformed),
        [("first", {"text": value}), ("second", {"text": "y"})],
    )


@pytest.mark.parametrize("family", DR03_FAMILIES)
def test_dr07_phantom_opener_prefix_eos_partial_next_header_is_invariant(family):
    # At EOS a phantom value span must not swallow the trailing closer or the
    # incomplete next header; the value still ends at the first boundary.
    detector = FunctionCallParser(_dr07_tools(), family).detector
    wire = _dr07_phantom_wire(family, wellformed=False)
    cut = wire.index(detector._ps_invoke_close) + len(detector._ps_invoke_close) + 4
    _dr07_two_chunk_invariant(
        family,
        _dr07_tools(),
        wire[:cut],
        [("first", {"text": "x call " + detector._ps_invoke_open_prefix + " to see"})],
    )


@pytest.mark.parametrize(("label", "prop", "value"), DR03_MISMATCH_CASES[:6])
def test_dr03_mismatched_typed_value_minimax_matches(label, prop, value):
    tools = _dr03_tools({"value": prop})
    wire = _dr03_wire("minimax", [[("value", value)]])
    expected = [("demo", {"value": value})]
    assert _dr03_stream(tools, "minimax", wire) == expected
    assert _dr03_oneshot(tools, "minimax", wire) == expected


def test_dr03_nullable_null_text_resolves_in_both_modes():
    # Parity that already held before DR-03: "null" for a nullable integer
    # resolves to None in both modes.
    tools = _dr03_tools({"value": {"type": "integer", "nullable": True}})
    wire = _dr03_wire("qwen3_coder", [[("value", "null")]])
    expected = [("demo", {"value": None})]
    assert _dr03_stream(tools, "qwen3_coder", wire) == expected
    assert _dr03_oneshot(tools, "qwen3_coder", wire) == expected


def test_dr03_declared_string_never_rejected():
    # Control: declared strings accept any text in both modes.
    tools = _dr03_tools({"value": {"type": "string"}})
    wire = _dr03_wire("qwen3_coder", [[("value", "perhaps")]])
    expected = [("demo", {"value": "perhaps"})]
    assert _dr03_stream(tools, "qwen3_coder", wire) == expected
    assert _dr03_oneshot(tools, "qwen3_coder", wire) == expected


def test_dr03_truncated_valid_partial_converges():
    # A truncation whose partial text is already schema-valid converges; the
    # streamed value closes via finalize_streaming's pbuf branch.
    wire = "<tool_call><function=demo><parameter=count>12"
    expected = [("demo", {"count": 12})]
    assert _dr03_stream(_DR03_BOOL_INT_TOOLS, "qwen3_coder", wire) == expected
    assert _dr03_oneshot(_DR03_BOOL_INT_TOOLS, "qwen3_coder", wire) == expected


@pytest.mark.parametrize(
    ("prop", "partial"),
    [
        ({"type": "boolean"}, "perha"),
        ({"type": "array"}, "[1,2"),
    ],
)
def test_dr03_truncated_invalid_partial_preserved_raw_both_modes(prop, partial):
    # A truncation whose partial text is NOT schema-valid preserves the raw
    # partial in both modes (streaming via the finalize always-emit).
    tools = _dr03_tools({"value": prop})
    wire = "<tool_call><function=demo><parameter=value>" + partial
    expected = [("demo", {"value": partial})]
    assert _dr03_stream(tools, "qwen3_coder", wire) == expected
    assert _dr03_oneshot(tools, "qwen3_coder", wire) == expected
