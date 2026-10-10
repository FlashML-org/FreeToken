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
        ("enabled", "tru", {}),
        ("items", "[1,", {}),
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
