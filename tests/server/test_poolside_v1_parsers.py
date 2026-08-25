from __future__ import annotations

import json

from freetoken.server.api_models import Tool
from freetoken.server.function_call_parser import FunctionCallParser
from freetoken.server.reasoning_parser import ReasoningParser

TOOLS: list[Tool] = [
    Tool.model_validate(
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "mode": {"type": "integer"},
                    },
                },
            },
        }
    )
]


def _tool_block(content: str = "    def f():\n        return 1\n") -> str:
    return (
        "<tool_call>write_file"
        "<arg_key>path</arg_key><arg_value>/tmp/a.py</arg_value>"
        f"<arg_key>content</arg_key><arg_value>{content}</arg_value>"
        "<arg_key>mode</arg_key><arg_value>420</arg_value>"
        "</tool_call>"
    )


def test_poolside_v1_reasoning_implicit_open_non_stream() -> None:
    parser = ReasoningParser("poolside_v1", force_reasoning=True)
    reasoning, content = parser.parse_non_stream(
        "I should inspect the file.</think>Here is the answer."
    )
    assert reasoning == "I should inspect the file."
    assert content == "Here is the answer."


def test_poolside_v1_reasoning_stream_marker_can_split_across_chunks() -> None:
    parser = ReasoningParser("poolside_v1", force_reasoning=True)
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    for chunk in ["Need ", "the file.</thi", "nk>Answer", "."]:
        reasoning, content = parser.parse_stream_chunk(chunk)
        reasoning_parts.append(reasoning)
        content_parts.append(content)
    reasoning, content = parser.flush()
    reasoning_parts.append(reasoning)
    content_parts.append(content)
    assert "".join(reasoning_parts) == "Need the file."
    assert "".join(content_parts) == "Answer."


def test_poolside_v1_tool_non_stream_preserves_string_whitespace() -> None:
    source = "    def f():\n        return 1\n"
    parser = FunctionCallParser(TOOLS, tool_call_parser="poolside_v1")
    result = parser.parse_non_stream("Calling it. " + _tool_block(source))
    assert result.normal_text == "Calling it. "
    assert len(result.calls) == 1
    assert result.calls[0].name == "write_file"
    assert json.loads(result.calls[0].parameters) == {
        "path": "/tmp/a.py",
        "content": source,
        "mode": 420,
    }


def test_poolside_v1_tool_stream_preserves_string_whitespace_and_types() -> None:
    source = "    def f():\n        return 1\n"
    parser = FunctionCallParser(TOOLS, tool_call_parser="poolside_v1")
    names: list[str] = []
    arg_fragments: list[str] = []
    normal: list[str] = []
    text = "Calling it. " + _tool_block(source)
    for i in range(0, len(text), 3):
        visible, calls = parser.parse_stream_chunk(text[i : i + 3])
        normal.append(visible)
        for call in calls:
            if call.name:
                names.append(call.name)
            arg_fragments.append(call.parameters)
    normal.append(parser.finish_stream())
    assert "".join(normal) == "Calling it. "
    assert "".join(names) == "write_file"
    assert json.loads("".join(arg_fragments)) == {
        "path": "/tmp/a.py",
        "content": source,
        "mode": 420,
    }


def _schema_tools() -> list[Tool]:
    return [
        Tool.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "typed",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "exact": {"type": "string"},
                            "enum_only": {"enum": ["a", "b"]},
                            "typeless": {},
                            "integer": {"type": "integer"},
                            "boolean": {"type": "boolean"},
                            "object": {"type": "object"},
                            "array": {"type": "array"},
                        },
                    },
                },
            }
        )
    ]


def _schema_block() -> str:
    return (
        "<tool_call>typed"
        "<arg_key>exact</arg_key><arg_value>  keep me  \n</arg_value>"
        "<arg_key>enum_only</arg_key><arg_value> a </arg_value>"
        "<arg_key>typeless</arg_key><arg_value> 5 </arg_value>"
        "<arg_key>integer</arg_key><arg_value> 7 </arg_value>"
        "<arg_key>boolean</arg_key><arg_value> true </arg_value>"
        '<arg_key>object</arg_key><arg_value> {"x": 1} </arg_value>'
        "<arg_key>array</arg_key><arg_value> [1, 2] </arg_value>"
        "</tool_call>"
    )


EXPECTED_SCHEMA_ARGS = {
    "exact": "  keep me  \n",
    "enum_only": "a",
    "typeless": 5,
    "integer": 7,
    "boolean": True,
    "object": {"x": 1},
    "array": [1, 2],
}


def test_poolside_v1_exact_string_schema_semantics_non_stream() -> None:
    parser = FunctionCallParser(_schema_tools(), tool_call_parser="poolside_v1")
    result = parser.parse_non_stream(_schema_block())
    assert len(result.calls) == 1
    assert json.loads(result.calls[0].parameters) == EXPECTED_SCHEMA_ARGS


def test_poolside_v1_exact_string_schema_semantics_one_char_streaming() -> None:
    parser = FunctionCallParser(_schema_tools(), tool_call_parser="poolside_v1")
    fragments: list[str] = []
    for char in _schema_block():
        _visible, calls = parser.parse_stream_chunk(char)
        fragments.extend(call.parameters for call in calls)
    parser.finish_stream()
    assert json.loads("".join(fragments)) == EXPECTED_SCHEMA_ARGS


def test_poolside_v1_preserves_escaped_source_under_one_char_streaming() -> None:
    source = '  print("C:\\\\tmp")  \n'
    parser = FunctionCallParser(TOOLS, tool_call_parser="poolside_v1")
    fragments: list[str] = []
    for char in _tool_block(source):
        _visible, calls = parser.parse_stream_chunk(char)
        fragments.extend(call.parameters for call in calls)
    parser.finish_stream()
    assert json.loads("".join(fragments))["content"] == source


def test_glm47_string_arguments_keep_legacy_trimming() -> None:
    source = "  padded  \n"
    parser = FunctionCallParser(TOOLS, tool_call_parser="glm47")
    result = parser.parse_non_stream(_tool_block(source))
    assert json.loads(result.calls[0].parameters)["content"] == source.strip()
