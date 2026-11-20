"""JSON Schema resolution in schema-aware tool argument parsing
(server/function_call_parser.py): $ref chains, unions and implicit containers,
plus MiniMax-M3's recursive nesting -- one-shot and streaming.
"""

from __future__ import annotations

import json

import pytest

from freetoken.server.function_call_parser import Function, FunctionCallParser, Tool


# Formats that route parameter values through _convert_param_value(); the rest
# parse arguments from their own JSON grammar and ignore the declared schema.
SCHEMA_AWARE_PARSERS = [
    "glm47",
    "qwen3_coder",
    "minimax",
    "minimax_m3",
    "muse_glimmer",
]

MINIMAX_M3_NS = "]<]minimax[>["


def _tool(prop_schema: dict, root_extra: dict | None = None, *, param_name: str = "limit") -> list[Tool]:
    parameters = {
        "type": "object",
        "properties": {
            param_name: prop_schema,
        },
    }

    if root_extra:
        parameters.update(root_extra)

    return [
        Tool(
            function=Function(
                name="schema_test",
                parameters=parameters,
            )
        )
    ]


def _wire(parser_name: str, raw: str, *, param_name: str = "limit") -> str:
    if parser_name == "qwen3_coder":
        return (
            "<tool_call>"
            "<function=schema_test>"
            f"<parameter={param_name}>{raw}</parameter>"
            "</function>"
            "</tool_call>"
        )

    if parser_name == "glm47":
        return (
            "<tool_call>schema_test"
            f"<arg_key>{param_name}</arg_key>"
            f"<arg_value>{raw}</arg_value>"
            "</tool_call>"
        )

    if parser_name == "minimax":
        return (
            "<minimax:tool_call>"
            '<invoke name="schema_test">'
            f'<parameter name="{param_name}">{raw}</parameter>'
            "</invoke>"
            "</minimax:tool_call>"
        )

    if parser_name == "minimax_m3":
        ns = MINIMAX_M3_NS
        return (
            f"{ns}<tool_call>\n"
            f'{ns}<invoke name="schema_test">'
            f"{ns}<{param_name}>{raw}{ns}</{param_name}>"
            f"{ns}</invoke>\n"
            f"{ns}</tool_call>"
        )

    if parser_name == "muse_glimmer":
        return (
            "<|start|>assistant to=schema_test<|message|>"
            "<atem:function_calls>\n"
            '<atem:invoke name="schema_test">\n'
            f'<atem:parameter name="{param_name}">{raw}</atem:parameter>\n'
            "</atem:invoke>\n"
            "</atem:function_calls>"
            "<|eot|>"
        )

    raise AssertionError(f"Missing wire fixture for parser {parser_name!r}")


def _assemble_streamed_calls(calls) -> list[tuple[str, dict]]:
    """Reassemble streamed calls the way a client does: a fragment naming a tool
    opens a call, the argument fragments after it concatenate into its JSON."""
    assembled: list[list] = []

    for call in calls:
        if call.name is not None:
            assembled.append([call.name, []])

        if call.parameters:
            assert assembled, f"argument fragment before any call name: {call!r}"
            assembled[-1][1].append(call.parameters)

    return [(name, json.loads("".join(parts) or "{}")) for name, parts in assembled]


def _parse_args(parser_name: str, prop_schema: dict, root_extra: dict | None, raw: str, *, streaming: bool, param_name: str = "limit") -> dict:
    tools = _tool(prop_schema, root_extra, param_name=param_name)
    text = _wire(parser_name, raw, param_name=param_name)

    parser = FunctionCallParser(tools, tool_call_parser=parser_name)

    if not streaming:
        result = parser.parse_non_stream(text)

        assert len(result.calls) == 1, result
        assert result.calls[0].name == "schema_test"

        return json.loads(result.calls[0].parameters)

    calls = []

    # Small chunks deliberately split XML/control markers and parameter values.
    for i in range(0, len(text), 7):
        _, emitted = parser.parse_stream_chunk(text[i : i + 7])
        calls.extend(emitted)

    # Drain any wire-order-deferred output.
    for _ in range(4):
        normal, emitted = parser.parse_stream_chunk("")
        calls.extend(emitted)

        if not normal and not emitted:
            break

    # End-of-stream hooks in the serving layer's order (generation.py): finalize
    # closes a call cut off mid-arguments, finish releases held-back text.
    calls.extend(parser.finalize_stream())
    parser.finish_stream()

    assembled = _assemble_streamed_calls(calls)

    assert len(assembled) == 1, (parser_name, calls)
    assert assembled[0][0] == "schema_test"

    return assembled[0][1]


SCHEMA_CASES = [
    # Existing behaviour: direct concrete types.
    pytest.param(
        {"type": "integer"},
        None,
        "5",
        5,
        id="direct-integer",
    ),
    pytest.param(
        {"type": "string"},
        None,
        "5",
        "5",
        id="direct-string-preserved",
    ),

    # Single local $ref.
    pytest.param(
        {"$ref": "#/$defs/Limit"},
        {
            "$defs": {
                "Limit": {"type": "integer"},
            }
        },
        "5",
        5,
        id="ref-integer",
    ),

    # Chained $ref.
    pytest.param(
        {"$ref": "#/$defs/A"},
        {
            "$defs": {
                "A": {"$ref": "#/$defs/B"},
                "B": {"type": "integer"},
            }
        },
        "5",
        5,
        id="ref-chain",
    ),

    # Draft-07 style definitions should work too: the resolver accepts generic
    # local JSON pointers, not only $defs.
    pytest.param(
        {"$ref": "#/definitions/Limit"},
        {
            "definitions": {
                "Limit": {"type": "integer"},
            }
        },
        "5",
        5,
        id="legacy-definitions-ref",
    ),

    # JSON Pointer escaping: ~1 => / and ~0 => ~.
    pytest.param(
        {"$ref": "#/$defs/A~1B~0C"},
        {
            "$defs": {
                "A/B~C": {"type": "integer"},
            }
        },
        "5",
        5,
        id="json-pointer-escaping",
    ),

    # oneOf / anyOf concrete type.
    pytest.param(
        {
            "oneOf": [
                {"type": "integer"},
            ]
        },
        None,
        "5",
        5,
        id="oneof-integer",
    ),
    pytest.param(
        {
            "anyOf": [
                {"type": "integer"},
            ]
        },
        None,
        "5",
        5,
        id="anyof-integer",
    ),

    # Nullable unions.
    pytest.param(
        {
            "oneOf": [
                {"type": "integer"},
                {"type": "null"},
            ]
        },
        None,
        "5",
        5,
        id="oneof-nullable-integer",
    ),
    pytest.param(
        {
            "anyOf": [
                {"type": "integer"},
                {"type": "null"},
            ]
        },
        None,
        "5",
        5,
        id="anyof-nullable-integer",
    ),
    pytest.param(
        {
            "type": ["integer", "null"],
        },
        None,
        "5",
        5,
        id="type-array-nullable-integer",
    ),

    # A union member may itself be a $ref.
    pytest.param(
        {
            "oneOf": [
                {"$ref": "#/$defs/Limit"},
                {"type": "null"},
            ]
        },
        {
            "$defs": {
                "Limit": {"type": "integer"},
            }
        },
        "5",
        5,
        id="oneof-ref-nullable",
    ),

    # Referenced structured values must remain JSON structures rather than
    # escaped strings.
    pytest.param(
        {"$ref": "#/$defs/Ids"},
        {
            "$defs": {
                "Ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                }
            }
        },
        "[1, 2, 3]",
        [1, 2, 3],
        id="ref-array",
    ),
    pytest.param(
        {"$ref": "#/$defs/Options"},
        {
            "$defs": {
                "Options": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer"},
                    },
                }
            }
        },
        '{"limit": 5}',
        {"limit": 5},
        id="ref-object",
    ),

    # JSON Schema does not require an explicit "type" if structure already
    # establishes the effective container kind.
    pytest.param(
        {
            "properties": {
                "limit": {"type": "integer"},
            }
        },
        None,
        '{"limit": 5}',
        {"limit": 5},
        id="implicit-object",
    ),
    pytest.param(
        {
            "items": {
                "type": "integer",
            }
        },
        None,
        "[1, 2]",
        [1, 2],
        id="implicit-array",
    ),

    # The PR intentionally uses loose JSON parsing when no single concrete type
    # can be inferred.
    pytest.param(
        {
            "oneOf": [
                {"type": "integer"},
                {"type": "string"},
            ]
        },
        None,
        "5",
        5,
        id="ambiguous-union-loose-json",
    ),
]


@pytest.mark.parametrize("parser_name", SCHEMA_AWARE_PARSERS)
@pytest.mark.parametrize("streaming", [False, True], ids=["one-shot", "streaming"])
@pytest.mark.parametrize(
    ("prop_schema", "root_extra", "raw", "expected"),
    SCHEMA_CASES,
)
def test_schema_aware_parsers_resolve_indirect_schema_types(
    parser_name,
    streaming,
    prop_schema,
    root_extra,
    raw,
    expected,
):
    args = _parse_args(
        parser_name,
        prop_schema,
        root_extra,
        raw,
        streaming=streaming,
    )

    assert args == {"limit": expected}


# ---------------------------------------------------------------------------
# Broken / recursive references must never take the request path down.
# ---------------------------------------------------------------------------
REF_EDGE_CASES = [
    pytest.param(
        {"$ref": "#/$defs/Missing"},
        None,
        id="dangling-local-ref",
    ),
    pytest.param(
        {"$ref": "https://example.invalid/schema.json#/Limit"},
        None,
        id="external-ref",
    ),
    pytest.param(
        {"$ref": "#/$defs/A"},
        {
            "$defs": {
                "A": "not-a-schema",
            }
        },
        id="ref-to-non-object-def",
    ),
    pytest.param(
        {"$ref": "#/$defs/A/name"},
        {
            "$defs": {
                "A": "not-a-container",
            }
        },
        id="ref-through-non-object-def",
    ),
    pytest.param(
        {"$ref": "#/$defs/A"},
        {
            "$defs": {
                "A": {"$ref": "#/$defs/A"},
            }
        },
        id="self-referential-ref",
    ),
    pytest.param(
        {"$ref": "#/$defs/A"},
        {
            "$defs": {
                "A": {"$ref": "#/$defs/B"},
                "B": {"$ref": "#/$defs/A"},
            }
        },
        id="mutually-recursive-ref",
    ),
]


@pytest.mark.parametrize("parser_name", SCHEMA_AWARE_PARSERS)
@pytest.mark.parametrize("streaming", [False, True], ids=["one-shot", "streaming"])
@pytest.mark.parametrize(
    ("prop_schema", "root_extra"),
    REF_EDGE_CASES,
)
def test_schema_ref_edge_cases_terminate_without_crashing(parser_name, streaming, prop_schema, root_extra):
    args = _parse_args(
        parser_name,
        prop_schema,
        root_extra,
        "5",
        streaming=streaming,
    )

    # An unresolvable ref leaves the parameter untyped: the same loose JSON
    # fallback as an ambiguous union, so "5" -> 5 and never a dropped parameter.
    assert args == {"limit": 5}


# ---------------------------------------------------------------------------
# Compatibility guards
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("parser_name", SCHEMA_AWARE_PARSERS)
@pytest.mark.parametrize("streaming", [False, True], ids=["one-shot", "streaming"])
def test_explicit_string_schema_never_loose_parses_json_literals(parser_name, streaming):
    args = _parse_args(
        parser_name,
        {"type": "string"},
        None,
        "123",
        streaming=streaming,
    )

    assert args == {"limit": "123"}


@pytest.mark.parametrize("parser_name", SCHEMA_AWARE_PARSERS)
@pytest.mark.parametrize("streaming", [False, True], ids=["one-shot", "streaming"])
def test_nullable_string_schema_preserves_numeric_looking_string(parser_name, streaming):
    args = _parse_args(
        parser_name,
        {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ]
        },
        None,
        "123",
        streaming=streaming,
    )

    assert args == {"limit": "123"}


# ---------------------------------------------------------------------------
# MiniMax-M3 threads the schema through nested XML
# ---------------------------------------------------------------------------
def _m3_tools(properties: dict, **root_extra) -> list[Tool]:
    """One ``schema_test`` tool: ``properties`` plus any extra root keywords."""
    return [
        Tool(
            function=Function(
                name="schema_test",
                parameters={"type": "object", "properties": properties, **root_extra},
            )
        )
    ]


def _m3_call(body: str) -> str:
    """Wrap an invoke body in MiniMax-M3's namespace-delimited wire form."""
    ns = MINIMAX_M3_NS

    return (
        f"{ns}<tool_call>\n"
        f'{ns}<invoke name="schema_test">'
        f"{body}"
        f"{ns}</invoke>\n"
        f"{ns}</tool_call>"
    )


_M3_NODE_DEFS = {
    "$defs": {
        "Node": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "count": {"type": "integer"},
                "child": {"$ref": "#/$defs/Node"},
            },
        }
    }
}


def _minimax_m3_recursive_tools() -> list[Tool]:
    return _m3_tools({"node": {"$ref": "#/$defs/Node"}}, **_M3_NODE_DEFS)


def _minimax_m3_recursive_wire() -> str:
    ns = MINIMAX_M3_NS

    return _m3_call(
        f"{ns}<node>"
        f"{ns}<code>1{ns}</code>"
        f"{ns}<count>2{ns}</count>"
        f"{ns}<child>"
        f"{ns}<code>5{ns}</code>"
        f"{ns}<count>6{ns}</count>"
        f"{ns}</child>"
        f"{ns}</node>"
    )


def _parse_minimax_m3(tools: list[Tool], text: str, *, streaming: bool) -> dict:
    """Parse one MiniMax-M3 call through whichever path is under test."""
    parser = FunctionCallParser(tools, tool_call_parser="minimax_m3")

    if not streaming:
        result = parser.parse_non_stream(text)

        assert len(result.calls) == 1, result
        assert result.calls[0].name == "schema_test"

        return json.loads(result.calls[0].parameters)

    calls = []

    # Small chunks deliberately split NS markers and element boundaries. M3 emits
    # each call in one piece at its closing marker, so no extra drain is needed.
    for i in range(0, len(text), 7):
        _, emitted = parser.parse_stream_chunk(text[i : i + 7])
        calls.extend(emitted)

    calls.extend(parser.finalize_stream())
    parser.finish_stream()

    assembled = _assemble_streamed_calls(calls)

    assert len(assembled) == 1, (calls,)
    assert assembled[0][0] == "schema_test"

    return assembled[0][1]


@pytest.mark.parametrize("streaming", [False, True], ids=["one-shot", "streaming"])
def test_minimax_m3_nested_ref_keeps_nested_schema_typing(streaming):
    args = _parse_minimax_m3(
        _minimax_m3_recursive_tools(),
        _minimax_m3_recursive_wire(),
        streaming=streaming,
    )

    assert args == {
        "node": {
            "code": "1",
            "count": 2,
            "child": {
                "code": "5",
                "count": 6,
            },
        }
    }


# Below the top level: an array's "items" schema and an implicitly-array nested
# property both used to lose typing and arrive as the verbatim string.
MINIMAX_M3_NESTED_CASES = [
    pytest.param(
        _m3_tools(
            {
                "ids": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Id"},
                }
            },
            **{"$defs": {"Id": {"type": "integer"}}},
        ),
        _m3_call(
            f"{MINIMAX_M3_NS}<ids>"
            f"{MINIMAX_M3_NS}<item>1{MINIMAX_M3_NS}</item>"
            f"{MINIMAX_M3_NS}<item>2{MINIMAX_M3_NS}</item>"
            f"{MINIMAX_M3_NS}</ids>"
        ),
        {"ids": [1, 2]},
        id="array-items-ref",
    ),
    pytest.param(
        _m3_tools(
            {
                "payload": {
                    "type": "object",
                    "properties": {
                        # No explicit "type": only the structure says array.
                        "cell": {"items": {"type": "integer"}},
                    },
                }
            }
        ),
        _m3_call(
            f"{MINIMAX_M3_NS}<payload>"
            f"{MINIMAX_M3_NS}<cell>1{MINIMAX_M3_NS}</cell>"
            f"{MINIMAX_M3_NS}<cell>2{MINIMAX_M3_NS}</cell>"
            f"{MINIMAX_M3_NS}</payload>"
        ),
        {"payload": {"cell": [1, 2]}},
        id="repeated-key-implicit-array",
    ),
]


@pytest.mark.parametrize("streaming", [False, True], ids=["one-shot", "streaming"])
@pytest.mark.parametrize(("tools", "text", "expected"), MINIMAX_M3_NESTED_CASES)
def test_minimax_m3_nested_schema_typing_below_the_top_level(tools, text, expected, streaming):
    assert _parse_minimax_m3(tools, text, streaming=streaming) == expected


# ---------------------------------------------------------------------------
# Frontier: only MiniMax-M3 recurses into nested markup
# ---------------------------------------------------------------------------
# The other formats read a parameter value as opaque text; pinning that makes
# teaching them to recurse a deliberate change.
FLAT_SCHEMA_AWARE_PARSERS = [name for name in SCHEMA_AWARE_PARSERS if name != "minimax_m3"]


@pytest.mark.parametrize("parser_name", FLAT_SCHEMA_AWARE_PARSERS)
@pytest.mark.parametrize("streaming", [False, True], ids=["one-shot", "streaming"])
def test_nested_markup_in_parameter_value_stays_opaque_outside_minimax_m3(parser_name, streaming):
    args = _parse_args(
        parser_name,
        {
            "type": "object",
            "properties": {"code": {"type": "string"}},
        },
        None,
        "<code>1</code>",
        streaming=streaming,
        param_name="node",
    )

    assert args == {"node": "<code>1</code>"}