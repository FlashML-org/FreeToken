"""MiniMax-M3 reasoning parser (<mm:think>, 3 thinking modes) and tool-call
detector (namespace-delimited recursive XML) -- one-shot and streaming."""

from __future__ import annotations

import json

import pytest

from freetoken.server.function_call_parser import (
    Function,
    MiniMaxM3Detector,
    Tool,
)
from freetoken.server.reasoning_parser import MiniMaxM3ReasoningParser, ReasoningParser

NS = "]<]minimax[>["


# ---------------------------------------------------------------------------
# Reasoning parser
# ---------------------------------------------------------------------------
def test_reasoning_registered():
    assert ReasoningParser.ReasoningParserEnum["minimax_m3"] is MiniMaxM3ReasoningParser


def test_reasoning_adaptive_with_tags():
    p = MiniMaxM3ReasoningParser(force_reasoning=False)
    r = p.detect_and_parse("<mm:think>chain of thought</mm:think>The answer is 4.")
    assert r.reasoning_text == "chain of thought"
    assert r.normal_text == "The answer is 4."


def test_reasoning_adaptive_without_tags():
    p = MiniMaxM3ReasoningParser(force_reasoning=False)
    r = p.detect_and_parse("Plain answer, the model chose not to think.")
    assert r.reasoning_text == ""
    assert r.normal_text == "Plain answer, the model chose not to think."


def test_reasoning_enabled_implicit_open():
    # thinking_mode=enabled: the template pre-opens <mm:think>, the model emits
    # only the closing tag -> the caller constructs with force_reasoning=True.
    p = MiniMaxM3ReasoningParser(force_reasoning=True)
    r = p.detect_and_parse("thinking hard</mm:think>final answer")
    assert r.reasoning_text == "thinking hard"
    assert r.normal_text == "final answer"


def test_reasoning_streaming_split_marker():
    p = MiniMaxM3ReasoningParser(force_reasoning=False)
    reasoning, content = [], []
    for chunk in ["<mm:th", "ink>let me ", "think</mm:t", "hink>done: 42"]:
        r = p.parse_streaming_increment(chunk)
        reasoning.append(r.reasoning_text)
        content.append(r.normal_text)
    r = p.flush()
    reasoning.append(r.reasoning_text)
    content.append(r.normal_text)
    assert "".join(reasoning) == "let me think"
    assert "".join(content) == "done: 42"


def test_reasoning_tool_block_ends_missing_close():
    # A malformed turn skipping </mm:think> straight into a tool block: the block
    # must land in normal_text for the tool parser (dsv4 precedent).
    p = MiniMaxM3ReasoningParser(force_reasoning=True)
    text = f"half a thought{NS}<tool_call>...{NS}</tool_call>"
    r = p.detect_and_parse(text)
    assert r.reasoning_text == "half a thought"
    assert r.normal_text.startswith(f"{NS}<tool_call>")


# ---------------------------------------------------------------------------
# Tool-call detector
# ---------------------------------------------------------------------------
def _tools():
    return [
        Tool(
            function=Function(
                name="create_order",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "integer"},
                        "note": {"type": "string"},
                        "shipping": {"type": "object"},
                        "items": {"type": "array"},
                    },
                },
            )
        ),
        Tool(function=Function(name="get_weather", parameters={
            "type": "object", "properties": {"city": {"type": "string"}},
        })),
    ]


def _block(inner: str) -> str:
    return f"{NS}<tool_call>\n{inner}{NS}</tool_call>"


_NESTED_INVOKE = (
    f'{NS}<invoke name="create_order">'
    f"{NS}<user_id>42{NS}</user_id>"
    f"{NS}<note>two books{NS}</note>"
    f"{NS}<shipping>"
    f"{NS}<city>Singapore{NS}</city>"
    f"{NS}<zip>018956{NS}</zip>"
    f"{NS}</shipping>"
    f"{NS}<items>"
    f"{NS}<item>"
    f"{NS}<sku>book-001{NS}</sku>"
    f"{NS}<qty>2{NS}</qty>"
    f"{NS}</item>"
    f"{NS}</items>"
    f"{NS}</invoke>\n"
)


def test_detect_and_parse_recursive_args():
    det = MiniMaxM3Detector()
    text = "I'll place the order." + _block(_NESTED_INVOKE)
    assert det.has_tool_call(text)
    res = det.detect_and_parse(text, _tools())
    assert res.normal_text == "I'll place the order."
    assert len(res.calls) == 1
    call = res.calls[0]
    assert call.name == "create_order"
    args = json.loads(call.parameters)
    assert args["user_id"] == 42  # schema-typed integer
    assert args["note"] == "two books"  # schema-typed string
    # nested leaves are loose-JSON typed: numbers parse, but a leading-zero token
    # like "018956" is not valid JSON and stays a string (never silently mangled)
    assert args["shipping"] == {"city": "Singapore", "zip": "018956"}
    assert args["items"] == [{"sku": "book-001", "qty": 2}]


def test_detect_and_parse_multiple_invokes():
    det = MiniMaxM3Detector()
    two = (
        _NESTED_INVOKE
        + f'{NS}<invoke name="get_weather">{NS}<city>Paris{NS}</city>{NS}</invoke>\n'
    )
    res = det.detect_and_parse(_block(two), _tools())
    assert [c.name for c in res.calls] == ["create_order", "get_weather"]
    assert json.loads(res.calls[1].parameters) == {"city": "Paris"}


def test_detect_and_parse_unknown_tool_follows_forwarding_policy():
    # FreeToken forwards unknown tool names by default (FORWARD_UNKNOWN_TOOLS);
    # the detector must honor the same policy switch as every other family.
    import freetoken.server.function_call_parser as fcp

    det = MiniMaxM3Detector()
    text = _block(f'{NS}<invoke name="nope">{NS}<a>1{NS}</a>{NS}</invoke>\n')
    res = det.detect_and_parse(text, _tools())
    assert [c.name for c in res.calls] == ["nope"]  # default: forwarded

    orig = fcp.FORWARD_UNKNOWN_TOOLS
    try:
        fcp.FORWARD_UNKNOWN_TOOLS = False
        res = MiniMaxM3Detector().detect_and_parse(text, _tools())
        assert res.calls == []  # strict mode: dropped with a warning
    finally:
        fcp.FORWARD_UNKNOWN_TOOLS = orig


def test_quoted_invoke_opener_is_data_not_call():
    """A parameter value quoting the invoke wire syntax must parse as data --
    a raw regex over the block used to spawn a phantom call and diverge from
    streaming."""
    quoted = f'see {NS}<invoke name="write_file"> for the syntax'
    text = _block(
        f'{NS}<invoke name="get_weather">{NS}<city>{quoted}{NS}</city>{NS}</invoke>\n'
    )
    res = MiniMaxM3Detector().detect_and_parse(text, _tools())
    assert [c.name for c in res.calls] == ["get_weather"]
    assert json.loads(res.calls[0].parameters) == {"city": quoted}
    # streaming agrees (it already produced one call before the fix; pin it)
    det = MiniMaxM3Detector()
    r = det.parse_streaming_increment(text, _tools())
    assert [c.name for c in r.calls if c.name is not None] == ["get_weather"]


def test_quoted_wrapper_closer_is_data_not_block_end():
    """A value quoting NS</tool_call> must not end the wrapper: before the fix
    the parameter was dropped ({}) and the rest of the block leaked as raw
    markup content."""
    quoted = f"wrap calls in {NS}</tool_call> at the end"
    inner = f'{NS}<invoke name="get_weather">{NS}<city>{quoted}{NS}</city>{NS}</invoke>\n'
    text = _block(inner) + " done."
    res = MiniMaxM3Detector().detect_and_parse(text, _tools())
    assert [c.name for c in res.calls] == ["get_weather"]
    assert json.loads(res.calls[0].parameters) == {"city": quoted}
    assert res.normal_text == "done."
    # chunked streaming: same call, no markup leak
    det = MiniMaxM3Detector()
    calls, texts = [], []
    for i in range(0, len(text), 7):
        r = det.parse_streaming_increment(text[i : i + 7], _tools())
        calls += [c for c in r.calls if c.name is not None]
        texts.append(r.normal_text)
    assert [c.name for c in calls] == ["get_weather"]
    assert NS not in "".join(texts)


def test_multiline_string_argument_verbatim():
    """The template renders leaf values verbatim; the round-trip must preserve
    them verbatim too (stripping ate the trailing newline of multi-line string
    arguments)."""
    tools = [
        Tool(function=Function(name="write_file", parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        }))
    ]
    content = "line1\nline2\n"
    text = _block(
        f'{NS}<invoke name="write_file">'
        f"{NS}<path>/tmp/x{NS}</path>"
        f"{NS}<content>{content}{NS}</content>"
        f"{NS}</invoke>\n"
    )
    res = MiniMaxM3Detector().detect_and_parse(text, tools)
    assert json.loads(res.calls[0].parameters) == {"path": "/tmp/x", "content": content}


def test_nested_schema_types_string_leaves():
    """Nested leaves honor the declared schema (string-typed fields stay
    strings instead of loose-JSON numbers/bools); undeclared nested leaves keep
    the loose-JSON fallback."""
    tools = [
        Tool(function=Function(name="create_order", parameters={
            "type": "object",
            "properties": {
                "shipping": {
                    "type": "object",
                    "properties": {
                        "zip": {"type": "string"},
                        "note": {"type": "string"},
                        "floor": {"type": "integer"},
                    },
                },
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        }))
    ]
    text = _block(
        f'{NS}<invoke name="create_order">'
        f"{NS}<shipping>"
        f"{NS}<zip>94107{NS}</zip>"
        f"{NS}<note>true{NS}</note>"
        f"{NS}<floor>3{NS}</floor>"
        f"{NS}</shipping>"
        f"{NS}<tags>"
        f"{NS}<item>1{NS}</item>"
        f"{NS}<item>two{NS}</item>"
        f"{NS}</tags>"
        f"{NS}</invoke>\n"
    )
    args = json.loads(MiniMaxM3Detector().detect_and_parse(text, tools).calls[0].parameters)
    assert args["shipping"] == {"zip": "94107", "note": "true", "floor": 3}
    assert args["tags"] == ["1", "two"]


def test_streaming_text_then_call():
    det = MiniMaxM3Detector()
    full = "Let me check." + _block(
        f'{NS}<invoke name="get_weather">{NS}<city>Paris{NS}</city>{NS}</invoke>\n'
    )
    # Chop into small chunks that split the namespace marker mid-token.
    chunks = [full[i : i + 7] for i in range(0, len(full), 7)]
    normal, calls = "", []
    for ch in chunks:
        r = det.parse_streaming_increment(ch, _tools())
        normal += r.normal_text
        calls.extend(r.calls)
    assert normal == "Let me check."
    named = [c for c in calls if c.name]
    frags = [c for c in calls if c.name is None and c.parameters]
    assert [c.name for c in named] == ["get_weather"]
    assert json.loads("".join(c.parameters for c in frags)) == {"city": "Paris"}
    # ledgers the serving layer reads at stream end
    assert det.prev_tool_call_arr[0]["name"] == "get_weather"
    assert det.prev_tool_call_arr[0]["arguments"] == {"city": "Paris"}


def test_streaming_multiple_invokes_and_indices():
    det = MiniMaxM3Detector()
    two = (
        _NESTED_INVOKE
        + f'{NS}<invoke name="get_weather">{NS}<city>Paris{NS}</city>{NS}</invoke>\n'
    )
    full = _block(two)
    chunks = [full[i : i + 13] for i in range(0, len(full), 13)]
    calls = []
    for ch in chunks:
        calls.extend(det.parse_streaming_increment(ch, _tools()).calls)
    named = [(c.tool_index, c.name) for c in calls if c.name]
    assert named == [(0, "create_order"), (1, "get_weather")]


def test_streaming_plain_text_never_held():
    det = MiniMaxM3Detector()
    r = det.parse_streaming_increment("Just a normal answer, no tools.", _tools())
    assert r.normal_text == "Just a normal answer, no tools."
    assert r.calls == []


def test_auto_selection_picks_minimax_m3():
    from unittest.mock import patch

    from freetoken.server.args import parse_args

    class _Config:
        def to_dict(self):
            return {
                "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
                "model_type": "minimax_m3_vl",
                "text_config": {"model_type": "minimax_m2"},
            }

    with patch("freetoken.utils.cached_load_hf_config", lambda _p: _Config()):
        args, _ = parse_args(["--model", "/models/anon"])
    assert args.tool_call_parser == "minimax_m3"
    assert args.reasoning_parser == "minimax_m3"


# ---------------------------------------------------------------------------
# Review regressions
# ---------------------------------------------------------------------------
def test_reasoning_adaptive_leading_bare_closer_one_shot():
    # Adaptive non-thinking turns START with a bare </mm:think> written by the
    # model; it must never leak into content (position 0 only).
    p = MiniMaxM3ReasoningParser(force_reasoning=False)
    r = p.detect_and_parse("</mm:think>Here is the answer.")
    assert r.reasoning_text == ""
    assert r.normal_text == "Here is the answer."


def test_reasoning_adaptive_leading_bare_closer_streaming_split():
    p = MiniMaxM3ReasoningParser(force_reasoning=False)
    normal = ""
    for chunk in ["</mm:th", "ink>Hello,", " world!"]:
        r = p.parse_streaming_increment(chunk)
        assert r.reasoning_text == ""
        normal += r.normal_text
    normal += p.flush().normal_text
    assert normal == "Hello, world!"


def test_reasoning_leading_closer_after_whitespace():
    # The detokenizer may open with a newline before the model's bare closer;
    # the whitespace-tolerant head still strips it (one-shot and streaming).
    p = MiniMaxM3ReasoningParser(force_reasoning=False)
    r = p.detect_and_parse("\n</mm:think>Here is the answer.")
    assert r.reasoning_text == "" and r.normal_text == "Here is the answer."

    p = MiniMaxM3ReasoningParser(force_reasoning=False)
    normal = ""
    for chunk in ["\n", "</mm:think>", "Hi."]:
        out = p.parse_streaming_increment(chunk)
        assert out.reasoning_text == ""
        normal += out.normal_text
    normal += p.flush().normal_text
    assert normal == "Hi."


def test_reasoning_later_closer_stays_visible():
    # Only a position-0 closer is stripped; one appearing later is content.
    p = MiniMaxM3ReasoningParser(force_reasoning=False)
    r = p.detect_and_parse("The tag </mm:think> is literal here.")
    assert "</mm:think>" in r.normal_text


def test_reasoning_forced_mode_closer_not_stripped():
    # thinking_mode=enabled: the template pre-opened the block, so a leading
    # closer legitimately ends (empty) reasoning -- content follows.
    p = MiniMaxM3ReasoningParser(force_reasoning=True)
    r = p.detect_and_parse("</mm:think>answer")
    assert r.reasoning_text == "" and r.normal_text == "answer"


def test_reasoning_streaming_head_prefix_replayed_on_divergence():
    p = MiniMaxM3ReasoningParser(force_reasoning=False)
    out = p.parse_streaming_increment("</mm")  # closer prefix: held
    assert out.normal_text == "" and out.reasoning_text == ""
    out = p.parse_streaming_increment("ory says hi")  # diverges: replayed
    assert out.normal_text == "</mmory says hi"


def test_streaming_wire_order_trailing_text_defers():
    # "text + block + trailing text" in ONE chunk: the trailing text must not be
    # emitted in the same increment as (i.e. wire-ahead of) the tool call.
    det = MiniMaxM3Detector()
    chunk = (
        "Before. "
        + _block(f'{NS}<invoke name="get_weather">{NS}<city>Paris{NS}</city>{NS}</invoke>\n')
        + " After."
    )
    r1 = det.parse_streaming_increment(chunk, _tools())
    assert r1.normal_text == "Before. "
    assert [c.name for c in r1.calls if c.name] == ["get_weather"]
    r2 = det.parse_streaming_increment("", _tools())
    assert r2.normal_text == " After." and r2.calls == []


def test_streaming_second_block_in_same_chunk():
    det = MiniMaxM3Detector()
    one = _block(f'{NS}<invoke name="get_weather">{NS}<city>Paris{NS}</city>{NS}</invoke>\n')
    two = _block(f'{NS}<invoke name="get_weather">{NS}<city>Tokyo{NS}</city>{NS}</invoke>\n')
    calls = []
    calls += det.parse_streaming_increment(one + two, _tools()).calls
    calls += det.parse_streaming_increment("", _tools()).calls
    args = [json.loads(c.parameters) for c in calls if c.name is None and c.parameters]
    assert args == [{"city": "Paris"}, {"city": "Tokyo"}]


def test_streaming_residue_after_close_holds_partial_marker():
    # After a closed wrapper, a split second-block marker in the residue must be
    # held (idle-path hold), never leaked as content.
    det = MiniMaxM3Detector()
    one = _block(f'{NS}<invoke name="get_weather">{NS}<city>Paris{NS}</city>{NS}</invoke>\n')
    r1 = det.parse_streaming_increment(one + "]<]mini", _tools())
    assert "]<]mini" not in r1.normal_text
    r2 = det.parse_streaming_increment(
        f'max[>[<tool_call>\n{NS}<invoke name="get_weather">{NS}<city>Rome{NS}</city>'
        f"{NS}</invoke>\n{NS}</tool_call>",
        _tools(),
    )
    frags = [c.parameters for c in r2.calls if c.name is None and c.parameters]
    assert json.loads(frags[-1]) == {"city": "Rome"}


def test_streaming_truncated_call_suppressed_and_recovered():
    # Generation cut mid-invoke (max_tokens): finish_streaming must not leak the
    # raw markup, and recover_truncated_call must salvage the complete params.
    from freetoken.server.function_call_parser import FunctionCallParser

    parser = FunctionCallParser(_tools(), "minimax_m3")
    det = parser.detector
    truncated = (
        "Let me order. " + NS + "<tool_call>\n"
        + f'{NS}<invoke name="create_order">{NS}<user_id>42{NS}</user_id>{NS}<note>hi'
    )
    normal = ""
    for i in range(0, len(truncated), 11):
        r = det.parse_streaming_increment(truncated[i : i + 11], _tools())
        normal += r.normal_text
        assert not r.calls
    assert normal == "Let me order. "
    # Serving order (generation.py): recover first (consumes the buffer on
    # success), then finish_stream drains what's left.
    recovered = parser.recover_truncated_call()
    assert [c.name for c in recovered] == ["create_order"]
    args = json.loads(recovered[0].parameters)
    assert args["user_id"] == 42  # the complete param survives; truncated one dropped
    assert det.finish_streaming() == ""  # raw markup never reaches the client


def test_args_stray_text_keeps_parsed_params():
    det = MiniMaxM3Detector()
    body = f"{NS}<city>Paris{NS}</city>zzz-stray"
    text = _block(f'{NS}<invoke name="get_weather">{body}{NS}</invoke>\n')
    res = det.detect_and_parse(text, _tools())
    assert json.loads(res.calls[0].parameters) == {"city": "Paris"}


def test_args_empty_value_is_empty_string():
    det = MiniMaxM3Detector()
    text = _block(f'{NS}<invoke name="get_weather">{NS}<city>{NS}</city>{NS}</invoke>\n')
    res = det.detect_and_parse(text, _tools())
    assert json.loads(res.calls[0].parameters) == {"city": ""}


def test_args_repeated_siblings_keep_parent_key():
    """Only <item> children render as a bare array; repeated same-name siblings
    under any other tag stay an object with an array-valued key (the bare-list
    collapse dropped the parent key the references keep)."""
    det = MiniMaxM3Detector()
    body = (
        f"{NS}<items>{NS}<sku>a{NS}</sku>{NS}<sku>b{NS}</sku>{NS}</items>"
        f"{NS}<note>x{NS}</note>{NS}<note>y{NS}</note>"
    )
    text = _block(f'{NS}<invoke name="create_order">{body}{NS}</invoke>\n')
    res = det.detect_and_parse(text, _tools())
    args = json.loads(res.calls[0].parameters)
    assert args["items"] == {"sku": ["a", "b"]}  # parent key kept, values aggregate
    assert args["note"] == ["x", "y"]  # repeated top-level params still aggregate


def test_element_semantics_reference_batch():
    """Element-grammar batch: empty container-typed params, anyOf coercion,
    dangling-closer leniency, single-quoted invoke name, $text for mixed
    content, and float preservation."""
    tools = [
        Tool(function=Function(name="t", parameters={
            "type": "object",
            "properties": {
                "obj": {"type": "object"},
                "arr": {"type": "array"},
                "uni": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                "price": {"type": "number"},
            },
        }))
    ]
    det = MiniMaxM3Detector()
    body = (
        f"{NS}<obj>{NS}</obj>"
        f"{NS}<arr>{NS}</arr>"
        f"{NS}<uni>42{NS}</uni>"
        f"{NS}<price>5.0{NS}</price>"
    )
    text = _block(f'{NS}<invoke name="t">{body}{NS}</invoke>\n')
    args = json.loads(det.detect_and_parse(text, tools).calls[0].parameters)
    assert args["obj"] == {} and args["arr"] == []  # typed empty containers
    assert args["uni"] == 42  # anyOf: integer member coerces
    assert args["price"] == 5.0 and isinstance(args["price"], float)  # not 5

    # anyOf falls back to the string member verbatim
    text2 = _block(f'{NS}<invoke name="t">{NS}<uni>hello{NS}</uni>{NS}</invoke>\n')
    args2 = json.loads(MiniMaxM3Detector().detect_and_parse(text2, tools).calls[0].parameters)
    assert args2["uni"] == "hello"

    # a dangling closer no longer drops the well-formed siblings after it
    body3 = f"{NS}</ghost>{NS}<uni>7{NS}</uni>"
    text3 = _block(f'{NS}<invoke name="t">{body3}{NS}</invoke>\n')
    args3 = json.loads(MiniMaxM3Detector().detect_and_parse(text3, tools).calls[0].parameters)
    assert args3["uni"] == 7

    # single-quoted name= parses (double quotes are the template's rendering,
    # but models emit single quotes too)
    text4 = _block(f"{NS}<invoke name='t'>{NS}<uni>1{NS}</uni>{NS}</invoke>\n")
    res4 = MiniMaxM3Detector().detect_and_parse(text4, tools)
    assert [c.name for c in res4.calls] == ["t"]

    # mixed text+children keeps the text under $text
    body5 = f"{NS}<obj>freeform {NS}<k>v{NS}</k>{NS}</obj>"
    text5 = _block(f'{NS}<invoke name="t">{body5}{NS}</invoke>\n')
    args5 = json.loads(MiniMaxM3Detector().detect_and_parse(text5, tools).calls[0].parameters)
    assert args5["obj"] == {"k": "v", "$text": "freeform"}


def test_reasoning_one_shot_matches_streaming_positionally():
    """One-shot and streaming must agree positionally: prose before a
    mid-content <mm:think> stays content, and a later quoted marker occurrence
    is data."""
    text = "intro <mm:think>deep thought</mm:think> answer quoting <mm:think> literally"
    one = MiniMaxM3ReasoningParser(force_reasoning=False).detect_and_parse(text)
    assert one.reasoning_text == "deep thought"
    assert one.normal_text == "intro  answer quoting <mm:think> literally"

    p = MiniMaxM3ReasoningParser(force_reasoning=False)
    reasoning, content = [], []
    for i in range(0, len(text), 3):
        r = p.parse_streaming_increment(text[i : i + 3])
        reasoning.append(r.reasoning_text)
        content.append(r.normal_text)
    r = p.flush()
    reasoning.append(r.reasoning_text)
    content.append(r.normal_text)
    assert "".join(reasoning) == one.reasoning_text
    assert "".join(content) == one.normal_text


def test_reasoning_one_shot_verbatim_no_strip():
    # whitespace around reasoning/content is data (references are verbatim)
    r = MiniMaxM3ReasoningParser(force_reasoning=False).detect_and_parse(
        "<mm:think>\nthink\n</mm:think>\nanswer\n"
    )
    assert r.reasoning_text == "\nthink\n"
    assert r.normal_text == "\nanswer\n"
    r2 = MiniMaxM3ReasoningParser(force_reasoning=True).detect_and_parse(
        "th\n</mm:think>\nans\n"
    )
    assert r2.reasoning_text == "th\n" and r2.normal_text == "\nans\n"


def test_detect_and_parse_multiple_wrappers_and_inter_block_text():
    det = MiniMaxM3Detector()
    one = _block(f'{NS}<invoke name="get_weather">{NS}<city>Paris{NS}</city>{NS}</invoke>\n')
    two = _block(f'{NS}<invoke name="get_weather">{NS}<city>Rome{NS}</city>{NS}</invoke>\n')
    res = det.detect_and_parse("A. " + one + " middle " + two + " B.", _tools())
    assert [json.loads(c.parameters)["city"] for c in res.calls] == ["Paris", "Rome"]
    assert "middle" in res.normal_text and "A." in res.normal_text and "B." in res.normal_text


@pytest.mark.parametrize(
    "gear,mode", [("off", "disabled"), ("adaptive", "adaptive"), ("on", "enabled")]
)
def test_think_gears(gear, mode):
    """M3's three thinking states are discovered from its template behavior
    (thinking_mode disabled/adaptive/enabled, template default adaptive)."""
    from freetoken.server.model_meta import derive_think_gears
    from freetoken.tokenizer.effort import EffortProfile, probe_thinking_profile

    def m3_render(kwargs, tools):
        # The template reads thinking_mode only; adaptive is its own default.
        return f"m3|{kwargs.get('thinking_mode', 'adaptive')}"

    no_efforts = EffortProfile(supported=frozenset(), default=None, consumes_effort=False)
    profile = probe_thinking_profile(m3_render, no_efforts)
    gears, default, kwargs = derive_think_gears(profile, parser_configured=True)
    assert gears == ("off", "adaptive", "on") and default == "adaptive"
    assert kwargs[gear]["thinking_mode"] == mode
