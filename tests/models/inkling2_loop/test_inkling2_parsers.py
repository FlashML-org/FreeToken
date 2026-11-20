"""Inkling 2's reasoning and tool-call parsers against the checkpoint's chat_template.jinja: a turn the template
renders, served back through the parsers streamed or not, renders to the same text again."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

CHECKPOINT = os.environ.get("FREETOKEN_INKLING2_MODEL", "")
END_SAMPLING = "<|content_model_end_sampling|>"

TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read a file.", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "write", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}, "lines": {"type": "array"}, "opts": {"type": "object"}}}}},
]


def _call(name: str, arguments: dict) -> dict:
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


# assistant turns as a client sends them back; the template renders each part as its own message
_TURNS = (
    {"content": "Plain answer."},
    {"reasoning_content": "Think about a < b first.", "content": "Use <b>bold</b> and a<|tag here.\n\nBye."},
    {"reasoning_content": "Read it first.", "content": "", "tool_calls": [_call("read", {"path": "a.py"})]},
    {
        "reasoning_content": "Both, then write.\nTwo lines.",
        "content": "Let me check.\n",
        "tool_calls": [
            _call("read", {"path": "a.py"}),
            _call("write", {"path": "b \"q\".py", "lines": [1, 2.5, None], "opts": {"z": True, "a": {"x": "日本 \\ }"}}}),
        ],
    },
    {"content": "", "tool_calls": [_call("read", {})]},
)


def _serve(text: str, chunk: int | None):
    """(reasoning, content, calls) the server returns for ``text``, streamed in ``chunk``-char deltas or not."""
    import asyncio

    from freetoken.message.frontend import UserReply
    from freetoken.server.generation import (
        ContentDelta, GenSpec, ReasoningDelta, ToolCallsDelta, generate_events, generate_full,
    )

    pieces = [text] if chunk is None else [text[i:i + chunk] for i in range(0, len(text), chunk)]

    class State:
        config = SimpleNamespace(tool_call_parser="inkling2", reasoning_parser="inkling2")

        async def wait_for_ack(self, uid):
            for i, piece in enumerate(pieces):
                yield UserReply(uid=uid, incremental_output=piece, finished=i == len(pieces) - 1, finish_reason=None,
                                prompt_tokens_delta=0, completion_tokens_delta=1)

    spec = GenSpec(messages=[], sampling_params=None, parser_tools=TOOLS, template_tools=TOOLS)
    if chunk is None:
        r = asyncio.run(generate_full(0, spec, State()))
        return r.reasoning, r.content, [(c.name, c.parameters) for c in r.tool_calls]

    async def events():
        return [ev async for ev in generate_events(0, spec, State())]

    evs = asyncio.run(events())
    return (
        "".join(ev.text for ev in evs if isinstance(ev, ReasoningDelta)),
        "".join(ev.text for ev in evs if isinstance(ev, ContentDelta)),
        [(c.name, c.parameters) for ev in evs if isinstance(ev, ToolCallsDelta) for c in ev.calls],
    )


@pytest.mark.needs_weights
@pytest.mark.parametrize("chunk", [None, 1, 3])
def test_a_parsed_turn_sent_back_renders_to_the_generated_text(chunk):
    from transformers import AutoTokenizer

    from freetoken.server.generation import render_messages

    if not os.path.exists(os.path.join(CHECKPOINT, "chat_template.jinja")):
        pytest.skip("set FREETOKEN_INKLING2_MODEL")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    user = {"role": "user", "content": "Hi."}
    prompt = tokenizer.apply_chat_template([user], tools=TOOLS, tokenize=False)
    for turn in _TURNS:
        full = tokenizer.apply_chat_template([user, {"role": "assistant", **turn}], tools=TOOLS, tokenize=False)
        assert full.startswith(prompt) and full.endswith(END_SAMPLING)
        # the model opens its own turn after the prompt; the detokenizer drops the stop token
        generated = full[len(prompt):-len(END_SAMPLING)]
        reasoning, content, calls = _serve(generated, chunk)
        back = {"role": "assistant", "content": content, "reasoning_content": reasoning or None,
                "tool_calls": [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls] or None}
        rendered = tokenizer.apply_chat_template(render_messages([user, back]), tools=TOOLS, tokenize=False)
        assert rendered == full, (generated, reasoning, content, calls)
