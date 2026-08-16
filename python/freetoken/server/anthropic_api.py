"""Anthropic Messages API (`/v1/messages`) — the endpoint Claude Code talks to.

Adapted from vLLM's ``AnthropicServingMessages``. FreeToken's serving layer is
function-based, so instead of subclassing the chat server this module drives the
shared protocol-neutral generation primitive
(:func:`freetoken.server.openai_api.submit_generation` + ``generate_events`` /
``generate_full``) and formats its semantic events into the Anthropic wire shape.
It consumes typed events, not a re-parsed OpenAI stream.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from .anthropic_models import (
    AnthropicContentBlock,
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicDelta,
    AnthropicError,
    AnthropicErrorResponse,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicStreamEvent,
    AnthropicUsage,
)
from .generation import (
    KEEPALIVE,
    ContentDelta,
    GenDone,
    GenerationError,
    GenResult,
    GenSpec,
    ReasoningDelta,
    ToolCallArgsDelta,
    ToolCallsDelta,
    ToolCallStart,
    count_prompt_tokens,
    generate_events,
    generate_full,
    render_messages,
    resolve_sampling,
    split_tool_lists,
    submit_generation,
    with_keepalive,
)
from .request_logger import log_request

# Emit a protocol-native `ping` event after this many seconds of stream silence,
# bridging long queue/prefill/decode gaps for clients with stream-idle timeouts.
KEEPALIVE_INTERVAL_S = 15.0

# OpenAI finish_reason -> Anthropic stop_reason (vLLM's stop_reason_map).
STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


def _anthropic_stop(finish_reason: str | None, matched_stop: str | None) -> tuple[str | None, str | None]:
    """(stop_reason, stop_sequence). A stop-string hit is reported as 'stop_sequence'
    with the matched string, otherwise the finish_reason is mapped normally."""
    if matched_stop is not None:
        return "stop_sequence", matched_stop
    return STOP_REASON_MAP.get(finish_reason or "stop"), None


def register_anthropic_routes(
    app: FastAPI,
    get_state: Callable[[], Any],
    get_model_sampling: Callable[[], dict[str, Any]],
) -> None:
    @app.post("/v1/messages")
    async def v1_messages(req: AnthropicMessagesRequest, request: Request):
        log_request("/v1/messages", req, request)
        state = get_state()
        mstate = getattr(state, "maintenance_state", "serving")
        if mstate != "serving":
            detail = "model is still loading" if mstate == "loading" else "cache rebuild in progress"
            return _anthropic_error_response(503, "overloaded_error", detail)
        return await handle_anthropic_messages(req, request, state, get_model_sampling())

    @app.post("/v1/messages/count_tokens")
    async def v1_messages_count_tokens(req: AnthropicCountTokensRequest, request: Request):
        log_request("/v1/messages/count_tokens", req, request)
        return await handle_anthropic_count_tokens(req, get_state())

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        # Anthropic clients parse errors as {"type": "error", "error": {...}}; FastAPI's
        # default 422 {"detail": [...]} is unreadable to them. Scoped to /v1/messages*
        # so the OpenAI-protocol routes keep their existing wire behavior.
        if request.url.path.startswith("/v1/messages"):
            return _anthropic_error_response(
                400, "invalid_request_error", _validation_error_message(exc)
            )
        return await request_validation_exception_handler(request, exc)


async def handle_anthropic_messages(
    req: AnthropicMessagesRequest,
    request: Request | None,
    state: Any,
    model_sampling: dict[str, Any],
):
    try:
        spec = convert_anthropic_to_genspec(
            req, model_sampling,
            reasoning_parser=getattr(state.config, "reasoning_parser", None),
        )
        uid = await submit_generation(spec, state)
    except ValueError as exc:
        return _anthropic_error_response(400, "invalid_request_error", str(exc))

    cache_report = getattr(state.config, "enable_cache_report", False)
    if req.stream:
        events = anthropic_event_stream(
            generate_events(uid, spec, state, source="/v1/messages"),
            req.model, uid, cache_report=cache_report,
        )
        if request is not None:
            events = state.stream_with_cancellation(events, request, uid)
        return StreamingResponse(events, media_type="text/event-stream")

    try:
        result = await generate_full(uid, spec, state, source="/v1/messages")
    except GenerationError as exc:
        return _anthropic_error_response(400, "invalid_request_error", str(exc))
    response = anthropic_full_response(result, req.model, uid, cache_report=cache_report)
    return JSONResponse(content=response.model_dump(exclude_none=True))


async def handle_anthropic_count_tokens(req: AnthropicCountTokensRequest, state: Any):
    if not req.messages:
        return _anthropic_error_response(
            400, "invalid_request_error", "messages: at least one message is required"
        )
    # Client-input validation (400) is kept strictly separate from tokenizer execution (500):
    # a checkpoint whose chat template raises ValueError is a server fault, not a bad request,
    # so it must not fall into the convert/empty-prompt ValueError branch.
    try:
        messages, template_tools, _, ctk = convert_anthropic_prompt(
            req, reasoning_parser=getattr(state.config, "reasoning_parser", None)
        )
    except ValueError as exc:
        return _anthropic_error_response(400, "invalid_request_error", str(exc))
    if not messages:
        # Non-empty on the wire but nothing survived conversion (e.g. image-only blocks on
        # this text-only server) — a client error, not a tokenizer fault.
        return _anthropic_error_response(
            400, "invalid_request_error", "messages: no tokenizable content"
        )
    try:
        n_tokens = await count_prompt_tokens(messages, template_tools, ctk, state)
    except GenerationError as exc:
        # The chat template could not render this conversation (bad role ordering, an unmatched
        # tool_result, ...) — a client error, exactly as /v1/messages classifies the same failure.
        return _anthropic_error_response(400, "invalid_request_error", str(exc))
    except Exception as exc:  # noqa: BLE001 — tokenizer init / other failure -> server error
        return _anthropic_error_response(500, "api_error", f"token counting failed: {exc}")
    response = AnthropicCountTokensResponse(input_tokens=n_tokens)
    return JSONResponse(content=response.model_dump())


# --------------------------------------------------------------------------- #
# Request conversion: Anthropic Messages -> GenSpec (the neutral generation spec;
# built directly, like vLLM's Responses path — no ChatCompletionRequest pivot).
# --------------------------------------------------------------------------- #
def convert_anthropic_prompt(
    req: AnthropicMessagesRequest | AnthropicCountTokensRequest,
    reasoning_parser: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, list[dict[str, Any]] | None, dict[str, Any]]:
    """(messages, template_tools, parser_tools, chat_template_kwargs) — the prompt
    side of the conversion, shared by /v1/messages and /v1/messages/count_tokens so
    a counted prompt is exactly the prompt a generation would tokenize."""
    # Collect all system content (top-level `system` + any system-role messages
    # Claude Code interleaves in the array) and emit ONE system message at the
    # front: strict chat templates (e.g. Qwen3.5) require system at the beginning.
    system_texts: list[str] = []
    if req.system:
        if isinstance(req.system, str):
            system_texts.append(req.system)
        else:
            system_texts.append(
                "".join(b.text for b in req.system if b.type == "text" and b.text)
            )

    other: list[dict[str, Any]] = []
    for msg in req.messages:
        if msg.role == "system":
            system_texts.append(_content_text(msg.content))
            continue

        if isinstance(msg.content, str):
            other.append({"role": msg.role, "content": msg.content})
            continue

        content_parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        thinking_parts: list[str] = []
        for block in msg.content:
            if block.type == "text" and block.text:
                content_parts.append({"type": "text", "text": block.text})
            elif block.type == "thinking" and block.thinking:
                # -> reasoning_content; redacted_thinking stays skipped (opaque payload).
                thinking_parts.append(block.thinking)
            elif block.type == "image":
                # Text-only server: drop image blocks rather than failing the request.
                continue
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id or f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": block.name or "",
                            "arguments": json.dumps(block.input or {}),
                        },
                    }
                )
            elif block.type == "tool_result":
                if msg.role == "user":
                    other.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.tool_use_id or block.id or "",
                            "content": _tool_result_text(block.content),
                        }
                    )
                else:
                    content_parts.append(
                        {
                            "type": "text",
                            "text": f"Tool result: {_tool_result_text(block.content)}",
                        }
                    )

        openai_msg: dict[str, Any] = {"role": msg.role}
        if thinking_parts:
            openai_msg["reasoning_content"] = "\n\n".join(thinking_parts)
        if tool_calls:
            openai_msg["tool_calls"] = tool_calls
        if content_parts:
            if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                openai_msg["content"] = content_parts[0]["text"]
            else:
                openai_msg["content"] = content_parts
        elif not tool_calls and not thinking_parts:
            # Nothing usable in this message (e.g. image-only) — skip it.
            continue
        other.append(openai_msg)

    messages: list[dict[str, Any]] = []
    system_text = "\n\n".join(t for t in system_texts if t)
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.extend(other)

    raw_tools = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in (req.tools or [])
    ]
    # Anthropic tool_choice: "none" hides the tools from the model entirely;
    # "auto"/"any" parse all tools, "tool" forces one.
    if req.tool_choice and req.tool_choice.type == "none":
        template_tools, parser_tools = None, None
    else:
        selected = req.tool_choice.name if (req.tool_choice and req.tool_choice.type == "tool") else None
        template_tools, parser_tools = split_tool_lists(raw_tools, selected)

    # Native extended-thinking toggle -> template kwargs, broadcast in every
    # spelling the ecosystem's templates read (a bare enable_thinking bool is
    # inert for templates that read a different knob, e.g. M3's thinking_mode).
    from .model_meta import thinking_toggle_kwargs

    ctk: dict[str, Any] = {}
    if req.thinking:
        if req.thinking.get("type") == "enabled":
            ctk = thinking_toggle_kwargs(True)
        elif req.thinking.get("type") == "disabled":
            ctk = thinking_toggle_kwargs(False)

    return render_messages(messages), template_tools, parser_tools, ctk


def convert_anthropic_to_genspec(
    req: AnthropicMessagesRequest,
    model_sampling: dict[str, Any],
    reasoning_parser: str | None = None,
) -> GenSpec:
    messages, template_tools, parser_tools, ctk = convert_anthropic_prompt(
        req, reasoning_parser=reasoning_parser
    )
    return GenSpec(
        messages=messages,
        sampling_params=resolve_sampling(
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            ignore_eos=False,
            model_sampling=model_sampling,
            stop=req.stop_sequences,
        ),
        chat_template_kwargs=ctk,
        template_tools=template_tools,
        parser_tools=parser_tools,
    )


def _content_text(content) -> str:
    """Plain text of an Anthropic message content (str or list of blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(b.text for b in content if getattr(b, "type", None) == "text" and b.text)


def _tool_result_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            parts.append(item.get("text") or "")
        else:
            parts.append(str(item))
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Output formatting: GenResult / GenEvent -> Anthropic response / events
# --------------------------------------------------------------------------- #
def anthropic_full_response(
    result: GenResult, model: str, uid: int, cache_report: bool = False
) -> AnthropicMessagesResponse:
    content: list[AnthropicContentBlock] = []
    if result.reasoning:
        content.append(
            AnthropicContentBlock(type="thinking", thinking=result.reasoning, signature="")
        )
    content.append(AnthropicContentBlock(type="text", text=result.content or ""))
    for index, call in enumerate(result.tool_calls):
        content.append(
            AnthropicContentBlock(
                type="tool_use",
                id=_tool_use_id(call.name, index),
                name=call.name,
                input=_parse_json_args(call.parameters),
            )
        )
    stop_reason, stop_sequence = _anthropic_stop(result.finish_reason, result.matched_stop)
    return AnthropicMessagesResponse(
        id=f"msg_{uid}",
        content=content,
        model=model,
        stop_reason=stop_reason,
        stop_sequence=stop_sequence,
        usage=_anthropic_usage(
            result.prompt_tokens, result.completion_tokens, result.cached_tokens, cache_report
        ),
    )


def _anthropic_usage(
    prompt_tokens: int, completion_tokens: int, cached_tokens: int, cache_report: bool
) -> AnthropicUsage:
    """Anthropic billing semantics under --enable-cache-report: input_tokens EXCLUDES
    the cached prefix and cache_read_input_tokens carries it (absent when 0, matching
    sglang). With the flag off, input_tokens is the full prompt length."""
    cached = cached_tokens if cache_report else 0
    return AnthropicUsage(
        input_tokens=max(prompt_tokens - cached, 0),
        output_tokens=completion_tokens,
        cache_read_input_tokens=cached or None,
    )


async def anthropic_event_stream(
    events: AsyncIterator[Any], model: str, uid: int, cache_report: bool = False
) -> AsyncIterator[str]:
    """Format the protocol-neutral GenEvent stream into Anthropic SSE events.

    No re-parsing of an OpenAI stream: this consumes ReasoningDelta / ContentDelta /
    ToolCallsDelta / GenDone directly. The terminal message_delta + message_stop are
    driven by GenDone (always present), so there is no usage-chunk dependency.
    """
    block_index = 0
    block_open: str | None = None  # "text" | "thinking" (tool_use blocks open+close atomically)

    def _open_text() -> str:
        nonlocal block_open
        block_open = "text"
        return _event(AnthropicStreamEvent(
            type="content_block_start", index=block_index,
            content_block=AnthropicContentBlock(type="text", text=""),
        ))

    def _open_thinking() -> str:
        nonlocal block_open
        block_open = "thinking"
        return _event(AnthropicStreamEvent(
            type="content_block_start", index=block_index,
            content_block=AnthropicContentBlock(type="thinking", thinking=""),
        ))

    def _stop_block() -> list[str]:
        nonlocal block_open, block_index
        frames: list[str] = []
        if block_open == "thinking":
            # Real thinking blocks end with a signature_delta; emit an empty one for
            # shape compliance (this server has no signing key and never verifies
            # signatures on replayed thinking blocks).
            frames.append(_event(AnthropicStreamEvent(
                type="content_block_delta", index=block_index,
                delta=AnthropicDelta(type="signature_delta", signature=""),
            )))
        frames.append(_event(AnthropicStreamEvent(type="content_block_stop", index=block_index)))
        block_open = None
        block_index += 1
        return frames

    # message_start (input_tokens filled in by the terminal message_delta).
    yield _event(AnthropicStreamEvent(
        type="message_start",
        message=AnthropicMessagesResponse(
            id=f"msg_{uid}", content=[], model=model,
            usage=AnthropicUsage(input_tokens=0, output_tokens=0),
        ),
    ))

    def _open_tool(name: str | None, ordinal: int | None, stable: bool = True) -> list[str]:
        nonlocal block_open, tool_args_sent, tool_ordinal, tool_stable
        frames: list[str] = []
        if block_open:
            frames.extend(_stop_block())
        frames.append(_event(AnthropicStreamEvent(
            type="content_block_start", index=block_index,
            content_block=AnthropicContentBlock(
                type="tool_use", id=_tool_use_id(name, block_index),
                name=name, input={},
            ),
        )))
        block_open = "tool"
        tool_args_sent = ""
        tool_ordinal = ordinal
        tool_stable = stable
        return frames

    def _tool_args_delta(fragment: str) -> str:
        nonlocal tool_args_sent
        tool_args_sent += fragment
        return _event(AnthropicStreamEvent(
            type="content_block_delta", index=block_index,
            delta=AnthropicDelta(type="input_json_delta", partial_json=fragment),
        ))

    tool_args_sent = ""
    tool_ordinal: int | None = None
    tool_stable = True
    events = with_keepalive(events, KEEPALIVE_INTERVAL_S)
    try:
        async for ev in events:
            if ev is KEEPALIVE:
                yield _event(AnthropicStreamEvent(type="ping"))

            elif isinstance(ev, ReasoningDelta):
                # Streamed as a native thinking block (Claude Code renders these).
                if ev.text == "":
                    continue
                if block_open != "thinking":
                    if block_open:
                        for f in _stop_block():
                            yield f
                    yield _open_thinking()
                yield _event(AnthropicStreamEvent(
                    type="content_block_delta", index=block_index,
                    delta=AnthropicDelta(type="thinking_delta", thinking=ev.text),
                ))

            elif isinstance(ev, ContentDelta):
                if ev.text == "":
                    continue
                if block_open != "text":
                    if block_open:
                        for f in _stop_block():
                            yield f
                    yield _open_text()
                yield _event(AnthropicStreamEvent(
                    type="content_block_delta", index=block_index,
                    delta=AnthropicDelta(type="text_delta", text=ev.text),
                ))

            elif isinstance(ev, ToolCallStart):
                for f in _open_tool(ev.name, ev.tool_index, ev.args_prefix_stable):
                    yield f

            elif isinstance(ev, ToolCallArgsDelta):
                if block_open != "tool":
                    continue  # defensive: fragment without an open tool block
                if not tool_stable:
                    continue  # fragments don't concatenate cleanly: full args at close
                yield _tool_args_delta(ev.fragment)

            elif isinstance(ev, ToolCallsDelta):
                for call in ev.calls:
                    if block_open == "tool" and tool_ordinal == call.tool_index:
                        # Close of a ToolCallStart-opened block: top up whatever of
                        # the final (authoritative) arguments wasn't streamed yet.
                        final = call.parameters or ""
                        if final.startswith(tool_args_sent):
                            remainder = final[len(tool_args_sent):]
                            if remainder:
                                yield _tool_args_delta(remainder)
                        for f in _stop_block():
                            yield f
                        continue
                    # Standalone complete call (buffered fallback path).
                    for f in _open_tool(call.name, call.tool_index):
                        yield f
                    yield _tool_args_delta(call.parameters or "")
                    for f in _stop_block():
                        yield f

            elif isinstance(ev, GenDone):
                if block_open:
                    for f in _stop_block():
                        yield f
                stop_reason, stop_sequence = _anthropic_stop(ev.finish_reason, ev.matched_stop)
                yield _event(AnthropicStreamEvent(
                    type="message_delta",
                    delta=AnthropicDelta(stop_reason=stop_reason, stop_sequence=stop_sequence),
                    usage=_anthropic_usage(
                        ev.prompt_tokens, ev.completion_tokens, ev.cached_tokens, cache_report
                    ),
                ))
                yield _event(AnthropicStreamEvent(type="message_stop"))
                # Anthropic streams terminate on message_stop — no OpenAI-style
                # `data: [DONE]` sentinel (it would be an unknown event to strict clients).
    except GenerationError as exc:
        # Request-side failure (template rejection / over-length prompt). Match the
        # non-streaming path's classification so Claude Code treats it as a client error
        # rather than a server fault to retry.
        if block_open:
            for f in _stop_block():
                yield f
        yield _event(AnthropicStreamEvent(
            type="error", error=AnthropicError(type="invalid_request_error", message=str(exc)),
        ))
    except Exception as exc:  # noqa: BLE001 — surface as an Anthropic error event
        if block_open:
            for f in _stop_block():
                yield f
        yield _event(AnthropicStreamEvent(
            type="error", error=AnthropicError(type="internal_error", message=str(exc)),
        ))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _event(event: AnthropicStreamEvent) -> str:
    """Encode an event as a named Anthropic SSE frame: ``event: <type>\\ndata: <json>``."""
    return f"event: {event.type}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"


def _tool_use_id(name: str | None, index: int) -> str:
    prefix = (name or "tool").replace("_", "-")[:24]
    return f"toolu_{prefix}_{index}_{uuid.uuid4().hex[:8]}"


def _parse_json_args(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments:
        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _validation_error_message(exc: RequestValidationError) -> str:
    """First pydantic error as an Anthropic-style 'field.path: message' line."""
    try:
        err = exc.errors()[0]
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        msg = err.get("msg") or "invalid request"
        return f"{loc}: {msg}" if loc else msg
    except Exception:  # noqa: BLE001 — never let error formatting raise
        return "invalid request"


def _anthropic_error_response(
    status_code: int, err_type: str, message: str
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=AnthropicErrorResponse(
            error=AnthropicError(type=err_type, message=message)
        ).model_dump(),
    )
