"""OpenAI Responses API (`/v1/responses`) — the endpoint codex talks to.

Adapted from vLLM's ``OpenAIServingResponses`` (the *simple* rail; the gpt-oss
Harmony rail is not ported). Like the Anthropic adapter, this drives the shared
protocol-neutral primitive (:func:`freetoken.server.openai_api.submit_generation` +
``generate_events`` / ``generate_full``) and formats its semantic events into
Responses typed items + stream events using the real ``openai.types.responses``
SDK models. It consumes typed events, not a re-parsed OpenAI stream.

Scope: a STATELESS subset (``store``/``previous_response_id``/``background`` are
ignored; ``GET /v1/responses/{id}`` and ``/cancel`` are stubbed) — enough to drive
codex, which resends full context as ``input`` when not storing. Reasoning streams
as a first-class ``reasoning`` item (``response.reasoning_text.delta``, the shape
vLLM emits and codex renders); reasoning items replayed as ``input`` are folded
back into their assistant turn as ``reasoning_content``.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseError,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseInProgressEvent,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_reasoning_item import Content as ReasoningTextContent
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)
from pydantic import BaseModel, ConfigDict

from .generation import (
    DEFAULT_MAX_OUTPUT_TOKENS,
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
    generate_events,
    generate_full,
    render_messages,
    resolve_sampling,
    split_tool_lists,
    submit_generation,
    with_keepalive,
)
from .request_logger import log_request

# Seconds of event silence before a keep-alive frame is emitted on the stream.
# codex's stream-idle timeout (default 300s) only resets on a data-bearing SSE
# frame, so long queue/prefill or decode gaps must be bridged with real events.
KEEPALIVE_INTERVAL_S = 15.0

class ResponsesRequest(BaseModel):
    """The subset of the Responses request FreeToken honors (extra fields allowed)."""

    model_config = ConfigDict(extra="allow")

    model: str
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    reasoning: dict[str, Any] | None = None
    # Stateful features are accepted but ignored in this subset.
    store: bool = False
    previous_response_id: str | None = None
    background: bool = False
    metadata: dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None


def register_responses_routes(
    app: FastAPI,
    get_state: Callable[[], Any],
    get_model_sampling: Callable[[], dict[str, Any]],
) -> None:
    @app.post("/v1/responses")
    async def v1_responses(req: ResponsesRequest, request: Request):
        log_request("/v1/responses", req, request)
        state = get_state()
        mstate = getattr(state, "maintenance_state", "serving")
        if mstate != "serving":
            detail = "model is still loading" if mstate == "loading" else "cache rebuild in progress"
            return _error_response(503, detail)
        if req.background:
            return _error_response(400, "background mode is not supported")
        if req.previous_response_id:
            return _error_response(
                400,
                "previous_response_id is not supported (stateless server); "
                "resend full context in 'input'",
            )
        return await handle_responses(req, request, state, get_model_sampling())

    @app.get("/v1/responses/{response_id}")
    async def v1_responses_get(response_id: str):
        return _error_response(404, f"response {response_id!r} not found (stateless server)")

    @app.post("/v1/responses/{response_id}/cancel")
    async def v1_responses_cancel(response_id: str):
        return _error_response(404, f"response {response_id!r} not found (stateless server)")


async def handle_responses(
    req: ResponsesRequest,
    request: Request | None,
    state: Any,
    model_sampling: dict[str, Any],
):
    response_id = f"resp_{uuid.uuid4().hex}"
    created = int(time.time())
    if req.max_output_tokens is not None and req.max_output_tokens < 1:
        return _error_response(400, "max_output_tokens must be a positive integer")
    default_max = getattr(state.config, "max_output_tokens", None) or DEFAULT_MAX_OUTPUT_TOKENS
    try:
        spec = convert_responses_to_genspec(
            req, model_sampling, default_max_tokens=default_max,
            reasoning_parser=getattr(state.config, "reasoning_parser", None),
        )
        uid = await submit_generation(spec, state)
    except ValueError as exc:
        return _error_response(400, str(exc))

    cache_report = getattr(state.config, "enable_cache_report", False)
    if req.stream:
        events = responses_stream_generator(
            generate_events(uid, spec, state, source="/v1/responses"), req, response_id, created,
            cache_report=cache_report,
        )
        if request is not None:
            events = state.stream_with_cancellation(events, request, uid)
        return StreamingResponse(events, media_type="text/event-stream")

    try:
        result = await generate_full(uid, spec, state, source="/v1/responses")
    except GenerationError as exc:
        return _error_response(400, str(exc), exc.code)
    response = build_responses_response(result, req, response_id, created, cache_report=cache_report)
    return JSONResponse(content=response.model_dump(mode="json"))


# --------------------------------------------------------------------------- #
# Request conversion: Responses -> GenSpec (the neutral generation spec; built
# directly with the Responses' OWN max-tokens default — so the OpenAI-chat 16-token
# floor can never leak in, exactly as vLLM keeps a per-protocol to_sampling_params).
# --------------------------------------------------------------------------- #
def convert_responses_to_genspec(
    req: ResponsesRequest,
    model_sampling: dict[str, Any],
    default_max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    reasoning_parser: str | None = None,
) -> GenSpec:
    # Collect every system/developer text — the top-level `instructions` PLUS any
    # system/developer-role input items (codex sends both: a system prompt as `instructions`
    # and a `developer` permissions message) — into ONE leading system message. Strict chat
    # templates (e.g. Qwen3.5) reject a second system message with "System message must be at
    # the beginning"; merging mirrors the Anthropic adapter (which is why Qwen3.5 already works
    # through Claude Code).
    system_texts: list[str] = []
    if req.instructions:
        system_texts.append(req.instructions)

    other: list[dict[str, Any]] = []
    if isinstance(req.input, str):
        other.append({"role": "user", "content": req.input})
    else:
        for item in req.input:
            for m in _convert_input_item(item):
                if m.get("role") == "system":
                    system_texts.append(m.get("content") or "")
                else:
                    other.append(m)
        other = _merge_assistant_run(other)

    messages: list[dict[str, Any]] = []
    system_text = "\n\n".join(t for t in system_texts if t)
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.extend(other)

    raw_tools = _convert_tools(req.tools) or []
    selected: str | None = None
    disabled = False
    tc = req.tool_choice
    if isinstance(tc, str):
        disabled = tc == "none"
    elif isinstance(tc, dict) and tc.get("type") == "function":
        selected = tc.get("name") or (tc.get("function") or {}).get("name")
    if disabled:
        template_tools, parser_tools = None, None
    else:
        template_tools, parser_tools = split_tool_lists(raw_tools, selected)

    from .model_meta import effort_toggle_kwargs

    ctk = dict(getattr(req, "chat_template_kwargs", None) or {})
    if req.reasoning:
        ctk = effort_toggle_kwargs(req.reasoning.get("effort"), ctk)

    return GenSpec(
        messages=render_messages(messages),
        sampling_params=resolve_sampling(
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            max_tokens=req.max_output_tokens or default_max_tokens,
            ignore_eos=False,
            model_sampling=model_sampling,
        ),
        chat_template_kwargs=ctk,
        template_tools=template_tools,
        parser_tools=parser_tools,
    )


def _convert_input_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    itype = item.get("type", "message")
    if itype == "message" or ("role" in item and "type" not in item):
        # codex sends a "developer" role (Responses instructions). Chat templates only
        # know system/user/assistant/tool, so fold developer -> system.
        role = item.get("role", "user")
        if role == "developer":
            role = "system"
        return [{"role": role, "content": _input_text(item.get("content"))}]
    if itype == "function_call":
        return [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "") or "",
                        },
                    }
                ],
            }
        ]
    if itype == "function_call_output":
        return [
            {
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": _stringify(item.get("output")),
            }
        ]
    if itype == "reasoning":
        # Folded into its assistant turn by _merge_assistant_run; summary-only /
        # encrypted items carry no recoverable text.
        text = "".join(
            c.get("text") or ""
            for c in (item.get("content") or [])
            if isinstance(c, dict) and c.get("type") == "reasoning_text"
        )
        if text:
            return [{"role": "assistant", "reasoning_content": text}]
        return []
    return []


def _merge_assistant_run(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coalesce one assistant turn's items (reasoning / message / function_call)
    into a single message — separate messages would render as extra assistant
    turns. Slot conflicts start a new message: a reasoning item always opens a
    turn, a message item only fills a turn with no content/tool_calls yet, so
    distinct assistant messages never merge."""
    merged: list[dict[str, Any]] = []
    for m in messages:
        prev = merged[-1] if merged else None
        mergeable = (
            prev is not None
            and m.get("role") == "assistant"
            and prev.get("role") == "assistant"
            and not m.get("reasoning_content")
            and (
                m.get("tool_calls")
                or (m.get("content") and not prev.get("content") and not prev.get("tool_calls"))
            )
        )
        if not mergeable:
            merged.append(dict(m))
            continue
        if m.get("content"):
            prev["content"] = m["content"]
        if m.get("tool_calls"):
            prev["tool_calls"] = (prev.get("tool_calls") or []) + m["tool_calls"]
    return merged


def _input_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") in ("input_text", "output_text", "text") or "text" in part:
                parts.append(part.get("text") or "")
        else:
            parts.append(str(part))
    return "".join(parts)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") not in (None, "function"):
            # Built-in tools (web_search, code_interpreter, ...) are unsupported; skip.
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "description": fn.get("description"),
                    "parameters": fn.get("parameters") or {"type": "object"},
                },
            }
        )
    return converted or None


# --------------------------------------------------------------------------- #
# Non-streaming response assembly (GenResult -> Response)
# --------------------------------------------------------------------------- #
def build_responses_response(
    result: GenResult,
    req: ResponsesRequest,
    response_id: str,
    created: int,
    cache_report: bool = False,
) -> Response:
    truncated = result.finish_reason == "length"
    item_status = "incomplete" if truncated else "completed"
    output: list[Any] = []
    if result.reasoning:
        output.append(
            ResponseReasoningItem(
                id=f"rs_{uuid.uuid4().hex}",
                type="reasoning",
                summary=[],
                content=[ReasoningTextContent(type="reasoning_text", text=result.reasoning)],
                status="completed",
            )
        )
    if result.content:
        output.append(
            ResponseOutputMessage(
                id=f"msg_{uuid.uuid4().hex}",
                role="assistant",
                status=item_status,
                type="message",
                content=[ResponseOutputText(type="output_text", text=result.content, annotations=[])],
            )
        )
    for call in result.tool_calls:
        output.append(
            ResponseFunctionToolCall(
                type="function_call",
                id=f"fc_{uuid.uuid4().hex}",
                call_id=f"call_{uuid.uuid4().hex[:24]}",
                name=call.name or "",
                arguments=call.parameters or "",
                status="completed",
            )
        )

    return _response_obj(
        response_id, created, req.model, output,
        status="incomplete" if truncated else "completed",
        usage=_usage(
            result.prompt_tokens, result.completion_tokens,
            result.cached_tokens if cache_report else 0,
        ),
        incomplete_reason="max_output_tokens" if truncated else None,
    )


def _response_obj(
    response_id: str, created: int, model: str, output: list[Any],
    *, status: str, usage: ResponseUsage | None, error: ResponseError | None = None,
    incomplete_reason: str | None = None,
) -> Response:
    return Response(
        id=response_id,
        created_at=created,
        model=model,
        object="response",
        output=output,
        status=status,
        usage=usage,
        error=error,
        incomplete_details=IncompleteDetails(reason=incomplete_reason) if incomplete_reason else None,
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _usage(prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> ResponseUsage:
    # input_tokens stays inclusive of the cached prefix (OpenAI semantics);
    # cached_tokens is nonzero only under --enable-cache-report.
    return ResponseUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=cached_tokens, cache_write_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


# --------------------------------------------------------------------------- #
# Streaming: GenEvent stream -> Responses semantic events
# --------------------------------------------------------------------------- #
async def responses_stream_generator(
    events: AsyncIterator[Any],
    req: ResponsesRequest,
    response_id: str,
    created: int,
    cache_report: bool = False,
) -> AsyncIterator[str]:
    seq = _Seq()

    def snapshot(status, output, usage=None, incomplete_reason=None):
        return _response_obj(
            response_id, created, req.model, output,
            status=status, usage=usage, incomplete_reason=incomplete_reason,
        )

    yield _sse(ResponseCreatedEvent(
        type="response.created", sequence_number=seq.next(), response=snapshot("in_progress", []),
    ))
    yield _sse(ResponseInProgressEvent(
        type="response.in_progress", sequence_number=seq.next(), response=snapshot("in_progress", []),
    ))

    output_items: list[Any] = []
    output_index = 0
    current: dict[str, Any] | None = None
    finish_reason = "stop"
    usage_pt = 0
    usage_ct = 0
    usage_cached = 0

    def close_current() -> list[str]:
        nonlocal current, output_index
        frames: list[str] = []
        if current is None:
            return frames
        if current["kind"] == "reasoning":
            item_id, text = current["id"], current["text"]
            frames.append(_sse(ResponseReasoningTextDoneEvent(
                type="response.reasoning_text.done", sequence_number=seq.next(),
                item_id=item_id, output_index=output_index, content_index=0, text=text,
            )))
            done_item = ResponseReasoningItem(
                id=item_id, type="reasoning", summary=[],
                content=[ReasoningTextContent(type="reasoning_text", text=text)],
                status="completed",
            )
        elif current["kind"] == "message":
            item_id, text = current["id"], current["text"]
            frames.append(_sse(ResponseTextDoneEvent(
                type="response.output_text.done", sequence_number=seq.next(),
                item_id=item_id, output_index=output_index, content_index=0, text=text, logprobs=[],
            )))
            frames.append(_sse(ResponseContentPartDoneEvent(
                type="response.content_part.done", sequence_number=seq.next(),
                item_id=item_id, output_index=output_index, content_index=0,
                part=ResponseOutputText(type="output_text", text=text, annotations=[]),
            )))
            # finish_reason is only non-"stop" once GenDone has arrived, so only the final
            # (truncated) message item is marked incomplete; items closed mid-stream stay completed.
            done_item = ResponseOutputMessage(
                id=item_id, role="assistant",
                status="incomplete" if finish_reason == "length" else "completed",
                type="message",
                content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
            )
        else:  # function_call
            item_id, args = current["id"], current["args"]
            frames.append(_sse(ResponseFunctionCallArgumentsDoneEvent(
                type="response.function_call_arguments.done", sequence_number=seq.next(),
                item_id=item_id, output_index=output_index, name=current["name"], arguments=args,
            )))
            done_item = ResponseFunctionToolCall(
                type="function_call", id=item_id, call_id=current["call_id"],
                name=current["name"], arguments=args, status="completed",
            )
        frames.append(_sse(ResponseOutputItemDoneEvent(
            type="response.output_item.done", sequence_number=seq.next(),
            output_index=output_index, item=done_item,
        )))
        output_items.append(done_item)
        output_index += 1
        current = None
        return frames

    def open_function_call(name: str, ordinal: int | None) -> list[str]:
        nonlocal current
        frames = close_current()
        item_id = f"fc_{uuid.uuid4().hex}"
        call_id = f"call_{uuid.uuid4().hex[:24]}"
        current = {
            "kind": "function_call", "id": item_id,
            "call_id": call_id, "name": name, "args": "", "ordinal": ordinal,
        }
        frames.append(_sse(ResponseOutputItemAddedEvent(
            type="response.output_item.added", sequence_number=seq.next(),
            output_index=output_index,
            item=ResponseFunctionToolCall(
                type="function_call", id=item_id, call_id=call_id,
                name=name, arguments="", status="in_progress",
            ),
        )))
        return frames

    def args_delta_frame(fragment: str) -> str:
        return _sse(ResponseFunctionCallArgumentsDeltaEvent(
            type="response.function_call_arguments.delta", sequence_number=seq.next(),
            item_id=current["id"], output_index=output_index, delta=fragment,
        ))

    events = with_keepalive(events, KEEPALIVE_INTERVAL_S)
    try:
        async for ev in events:
            if ev is KEEPALIVE:
                # Data-bearing frame codex ignores but whose arrival resets its
                # stream-idle countdown (SSE comments would not).
                yield _sse(ResponseInProgressEvent(
                    type="response.in_progress", sequence_number=seq.next(),
                    response=snapshot("in_progress", output_items),
                ))

            elif isinstance(ev, ReasoningDelta):
                # Streamed as a first-class reasoning item (codex renders
                # response.reasoning_text.delta natively).
                if ev.text == "":
                    continue
                if current is None or current["kind"] != "reasoning":
                    for f in close_current():
                        yield f
                    item_id = f"rs_{uuid.uuid4().hex}"
                    current = {"kind": "reasoning", "id": item_id, "text": ""}
                    yield _sse(ResponseOutputItemAddedEvent(
                        type="response.output_item.added", sequence_number=seq.next(),
                        output_index=output_index,
                        item=ResponseReasoningItem(
                            id=item_id, type="reasoning", summary=[], status="in_progress",
                        ),
                    ))
                current["text"] += ev.text
                yield _sse(ResponseReasoningTextDeltaEvent(
                    type="response.reasoning_text.delta", sequence_number=seq.next(),
                    item_id=current["id"], output_index=output_index, content_index=0,
                    delta=ev.text,
                ))

            elif isinstance(ev, ContentDelta):
                if ev.text == "":
                    continue
                if current is None or current["kind"] != "message":
                    for f in close_current():
                        yield f
                    item_id = f"msg_{uuid.uuid4().hex}"
                    current = {"kind": "message", "id": item_id, "text": ""}
                    yield _sse(ResponseOutputItemAddedEvent(
                        type="response.output_item.added", sequence_number=seq.next(),
                        output_index=output_index,
                        item=ResponseOutputMessage(
                            id=item_id, role="assistant", status="in_progress", type="message", content=[],
                        ),
                    ))
                    yield _sse(ResponseContentPartAddedEvent(
                        type="response.content_part.added", sequence_number=seq.next(),
                        item_id=item_id, output_index=output_index, content_index=0,
                        part=ResponseOutputText(type="output_text", text="", annotations=[]),
                    ))
                current["text"] += ev.text
                yield _sse(ResponseTextDeltaEvent(
                    type="response.output_text.delta", sequence_number=seq.next(),
                    item_id=current["id"], output_index=output_index, content_index=0,
                    delta=ev.text, logprobs=[],
                ))

            elif isinstance(ev, ToolCallStart):
                for f in open_function_call(ev.name or "", ev.tool_index):
                    yield f

            elif isinstance(ev, ToolCallArgsDelta):
                if current is None or current["kind"] != "function_call":
                    continue  # defensive: fragment without an open call
                current["args"] += ev.fragment
                yield args_delta_frame(ev.fragment)

            elif isinstance(ev, ToolCallsDelta):
                for call in ev.calls:
                    if (
                        current is not None
                        and current["kind"] == "function_call"
                        and current.get("ordinal") == call.tool_index
                    ):
                        # Close of a ToolCallStart-opened call: the final arguments
                        # are authoritative — top up whatever wasn't streamed yet.
                        final = call.parameters or ""
                        if final.startswith(current["args"]):
                            remainder = final[len(current["args"]):]
                            if remainder:
                                yield args_delta_frame(remainder)
                        current["args"] = final
                        for f in close_current():
                            yield f
                        continue
                    # Standalone complete call (buffered fallback path): open,
                    # deliver the whole arguments, close immediately so codex
                    # persists the item even if the stream dies later.
                    for f in open_function_call(call.name or "", call.tool_index):
                        yield f
                    args = call.parameters or ""
                    current["args"] = args
                    yield args_delta_frame(args)
                    for f in close_current():
                        yield f

            elif isinstance(ev, GenDone):
                finish_reason = ev.finish_reason
                usage_pt, usage_ct = ev.prompt_tokens, ev.completion_tokens
                usage_cached = ev.cached_tokens if cache_report else 0
                for f in close_current():
                    yield f
                if finish_reason == "length":
                    yield _sse(ResponseIncompleteEvent(
                        type="response.incomplete", sequence_number=seq.next(),
                        response=snapshot(
                            "incomplete", output_items,
                            usage=_usage(usage_pt, usage_ct, usage_cached),
                            incomplete_reason="max_output_tokens",
                        ),
                    ))
                else:
                    yield _sse(ResponseCompletedEvent(
                        type="response.completed", sequence_number=seq.next(),
                        response=snapshot(
                            "completed", output_items,
                            usage=_usage(usage_pt, usage_ct, usage_cached),
                        ),
                    ))
    except GenerationError as exc:
        # Request failed before/instead of producing output (template render error, over-length
        # prompt). Emit the protocol's terminal failure event so codex sees a clean error rather
        # than a stalled stream that trips its 300s idle timeout into "Reconnecting".
        for f in close_current():
            yield f
        yield _sse(ResponseFailedEvent(
            type="response.failed", sequence_number=seq.next(),
            response=_response_obj(
                response_id, created, req.model, output_items, status="failed",
                usage=_usage(usage_pt, usage_ct, usage_cached),
                # codex reads this code to tell a blown context window from a generic failure.
                # model_construct because ResponseError.code is a closed Literal in the SDK.
                error=ResponseError.model_construct(
                    code=exc.code or "server_error", message=str(exc)
                ),
            ),
        ))
    except Exception as exc:  # noqa: BLE001 — never leave the client without a terminal event
        for f in close_current():
            yield f
        yield _sse(ResponseFailedEvent(
            type="response.failed", sequence_number=seq.next(),
            response=_response_obj(
                response_id, created, req.model, output_items, status="failed",
                usage=_usage(usage_pt, usage_ct, usage_cached),
                error=ResponseError(code="server_error", message=str(exc)),
            ),
        ))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _Seq:
    def __init__(self):
        self._n = -1

    def next(self) -> int:
        self._n += 1
        return self._n


def _sse(event) -> str:
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"


def _error_response(status_code: int, message: str, code: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": "invalid_request_error", "code": code}},
    )
