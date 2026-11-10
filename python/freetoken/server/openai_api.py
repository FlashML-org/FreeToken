from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from freetoken.core import SamplingParams
from freetoken.message import TokenizeMsg
from freetoken.tokenizer.effort import EFFORT_SCALE, KNOWN_REASONING_EFFORTS

from .api_models import (
    ChatCompletionRequest,
    CompletionRequest,
    ModelCard,
    ModelList,
    ToolChoiceObject,
)
from .function_call_parser import ToolCallItem
from .request_logger import log_request
from .generation import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    KEEPALIVE,
    ContentDelta,
    GenDone,
    GenerationError,
    GenSpec,
    GenUsage,
    ReasoningDelta,
    ToolCallArgsDelta,
    ToolCallsDelta,
    ToolCallStart,
    generate_events,
    generate_full,
    prerender_error,
    render_messages,
    resolve_sampling,
    timings_block,
    submit_generation,
    with_keepalive,
)

KEEPALIVE_INTERVAL_S = 15.0


#: The wire superset plus "off", DeepSeek's disable synonym that
#: effort_toggle_kwargs has always honored.
_ACCEPTED_EFFORTS = (*KNOWN_REASONING_EFFORTS, "off")


def _thinking_type(req: Any) -> str | None:
    """The DeepSeek-wire thinking toggle, or None for absent/foreign shapes
    (which stay ignored, as extra="allow" ignored them before the field existed)."""
    if isinstance(req.thinking, dict):
        value = req.thinking.get("type")
        if value in ("enabled", "disabled"):
            return value
    return None



def chat_request_to_genspec(
    req: ChatCompletionRequest,
    model_sampling: dict[str, Any],
    default_max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    timings_per_token: bool = False,
) -> GenSpec:
    """OpenAI ChatCompletionRequest -> GenSpec (the OpenAI 'to_sampling_params')."""
    from .model_meta import effort_toggle_kwargs

    ctk = req.chat_template_kwargs
    thinking_type = _thinking_type(req)
    if req.reasoning_effort or thinking_type:
        ctk = effort_toggle_kwargs(req.reasoning_effort, ctk, thinking_type=thinking_type)
    return GenSpec(
        messages=render_messages([m.model_dump(exclude_none=True) for m in req.messages]),
        sampling_params=resolve_sampling(
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            ignore_eos=req.ignore_eos,
            model_sampling=model_sampling,
            stop=req.stop,
            default_max_tokens=default_max_tokens,
            presence_penalty=req.presence_penalty,
            frequency_penalty=req.frequency_penalty,
            timings_per_token=timings_per_token,
        ),
        chat_template_kwargs=ctk,
        template_tools=_tools_for_template(req),
        parser_tools=(_all_tool_dicts(req.tools) if _should_parse_tools(req) else None),
    )


def _all_tool_dicts(tools) -> list[dict[str, Any]]:
    return [t.model_dump(exclude_none=True) for t in (tools or [])]


def _maintenance_gate(state: Any) -> JSONResponse | None:
    """503 while the engine is not serving. Distinguishes the startup "loading" phase from a
    runtime cache "rebuild"/"failed" so clients (and the desktop) get an actionable message.
    None when serving."""
    mstate = getattr(state, "maintenance_state", "serving")
    if mstate == "serving":
        return None
    if mstate == "loading":
        msg = "model is still loading"
    elif mstate == "failed":
        msg = "server unavailable: maintenance failed (restart required)"
    else:
        msg = "server unavailable: cache rebuild in progress"
    return JSONResponse({"error": msg}, status_code=503)


def register_openai_routes(
    app: FastAPI,
    get_state: Callable[[], Any],
    get_model_sampling: Callable[[], dict[str, Any]],
) -> None:
    @app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
    async def v1_root():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def v1_chat_completions(req: ChatCompletionRequest, request: Request):
        log_request("/v1/chat/completions", req, request)
        state = get_state()
        if (gate := _maintenance_gate(state)) is not None:
            return gate
        return await handle_chat_completion(req, request, state, get_model_sampling())

    @app.post("/v1/completions")
    async def v1_completions(req: CompletionRequest, request: Request):
        log_request("/v1/completions", req, request)
        state = get_state()
        if (gate := _maintenance_gate(state)) is not None:
            return gate
        return await handle_completion(req, request, state, get_model_sampling())

    @app.get("/v1/models")
    async def v1_models():
        state = get_state()
        model_id = _served_model_name(state)
        ctx = _model_context_length(state)
        efforts, default_effort = await _effort_fields(state)
        return ModelList(data=[ModelCard(
            id=model_id,
            root=state.config.model_path,
            max_model_len=ctx,
            context_length=ctx,
            supported_reasoning_efforts=efforts,
            default_reasoning_effort=default_effort,
        )])


async def handle_chat_completion(
    req: ChatCompletionRequest,
    request: Request | None,
    state: Any,
    model_sampling: dict[str, Any],
):
    if req.function_call is not None:
        return create_error_response("function_call is not supported; use tools/tool_choice instead")
    if req.logit_bias is not None:
        return create_error_response("logit_bias is not supported")
    if _response_format_unsupported(req.response_format):
        return create_error_response(
            "response_format json_object/json_schema is not supported (no constrained decoding)",
            param="response_format",
        )
    if req.n != 1:
        return create_error_response("Only n=1 is supported", param="n")
    # Case/whitespace and the "off" disable synonym stay accepted here because
    # effort_toggle_kwargs normalizes and honors them downstream.
    effort = req.reasoning_effort.strip().lower() if isinstance(req.reasoning_effort, str) else None
    if effort and effort not in _ACCEPTED_EFFORTS:
        return create_error_response(
            f"reasoning_effort must be one of {', '.join(_ACCEPTED_EFFORTS)}; "
            f"got {req.reasoning_effort!r}",
            param="reasoning_effort",
        )
    if isinstance(req.thinking, dict):
        thinking_type = req.thinking.get("type")
        if thinking_type is not None and thinking_type not in ("enabled", "disabled"):
            return create_error_response(
                f"thinking.type must be 'enabled' or 'disabled'; got {thinking_type!r}",
                param="thinking",
            )

    try:
        default_max_tokens = (
            getattr(state.config, "max_output_tokens", None) or DEFAULT_MAX_OUTPUT_TOKENS
        )
        spec = chat_request_to_genspec(
            req,
            model_sampling,
            default_max_tokens=default_max_tokens,
            # stamp per-token only when the server opted into timings too; llama.cpp
            # publishes partial timings on streams, so non-stream requests never pay
            timings_per_token=bool(
                req.stream
                and req.timings_per_token
                and getattr(state.config, "enable_timings_report", False)
            ),
        )
    except ValueError as exc:
        return create_error_response(str(exc))

    if req.stream:
        # Non-stream requests already surface render failures as a clean 400
        # through GenerationError; only the stream path needs the pre-check.
        err = await prerender_error(spec, state)
        if err is not None:
            return create_error_response(str(err), code=err.code)

    try:
        uid = await submit_generation(spec, state)
    except GenerationError as exc:
        return create_error_response(str(exc), code=exc.code)

    if req.stream:
        chunks = stream_chat_completion_chunks(uid, req, state, spec)
        if request is not None:
            chunks = state.stream_with_cancellation(chunks, request, uid)
        return StreamingResponse(chunks, media_type="text/event-stream")

    try:
        result = await generate_full(uid, spec, state, source="/v1/chat/completions")
    except GenerationError as exc:
        return create_error_response(str(exc), code=exc.code)
    message: dict[str, Any] = {"role": "assistant", "content": result.content}
    if result.reasoning:
        message["reasoning_content"] = result.reasoning
    if result.tool_calls:
        message["tool_calls"] = _tool_calls_to_openai(result.tool_calls)

    response: dict[str, Any] = {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": _usage(
            result.prompt_tokens,
            result.completion_tokens,
            _reported_cached(state, result.cached_tokens),
        ),
    }
    if (timings := timings_block(state, result.timings)) is not None:
        response["timings"] = timings
    return response


async def stream_chat_completion_chunks(
    uid: int,
    req: ChatCompletionRequest,
    state: Any,
    spec: GenSpec | None = None,
) -> AsyncIterator[bytes]:
    """Format generate_events() into the OpenAI chat.completion.chunk SSE stream."""
    if spec is None:
        spec = chat_request_to_genspec(req, {})
    created = int(time.time())
    yield _sse(
        _chat_chunk(
            req,
            uid,
            created,
            [{"delta": {"role": "assistant", "content": ""}, "index": 0, "finish_reason": None}],
        )
    )

    timings: dict[str, Any] | None = None
    # timings ride the stream's LAST chunk only: the usage chunk when the client asked
    # for one, otherwise the finish chunk
    include_usage = bool(req.stream_options and req.stream_options.include_usage)
    continuous_usage = include_usage and bool(req.stream_options.continuous_usage_stats)
    # the final usage chunk reads it too, so a stream cut by an error reports what it produced
    usage = GenUsage()

    def emit(chunk: dict[str, Any]) -> bytes:
        if continuous_usage:
            chunk["usage"] = _reported_usage(state, usage)
        # per-token: post-ack chunks carry the ack's snapshot. Request-side gate: the
        # terminal snapshot rides every request, so post-terminal residue must stay clean.
        if req.timings_per_token and (t := timings_block(state, usage.timings)) is not None:
            chunk["timings"] = t
        return _sse(chunk)

    tool_calls_sent = 0
    open_tool: dict[str, Any] | None = None
    output_started = False
    events = with_keepalive(
        generate_events(uid, spec, state, source="/v1/chat/completions", usage=usage),
        KEEPALIVE_INTERVAL_S,
    )
    try:
        while True:
            try:
                ev = await events.__anext__()
            except StopAsyncIteration:
                break
            except GenerationError as exc:
                # Request failed before producing output — emit an error chunk + [DONE] so the
                # client gets a terminal signal instead of a stalled stream.
                yield _sse(
                    {"error": {"message": str(exc), "type": "invalid_request_error", "code": exc.code}}
                )
                break
            if ev is KEEPALIVE:
                if not output_started:
                    # Event-based idle timers ignore SSE comments during queue/prefill silence.
                    # A chunk like any other after the role chunk: emit() adds the usage so
                    # far and, once an ack has delivered one, its timings snapshot.
                    yield emit(_chat_chunk(
                        req, uid, created, [{"delta": {}, "index": 0, "finish_reason": None}],
                    ))
                continue
            if isinstance(ev, (ContentDelta, ReasoningDelta)):
                output_started = output_started or bool(ev.text)
            else:
                output_started = True
            if isinstance(ev, ReasoningDelta):
                yield emit(
                    _chat_chunk(
                        req,
                        uid,
                        created,
                        [{"delta": {"reasoning_content": ev.text}, "index": 0, "finish_reason": None}],
                    )
                )
            elif isinstance(ev, ContentDelta):
                yield emit(
                    _chat_chunk(
                        req,
                        uid,
                        created,
                        [{"delta": {"content": ev.text}, "index": 0, "finish_reason": None}],
                    )
                )
            elif isinstance(ev, ToolCallStart):
                open_tool = {
                    "index": tool_calls_sent,
                    "ordinal": ev.tool_index,
                    "sent": "",
                    "stable": ev.args_prefix_stable,
                }
                yield emit(
                    _chat_chunk(
                        req, uid, created,
                        [{
                            "delta": {"tool_calls": [{
                                "index": open_tool["index"],
                                "id": _tool_call_id(ev.name, open_tool["index"]),
                                "type": "function",
                                "function": {"name": ev.name, "arguments": ""},
                            }]},
                            "index": 0, "finish_reason": None,
                        }],
                    )
                )
            elif isinstance(ev, ToolCallArgsDelta):
                # Clients concatenate argument fragments, so only prefix-stable
                # fragments stream; otherwise the full arguments arrive at close.
                if open_tool is not None and open_tool["stable"] and ev.fragment:
                    open_tool["sent"] += ev.fragment
                    yield emit(
                        _chat_chunk(
                            req, uid, created,
                            [{
                                "delta": {"tool_calls": [{
                                    "index": open_tool["index"],
                                    "function": {"arguments": ev.fragment},
                                }]},
                                "index": 0, "finish_reason": None,
                            }],
                        )
                    )
            elif isinstance(ev, ToolCallsDelta):
                for call in ev.calls:
                    if open_tool is not None and open_tool["ordinal"] == call.tool_index:
                        # Close of a ToolCallStart-opened call: send whatever of the
                        # final (authoritative) arguments wasn't streamed yet.
                        final = call.parameters or ""
                        remainder = (
                            final[len(open_tool["sent"]):]
                            if final.startswith(open_tool["sent"])
                            else final if not open_tool["sent"] else ""
                        )
                        if remainder:
                            yield emit(
                                _chat_chunk(
                                    req, uid, created,
                                    [{
                                        "delta": {"tool_calls": [{
                                            "index": open_tool["index"],
                                            "function": {"arguments": remainder},
                                        }]},
                                        "index": 0, "finish_reason": None,
                                    }],
                                )
                            )
                        open_tool = None
                        tool_calls_sent += 1
                        continue
                    # Standalone complete call (buffered fallback path).
                    for delta in _tool_call_deltas([call], start_index=tool_calls_sent):
                        yield emit(
                            _chat_chunk(
                                req, uid, created,
                                [{"delta": {"tool_calls": [delta]}, "index": 0, "finish_reason": None}],
                            )
                        )
                    tool_calls_sent += 1
            elif isinstance(ev, GenDone):
                timings = ev.timings
                done = _chat_chunk(req, uid, created, [{"delta": {}, "index": 0, "finish_reason": ev.finish_reason}])
                if not include_usage and (t := timings_block(state, timings)) is not None:
                    done["timings"] = t
                yield emit(done)
    finally:
        await events.aclose()

    if include_usage:
        final: dict[str, Any] = {
            "id": f"chatcmpl-{uid}",
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [],
            "usage": _reported_usage(state, usage),
        }
        if (t := timings_block(state, timings)) is not None:
            final["timings"] = t
        yield _sse(final)

    yield b"data: [DONE]\n\n"


async def handle_completion(
    req: CompletionRequest,
    request: Request | None,
    state: Any,
    model_sampling: dict[str, Any],
):
    unsupported = _completion_unsupported_reason(req)
    if unsupported is not None:
        return create_error_response(unsupported)
    if req.n != 1:
        return create_error_response("Only n=1 is supported", param="n")
    try:  # surfaces an out-of-range max_tokens as a 400 rather than a 500 from the worker
        default_max_tokens = (
            getattr(state.config, "max_output_tokens", None) or DEFAULT_MAX_OUTPUT_TOKENS
        )
        _resolve_sampling(req, model_sampling, default_max_tokens=default_max_tokens)
    except ValueError as exc:
        return create_error_response(str(exc), param="max_tokens")

    # stamp per-token only when the server opted into timings too; llama.cpp publishes
    # partial timings on streams, so non-stream requests never pay for the snapshots
    timings_pt = bool(
        req.stream
        and req.timings_per_token
        and getattr(state.config, "enable_timings_report", False)
    )
    prompts = [req.prompt] if isinstance(req.prompt, str) else req.prompt
    assert isinstance(prompts, list)
    if not prompts:
        return create_error_response("prompt must not be empty", param="prompt")
    if req.stream:
        if len(prompts) != 1:
            return create_error_response("Streaming completions only support a single text prompt")
        uid = state.new_user()
        await state.send_one(
            TokenizeMsg(uid=uid, text=prompts[0], sampling_params=_resolve_sampling(
                req, model_sampling, default_max_tokens=default_max_tokens,
                timings_per_token=timings_pt,
            ))
        )
        chunks = stream_completion_chunks(uid, req, state)
        if request is not None:
            chunks = state.stream_with_cancellation(chunks, request, uid)
        return StreamingResponse(chunks, media_type="text/event-stream")

    choices: list[dict[str, Any]] = []
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    # with several prompts the timings are the FIRST prompt's own spans, as llama.cpp
    # returns arr[0]; one clock across the sequential loop would mix the later prompts'
    # queue and prefill into the rates
    first_timings: dict[str, Any] | None = None
    for index, prompt in enumerate(prompts):
        uid = state.new_user()
        await state.send_one(
            TokenizeMsg(
                uid=uid,
                text=prompt,
                sampling_params=_resolve_sampling(
                    req, model_sampling, default_max_tokens=default_max_tokens,
                    timings_per_token=timings_pt,
                ),
            )
        )
        text = ""
        finish_reason = "stop"
        async for ack in state.wait_for_ack(uid):
            if getattr(ack, "error", None):
                return create_error_response(ack.error, code=getattr(ack, "error_code", None))
            prompt_tokens += ack.prompt_tokens_delta
            completion_tokens += ack.completion_tokens_delta
            cached_tokens += ack.cached_tokens
            text += ack.incremental_output
            if ack.finished:
                if index == 0:
                    first_timings = getattr(ack, "timings", None)
                finish_reason = getattr(ack, "finish_reason", None) or "stop"
                break
        choices.append({"index": index, "text": text, "finish_reason": finish_reason, "logprobs": None})

    response: dict[str, Any] = {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": choices,
        "usage": _usage(prompt_tokens, completion_tokens, _reported_cached(state, cached_tokens)),
    }
    if (timings := timings_block(state, first_timings)) is not None:
        response["timings"] = timings
    return response


async def stream_completion_chunks(uid: int, req: CompletionRequest, state: Any) -> AsyncIterator[bytes]:
    usage = GenUsage()
    timings: dict[str, Any] | None = None
    include_usage = bool(req.stream_options and req.stream_options.include_usage)
    continuous_usage = include_usage and bool(req.stream_options.continuous_usage_stats)

    def emit(chunk: dict[str, Any]) -> bytes:
        if continuous_usage:
            chunk["usage"] = _reported_usage(state, usage)
        # per-token: same request-side check as the chat stream - the terminal ack carries
        # timings for every request, only the flag decides who gets a block on each chunk
        if req.timings_per_token and (t := timings_block(state, usage.timings)) is not None:
            chunk["timings"] = t
        return _sse(chunk)

    finish_reason = "stop"
    async for ack in state.wait_for_ack(uid):
        if getattr(ack, "error", None):
            yield _sse({"error": {"message": ack.error, "type": "invalid_request_error", "code": getattr(ack, "error_code", None)}})
            yield b"data: [DONE]\n\n"
            return
        usage.prompt_tokens += ack.prompt_tokens_delta
        usage.completion_tokens += ack.completion_tokens_delta
        usage.cached_tokens += ack.cached_tokens
        if getattr(ack, "timings", None) is not None:
            timings = ack.timings
            usage.timings = timings
        if ack.incremental_output:
            yield emit(
                {
                    "id": f"cmpl-{uid}",
                    "object": "text_completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [
                        {
                            "text": ack.incremental_output,
                            "index": 0,
                            "finish_reason": None,
                            "logprobs": None,
                        }
                    ],
                }
            )
        if ack.finished:
            finish_reason = getattr(ack, "finish_reason", None) or "stop"
            break

    done_chunk: dict[str, Any] = {
        "id": f"cmpl-{uid}",
        "object": "text_completion.chunk",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"text": "", "index": 0, "finish_reason": finish_reason, "logprobs": None}],
    }
    if not include_usage and (t := timings_block(state, timings)) is not None:
        # as on the chat stream: the finish chunk is the last one only when no usage
        # chunk follows
        done_chunk["timings"] = t
    yield emit(done_chunk)
    if include_usage:
        final: dict[str, Any] = {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "created": int(time.time()),
            "model": req.model,
            "choices": [],
            "usage": _reported_usage(state, usage),
        }
        if (t := timings_block(state, timings)) is not None:
            final["timings"] = t
        yield _sse(final)
    yield b"data: [DONE]\n\n"


def create_error_response(
    message: str,
    status_code: int = 400,
    err_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": err_type,
                "param": param,
                "code": code,
            }
        },
    )


def _resolve_sampling(
    req: ChatCompletionRequest | CompletionRequest,
    model_sampling: dict[str, Any],
    default_max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    timings_per_token: bool = False,
) -> SamplingParams:
    return resolve_sampling(
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        ignore_eos=req.ignore_eos,
        model_sampling=model_sampling,
        stop=req.stop,
        default_max_tokens=default_max_tokens,
        presence_penalty=req.presence_penalty,
        frequency_penalty=req.frequency_penalty,
        timings_per_token=timings_per_token,
    )


def _tools_for_template(req: ChatCompletionRequest) -> list[dict[str, Any]] | None:
    if not _should_parse_tools(req):
        return None

    tools = req.tools or []
    if isinstance(req.tool_choice, ToolChoiceObject):
        selected = req.tool_choice.function.name
        tools = [tool for tool in tools if tool.function.name == selected]

    return [tool.model_dump(exclude_none=True) for tool in tools]


def _should_parse_tools(req: ChatCompletionRequest) -> bool:
    return bool(req.tools) and req.tool_choice != "none"


def _tool_calls_to_openai(calls: list[ToolCallItem]) -> list[dict[str, Any]]:
    result = []
    for index, call in enumerate(calls):
        result.append(
            {
                "id": _tool_call_id(call.name, index),
                "index": index,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.parameters,
                },
            }
        )
    return result


def _tool_call_deltas(calls: list[ToolCallItem], start_index: int = 0) -> list[dict[str, Any]]:
    """OpenAI tool_calls stream deltas. ``start_index`` offsets the slot index so
    calls arriving across multiple ToolCallsDelta events (streamed one per call as
    each closes) don't all collapse into slot 0."""
    deltas: list[dict[str, Any]] = []
    for offset, call in enumerate(calls):
        index = start_index + offset
        call_id = _tool_call_id(call.name, index)
        deltas.append(
            {
                "index": index,
                "id": call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": ""},
            }
        )
        deltas.append({"index": index, "function": {"arguments": call.parameters}})
    return deltas


def _tool_call_id(name: str | None, index: int) -> str:
    prefix = (name or "tool").replace("_", "-")[:24]
    return f"call_{prefix}_{index}_{uuid.uuid4().hex[:8]}"


def _chat_chunk(
    req: ChatCompletionRequest, uid: int, created: int, choices: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion.chunk",
        "created": created,
        "model": req.model,
        "choices": choices,
    }


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def _reported_cached(state: Any, cached_tokens: int) -> int:
    """The prefix-cache hit to report; 0 unless --enable-cache-report is set."""
    return cached_tokens if getattr(state.config, "enable_cache_report", False) else 0


def _reported_usage(state: Any, usage: GenUsage) -> dict[str, Any]:
    return _usage(usage.prompt_tokens, usage.completion_tokens, _reported_cached(state, usage.cached_tokens))


def _usage(prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    # sglang convention: the details object appears only for a nonzero hit, so a
    # disabled report and a 0-token hit serialize identically.
    if cached_tokens > 0:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return usage


def _response_format_unsupported(response_format: dict[str, Any] | None) -> bool:
    # We have no constrained/guided decoding; only plain text ('text' or unset) is honored.
    return response_format is not None and response_format.get("type") not in (None, "text")


def _completion_unsupported_reason(req: CompletionRequest) -> str | None:
    if _is_token_prompt(req.prompt):
        return "OpenAI token-id prompt inputs are not supported; pass text prompt strings instead"
    if req.logprobs is not None:
        return "logprobs is not supported"
    if req.echo:
        return "echo is not supported"
    if req.suffix is not None:
        return "suffix is not supported"
    if req.logit_bias is not None:
        return "logit_bias is not supported"
    if _response_format_unsupported(req.response_format):
        return "response_format json_object/json_schema is not supported (no constrained decoding)"
    return None


def _is_token_prompt(prompt: Any) -> bool:
    return (
        isinstance(prompt, list)
        and bool(prompt)
        and (
            all(isinstance(item, int) for item in prompt)
            or all(isinstance(item, list) and all(isinstance(token, int) for token in item) for item in prompt)
        )
    )


async def _effort_fields(state: Any) -> tuple[list[str] | None, str | None]:
    """The checkpoint's probed effort vocabulary for /v1/models, or (None, None)
    when there is no frontend tokenizer, it fails to build, or the model has no
    effort knob — a metadata route must never 500 over this."""
    build = getattr(state, "frontend_tokenizer", None)
    if build is None:
        return None, None
    try:
        manager = await asyncio.to_thread(build)
        profile = await asyncio.to_thread(manager.effort_profile)
    except Exception:  # noqa: BLE001 -- metadata only; the generation path reports real faults
        return None, None
    from freetoken.tokenizer.effort import effective_efforts

    served = effective_efforts(profile)
    if not served:
        return None, None
    ordered = sorted(served, key=lambda name: -EFFORT_SCALE.get(name, 0.0))
    return ordered, profile.default


def _served_model_name(state: Any) -> str:
    return getattr(state.config, "served_model_name", None) or state.config.model_path


def _model_context_length(state: Any) -> int | None:
    """The model ceiling, not `min(ceiling, KV budget)`: a rebuild moves the latter, and agents
    read this once at startup."""
    try:  # never 500 a metadata route: max_seq_len walks into the HF config on some builds
        value = int(state.config.max_seq_len)
    except Exception:  # noqa: BLE001
        return None
    return value if value > 0 else None
