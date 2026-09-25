"""Pydantic models for the Anthropic Messages API.

Adapted from vLLM's ``vllm/entrypoints/anthropic/protocol.py`` (Apache-2.0,
Copyright contributors to the vLLM project). Kept close to the upstream shapes
so the conversion logic in :mod:`freetoken.server.anthropic_api` mirrors vLLM's
``AnthropicServingMessages``.
"""

from __future__ import annotations

import time
from typing import Any, Literal, Optional

from pydantic import BaseModel, field_validator


class AnthropicError(BaseModel):
    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    type: Literal["error"] = "error"
    error: AnthropicError


class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class AnthropicContentBlock(BaseModel):
    """A content block in a message. ``type`` is left open (str) rather than a
    Literal: real clients (Claude Code) send blocks beyond text/thinking/image/
    tool_use/tool_result (e.g. ``redacted_thinking``); the converter handles the
    known types and skips the rest instead of rejecting the whole request with a 422."""

    type: str
    text: str | None = None
    # For thinking blocks.
    thinking: str | None = None
    signature: str | None = None
    # For image content.
    source: dict[str, Any] | None = None
    # For tool use / result. A tool_result names the call it answers with ``tool_use_id``,
    # not ``id`` -- without this field it is dropped at validation and every tool result
    # reaches the encoder unattributed (parallel results then cannot be matched or ordered).
    id: str | None = None
    tool_use_id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    content: str | list[dict[str, Any]] | None = None
    is_error: bool | None = None


class AnthropicMessage(BaseModel):
    # Anthropic's spec is user/assistant only, but Claude Code sends system-role
    # messages inside the array (not just the top-level `system` field); accept it
    # so the converter can render it as a system message instead of 422-ing.
    role: str
    content: str | list[AnthropicContentBlock]


class AnthropicTool(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any]

    @field_validator("input_schema")
    @classmethod
    def validate_input_schema(cls, v):
        if not isinstance(v, dict):
            raise ValueError("input_schema must be a dictionary")
        if "type" not in v:
            v["type"] = "object"
        return v


class AnthropicToolChoice(BaseModel):
    type: Literal["auto", "any", "tool", "none"]
    name: str | None = None


class AnthropicMessagesRequest(BaseModel):
    """Anthropic Messages API request (https://docs.anthropic.com/en/api/messages)."""

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    metadata: dict[str, Any] | None = None
    stop_sequences: list[str] | None = None
    stream: bool | None = False
    system: str | list[AnthropicContentBlock] | None = None
    temperature: float | None = None
    # Native extended-thinking toggle: {"type": "enabled"|"disabled", "budget_tokens": N}.
    # budget_tokens is accepted but not enforced (stateless subset).
    thinking: dict[str, Any] | None = None
    tool_choice: AnthropicToolChoice | None = None
    tools: list[AnthropicTool] | None = None
    top_k: int | None = None
    top_p: float | None = None

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v):
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


class AnthropicCountTokensRequest(BaseModel):
    """Anthropic count_tokens request (POST /v1/messages/count_tokens): the input
    side of a Messages request — no max_tokens/stream/sampling fields."""

    model: str
    messages: list[AnthropicMessage]
    system: str | list[AnthropicContentBlock] | None = None
    thinking: dict[str, Any] | None = None
    tool_choice: AnthropicToolChoice | None = None
    tools: list[AnthropicTool] | None = None


class AnthropicCountTokensResponse(BaseModel):
    input_tokens: int


class AnthropicDelta(BaseModel):
    """Delta payload for streaming events."""

    type: Literal["text_delta", "input_json_delta", "thinking_delta", "signature_delta"] | None = None
    text: str | None = None
    partial_json: str | None = None
    thinking: str | None = None
    signature: str | None = None

    # message_delta carries the terminal stop reason.
    stop_reason: (
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    ) = None
    stop_sequence: str | None = None


class AnthropicMessagesResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicContentBlock]
    model: str
    stop_reason: (
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    ) = None
    stop_sequence: str | None = None
    usage: AnthropicUsage | None = None
    # FreeToken extension, served only under --enable-metrics-report: per-request inference
    # timings. Anthropic's wire has no such field, so it stays absent by default rather than
    # being emitted as null -- both response paths dump with exclude_none.
    metrics: dict[str, Any] | None = None

    def model_post_init(self, __context):
        if not self.id:
            self.id = f"msg_{int(time.time() * 1000)}"


class AnthropicStreamEvent(BaseModel):
    type: Literal[
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "ping",
        "error",
    ]
    message: Optional[AnthropicMessagesResponse] = None
    delta: AnthropicDelta | None = None
    content_block: AnthropicContentBlock | None = None
    index: int | None = None
    error: AnthropicError | None = None
    usage: AnthropicUsage | None = None
    # Rides message_delta next to usage, under --enable-metrics-report only.
    metrics: dict[str, Any] | None = None
