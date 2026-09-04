"""OpenAI logprobs formatting: neutral engine entries -> wire shapes."""

from __future__ import annotations

from typing import Any


def chat_logprobs_error(req: Any) -> str | None:
    if req.top_logprobs is not None and not req.logprobs:
        return "top_logprobs requires logprobs=true"
    if req.top_logprobs is not None and not 0 <= req.top_logprobs <= 20:
        return "top_logprobs must be between 0 and 20"
    return None


def completion_logprobs_error(req: Any) -> str | None:
    if req.logprobs is not None and not 0 <= req.logprobs <= 5:
        return "logprobs must be between 0 and 5"
    if req.echo and req.logprobs is not None:
        return "echo with logprobs is not supported"
    return None


def chat_content_entry(entry: dict) -> dict:
    return {
        "token": entry["token"],
        "logprob": entry["logprob"],
        "bytes": entry["bytes"],
        "top_logprobs": [
            {
                "token": candidate["token"],
                "logprob": candidate["logprob"],
                "bytes": candidate["bytes"],
            }
            for candidate in entry["top"]
        ],
    }


def completions_logprobs(entries: list[dict], start_offset: int = 0) -> dict:
    tokens: list[str] = []
    token_logprobs: list[float] = []
    top_logprobs: list[dict[str, float]] = []
    text_offset: list[int] = []
    offset = start_offset

    for entry in entries:
        token = entry["token"]
        tokens.append(token)
        token_logprobs.append(entry["logprob"])
        top_logprobs.append({candidate["token"]: candidate["logprob"] for candidate in entry["top"]})
        text_offset.append(offset)
        offset += len(token)

    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": text_offset,
    }
