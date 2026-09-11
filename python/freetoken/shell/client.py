"""The FreeToken shell's view of a running server: an ordinary API client.

Chat goes through ``POST /v1/chat/completions`` (SSE) -- the same endpoint opencode and the
desktop app use -- so the shell shares one code path with every other client: prompt rendering,
sampling defaults, the reasoning split, tool-call parsing, request accounting. The status bar
reads the public control plane (``/health``, ``/v1/stats``, ``/v1/cache/status``, ``/v1/models``),
and ``/cache`` drives ``POST /v1/cache/rebuild``. There is no shell-private server interface, so
``ft shell`` can attach to any server, in this process or on another machine.

Transport follows what the repo already does: the ``openai`` SDK for generation (like
``freetoken.benchmark.client``), stdlib ``urllib`` for the control-plane JSON (like ``ft ctl``
and the daemon). Nothing here imports torch -- attaching to a remote server costs a few
milliseconds of imports, not a CUDA context.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import openai
from openai import AsyncOpenAI

# Any string is accepted by the server; sent so the SDK doesn't refuse to build a request.
LOCAL_API_KEY = "freetoken-local"
CONTROL_TIMEOUT = 10.0
# Per-read timeout on the chat stream. Generous: it has to cover the gap before the first
# token, which is a full prefill of the whole conversation on a cold cache.
CHAT_READ_TIMEOUT = 900.0
READY_POLL_INTERVAL = 0.5


class ShellClientError(Exception):
    """A request to the server failed, or the server refused it. Carries a message meant to be
    printed to the user as-is."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ReasoningDelta:
    text: str


@dataclass(frozen=True)
class ContentDelta:
    text: str


@dataclass(frozen=True)
class TurnDone:
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


ChatEvent = ReasoningDelta | ContentDelta | TurnDone


@dataclass(frozen=True)
class Sampling:
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None


def _error_text(payload: Any) -> str | None:
    """Pull a human-readable message out of the two error shapes the server emits: the OpenAI
    envelope ``{"error": {"message": ...}}`` and the plain ``{"error": "..."}`` / ``{"detail":
    ...}`` used by the maintenance gates."""
    if isinstance(payload, str):
        return payload or None
    if not isinstance(payload, dict):
        return None
    for key in ("error", "detail", "message"):
        value = payload.get(key)
        if isinstance(value, dict):
            nested = _error_text(value)
            if nested:
                return nested
        elif value:
            return str(value)
    return None


class ShellClient:
    def __init__(
        self,
        origin: str,
        *,
        api_key: str = LOCAL_API_KEY,
        timeout: float = CONTROL_TIMEOUT,
    ) -> None:
        self.origin = origin.rstrip("/")
        self.timeout = timeout
        # Sent on the control plane too (the OpenAI client already sends it on chat), so a
        # server started with --api-key accepts both halves of the shell. The default is the
        # placeholder an unauthenticated server ignores.
        self.api_key = api_key
        self._openai = AsyncOpenAI(
            base_url=f"{self.origin}/v1",
            api_key=api_key,
            max_retries=0,  # a retried chat request would generate the turn twice
            timeout=openai.Timeout(CHAT_READ_TIMEOUT, connect=10.0),
        )

    async def aclose(self) -> None:
        await self._openai.close()

    # ---------------------------------------------------------------- control plane

    def _request_json_blocking(
        self, method: str, path: str, body: dict[str, Any] | None, timeout: float
    ) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json", "Authorization": f"Bearer {self.api_key}"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.origin}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            # A rebuild rejection (409/503) is a normal answer, not a transport failure: the
            # body carries the status the caller wants to report, so hand it back as data.
            raw = exc.read()
            try:
                doc = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                doc = None
            if isinstance(doc, dict) and doc.get("status"):
                return doc
            raise ShellClientError(
                f"HTTP {exc.code}: {_error_text(doc) or exc.reason}", status=exc.code
            ) from exc
        except urllib.error.URLError as exc:
            raise ShellClientError(f"cannot reach {self.origin}: {exc.reason}") from exc
        except (TimeoutError, OSError) as exc:
            raise ShellClientError(f"cannot reach {self.origin}: {exc}") from exc
        if not raw:
            return {}
        try:
            doc = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ShellClientError(f"{path} returned invalid JSON") from exc
        if not isinstance(doc, dict):
            raise ShellClientError(f"{path} returned a non-object JSON document")
        return doc

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._request_json_blocking, method, path, body, timeout or self.timeout
        )

    async def health(self) -> dict[str, Any]:
        return await self._request_json("GET", "/health")

    async def stats(self) -> dict[str, Any]:
        return await self._request_json("GET", "/v1/stats")

    async def cache_status(self) -> dict[str, Any]:
        return await self._request_json("GET", "/v1/cache/status")

    async def cache_rebuild(
        self,
        *,
        moe_cache_size: int | None = None,
        num_pages: int | None = None,
        num_mamba_slots: int | None = None,
        num_swa_pages: int | None = None,
        wait: float = 300.0,
    ) -> dict[str, Any]:
        """Resize cache pools. Every count is in the endpoint's own unit -- slots for moe/mamba,
        pages for kv and for the window pool (whose page is P on DSV4, 1 token on radix-SWA).
        Untargeted pools are left out of the body entirely, which the server reads as 'keep'."""
        body: dict[str, Any] = {"timeout": wait}
        for key, value in (
            ("moe_cache_size", moe_cache_size),
            ("num_pages", num_pages),
            ("num_mamba_slots", num_mamba_slots),
            ("num_swa_pages", num_swa_pages),
        ):
            if value is not None:
                body[key] = value
        # The server blocks for the whole rebuild, so the client must outwait it.
        return await self._request_json(
            "POST", "/v1/cache/rebuild", body=body, timeout=wait + self.timeout
        )

    async def model_id(self) -> str | None:
        """The first id from ``/v1/models``. None when the server reports none (it is a single-
        model server, so this is the model the shell will be talking to)."""
        doc = await self._request_json("GET", "/v1/models")
        data = doc.get("data")
        if not isinstance(data, list):
            return None
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                return item["id"]
        return None

    async def wait_until_ready(
        self,
        *,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        connect_grace: float = 0.0,
    ) -> dict[str, Any]:
        """Block until ``/health`` reports ok, returning that document.

        ``connect_grace`` is how long a *connection* failure is tolerated before giving up --
        0 for attaching to a server that is supposed to be up already (so a typo'd address
        fails immediately with the reason), a few seconds when we just started the server
        ourselves and uvicorn may not have finished binding. A server that is up but still
        loading weights is always waited on: that phase has no useful timeout, and ^C quits.
        """
        deadline = time.monotonic() + connect_grace
        while True:
            try:
                doc = await self.health()
            except ShellClientError:
                if time.monotonic() >= deadline:
                    raise
                await asyncio.sleep(0.25)
                continue
            status = doc.get("status")
            if status == "ok":
                return doc
            if status == "error":
                raise ShellClientError(str(doc.get("message") or "the server reported an error"))
            if on_progress is not None:
                on_progress(doc)
            await asyncio.sleep(READY_POLL_INTERVAL)

    # ---------------------------------------------------------------- generation

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        sampling: Sampling,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> AsyncIterator[ChatEvent]:
        """Stream one turn. Yields reasoning/content deltas as the server splits them, then
        exactly one :class:`TurnDone` carrying the authoritative token usage.

        Cancelling the iteration (``break``, ^C) closes the HTTP response, which the server
        reads as a client disconnect and turns into an abort -- the engine stops decoding
        instead of finishing a turn nobody is reading.
        """
        extra_body: dict[str, Any] = {}
        if chat_template_kwargs:
            extra_body["chat_template_kwargs"] = chat_template_kwargs
        if sampling.top_k is not None:
            extra_body["top_k"] = sampling.top_k  # not an OpenAI field; FreeToken accepts it

        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if sampling.max_tokens is not None:
            request["max_tokens"] = sampling.max_tokens
        if sampling.temperature is not None:
            request["temperature"] = sampling.temperature
        if sampling.top_p is not None:
            request["top_p"] = sampling.top_p

        try:
            stream = await self._openai.chat.completions.create(
                **request, extra_body=extra_body or None
            )
        except openai.APIStatusError as exc:
            raise ShellClientError(
                _error_text(exc.body) or f"HTTP {exc.status_code}", status=exc.status_code
            ) from exc
        except openai.APIConnectionError as exc:
            raise ShellClientError(f"cannot reach {self.origin}: {exc}") from exc

        finish_reason: str | None = None
        prompt_tokens = 0
        completion_tokens = 0
        try:
            async for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                for choice in chunk.choices or []:
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                    delta = choice.delta
                    if delta is None:
                        continue
                    # reasoning_content is FreeToken's (and vLLM/SGLang's) split channel; the
                    # SDK keeps unknown fields, so read it off the model as an extra.
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        yield ReasoningDelta(reasoning)
                    if delta.content:
                        yield ContentDelta(delta.content)
        except openai.APIError as exc:
            raise ShellClientError(f"stream failed: {exc}") from exc
        finally:
            await stream.close()

        yield TurnDone(
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
