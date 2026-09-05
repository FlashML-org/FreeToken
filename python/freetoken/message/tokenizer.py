from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from freetoken.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseTokenizerMsg:
    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: List[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    uid: int
    next_token: int
    finished: bool
    finish_reason: str | None = None
    # The stop string that ended generation (if any), so the detokenizer can trim it
    # and everything after it from the final output.
    matched_stop: str | None = None
    # The request's stop strings (None when it has none), so the detokenizer can hold back
    # a trailing partial-stop prefix instead of streaming it and then needing to retract.
    stop_strs: list[str] | None = None
    # KV page-pool usage snapshot at this step (not-evictable used/total), passed
    # through to the frontend for the shell status bar. 0/0 for owned-KV models.
    kv_used_pages: int = 0
    kv_total_pages: int = 0
    # GDN (mamba) state-pool slot usage (used/total) for hybrid models, else 0/0.
    mamba_used_slots: int = 0
    mamba_total_slots: int = 0
    # Window (swa) pool token usage (used/total) for SWA models, else 0/0.
    swa_used_tokens: int = 0
    swa_total_tokens: int = 0
    # Bytes this engine process holds on the GPU (torch reserved pool). 0 on CPU.
    gpu_mem_bytes: int = 0


@dataclass
class PromptAdmittedMsg(BaseTokenizerMsg):
    """Scheduler -> tokenizer accounting signal for an admitted prompt.

    Emitted exactly once, after the request's first prefill batch has been prepared
    successfully. ``prompt_tokens`` is the complete prompt length, even when the
    prefill itself is chunked. ``cached_tokens`` is the prefix-cache hit — the subset
    of ``prompt_tokens`` served from cache instead of recomputed.
    """

    uid: int
    prompt_tokens: int
    cached_tokens: int = 0


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    uid: int
    text: str | List[Dict[str, Any]]
    sampling_params: SamplingParams
    chat_template_kwargs: Dict[str, Any] | None = None
    tools: List[Dict[str, Any]] | None = None
    # Encoded image files, in the order their ``{"type": "image"}`` parts appear in ``text``.
    images: List[bytes] | None = None


@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int


@dataclass
class CacheRebuildMsg(BaseTokenizerMsg):
    # api server -> tokenizer worker (pure passthrough to CacheRebuildBackendMsg).
    request_id: str
    moe_cache_size: int | None = None
    num_pages: int | None = None
    num_mamba_slots: int | None = None
    num_swa_pages: int | None = None
    mode: str = "if_idle"


@dataclass
class CacheRebuildResultMsg(BaseTokenizerMsg):
    # scheduler -> detokenizer worker (passthrough to CacheRebuildReply).
    request_id: str
    status: str  # "ok" | "busy" | "failed"
    moe_cache_size: int = 0
    num_pages: int = 0
    mamba_slots: int = 0
    num_swa_pages: int = 0
    error: str | None = None


@dataclass
class ErrorReplyMsg(BaseTokenizerMsg):
    # scheduler -> tokenizer/detokenizer worker -> frontend: a request the scheduler cannot
    # serve (e.g. its prompt exceeds the KV budget). The worker translates it into a terminal
    # UserReply(error=...) so the client gets a prompt error instead of hanging on a `finished`
    # reply that will never come. Mirrors the CacheRebuildResultMsg passthrough pattern.
    uid: int
    error: str
    # Stable machine-readable reason, mirrored onto the wire as OpenAI's error `code` so a client
    # can react without parsing prose. Today only "context_length_exceeded" (prompt longer than
    # the servable context); None = no specific class, the message is all there is.
    code: str | None = None
