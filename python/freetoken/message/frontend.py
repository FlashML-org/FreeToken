from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .utils import deserialize_type, serialize_type


@dataclass
class BaseFrontendMsg:
    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    data: List[BaseFrontendMsg]


@dataclass
class UserReply(BaseFrontendMsg):
    uid: int
    incremental_output: str
    finished: bool
    prompt_tokens_delta: int = 0
    completion_tokens_delta: int = 0
    # Prefix-cache hit length (tokens of the prompt served from cache instead of
    # recomputed). Arrives once, on the same reply as prompt_tokens_delta.
    cached_tokens: int = 0
    # KV page-pool usage snapshot (not-evictable used/total) for the shell status bar.
    # 0/0 when not reported (e.g. the prompt-tokens reply, or owned-KV models with no
    # shared page pool).
    kv_used_pages: int = 0
    kv_total_pages: int = 0
    # GDN (mamba) state-pool slot usage (used/total) for hybrid models, else 0/0.
    mamba_used_slots: int = 0
    mamba_total_slots: int = 0
    # Window (swa) pool token usage (used/total) for SWA models, else 0/0.
    swa_used_tokens: int = 0
    swa_total_tokens: int = 0
    # Bytes the engine process holds on the GPU (torch reserved pool). 0 when not reported.
    gpu_mem_bytes: int = 0
    # Set (with finished=True) when a request failed before producing output — e.g. a chat
    # template that the tokenizer cannot render, or a prompt that exceeds the KV budget the
    # scheduler can serve. Carries a human-readable reason. Without this, such a request would
    # send no `finished` reply at all and the API layer's wait_for_ack would hang forever.
    error: str | None = None
    # Machine-readable class for `error` (see ErrorReplyMsg.code), surfaced as OpenAI's error
    # `code`. None when the failure has no specific class.
    error_code: str | None = None
    finish_reason: str | None = None
    # The stop string that ended generation (Anthropic reports it as stop_reason='stop_sequence').
    matched_stop: str | None = None
    # Neutral sampled-token logprobs entry for this token (see
    # tokenizer.detokenize.build_logprobs_entry); None when the request did not ask.
    logprobs: dict | None = None


@dataclass
class CacheRebuildReply(BaseFrontendMsg):
    # detokenizer worker -> api server: result of a /v1/cache/rebuild request.
    request_id: str
    status: str  # "ok" | "busy" | "failed"
    moe_cache_size: int = 0
    num_pages: int = 0
    mamba_slots: int = 0
    num_swa_pages: int = 0
    error: str | None = None
