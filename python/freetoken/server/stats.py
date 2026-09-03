"""Runtime metrics for /v1/stats. The FrontendManager owns one StatsTracker and feeds it
every UserReply (the single chokepoint in listen()). kv/mamba/vram keep their last-known-value
semantics like ShellStats; throughput uses an independent sliding-window rate (NOT cumulative
average like tok_s), so idle polls decay to zero by wall clock."""

from __future__ import annotations

import time
from collections import deque
from typing import Any


class StatsTracker:
    def __init__(self, window_s: float = 5.0) -> None:
        self.window_s = window_s
        # maxlen bounds memory on the headless path: stale-sample eviction is poll-driven
        # (only _rate() trims to window_s), and clients that never hit /v1/stats (e.g.
        # codex/claude via /v1/chat/completions) would otherwise grow these unbounded.
        # 4096 is generous vs the sliding window's span at any realistic reply rate.
        self._decode: "deque[tuple[float, int]]" = deque(maxlen=4096)
        self._prefill: "deque[tuple[float, int]]" = deque(maxlen=4096)
        self._inflight: set[int] = set()
        # Requests for which an abort was dispatched but the scheduler's explicit terminal
        # acknowledgement has not arrived yet. They remain active until that barrier, while
        # any sampled tokens racing the abort continue to count toward lifetime totals.
        self._aborting: set[int] = set()
        self.completed = 0
        # Cumulative prompt/completion tokens since this process started (lifetime for THIS served
        # model). Exposed in /v1/stats so the desktop can diff consecutive polls into per-model
        # "cost saved by running locally" accounting. Monotonic; resets when the process restarts.
        self.prompt_tokens_total = 0
        self.completion_tokens_total = 0
        self.kv_used_pages = 0
        self.kv_total_pages = 0
        self.mamba_used_slots = 0
        self.mamba_total_slots = 0
        self.swa_used_tokens = 0
        self.swa_total_tokens = 0
        self.vram_bytes = 0

    @property
    def active(self) -> int:
        return len(self._inflight)

    @property
    def inflight_uids(self) -> tuple[int, ...]:
        """Stable snapshot used by prepare-stop to abort every still-admitted request."""
        return tuple(sorted(self._inflight))

    def on_new_user(self, uid: int) -> None:
        self._inflight.add(uid)
        self._aborting.discard(uid)

    def on_abort(self, uid: int) -> None:
        if uid in self._inflight:
            self._aborting.add(uid)

    def observe(self, reply: Any, now: float | None = None) -> None:
        t = time.monotonic() if now is None else now
        if getattr(reply, "completion_tokens_delta", 0) > 0:
            self._decode.append((t, reply.completion_tokens_delta))
            self.completion_tokens_total += reply.completion_tokens_delta
        if getattr(reply, "prompt_tokens_delta", 0) > 0:
            self._prefill.append((t, reply.prompt_tokens_delta))
            self.prompt_tokens_total += reply.prompt_tokens_delta
        if getattr(reply, "kv_total_pages", 0) > 0:  # ignore 0/0 (prompt reply, owned-KV)
            self.kv_used_pages = reply.kv_used_pages
            self.kv_total_pages = reply.kv_total_pages
        if getattr(reply, "mamba_total_slots", 0) > 0:  # hybrid (GDN) only
            self.mamba_used_slots = reply.mamba_used_slots
            self.mamba_total_slots = reply.mamba_total_slots
        if getattr(reply, "swa_total_tokens", 0) > 0:  # SWA (window pool) only
            self.swa_used_tokens = reply.swa_used_tokens
            self.swa_total_tokens = reply.swa_total_tokens
        if getattr(reply, "gpu_mem_bytes", 0) > 0:
            self.vram_bytes = reply.gpu_mem_bytes
        if getattr(reply, "finished", False):
            uid = getattr(reply, "uid", None)
            if uid in self._inflight:
                self._inflight.discard(uid)
                if uid in self._aborting:
                    self._aborting.discard(uid)
                else:
                    self.completed += 1

    def _rate(self, window: "deque[tuple[float, int]]", now: float | None) -> float:
        t = time.monotonic() if now is None else now
        cutoff = t - self.window_s
        while window and window[0][0] < cutoff:
            window.popleft()
        if not window:
            return 0.0
        total = sum(n for _ts, n in window)
        span = max(t - window[0][0], 1e-9)
        return total / span

    def decode_tps(self, now: float | None = None) -> float:
        return self._rate(self._decode, now)

    def prefill_tps(self, now: float | None = None) -> float:
        return self._rate(self._prefill, now)


def derive_model_card(config: Any) -> dict:
    """attn enum + moe bool + ctx from the model config."""
    mc = config.model_config
    if getattr(mc, "has_linear_attention", False):
        attn = "hybrid_linear"
    elif getattr(mc, "has_swa_attention", False):
        attn = "hybrid_swa"
    else:
        attn = "mha"
    return {
        "id": config.served_model_name,
        "ctx": config.max_seq_len,
        "attn": attn,
        "moe": bool(getattr(mc, "is_moe", False)),
    }


def _swa_page_size(config: Any) -> int:
    """The window pool's own page unit: P (window_size) for DSV4, 1 token for radix-SWA.
    Mirrors compute_cache_pools' swa_page_size."""
    dsv4 = getattr(getattr(config, "model_config", None), "dsv4_args", None)
    if dsv4 is not None:
        return int(getattr(dsv4, "window_size", 0) or 1)
    return 1


def build_stats(state: Any, p95_ms: int, ttft_mean_ms: int) -> dict:
    """Full /v1/stats doc. throughput is 0 when idle; kv/mamba/swa are null
    when their total is 0 (owned-KV / non-hybrid / non-SWA). kv and swa share one shape:
    pages + the pool's own page_size (tokens = pages x page_size). gpus: the engine's GPU as
    [{index, name, uuid, total_bytes}] (the primary rank's; a list so TP can extend it), []
    until the readiness meta arrives."""
    tr: StatsTracker = state.stats
    config = state.config
    ready_at = getattr(state, "ready_at", None)
    uptime_s = max(0, int(time.monotonic() - ready_at)) if ready_at is not None else 0
    kv = (
        {"used_pages": tr.kv_used_pages, "total_pages": tr.kv_total_pages,
         "page_size": getattr(config, "page_size", 1)}
        if tr.kv_total_pages > 0 else None
    )
    mamba = (
        {"used_slots": tr.mamba_used_slots, "total_slots": tr.mamba_total_slots}
        if tr.mamba_total_slots > 0 else None
    )
    sps = _swa_page_size(config)
    swa = (
        {"used_pages": tr.swa_used_tokens // sps, "total_pages": tr.swa_total_tokens // sps,
         "page_size": sps}
        if tr.swa_total_tokens > 0 else None
    )
    return {
        "instance_id": getattr(state, "instance_id", None),
        "model": derive_model_card(config),
        "uptime_s": uptime_s,
        "kv": kv,
        "mamba": mamba,
        "swa": swa,
        "vram_bytes": tr.vram_bytes,
        "gpus": list(getattr(state, "gpus", None) or []),
        "throughput": {
            "decode_tps": round(tr.decode_tps(), 1),
            "prefill_tps": round(tr.prefill_tps(), 1),
        },
        "requests": {
            "active": tr.active,
            "completed": tr.completed,
            "p95_ms": p95_ms,
            "ttft_mean_ms": ttft_mean_ms,
            "prompt_tokens_total": tr.prompt_tokens_total,
            "completion_tokens_total": tr.completion_tokens_total,
        },
    }
