from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from freetoken.core import Batch


@dataclass
class SchedulerStatusReporter:
    log: Callable[[str], None]
    clock: Callable[[], float] = time.perf_counter
    decode_log_interval: int = 40
    _last_prefill_time: float = field(init=False)
    _last_decode_time: float = field(init=False)
    _decode_forward_count: int = field(default=0, init=False)
    _decode_generated_tokens: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        now = self.clock()
        self._last_prefill_time = now
        self._last_decode_time = now
        self.decode_log_interval = max(1, self.decode_log_interval)

    def report_batch(
        self,
        batch: Batch,
        *,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        page_size: int,
        mamba_slots: tuple[int, int] | None = None,
        swa_tokens: tuple[int, int] | None = None,
        moe_stats: dict | None = None,
    ) -> None:
        if batch.is_prefill:
            self._report_prefill(
                batch,
                running_reqs=running_reqs,
                queue_reqs=queue_reqs,
                kv_used_pages=kv_used_pages,
                kv_total_pages=kv_total_pages,
                mamba_slots=mamba_slots,
                swa_tokens=swa_tokens,
            )
        elif batch.is_decode:
            self._report_decode(
                batch,
                running_reqs=running_reqs,
                queue_reqs=queue_reqs,
                kv_used_pages=kv_used_pages,
                kv_total_pages=kv_total_pages,
                page_size=page_size,
                mamba_slots=mamba_slots,
                swa_tokens=swa_tokens,
                moe_stats=moe_stats,
            )

    def decode_stats_due(self, batch: Batch) -> bool:
        """Whether next decode report will log and may safely read device counters."""
        return (
            batch.is_decode
            and (self._decode_forward_count + 1) % self.decode_log_interval == 0
        )

    def _report_prefill(
        self,
        batch: Batch,
        *,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        mamba_slots: tuple[int, int] | None = None,
        swa_tokens: tuple[int, int] | None = None,
    ) -> None:
        now = self.clock()
        gap = now - self._last_prefill_time
        self._last_prefill_time = now
        # Read the schedule-time snapshot: by report time the forward's complete_one() has
        # advanced each req's cached_len to device_len, so reading the reqs here would log
        # decode-state values (#new-token == #reqs, #cached-token == full prompt).
        new_tokens = batch.log_new_tokens
        cached_tokens = batch.log_cached_tokens
        input_throughput = new_tokens / gap if gap > 0 else 0.0
        self.log(
            f"Prefill batch, "
            f"#new-seq: {len(batch.reqs)}, "
            f"#new-token: {new_tokens}, "
            f"#cached-token: {cached_tokens}, "
            f"token usage: {_usage_ratio(kv_used_pages, kv_total_pages):.2f}, "
            f"{_swa_msg(swa_tokens)}"
            f"{_mamba_msg(mamba_slots)}"
            f"#running-req: {running_reqs}, "
            f"#queue-req: {queue_reqs}, "
            f"input throughput (token/s): {input_throughput:.2f}"
        )

    def _report_decode(
        self,
        batch: Batch,
        *,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        page_size: int,
        mamba_slots: tuple[int, int] | None = None,
        swa_tokens: tuple[int, int] | None = None,
        moe_stats: dict | None = None,
    ) -> None:
        self._decode_forward_count += 1
        self._decode_generated_tokens += len(batch.reqs)
        if self._decode_forward_count % self.decode_log_interval != 0:
            return

        now = self.clock()
        gap = now - self._last_decode_time
        self._last_decode_time = now
        gen_throughput = self._decode_generated_tokens / gap if gap > 0 else 0.0
        self._decode_generated_tokens = 0
        self.log(
            f"Decode batch, "
            f"#running-req: {running_reqs}, "
            f"#token: {kv_used_pages * page_size}, "
            f"token usage: {_usage_ratio(kv_used_pages, kv_total_pages):.2f}, "
            f"{_swa_msg(swa_tokens)}"
            f"{_mamba_msg(mamba_slots)}"
            f"gen throughput (token/s): {gen_throughput:.2f}, "
            f"#queue-req: {queue_reqs}"
            f"{_moe_msg(moe_stats)}"
        )


def _usage_ratio(used: int, total: int) -> float:
    return used / total if total > 0 else 0.0


def _moe_msg(stats: dict | None) -> str:
    """MoE cache hit/miss summary for the decode log line (empty when stats are off)."""
    if not stats:
        return ""
    miss = stats.get("miss_rate")
    fetch = stats.get("fetch_rate")
    cpu = stats.get("cpu_per_layer")
    hit = (1.0 - miss) if miss is not None else None
    parts = [f"moe hit: {hit:.3f}" if hit is not None else "moe hit: n/a"]
    if miss is not None:
        parts.append(f"miss: {miss:.3f}")
    if fetch is not None:
        parts.append(f"fetch: {fetch:.3f}")
    if cpu is not None:
        parts.append(f"cpu: {cpu:.3f}")
    return ", " + ", ".join(parts)


def _mamba_msg(mamba_slots: tuple[int, int] | None) -> str:
    """GDN-state (mamba) pool occupancy for hybrid models; empty for the rest."""
    if mamba_slots is None:
        return ""
    used, total = mamba_slots
    return f"#mamba-slot: {used}/{total}, mamba usage: {_usage_ratio(used, total):.2f}, "


def _swa_msg(swa_tokens: tuple[int, int] | None) -> str:
    """Window (swa) pool occupancy for SWA models; empty for the rest."""
    if swa_tokens is None:
        return ""
    used, total = swa_tokens
    return f"#swa-token: {used}/{total}, swa usage: {_usage_ratio(used, total):.2f}, "
