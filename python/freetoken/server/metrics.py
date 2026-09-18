"""Prometheus projection of the frontend's one StatsTracker.

Counters/gauges are collected directly from that tracker; only timing distributions
need new storage. Registries are per tracker, not process globals, so tests and a
replacement engine cannot inherit another instance's counters or histogram buckets.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Histogram, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

# Seconds, spanning fast local decode through long-prefill/offload workloads.
_LATENCY_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)


class _TrackerCollector:
    def __init__(self, tracker: Any, model_name: str) -> None:
        self.tracker = tracker
        self.model_name = model_name

    def collect(self):
        tr = self.tracker
        for name, value, help_text in (
            ("prompt_tokens", tr.prompt_tokens_total, "Prompt tokens admitted by the scheduler."),
            ("generation_tokens", tr.completion_tokens_total, "Tokens actually sampled by the engine."),
        ):
            metric = CounterMetricFamily(f"freetoken:{name}", help_text, labels=["model_name"])
            metric.add_metric([self.model_name], value)
            yield metric
        # Unknown is absent, never a fabricated empty queue. Scheduler telemetry
        # arrives independently of generation replies, including idle/abort paths.
        for name, value in (
            ("num_requests_running", tr.scheduler_running),
            ("num_requests_waiting", tr.scheduler_waiting),
            ("kv_cache_usage_perc",
             tr.kv_used_pages / tr.kv_total_pages if tr.kv_total_pages else None),
        ):
            if value is not None:
                metric = GaugeMetricFamily(
                    f"freetoken:{name}", "Last scheduler snapshot; KV usage is a 0..1 ratio.",
                    labels=["model_name"],
                )
                metric.add_metric([self.model_name], value)
                yield metric


class TrackerMetrics:
    def __init__(self, tracker: Any, model_name: str) -> None:
        self.registry = CollectorRegistry()
        self.registry.register(_TrackerCollector(tracker, model_name))
        self._histograms = {}
        for name, help_text in (
            ("time_to_first_token_seconds", "Recorded streaming time to first output event."),
            ("time_per_output_token_seconds", "Mean time per output token after the first, per completed stream."),
            ("e2e_request_latency_seconds", "Shared generation request lifetime, including failures."),
        ):
            metric = Histogram(
                f"freetoken:{name}", help_text, labelnames=["model_name"],
                buckets=_LATENCY_BUCKETS, registry=self.registry,
            )
            self._histograms[name] = metric.labels(model_name=model_name)

    def observe_request(self, record: Any) -> None:
        # Consume the SAME record as the desktop ring. Never re-observe a ring
        # snapshot on scrape: eviction/repeated scrapes must not reset/double counts.
        self._histograms["e2e_request_latency_seconds"].observe(record.duration_ms / 1000)
        if record.ttft_ms is not None:
            self._histograms["time_to_first_token_seconds"].observe(record.ttft_ms / 1000)
            if not record.error and (record.completion_tokens or 0) > 1:
                decode_s = max(0, record.duration_ms - record.ttft_ms) / 1000
                self._histograms["time_per_output_token_seconds"].observe(
                    decode_s / (record.completion_tokens - 1)
                )

    def render(self) -> bytes:
        return generate_latest(self.registry)


def register_metrics_routes(app: FastAPI, get_state: Callable[[], Any]) -> None:
    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        # This async handler collects without yielding: listen() cannot mutate
        # counters halfway through one scrape on the same frontend event loop.
        return Response(
            get_state().stats.metrics.render(),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )
