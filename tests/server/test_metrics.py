"""Read the exposition with Prometheus' parser, and compare to existing accounting."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from freetoken.message import UserReply
from freetoken.server import api_server, request_ring
from freetoken.server.control_api import register_control_routes
from freetoken.server.generation import GenSpec, generate_events, generate_full
from freetoken.server.metrics import register_metrics_routes
from freetoken.server.request_ring import RequestRecord
from freetoken.server.stats import StatsTracker
from prometheus_client.parser import text_string_to_metric_families


def samples(tracker):
    return {
        sample.name: sample.value
        for metric in text_string_to_metric_families(tracker.metrics.render().decode())
        for sample in metric.samples
        if "le" not in sample.labels
    }


def record(**kwargs):
    fields = dict(
        ts="2026-01-01T00:00:00Z", method="POST", path="/v1/messages", status=200,
        model="test", duration_ms=1100, ttft_ms=100, prompt_tokens=12,
        completion_tokens=3, stream=True, error=None,
    )
    return RequestRecord(**(fields | kwargs))


def test_counters_are_the_accounting_totals_including_late_aborted_work():
    tr = StatsTracker(model_name="test")
    tr.on_new_user(1)
    tr.observe(UserReply(1, "", False, prompt_tokens_delta=12))
    tr.on_abort(1)
    tr.observe(UserReply(1, "x", False, completion_tokens_delta=2))
    assert tr.active == 1
    tr.observe(UserReply(1, "", True, completion_tokens_delta=1, error="request aborted"))
    data = samples(tr)
    assert tr.active == 0 and tr.completed == 0
    assert data["freetoken:prompt_tokens_total"] == tr.prompt_tokens_total == 12
    assert data["freetoken:generation_tokens_total"] == tr.completion_tokens_total == 3
    assert samples(tr) == data  # scraping is read-only


def test_unknown_queue_and_capacity_are_absent_until_real_snapshots():
    tr = StatsTracker()
    data = samples(tr)
    assert "freetoken:num_requests_waiting" not in data
    assert "freetoken:kv_cache_usage_perc" not in data
    tr.observe_queue(2, 5)
    tr.observe(UserReply(1, "", False, kv_used_pages=3, kv_total_pages=12))
    data = samples(tr)
    assert data["freetoken:num_requests_running"] == 2
    assert data["freetoken:num_requests_waiting"] == 5
    assert data["freetoken:kv_cache_usage_perc"] == 0.25
    tr.observe_queue(0, 0)
    assert samples(tr)["freetoken:num_requests_waiting"] == 0


def test_histograms_consume_each_record_once_and_survive_ring_eviction():
    tr = StatsTracker()
    ring = request_ring.RequestRing(capacity=1)
    for rec in (record(), record(duration_ms=2100)):
        ring.add(rec)
        tr.observe_request(rec)
    data = samples(tr)
    assert data["freetoken:time_to_first_token_seconds_count"] == 2
    assert data["freetoken:time_to_first_token_seconds_sum"] == pytest.approx(0.2)
    assert data["freetoken:time_per_output_token_seconds_sum"] == pytest.approx(1.5)
    assert data["freetoken:e2e_request_latency_seconds_sum"] == pytest.approx(3.2)
    assert samples(tr) == data


def test_missing_ttft_single_token_and_failure_do_not_invent_decode_latency():
    tr = StatsTracker()
    for rec in (
        record(ttft_ms=None, stream=False),
        record(completion_tokens=1),
        record(error="aborted", completion_tokens=0),
    ):
        tr.observe_request(rec)
    data = samples(tr)
    assert data["freetoken:e2e_request_latency_seconds_count"] == 3
    assert data["freetoken:time_to_first_token_seconds_count"] == 2
    assert data["freetoken:time_per_output_token_seconds_count"] == 0
    assert samples(StatsTracker())["freetoken:e2e_request_latency_seconds_count"] == 0


def test_metrics_and_stats_read_the_same_instance():
    config = SimpleNamespace(
        served_model_name="test", max_seq_len=100,
        model_config=SimpleNamespace(is_moe=False),
    )
    tr = StatsTracker(model_name="test")
    state = SimpleNamespace(stats=tr, config=config)
    app = FastAPI()
    register_control_routes(app, lambda: state)
    register_metrics_routes(app, lambda: state)
    tr.observe(UserReply(1, "hi", True, prompt_tokens_delta=7, completion_tokens_delta=2))
    tr.observe_queue(0, 0)
    with TestClient(app) as http:
        stats = http.get("/v1/stats").json()["requests"]
        response = http.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert stats["prompt_tokens_total"] == samples(tr)["freetoken:prompt_tokens_total"]
    assert stats["completion_tokens_total"] == samples(tr)["freetoken:generation_tokens_total"]
    assert 'model_name="test"' in response.text


@pytest.mark.parametrize("stream", [False, True])
def test_shared_generation_observes_the_identical_desktop_record(monkeypatch, stream):
    tr = StatsTracker(model_name="test")
    config = SimpleNamespace(served_model_name="test", reasoning_parser=None, tool_call_parser="llama3")

    async def acks(uid):
        yield UserReply(uid, "a", False, prompt_tokens_delta=12, completion_tokens_delta=1)
        yield UserReply(uid, "bc", True, completion_tokens_delta=2)

    state = SimpleNamespace(config=config, stats=tr, wait_for_ack=acks)
    monkeypatch.setattr(api_server, "_GLOBAL_STATE", state)
    request_ring.reset()
    spec = GenSpec(messages=[], sampling_params=SimpleNamespace())

    async def run():
        if stream:
            async for _ in generate_events(1, spec, state, source="/v1/messages"):
                pass
        else:
            await generate_full(1, spec, state, source="/v1/messages")

    asyncio.run(run())
    rows, _ = request_ring.requests_since(0, 10)
    assert len(rows) == 1
    data = samples(tr)
    assert data["freetoken:e2e_request_latency_seconds_count"] == 1
    assert data["freetoken:e2e_request_latency_seconds_sum"] == rows[0]["duration_ms"] / 1000
    assert data["freetoken:time_to_first_token_seconds_count"] == int(stream)
    # Recording request totals must NOT increment engine accounting a second time.
    assert tr.prompt_tokens_total == tr.completion_tokens_total == 0


def test_frontend_listener_consumes_queue_telemetry_without_admitting_work():
    from freetoken.message import QueueStatsReply

    class Source:
        def __init__(self):
            self.messages = iter([QueueStatsReply(4, 7)])

        async def get(self):
            msg = next(self.messages, None)
            if msg is None:
                raise RuntimeError("test queue drained")
            return msg

    manager = api_server.FrontendManager(
        config=SimpleNamespace(served_model_name="test"),
        send_tokenizer=None, recv_tokenizer=Source(),
    )
    with pytest.raises(RuntimeError, match="test queue drained"):
        asyncio.run(manager.listen())
    assert manager.stats.scheduler_running == 4
    assert manager.stats.scheduler_waiting == 7
    assert manager.stats.active == 0
    assert manager.ack_map == manager.event_map == {}
