"""Cumulative prefill/decode timing on /v1/stats.

The sliding-window rates decay to zero between polls, so an external poller cannot turn them
into per-session speeds. These totals let it diff consecutive polls instead: tokens over the
seconds that produced them, the same shape as llama.cpp's prompt/predicted seconds counters."""

from types import SimpleNamespace

import pytest

from freetoken.server.stats import StatsTracker, build_stats


def reply(uid, *, prompt=0, completion=0, cached=0, finished=False):
    return SimpleNamespace(
        uid=uid,
        prompt_tokens_delta=prompt,
        completion_tokens_delta=completion,
        cached_tokens=cached,
        finished=finished,
    )


def test_prefill_runs_from_admission_to_first_output_and_decode_from_there():
    tr = StatsTracker()
    tr.on_new_user(7, now=100.0)
    tr.observe(reply(7, prompt=1000, cached=200), now=100.1)
    tr.observe(reply(7, completion=1), now=102.0)  # first token: 2.0 s after admission
    tr.observe(reply(7, completion=3), now=102.5)
    tr.observe(reply(7, completion=2, finished=True), now=103.0)

    assert tr.prefill_seconds_total == pytest.approx(2.0)
    assert tr.decode_seconds_total == pytest.approx(1.0)
    # The first token falls out of prefill; only tokens after it are decode work.
    assert tr.decode_tokens_total == 5
    assert tr.completion_tokens_total == 6
    assert tr.prompt_tokens_total == 1000
    assert tr.cached_prompt_tokens_total == 200


def test_tokens_riding_on_the_first_output_reply_are_not_decode_work():
    # Overlap scheduling can deliver several tokens on the first reply; no decode interval
    # measured them, so counting them would inflate the decode rate.
    tr = StatsTracker()
    tr.on_new_user(1, now=0.0)
    tr.observe(reply(1, completion=4), now=1.0)
    tr.observe(reply(1, completion=2, finished=True), now=1.5)

    assert tr.prefill_seconds_total == pytest.approx(1.0)
    assert tr.decode_seconds_total == pytest.approx(0.5)
    assert tr.decode_tokens_total == 2


def test_timing_is_summed_per_request_and_state_is_released_on_finish():
    tr = StatsTracker()
    tr.on_new_user(1, now=0.0)
    tr.on_new_user(2, now=0.0)
    tr.observe(reply(1, completion=1), now=1.0)
    tr.observe(reply(2, completion=1), now=2.0)
    tr.observe(reply(1, completion=1, finished=True), now=3.0)
    tr.observe(reply(2, completion=1, finished=True), now=3.0)

    assert tr.prefill_seconds_total == pytest.approx(3.0)
    assert tr.decode_seconds_total == pytest.approx(3.0)
    assert tr.decode_tokens_total == 2
    assert tr.active == 0
    assert not tr._admitted_at and not tr._last_output_at


def test_a_request_that_fails_before_output_adds_no_time():
    tr = StatsTracker()
    tr.on_new_user(3, now=0.0)
    tr.observe(reply(3, finished=True), now=5.0)

    assert tr.prefill_seconds_total == 0.0
    assert tr.decode_seconds_total == 0.0
    assert not tr._admitted_at


def test_stats_document_publishes_the_timing_totals():
    tr = StatsTracker()
    tr.on_new_user(1, now=0.0)
    tr.observe(reply(1, prompt=10, cached=4), now=0.1)
    tr.observe(reply(1, completion=1), now=0.25)
    tr.observe(reply(1, completion=1, finished=True), now=0.75)
    state = SimpleNamespace(
        stats=tr,
        config=SimpleNamespace(
            served_model_name="m",
            max_seq_len=4096,
            served_modalities=set(),
            model_config=SimpleNamespace(),
        ),
    )

    requests = build_stats(state, p95_ms=0, ttft_mean_ms=0)["requests"]

    assert requests["prompt_tokens_total"] == 10
    assert requests["cached_prompt_tokens_total"] == 4
    assert requests["completion_tokens_total"] == 2
    assert requests["decode_tokens_total"] == 1
    assert requests["prefill_seconds_total"] == pytest.approx(0.25)
    assert requests["decode_seconds_total"] == pytest.approx(0.5)
