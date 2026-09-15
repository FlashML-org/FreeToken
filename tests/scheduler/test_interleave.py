"""Tests for DecodeInterleavePolicy and its wiring into Scheduler._schedule_next_batch.

The production symptom this exists for (measured 2026-09-15): a 300k-token prompt
is chunked at the window pool budget (384 tok), so it becomes 797 consecutive
prefill steps, and an already-decoding request is not scheduled once during the
267-317 s that takes.  Prefill must keep priority (a request that never prefills
never starts) but must yield to decode periodically.
"""

from __future__ import annotations

import pytest

from freetoken.scheduler.interleave import DecodeInterleavePolicy


# --------------------------------------------------------------------------- #
# policy, in isolation
# --------------------------------------------------------------------------- #
def test_disabled_by_default_and_never_forces_decode():
    """Unset reproduces historical prefill-first behaviour exactly."""
    p = DecodeInterleavePolicy()
    assert not p.enabled
    for _ in range(10_000):
        p.note_prefill()
    assert not p.wants_decode()


@pytest.mark.parametrize("every", [1, 2, 8, 16])
def test_forces_decode_after_exactly_every_prefills(every):
    p = DecodeInterleavePolicy(every)
    assert p.enabled
    for i in range(1, every):
        p.note_prefill()
        assert not p.wants_decode(), f"fired early at {i} < {every}"
    p.note_prefill()
    assert p.wants_decode(), f"did not fire at {every}"


def test_decode_resets_the_streak():
    p = DecodeInterleavePolicy(4)
    for _ in range(4):
        p.note_prefill()
    assert p.wants_decode()
    p.note_decode()
    assert not p.wants_decode(), "streak must reset after a decode step"
    for _ in range(3):
        p.note_prefill()
    assert not p.wants_decode()
    p.note_prefill()
    assert p.wants_decode()


def test_every_one_interleaves_strictly():
    """every=1 is the extreme: decode after every single prefill step."""
    p = DecodeInterleavePolicy(1)
    p.note_prefill()
    assert p.wants_decode()


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_non_positive_disables(bad):
    p = DecodeInterleavePolicy(bad)
    assert not p.enabled
    for _ in range(50):
        p.note_prefill()
    assert not p.wants_decode()


def test_reset_clears_credit():
    p = DecodeInterleavePolicy(8)
    for _ in range(7):
        p.note_prefill()
    p.reset()
    assert not p.wants_decode()
    p.note_prefill()
    assert not p.wants_decode()


# --------------------------------------------------------------------------- #
# the decision the scheduler actually makes
# --------------------------------------------------------------------------- #
def _decide(policy, has_prefill, has_decode, every=8):
    """Mirror of the scheduler's new decision, as a pure function.

    prefill wins unless the policy says decode is due AND a decode is available.
    """
    if has_prefill and not (policy.wants_decode() and has_decode):
        policy.note_prefill()
        return "prefill"
    if has_decode:
        policy.note_decode()
        return "decode"
    if has_prefill:
        policy.note_prefill()
        return "prefill"
    return None


def test_burst_is_punctuated_by_decode_every_n():
    """797 prefill steps in a row -- the production shape -- must contain decode steps."""
    policy = DecodeInterleavePolicy(8)
    seq = []
    for _ in range(797):
        seq.append(_decide(policy, has_prefill=True, has_decode=True))
    n_dec = seq.count("decode")
    # A decode step resets the streak, so a cycle is every+1 slots (8 prefill + 1 decode):
    # 797 slots hold 88 full cycles, i.e. 88 decode steps.
    assert n_dec == 797 // 9, f"expected {797 // 9} decode steps, got {n_dec}"
    # and no run of prefills exceeds the threshold
    run = 0
    worst = 0
    for s in seq:
        run = run + 1 if s == "prefill" else 0
        worst = max(worst, run)
    assert worst == 8, f"longest prefill run {worst} exceeds the threshold"


def test_disabled_policy_yields_the_historical_sequence():
    policy = DecodeInterleavePolicy()
    seq = [_decide(policy, True, True) for _ in range(100)]
    assert seq.count("decode") == 0, "disabled must never take a decode slot"


def test_decode_is_not_forced_when_none_is_runnable():
    """Nothing to decode -> prefill keeps the slot (no idle step)."""
    policy = DecodeInterleavePolicy(2)
    seq = [_decide(policy, has_prefill=True, has_decode=False) for _ in range(10)]
    assert seq == ["prefill"] * 10


def test_prefill_keeps_priority_below_the_threshold():
    """The first N-1 slots of a burst must still go to prefill."""
    policy = DecodeInterleavePolicy(8)
    seq = [_decide(policy, True, True) for _ in range(7)]
    assert seq.count("decode") == 0


def test_pure_decode_phase_unaffected():
    """With nothing to prefill, every slot is decode (no policy interference)."""
    policy = DecodeInterleavePolicy(8)
    seq = [_decide(policy, has_prefill=False, has_decode=True) for _ in range(50)]
    assert seq == ["decode"] * 50


def test_scheduler_without_policy_keeps_prefill_first():
    """A Scheduler built without __init__ (as the accounting tests do) must not break.

    _schedule_next_batch reads the policy through getattr because the accounting tests
    build a Scheduler this way; a stripped object keeps the historical prefill-first
    order instead of raising.
    """
    from types import SimpleNamespace

    from freetoken.scheduler.scheduler import Scheduler

    prefill_batch = SimpleNamespace(is_prefill=True, prompt_admissions=[])
    decode_batch = SimpleNamespace(is_prefill=False, prompt_admissions=[])
    s = Scheduler.__new__(Scheduler)
    s.prefill_budget = 384
    s.prefill_manager = SimpleNamespace(schedule_next_batch=lambda budget: prefill_batch)
    s.decode_manager = SimpleNamespace(schedule_next_batch=lambda: decode_batch)
    s._prepare_batch = lambda value: value
    s.send_result = lambda messages: None
    # no _interleave attribute at all
    assert not hasattr(s, "_interleave")
    assert Scheduler._schedule_next_batch(s) is prefill_batch, "must stay prefill-first"


def test_scheduler_with_policy_and_no_decode_stays_prefill():
    """When no decode is runnable the policy must not steal a slot."""
    from types import SimpleNamespace

    from freetoken.scheduler.scheduler import Scheduler
    from freetoken.scheduler.interleave import DecodeInterleavePolicy

    prefill_batch = SimpleNamespace(is_prefill=True, prompt_admissions=[])
    s = Scheduler.__new__(Scheduler)
    s.prefill_budget = 384
    s.prefill_manager = SimpleNamespace(schedule_next_batch=lambda budget: prefill_batch)
    s.decode_manager = SimpleNamespace(schedule_next_batch=lambda: None)
    s._prepare_batch = lambda value: value
    s.send_result = lambda messages: None
    s._interleave = DecodeInterleavePolicy(1)      # fires every step
    for _ in range(5):
        assert Scheduler._schedule_next_batch(s) is prefill_batch


def test_scheduler_with_policy_takes_decode_when_due():
    """Once the streak reaches the threshold, a runnable decode gets the slot."""
    from types import SimpleNamespace

    from freetoken.scheduler.scheduler import Scheduler
    from freetoken.scheduler.interleave import DecodeInterleavePolicy

    prefill_batch = SimpleNamespace(is_prefill=True, prompt_admissions=[])
    decode_batch = SimpleNamespace(is_prefill=False, prompt_admissions=[])
    s = Scheduler.__new__(Scheduler)
    s.prefill_budget = 384
    s.prefill_manager = SimpleNamespace(schedule_next_batch=lambda budget: prefill_batch)
    s.decode_manager = SimpleNamespace(schedule_next_batch=lambda: decode_batch)
    s._prepare_batch = lambda value: value
    s.send_result = lambda messages: None
    s._interleave = DecodeInterleavePolicy(3)
    got = [Scheduler._schedule_next_batch(s) for _ in range(4)]
    assert got[0] is prefill_batch and got[1] is prefill_batch and got[2] is prefill_batch
    assert got[3] is decode_batch, "4th slot must be the forced decode"
