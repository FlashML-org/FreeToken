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


# --------------------------------------------------------------------------- #
# Truth table against the REAL scheduler (codex review: _decide is a model of
# the decision, not the decision -- it can drift from the code it describes).
# --------------------------------------------------------------------------- #
def _real_scheduler(every=None, prefill=True, decode=True):
    """Build a Scheduler whose two managers are stubs with call counters."""
    from types import SimpleNamespace

    from freetoken.scheduler.scheduler import Scheduler
    from freetoken.scheduler.interleave import DecodeInterleavePolicy

    calls = {"prefill": 0, "decode": 0}
    pf = SimpleNamespace(is_prefill=True, prompt_admissions=[])
    dc = SimpleNamespace(is_prefill=False, prompt_admissions=[])

    def _pf(budget):
        calls["prefill"] += 1
        return pf if prefill else None

    def _dc():
        calls["decode"] += 1
        return dc if decode else None

    s = Scheduler.__new__(Scheduler)
    s.prefill_budget = 384
    s.prefill_manager = SimpleNamespace(schedule_next_batch=_pf)
    s.decode_manager = SimpleNamespace(schedule_next_batch=_dc)
    s._prepare_batch = lambda value: value
    s.send_result = lambda messages: None
    if every is not None:
        s._interleave = DecodeInterleavePolicy(every)
    return s, calls, pf, dc


@pytest.mark.parametrize("every,expected", [
    # (policy N, expected sequence of slot winners over 9 slots)
    (None, ["p"] * 9),                       # disabled -> always prefill
    # streak starts at 0, so slot 0 is prefill; then strict alternation
    (1, ["p", "d", "p", "d", "p", "d", "p", "d", "p"]),
    (3, ["p", "p", "p", "d", "p", "p", "p", "d", "p"]),
    (8, ["p"] * 8 + ["d"]),
])
def test_truth_table_against_real_scheduler(every, expected):
    """The real _schedule_next_batch must produce exactly this slot sequence."""
    from freetoken.scheduler.scheduler import Scheduler

    s, calls, pf, dc = _real_scheduler(every=every)
    got = []
    for _ in range(9):
        b = Scheduler._schedule_next_batch(s)
        got.append("p" if b is pf else ("d" if b is dc else "?"))
    assert got == expected, f"every={every}: got {got}, want {expected}"


def test_every_one_alternates_while_both_stay_runnable():
    """every=1 is the degenerate case: one decode between every prefill."""
    from freetoken.scheduler.scheduler import Scheduler

    s, calls, pf, dc = _real_scheduler(every=1)
    seq = []
    for _ in range(6):
        b = Scheduler._schedule_next_batch(s)
        seq.append("p" if b is pf else "d")
    # slot 0 prefill (streak starts at 0), then alternation
    assert seq == ["p", "d", "p", "d", "p", "d"]


def test_latch_releases_when_decode_becomes_available():
    """Due decode unavailable -> keep prefilling -> decode becomes available -> it fires.

    Exercises the subtle path: the policy must not lose its 'decode is due' state
    while waiting, and must not fire twice once it does.
    """
    from freetoken.scheduler.scheduler import Scheduler

    s, calls, pf, dc = _real_scheduler(every=2)
    # phase 1: no decode runnable at all
    s.decode_manager.schedule_next_batch = lambda: None
    for _ in range(5):
        assert Scheduler._schedule_next_batch(s) is pf
    # phase 2: decode becomes available -> next slot must be decode
    s.decode_manager.schedule_next_batch = lambda: dc
    assert Scheduler._schedule_next_batch(s) is dc, "latch did not release"


def test_no_work_returns_none():
    from freetoken.scheduler.scheduler import Scheduler

    s, _, _, _ = _real_scheduler(every=8, prefill=False, decode=False)
    assert Scheduler._schedule_next_batch(s) is None


def test_pure_decode_fallback_when_nothing_to_prefill():
    """With no prefill available every slot is decode (no policy interference)."""
    from freetoken.scheduler.scheduler import Scheduler

    s, calls, pf, dc = _real_scheduler(every=2, prefill=False, decode=True)
    got = [Scheduler._schedule_next_batch(s) for _ in range(5)]
    assert all(b is dc for b in got)


def test_disabled_policy_object_not_just_missing_attribute():
    """codex: cover a policy that EXISTS but is disabled, not only an absent one."""
    from freetoken.scheduler.scheduler import Scheduler

    s, calls, pf, dc = _real_scheduler(every=None)
    s._interleave = __import__(
        "freetoken.scheduler.interleave", fromlist=["DecodeInterleavePolicy"]
    ).DecodeInterleavePolicy(0)          # present but disabled
    got = [Scheduler._schedule_next_batch(s) for _ in range(6)]
    assert all(b is pf for b in got), "a disabled policy must never take a slot"


def test_decode_manager_not_probed_when_policy_disabled():
    """When disabled, the decode manager must not be consulted at all (no extra call)."""
    from freetoken.scheduler.scheduler import Scheduler

    s, calls, pf, dc = _real_scheduler(every=None)
    Scheduler._schedule_next_batch(s)
    assert calls["decode"] == 0, "disabled policy should not probe decode"


def test_config_to_scheduler_wiring():
    """codex: cover CLI/config -> Scheduler.__init__ wiring."""
    import inspect

    from freetoken.engine.config import EngineConfig
    from freetoken.scheduler import scheduler as sched_mod

    assert "decode_interleave_every" in {f.name for f in __import__("dataclasses").fields(EngineConfig)}
    src = inspect.getsource(sched_mod.Scheduler.__init__)
    assert "_interleave" in src and "decode_interleave_every" in src, (
        "Scheduler.__init__ must build the policy from the config field"
    )
    # and the CLI exposes it
    from freetoken.server.args import ServerArgs
    assert hasattr(ServerArgs, "decode_interleave_every")
