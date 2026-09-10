"""`ft bench bw` must not present a coin flip as a fact.

The hybrid-vs-offload verdict is a bandwidth ratio against a threshold. Some formats
measure the same to within a percent; others swing far enough between runs to land on
both sides of it -- mxfp4 was observed anywhere from 19 to 71 GB/s on one machine,
flipping its recommendation run to run while each individual run reported a single
confident number. `verdict` bounds the ratio by the extremes actually observed and
withholds the call when that interval straddles the threshold.
"""

from __future__ import annotations

from freetoken.moe.benchbw import verdict


def test_clearly_above_threshold_is_hybrid():
    pick, confident, (lo, hi) = verdict([96.0, 97.0, 95.5], [25.0, 25.1, 25.0], 2.0)
    assert (pick, confident) == ("hybrid", True)
    assert lo > 2.0 and hi > lo


def test_clearly_below_threshold_is_offload():
    pick, confident, _ = verdict([40.0, 41.0, 39.5], [25.0, 25.1, 25.0], 2.0)
    assert (pick, confident) == ("offload", True)


def test_straddling_the_threshold_is_withheld():
    """The real mxfp4 case: the median says hybrid, the spread says nobody knows."""
    pick, confident, (lo, hi) = verdict([42.2, 50.2, 60.0], [25.2, 25.0, 25.1], 2.0)
    assert lo < 2.0 < hi, (lo, hi)
    assert confident is False
    assert pick == "offload", "an undecided measurement must fall back to the safe backend"


def test_a_single_run_still_decides():
    """One rep has no spread to speak of, so it behaves exactly as before."""
    assert verdict([96.0], [25.0], 2.0)[:2] == ("hybrid", True)
    assert verdict([40.0], [25.0], 2.0)[:2] == ("offload", True)


def test_exactly_at_the_threshold_is_not_hybrid():
    """`recommend` is a strict >, so the boundary resolves to offload, not a coin flip."""
    pick, confident, _ = verdict([50.0], [25.0], 2.0)
    assert (pick, confident) == ("offload", True)
