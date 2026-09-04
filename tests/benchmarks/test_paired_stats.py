from __future__ import annotations

import sys
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
if str(BENCHMARKS) in sys.path:
    sys.path.remove(str(BENCHMARKS))
sys.path.insert(0, str(BENCHMARKS))

from lib.paired_stats import bootstrap_median_interval, paired_summary


def test_paired_summary_reports_positive_candidate_recovery():
    result = paired_summary([10, 12, 11], [8, 9, 10])
    assert result["pairs"] == 3
    assert result["median_recovery_us"] == 2
    assert result["recovery_p02_5_us"] <= 2 <= result["recovery_p97_5_us"]
    assert result["candidate_speedup_pct"] > 0


def test_paired_summary_rejects_unpaired_inputs():
    with pytest.raises(ValueError, match="equal length"):
        paired_summary([1], [1, 2])


def test_empty_bootstrap_is_explicit():
    assert bootstrap_median_interval([]) == (None, None)
