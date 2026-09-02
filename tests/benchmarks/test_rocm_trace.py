import importlib.util
import sys
from pathlib import Path

_TRACE_PATH = Path(__file__).parents[2] / "benchmarks" / "lib" / "rocm_trace.py"
_SPEC = importlib.util.spec_from_file_location("_freetoken_rocm_trace", _TRACE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

from _freetoken_rocm_trace import (
    calibrate_clocks,
    correlate_events,
    intersection,
    token_ledger,
    union,
    warm_offload_summary,
)


def test_clock_calibration_rejects_large_residual():
    good = calibrate_clocks([(0, 100), (1000, 1100), (2000, 2100)])
    assert good.accepted
    bad = calibrate_clocks([(0, 0), (1000, 100000), (2000, 0)])
    assert not bad.accepted


def test_union_and_intersection_do_not_double_count_overlap():
    assert union([(0, 10), (5, 20), (30, 40)]) == [(0, 20), (30, 40)]
    assert intersection([(0, 10), (15, 30)], [(5, 20)]) == [(5, 10), (15, 20)]


def test_token_ledger_clips_async_events_and_separates_host_wait():
    rows = token_ledger(
        [{"token": 3, "start_ns": 100, "end_ns": 300}],
        [
            {"kind": "kernel", "start_ns": 50, "end_ns": 180},
            {"kind": "copy", "start_ns": 160, "end_ns": 260},
        ],
        [
            {"category": "host_active", "start_ns": 120, "end_ns": 220},
            {"category": "host_wait", "start_ns": 220, "end_ns": 280},
        ],
    )
    assert rows[0]["gpu_ns"] == 160
    assert rows[0]["host_active_ns"] == 100
    assert rows[0]["host_wait_ns"] == 60
    assert rows[0]["unattributed_ns"] == 20
    assert rows[0]["host_active_only_ns"] == 0


def test_correlate_events_preserves_unmatched_identity():
    rows = correlate_events(
        [{"correlation_id": 4, "stream_id": "s0"}],
        [{"correlation_id": 4, "start_ns": 1, "end_ns": 2}],
        [{"correlation_id": 9, "start_ns": 3, "end_ns": 4}],
    )
    assert rows[0]["correlation_matched"] is True
    assert rows[1]["correlation_matched"] is False


def test_warm_offload_summary_keeps_missing_bytes_explicit():
    summary = warm_offload_summary([
        {"kind": "kernel", "name": "ensure_experts", "start_ns": 0, "end_ns": 5},
        {"kind": "kernel", "name": "fast_index_copy_multi_jit", "start_ns": 5, "end_ns": 9},
    ])
    assert summary["status"] == "measured"
    assert summary["all_hit"] is False
    assert summary["copy_missing_bytes"] is None


def test_warm_offload_summary_reports_all_hit_zero_copy_count():
    summary = warm_offload_summary([
        {"kind": "kernel", "name": "ensure_experts", "start_ns": 0, "end_ns": 5},
    ])
    assert summary["all_hit"] is True
    assert summary["copy_missing_count"] == 0
