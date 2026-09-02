"""Pure marker pairing and stage-summary tests."""

from freetoken.utils.step_profiler import summarize_markers


def test_summarize_markers_keeps_outer_step_separate_from_overlapping_phases():
    markers = [
        {"step": 1, "phase": "scheduler", "event": "begin", "monotonic_ns": 0},
        {"step": 1, "phase": "attention_metadata", "event": "begin", "monotonic_ns": 10},
        {"step": 1, "phase": "attention_metadata", "event": "end", "monotonic_ns": 40},
        {"step": 1, "phase": "scheduler", "event": "end", "monotonic_ns": 100},
    ]
    summary = summarize_markers(markers)
    assert summary["complete"] is True
    assert summary["critical_step"]["median_ns"] == 100
    assert summary["phases"]["attention_metadata"]["median_ns"] == 30
    assert summary["note"].startswith("phase totals overlap")


def test_summarize_markers_rejects_unmatched_ranges():
    summary = summarize_markers([
        {"step": 1, "phase": "scheduler", "event": "end", "monotonic_ns": 1},
    ])
    assert summary["complete"] is False
    assert "unmatched end" in summary["errors"][0]
