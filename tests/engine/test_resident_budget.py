import pytest

from freetoken.engine.resident_budget import phase_memory, required_phase_bytes


def test_phase_uses_driver_high_water_without_double_counting_non_torch_bytes():
    phase = phase_memory(
        "capture",
        start_free_bytes=900,
        end_free_bytes=700,
        allocator_peak_allocated_bytes=100,
        allocator_peak_reserved_bytes=150,
        minimum_driver_free_bytes=650,
        total_driver_bytes=1000,
    )
    assert phase.driver_used_high_water_bytes == 350
    assert phase.non_torch_bytes == 200
    assert phase.required_bytes == 350
    assert required_phase_bytes([phase], safety_bytes=50) == 400


def test_phase_rejects_impossible_driver_counter():
    with pytest.raises(ValueError, match="within total"):
        phase_memory(
            "load", start_free_bytes=1, end_free_bytes=1,
            allocator_peak_allocated_bytes=0, allocator_peak_reserved_bytes=0,
            minimum_driver_free_bytes=2, total_driver_bytes=1,
        )
