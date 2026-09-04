from freetoken.engine.resident_budget import ResidentBudget


def test_observed_phase_can_raise_static_resident_requirement():
    budget = ResidentBudget(
        free_bytes=100, total_vram_bytes=100,
        packed_model_bytes=10, kv_bytes=10, gdn_state_bytes=0,
        page_table_bytes=0, graph_reserve_bytes=0, peak_load_scratch_bytes=0,
        safety_bytes=0,
        phases=(),
    )
    assert budget.required_bytes == 20
