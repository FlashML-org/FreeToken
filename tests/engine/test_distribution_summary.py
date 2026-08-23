from types import SimpleNamespace

import torch

from freetoken.engine.distribution import format_distribution_summary, host_expert_bytes


def test_host_expert_bytes_sums_every_rank_local_bank():
    cache = SimpleNamespace(
        bank_sources={
            "gate": [torch.empty(2, 8, dtype=torch.uint8), torch.empty(2, 8, dtype=torch.uint8)],
            "down": [torch.empty(2, 4, dtype=torch.bfloat16)],
        }
    )
    assert host_expert_bytes(cache) == 48
    assert host_expert_bytes(None) == 0


def test_summary_groups_ranks_by_numa_and_reports_cpu_gpu_split():
    gib = 1 << 30
    rows = [
        {
            "rank": 0,
            "gpu": 0,
            "gpu_name": "GPU",
            "gpu_used": 12 * gib,
            "gpu_total": 16 * gib,
            "node": 0,
            "node_memory_total": 128 * gib,
            "node_physical_cores": 20,
            "cpu_workers": 9,
            "cpu_coordinator": 1,
            "host_experts": 32 * gib,
            "system_memory_total": 256 * gib,
        },
        {
            "rank": 1,
            "gpu": 1,
            "gpu_name": "GPU",
            "gpu_used": 12 * gib,
            "gpu_total": 16 * gib,
            "node": 0,
            "node_memory_total": 128 * gib,
            "node_physical_cores": 20,
            "cpu_workers": 9,
            "cpu_coordinator": 1,
            "host_experts": 32 * gib,
            "system_memory_total": 256 * gib,
        },
    ]

    lines = format_distribution_summary(rows)

    assert lines[0] == "Model distribution (final, NUMA-aware):"
    assert "host experts 64.00 GiB/128.00 GiB (50.0% RAM)" in lines[1]
    assert "CPU cores 20/20 assigned" in lines[1]
    assert "GPU 0 12.00 GiB/16.00 GiB (75.0% VRAM)" in lines[2]
    assert "CPU-MoE 9 workers + 1 coordinator" in lines[2]
    assert "counted placement CPU 72.7% / GPU 27.3%" in lines[-1]
