"""Final CPU/GPU placement summary for a loaded engine.

The startup banner describes hardware. This module describes where the loaded model
actually landed after host expert banks, GPU caches, and CUDA graphs have reached their
serving sizes. The data is gathered from every TP rank immediately before readiness.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

import torch

from freetoken.utils import numa


def _gib(n_bytes: int) -> str:
    return f"{n_bytes / (1 << 30):.2f} GiB"


def _percent(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole > 0 else 0.0


def host_expert_bytes(cache: Any) -> int:
    """Pinned host expert bytes owned by one TP rank."""
    if cache is None:
        return 0
    return sum(
        tensor.numel() * tensor.element_size()
        for layers in getattr(cache, "bank_sources", {}).values()
        for tensor in layers
    )


def _thread_siblings(cpu: int) -> str | None:
    """Return the kernel topology key without depending on optional NUMA helpers."""
    try:
        with open(
            f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        ) as f:
            return f.read().strip()
    except OSError:
        return None


def _physical_core_count(cpus: list[int]) -> int:
    siblings = {_thread_siblings(cpu) or f"cpu:{cpu}" for cpu in cpus}
    return len(siblings)


def _node_memory_total(node: int | None) -> int:
    if node is None:
        return 0
    try:
        with open(f"/sys/devices/system/node/node{node}/meminfo") as f:
            for line in f:
                if " MemTotal:" in line:
                    return int(line.split()[-2]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _system_memory_total() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError):
        return 0


def local_distribution_snapshot(engine: Any) -> dict[str, Any]:
    """Collect one rank's final serving placement without synchronizing a hot path."""
    rank = int(engine.config.tp_info.rank)
    free, total = torch.cuda.mem_get_info(engine.device)
    placed = numa.placement()
    node = placed[0] if placed is not None else numa.device_numa_node(rank)
    node_cpus = (
        list(placed[1]) if placed is not None else sorted(numa.allowed_cpus())
    )
    executor = getattr(engine, "cpu_moe_executor", None)
    workers = int(getattr(executor, "num_threads", 0) or 0)
    coordinator = int(getattr(executor, "_coord_core", -1) >= 0)
    return {
        "rank": rank,
        "gpu": int(engine.device.index if engine.device.index is not None else rank),
        "gpu_name": torch.cuda.get_device_name(engine.device),
        "gpu_used": int(total - free),
        "gpu_total": int(total),
        "node": node,
        "node_memory_total": _node_memory_total(node),
        "node_physical_cores": _physical_core_count(node_cpus),
        "cpu_workers": workers,
        "cpu_coordinator": coordinator,
        "host_experts": host_expert_bytes(getattr(engine, "moe_offload_cache", None)),
        "system_memory_total": _system_memory_total(),
    }


def format_distribution_summary(rows: list[dict[str, Any]]) -> list[str]:
    """Format gathered TP snapshots as a compact NUMA-aware workstation map."""
    if not rows:
        return []
    rows = sorted(rows, key=lambda row: int(row["rank"]))
    by_node: dict[int | None, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_node[row.get("node")].append(row)

    lines = ["Model distribution (final, NUMA-aware):"]
    for node, node_rows in sorted(
        by_node.items(), key=lambda item: (-1 if item[0] is None else int(item[0]))
    ):
        host = sum(int(row["host_experts"]) for row in node_rows)
        memory_total = max(int(row["node_memory_total"]) for row in node_rows)
        cores = max(int(row["node_physical_cores"]) for row in node_rows)
        assigned = sum(
            int(row["cpu_workers"]) + int(row["cpu_coordinator"])
            for row in node_rows
        )
        label = "NUMA unknown" if node is None else f"NUMA node {node}"
        lines.append(
            f"  {label}: host experts {_gib(host)}/{_gib(memory_total)} "
            f"({_percent(host, memory_total):.1f}% RAM), CPU cores {assigned}/{cores} assigned"
        )
        for row in node_rows:
            coord = " + 1 coordinator" if row["cpu_coordinator"] else ""
            lines.append(
                f"    rank {row['rank']} -> GPU {row['gpu']} "
                f"{_gib(int(row['gpu_used']))}/{_gib(int(row['gpu_total']))} "
                f"({_percent(int(row['gpu_used']), int(row['gpu_total'])):.1f}% VRAM), "
                f"CPU-MoE {row['cpu_workers']} workers{coord}, "
                f"host experts {_gib(int(row['host_experts']))}"
            )

    host = sum(int(row["host_experts"]) for row in rows)
    gpu_used = sum(int(row["gpu_used"]) for row in rows)
    gpu_total = sum(int(row["gpu_total"]) for row in rows)
    system_memory = max(int(row["system_memory_total"]) for row in rows)
    counted = host + gpu_used
    lines.append(
        f"  total: CPU host experts {_gib(host)}/{_gib(system_memory)} "
        f"({_percent(host, system_memory):.1f}% RAM), GPU {_gib(gpu_used)}/{_gib(gpu_total)} "
        f"({_percent(gpu_used, gpu_total):.1f}% VRAM); counted placement "
        f"CPU {_percent(host, counted):.1f}% / GPU {_percent(gpu_used, counted):.1f}%"
    )
    return lines


__all__ = [
    "format_distribution_summary",
    "host_expert_bytes",
    "local_distribution_snapshot",
]
