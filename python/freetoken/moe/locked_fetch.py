"""Locked-layer fetch for hybrid decode.

A locked layer's host banks are OS-locked but not registered with CUDA (the pin quota ran out), so the GPU cannot
gather their rows. Each decode step a publish kernel hands the fetch list chosen by the hybrid ensure to CPU threads,
which copy each fetched row of every bank into pinned staging, and a gather kernel copies staging row i into
``evict_slots[i]`` (``_cpu_moe.LockedFetch``). The misses it does not fetch go to the CPU executor as on any hybrid layer.
"""

from __future__ import annotations

import os

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)


def staging_bytes(sources: dict[str, list[torch.Tensor]], max_fetch: int) -> dict[str, int]:
    """Per canonical bank role: ``max_fetch`` rows of the bank's widest row over every layer."""
    from freetoken.moe.legacy_format import canonical_role

    out = {}
    for name, per_layer in sources.items():
        rows = [t[0].numel() * t.element_size() for t in per_layer if t.numel()]
        out[canonical_role(name)] = max([16] + [r * max_fetch for r in rows])
    return out


def alloc_staging(sizes: dict[str, int]) -> dict[str, torch.Tensor]:
    """Pinned staging per bank role."""
    from freetoken.kernel.pinned import alloc_pinned_tensor
    from freetoken.moe.host_banks import charge_pin

    charge_pin(sum(sizes.values()))
    return {role: alloc_pinned_tensor(n, dtype=torch.uint8) for role, n in sizes.items()}


def _smt_siblings(cpus: list[int]) -> list[int]:
    out = []
    for cpu in cpus:
        try:
            with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as f:
                text = f.read().strip()
        except OSError:
            continue
        for part in text.split(","):
            lo, _, hi = part.partition("-")
            out.extend(i for i in range(int(lo), int(hi or lo) + 1) if i != cpu)
    return out


def _copy_threads() -> int:
    """FREETOKEN_LOCKED_FETCH_THREADS, else max(2, min(4, logical CPUs / 4)): a few threads already saturate memcpy."""
    if env := os.environ.get("FREETOKEN_LOCKED_FETCH_THREADS"):
        return max(1, int(env))
    logical = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return max(2, min(4, logical // 4))


def _copy_cpus(executor, count: int) -> list[int]:
    """CPUs for the copy threads: SMT siblings of the executor's workers, else physical cores the executor leaves free."""
    from freetoken.moe.cpu_executor import physical_core_cpus

    busy = set(executor.core_ids) | {getattr(executor, "_coord_core", -1)}
    cpus = _smt_siblings(executor.core_ids) or [c for c in physical_core_cpus() if c not in busy]
    return cpus[:count]


class LockedLayerFetch:
    """Staging and copy threads for the locked layers of one ``OffloadMoeCache``."""

    def __init__(self, cache, layers: list[int], plan: dict[int, tuple[int, float]], max_fetch: int, executor, staging: dict[str, torch.Tensor] | None = None) -> None:
        from freetoken.kernel import _cpu_moe
        from freetoken.kernel.pinned import device_ptr
        from freetoken.moe.legacy_format import canonical_role

        if not cache._copy_fused_ok:
            raise ValueError("locked layers fetch through the fused multi-bank copy, which this cache could not plan (bank alignment or FREETOKEN_FUSED_COPY=0)")
        if not 1 <= max_fetch <= cache.num_experts:
            raise ValueError(f"--moe-locked-max-fetch {max_fetch} must be in [1, {cache.num_experts}] (the experts of a layer)")
        # (max_fetch, fraction) of each locked layer for the hybrid ensure; max_fetch caps any of them
        self.plan = plan
        self.max_fetch = max_fetch
        roles = [canonical_role(name) for name in cache.bank_schema]
        nbanks = len(cache.banks)
        src = [[0] * nbanks for _ in range(cache.num_layers)]
        row_bytes = [[0] * nbanks for _ in range(cache.num_layers)]
        need = [16] * nbanks
        for b, (per_layer, _) in enumerate(cache.banks):
            for layer in layers:
                rb = int(cache._row_bytes[cache.layer_layout[layer]][b])
                if per_layer[layer].numel() == 0 or rb == 0:
                    continue
                src[layer][b] = per_layer[layer].data_ptr()
                row_bytes[layer][b] = rb
                need[b] = max(need[b], rb * max_fetch)
        if staging is None:
            staging = alloc_staging(dict(zip(roles, need)))
        for b, role in enumerate(roles):
            assert staging[role].numel() >= need[b], (role, staging[role].numel(), need[b])
        self.staging = [staging[role] for role in roles]
        self.staging_ptrs = torch.tensor([device_ptr(t) for t in self.staging], dtype=torch.int64, device=cache.device)
        cpus = _copy_cpus(executor, _copy_threads())
        # the executor's doorbell decision (FREETOKEN_CPU_MOE_FLAG_SYNC and the memop probe); without memops the gather spins on the GPU
        memops = bool(getattr(executor, "_flag_sync", False))
        self._ext = _cpu_moe.LockedFetch(src, row_bytes, [t.data_ptr() for t in self.staging], max_fetch, cpus, memops)
        logger.info_rank0(
            f"locked layers {layers}: GPU slots + fetch up to {max_fetch} misses per step through "
            f"{sum(t.numel() for t in self.staging) / 2**20:.1f} MiB of pinned staging, {max(1, len(cpus))} copy thread(s) on cpus {cpus or 'any'}"
            + ("" if memops else ", the GPU spin-waits on them (no stream memops)")
        )

    def copy(self, cache, layer_id: int) -> None:
        """Stage and gather the rows the last hybrid ensure scheduled for ``layer_id``; graph capturable."""
        self._ext.copy(
            torch.cuda.current_stream().cuda_stream,
            layer_id,
            cache.num_indices.data_ptr(),
            cache.src_indices.data_ptr(),
            cache.evict_slots.data_ptr(),
            cache._copy_dst_ptrs_pool[cache.pool_of_layer[layer_id]].data_ptr(),
            self.staging_ptrs.data_ptr(),
            cache._copy_feat_bytes[cache.layer_layout[layer_id]].data_ptr(),
            len(self.staging),
            cache._copy_blocks_per_bank,
        )
