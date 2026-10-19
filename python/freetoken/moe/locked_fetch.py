"""Locked-layer fetch for hybrid decode.

A locked layer's host banks are OS-locked but not registered with CUDA (the pin quota ran out), so the GPU cannot
gather their rows. Each decode step the GPU copies the fetch list chosen by the hybrid ensure (count + layer-local
rows) to pinned meta and rings a doorbell; CPU threads copy each fetched row of every bank into pinned staging
(``_cpu_moe.LockedFetch``) and the GPU gathers staging row i into ``evict_slots[i]``. The misses it does not fetch
go to the CPU executor as on any hybrid layer.
"""

from __future__ import annotations

import os

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)


def staging_bytes(sources: dict[str, list[torch.Tensor]], max_fetch: int, layers=None) -> dict[str, int]:
    """Per bank role: ``max_fetch`` rows of the widest row over ``layers`` (every layer when None)."""
    out = {}
    for role, per_layer in sources.items():
        rows = [t[0].numel() * t.element_size() for l, t in enumerate(per_layer) if t.numel() and (layers is None or l in layers)]
        out[role] = max([16] + [r * max_fetch for r in rows])
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


def _copy_cpus(executor, count: int) -> list[int]:
    """CPUs for the copy threads: SMT siblings of the executor's workers, else physical cores the executor leaves free."""
    from freetoken.moe.cpu_executor import physical_core_cpus

    busy = set(executor.core_ids) | {getattr(executor, "_coord_core", -1)}
    cpus = _smt_siblings(executor.core_ids) or [c for c in physical_core_cpus() if c not in busy]
    return cpus[:count]


class LockedLayerFetch:
    """Staging, meta and copy threads for the locked layers of one ``OffloadMoeCache``."""

    def __init__(self, cache, layers: list[int], fraction: float, max_fetch: int, executor, staging: dict[str, torch.Tensor] | None = None) -> None:
        from freetoken.kernel import _cpu_moe
        from freetoken.kernel.pinned import alloc_pinned_tensor, device_ptr
        from freetoken.moe.legacy_format import canonical_role

        if not cache._copy_fused_ok:
            raise ValueError("locked layers fetch through the fused multi-bank copy, which this cache could not plan (bank alignment or FREETOKEN_FUSED_COPY=0)")
        self.fraction = fraction
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
        self.count = alloc_pinned_tensor(1, dtype=torch.int64)
        self.rows = alloc_pinned_tensor(max_fetch, dtype=torch.int32)
        # the gather kernel takes src indices shaped like evict_slots; it reads only the first num_indices <= max_fetch
        self.row_ids = torch.arange(cache.evict_slots.numel(), dtype=torch.int32, device=cache.device)
        cpus = _copy_cpus(executor, int(os.environ.get("FREETOKEN_LOCKED_FETCH_THREADS", "8")))
        self._ext = _cpu_moe.LockedFetch(src, row_bytes, [t.data_ptr() for t in self.staging], self.count.data_ptr(), self.rows.data_ptr(), max_fetch, cpus)
        logger.info_rank0(
            f"locked layers {layers}: GPU slots + fetch {fraction:g} of each step's misses (at most {max_fetch}) through "
            f"{sum(t.numel() for t in self.staging) / 2**20:.1f} MiB of pinned staging, {max(1, len(cpus))} copy threads on cpus {cpus or 'any'}"
        )

    def copy(self, cache, layer_id: int) -> None:
        """Stage and gather the rows the last hybrid ensure scheduled for ``layer_id``; graph capturable."""
        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

        assert self.row_ids.numel() == cache.evict_slots.numel(), "the slot plan was rebuilt after the locked fetch"
        stream = torch.cuda.current_stream().cuda_stream
        self.count.copy_(cache.num_indices, non_blocking=True)
        self.rows.copy_(cache.src_indices[: self.max_fetch], non_blocking=True)
        self._ext.gpu_submit(stream, layer_id)
        self._ext.gpu_wait(stream)
        fast_index_copy_multi_jit(
            cache._copy_dst_ptrs_pool[cache.pool_of_layer[layer_id]],
            self.staging_ptrs,
            cache._copy_feat_bytes[cache.layer_layout[layer_id]],
            cache.evict_slots,
            self.row_ids,
            cache.num_indices,
            blocks_per_bank=cache._copy_blocks_per_bank,
        )
