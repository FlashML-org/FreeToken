"""Locked-layer fetch for hybrid decode.

A locked layer's host banks are OS-locked but not registered with CUDA (the pin quota ran out), so the GPU cannot
gather their rows. Each decode step a publish kernel hands the fetch list chosen by the hybrid ensure to CPU threads
(``_cpu_moe.LockedFetch``), which copy each fetched row of every bank into pinned staging, and fast_index_copy_multi
copies staging row i into ``evict_slots[i]``. The misses it does not fetch go to the CPU executor as on any hybrid layer.
"""

from __future__ import annotations

import collections
import os

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)


def locked_fetch_wanted(config) -> bool:
    """Whether locked layers fetch through pinned staging: hybrid, --moe-locked-fetch not 0, CUDA (the copy kernels are untested on ROCm)."""
    return config.moe_strategy == "hybrid" and config.moe_locked_fetch != 0 and torch.version.hip is None


def check_locked_fetch_config(config) -> None:
    """Reject --moe-locked-max-fetch past the experts of a layer before the banks are read (only with --moe-cpu-layers)."""
    num_experts = config.model_config.num_experts
    if config.moe_cpu_layers and locked_fetch_wanted(config) and not 1 <= config.moe_locked_max_fetch <= num_experts:
        raise ValueError(f"--moe-locked-max-fetch {config.moe_locked_max_fetch} must be in [1, {num_experts}] (the experts of a layer)")


def staging_reserve(config, bank_bytes) -> int:
    """Pin budget the staging takes when --moe-cpu-layers is settled before the load: ``max_fetch`` rows of twice the mean expert.

    ``bank_bytes()`` estimates the banks; it runs only when the staging is wanted."""
    if not (config.moe_cpu_layers and locked_fetch_wanted(config)) or not (total := bank_bytes()):
        return 0
    mc = config.model_config
    return 2 * (total // (mc.num_moe_layers * mc.num_experts)) * config.moe_locked_max_fetch


def settle_banks(config, banks, *, probe: bool, has_locked: bool):
    """Pin the staging when locked layers may fetch, then with ``probe`` settle the DEFERRED banks (expert_banks.probe_banks).

    Pinning the staging first keeps the banks from taking its share of the pin budget. Returns the banks, the locked layers
    the probe found (None without ``probe``) and the staging (None when unused); a staging that cannot be pinned raises."""
    from freetoken.moe.expert_banks import probe_banks

    staging = None
    if locked_fetch_wanted(config) and (probe or has_locked):
        staging = pin_staging(banks.sources, config.moe_locked_max_fetch)
    locked = None
    if probe:
        banks, locked = probe_banks(banks, config.model_config.num_moe_layers)
        if staging is not None and not locked:
            refund_staging(staging)  # every bank got pinned
            staging = None
    return banks, locked, staging


def locked_pool_tags(layer_tags: list | None, locked, num_layers: int) -> list:
    """Bank layer tags with the locked layers marked, so they group into slot pools of their own."""
    return [(t, "locked") if i in locked else t for i, t in enumerate(layer_tags or [None] * num_layers)]


def attach_locked_fetch(cache, config, staging: dict[str, torch.Tensor], layer_rows) -> None:
    """Set ``cache.locked_fetch`` for ``cache.locked_hybrid``; needs the cache's CPU executor. ``layer_rows`` as in fetch_plan."""
    locked = sorted(cache.locked_hybrid)
    plan = fetch_plan(config.moe_locked_fetch, locked, layer_rows)
    cache.locked_fetch = LockedLayerFetch(cache, locked, plan, config.moe_locked_max_fetch, cache.cpu_executor, staging=staging)


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
    from freetoken.moe.host_banks import charge_pin, refund_pin

    charge_pin(sum(sizes.values()))
    try:
        return {role: alloc_pinned_tensor(n, dtype=torch.uint8) for role, n in sizes.items()}
    except BaseException:
        refund_pin(sum(sizes.values()))
        raise


def pin_staging(sources: dict[str, list[torch.Tensor]], max_fetch: int) -> dict[str, torch.Tensor]:
    """The staging for ``sources``; raises when it cannot be pinned."""
    from freetoken.moe.host_banks import PinFailed

    sizes = staging_bytes(sources, max_fetch)
    try:
        return alloc_staging(sizes)
    except (PinFailed, RuntimeError) as exc:
        raise RuntimeError(
            f"could not pin the {sum(sizes.values()) / 2**20:.1f} MiB locked-layer staging ({exc}); free pinned host memory, "
            "lower --moe-locked-max-fetch, or pass --moe-locked-fetch 0 to keep locked layers CPU-only"
        ) from exc


def refund_staging(staging: dict[str, torch.Tensor]) -> None:
    """Give the staging's bytes back to the simulated pin cap; the memory itself goes with its last reference."""
    from freetoken.moe.host_banks import refund_pin

    refund_pin(sum(t.numel() for t in staging.values()))


def fetch_plan(share: float, layers: list[int], layer_rows) -> dict[int, tuple[int, float]]:
    """(max_fetch, fraction) per locked layer for the hybrid ensure.

    ``share`` > 0 fetches that share of every step's misses; otherwise each layer takes p / (2p + c) from its `ft bench bw`
    row, or a fixed cap of 1 without one. ``layer_rows()`` returns (row key, label) per MoE layer and the profile's rows, or None.
    """
    if share > 0:
        logger.info_rank0(f"--moe-locked-fetch {share:g}: locked layers fetch this share of each step's misses")
        return {layer: (0, share) for layer in layers}
    from freetoken.moe.bench_profile import locked_fetch_fraction

    rows = layer_rows()
    keys, labels, rows = rows if rows is not None else ([None] * (max(layers) + 1), [""] * (max(layers) + 1), {})
    fractions = {layer: locked_fetch_fraction(rows.get(keys[layer])) for layer in layers}
    groups = collections.Counter((labels[layer], f) for layer, f in fractions.items())
    if fetched := [f"{f:.1%} on {n} {label} layers" for (label, f), n in groups.items() if f is not None]:
        logger.info_rank0(f"--moe-locked-fetch auto: locked layers fetch this share of each step's misses (benched p / (2p + c)): {'; '.join(fetched)}")
    if capped := [layer for layer, f in fractions.items() if f is None]:
        logger.warning_rank0(
            f"--moe-locked-fetch auto: no `ft bench bw` row for locked layers {capped}; they fetch at most 1 expert per step. "
            "`ft bench bw --model <checkpoint>` benches each layer kind"
        )
    return {layer: (0, f) if f is not None else (1, 0.0) for layer, f in fractions.items()}


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
    cpus = [c for c in dict.fromkeys(_smt_siblings(executor.core_ids)) if c not in busy]
    cpus = cpus or [c for c in physical_core_cpus() if c not in busy]
    return cpus[:count]


class LockedLayerFetch:
    """Staging and copy threads for the locked layers of one ``OffloadMoeCache``."""

    def __init__(self, cache, layers: list[int], plan: dict[int, tuple[int, float]], max_fetch: int, executor, staging: dict[str, torch.Tensor] | None = None) -> None:
        from freetoken.kernel import _cpu_moe
        from freetoken.kernel.pinned import device_ptr
        from freetoken.moe.legacy_format import canonical_role

        if not cache._copy_fused_ok:
            raise ValueError("locked layers fetch through the fused multi-bank copy, which this cache could not plan (bank alignment or FREETOKEN_FUSED_COPY=0)")
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
        self._wait_flag = 0 if memops else self._ext.done_word()
        logger.info_rank0(
            f"locked layers {layers}: GPU slots + fetch up to {max_fetch} misses per step through "
            f"{sum(t.numel() for t in self.staging) / 2**20:.1f} MiB of pinned staging, {max(1, len(cpus))} copy thread(s) on cpus {cpus or 'any'}"
            + ("" if memops else ", the GPU spin-waits on them (no stream memops)")
        )

    def copy(self, cache, layer_id: int) -> None:
        """Stage and gather the rows the last hybrid ensure scheduled for ``layer_id``; graph capturable."""
        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

        # the gather reads num_indices rows of the staging, which has max_fetch
        assert cache._pending_fetch_cap == self.max_fetch, (cache._pending_fetch_cap, self.max_fetch)
        self._ext.publish(torch.cuda.current_stream().cuda_stream, layer_id, cache.num_indices.data_ptr(), cache.src_indices.data_ptr())
        fast_index_copy_multi_jit(
            cache._copy_dst_ptrs_pool[cache.pool_of_layer[layer_id]],
            self.staging_ptrs,
            cache._copy_feat_bytes[cache.layer_layout[layer_id]],
            cache.evict_slots,
            None,
            cache.num_indices,
            blocks_per_bank=cache._copy_blocks_per_bank,
            wait_flag=self._wait_flag,
        )
