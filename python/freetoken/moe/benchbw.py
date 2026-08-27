"""``ft bench bw``: measure host RAM vs PCIe bandwidth and pick the offload-family
MoE backend for a given model on this machine.

Two layers of measurement:

  * Ceilings (hardware-only, model-agnostic): a STREAM-style CPU DRAM read and a
    linear pinned<->device copy. Quick upper bounds.
  * Real kernels per selected workload + expert dtype: the actual CPU MoE GEMV
    (``CpuMoeExecutor``, what hybrid computes on the CPU) and the actual PCIe gather
    (``OffloadMoeCache.copy_missing`` / ``fast_index_copy``, what offload streams over
    PCIe on a decode miss). These are what the hybrid-vs-offload choice rides on.
    When both work, the same pair is also measured *overlapped* (running concurrently,
    contending for host DRAM) -- those two achieved bandwidths set the hybrid backend's
    per-step fetch split (``load_hybrid_fetch_fraction``).

The rule, per (workload, format): recommend ``hybrid`` when the CPU MoE kernel
bandwidth exceeds ``threshold`` x the PCIe gather bandwidth (default 2x), otherwise
``offload``. Formats with no CPU MoE weight path (e.g. block-fp8) can't be computed on
the CPU, so they always resolve to ``offload``.

The hybrid-vs-offload choice is dtype-dominated, so the default is a **per-dtype tuning bench**
(``--dtype``): one bench per expert format against a canonical geometry -- the minimal set the
runtime backend pick matches on. ``--model`` additionally benches specific model geometries for
per-model detail. Results are written to a JSON file per GPU (default ``$XDG_CACHE_HOME/
freetoken/benchbw/<gpu-uuid>.json``) so the choice is reproducible and a multi-GPU box keeps
one profile per card.

    ft bench bw                              # per-dtype tuning bench (default: all formats)
    ft bench bw --dtype nvfp4,bf16           # only these formats
    ft bench bw --model qwen3.6-moe          # per-model detail instead
    ft bench bw --dtype all --model all      # both
    ft bench bw --gpu GPU-2f3a...            # bench a specific GPU (UUID or nvidia-smi index)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import socket
import statistics
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace

import torch

from freetoken.gpu_select import (
    assign_gpu,
    bind_assigned_gpu,
    gpu_identity,
    single_gpu_arg,
)
from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.moe.cpu_executor import physical_core_cpus, resolve_threads_and_affinity
from freetoken.utils import init_logger

logger = init_logger(__name__)

# Formats the CPU MoE C++ kernel can compute AND this bench can build banks for; anything
# else is offload-only here. (The kernel also does q4_0, but this bench has no q4_0 banks.)
_CPU_MOE_FORMATS = frozenset({"bf16", "nvfp4", "mxfp4_triton", "ds_fp4"})
# Formats this bench can build synthetic (correctly-sized) banks for.
_BUILDABLE_FORMATS = frozenset({"bf16", "nvfp4", "fp8_block", "mxfp4_triton", "ds_fp4"})
# Friendlier CLI/display aliases for the internal quant_format strings.
_FORMAT_ALIASES = {"fp8": "fp8_block", "mxfp4": "mxfp4_triton"}
_FORMAT_DISPLAY = {"fp8_block": "fp8", "mxfp4_triton": "mxfp4"}
# Cap synthetic bank memory per (format) so big models stay feasible; bandwidth is
# per-byte so a smaller expert count doesn't change the measured GB/s (as long as the
# working set still exceeds the LLC, which this budget guarantees).
_SYNTH_BANK_BUDGET = 2 << 30

# CPU MoE ISA tiers, forced via FREETOKEN_CPU_MOE_ISA (the kernel caps down to hw/build
# support, so requesting a higher tier than the CPU has is safe -- it clamps).
_ISA_TIERS = ("scalar", "avx2", "avx512", "avx512bf16")


@contextlib.contextmanager
def _forced_isa(isa: str | None):
    """Force the CPU MoE ISA tier for executors constructed in this block (None = auto)."""
    key = "FREETOKEN_CPU_MOE_ISA"
    if isa is None:
        yield
        return
    prev = os.environ.get(key)
    os.environ[key] = isa
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


@dataclass(frozen=True)
class Workload:
    name: str
    hidden: int  # H
    inter: int  # moe_intermediate I
    experts: int  # E
    top_k: int
    formats: tuple[str, ...]  # native expert quant_format(s) to bench
    activation: str = "silu"
    swiglu_alpha: float = 1.702
    swiglu_limit: float | None = None


# Preset workloads. Dims from the model configs / benchmarks/bench_offload_cache_copy.py.
# E/top_k/H/I are what drive the per-expert byte size and thus the bandwidths.
WORKLOADS: dict[str, Workload] = {
    "qwen3.8-flash-next": Workload(
        "qwen3.8-flash-next", 2560, 640, 512, 10, ("nvfp4",)
    ),
    "qwen3.6-moe": Workload("qwen3.6-moe", 2048, 512, 256, 8, ("bf16", "nvfp4", "fp8_block")),
    "qwen3-30b": Workload("qwen3-30b", 2048, 768, 128, 8, ("bf16",)),
    "gemma4-26b": Workload("gemma4-26b", 2816, 704, 128, 8, ("bf16",), activation="gelu_tanh"),
    "gpt-oss-120b": Workload("gpt-oss-120b", 2880, 2880, 128, 4, ("mxfp4_triton",),
                             activation="gpt_oss_swiglu", swiglu_limit=7.0),
    "gpt-oss-20b": Workload("gpt-oss-20b", 2880, 2880, 32, 4, ("mxfp4_triton",),
                            activation="gpt_oss_swiglu", swiglu_limit=7.0),
    "dsv4": Workload("dsv4", 4096, 2048, 256, 6, ("ds_fp4",), swiglu_limit=7.0),
    "glm4.7-nvfp4": Workload("glm4.7-nvfp4", 5120, 1536, 160, 8, ("nvfp4",)),
    "minimax-m2.5": Workload("minimax-m2.5", 3072, 1536, 256, 8, ("nvfp4",)),
}

# Per-dtype canonical geometries for the TUNING bench (`--dtype`). The hybrid-vs-offload choice is
# dtype-dominated -- the CPU-MoE-vs-PCIe-gather bandwidth ratio rides on the expert *format* and
# the hardware, not the exact model -- so measuring one representative geometry per format is
# enough for the runtime backend pick (the engine also matches by dtype). This is the minimal set
# the Desktop "detect" flow runs; `--model` still benches specific model geometries for per-model
# detail. Geometries mirror the model families that ship each format (mxfp4 keeps gpt-oss's large
# experts -- the reason that format sits near the 2x threshold).
DTYPE_WORKLOADS: dict[str, Workload] = {
    "bf16": Workload("dtype:bf16", 2048, 768, 128, 8, ("bf16",)),
    "nvfp4": Workload("dtype:nvfp4", 3072, 1536, 128, 8, ("nvfp4",)),
    "fp8_block": Workload("dtype:fp8", 2048, 512, 128, 8, ("fp8_block",)),
    "mxfp4_triton": Workload("dtype:mxfp4", 2880, 2880, 128, 4, ("mxfp4_triton",),
                             activation="gpt_oss_swiglu", swiglu_limit=7.0),
    "ds_fp4": Workload("dtype:ds_fp4", 4096, 2048, 128, 6, ("ds_fp4",), swiglu_limit=7.0),
}


def default_out_path(gpu_uuid: str | None = None) -> str:
    # Single source of truth with the (torch-free) reader the engine consults.
    from freetoken.moe.bench_profile import default_profile_path

    return default_profile_path(gpu_uuid)


def _cgroup_mem_headroom() -> int | None:
    """Free bytes under this process's cgroup v2 memory limit, or None if unlimited."""
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            raw = f.read().strip()
        if raw == "max":
            return None
        limit = int(raw)
        with open("/sys/fs/cgroup/memory.current") as f:
            used = int(f.read().strip())
        return max(0, limit - used)
    except (OSError, ValueError):
        return None


def _available_ram_bytes() -> int:
    """Free host RAM, clamped to the cgroup memory limit so a container isn't over-estimated."""
    try:
        host = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        host = 8 << 30
    cg = _cgroup_mem_headroom()
    return min(host, cg) if cg is not None else host


# ============================== ceilings (hardware) ==============================


def measure_cpu_mem_bw(
    num_threads: int = 0,
    iters: int = 8,
    min_total_bytes: int = 2 << 30,
    per_core_bytes: int = 128 << 20,
) -> dict:
    """Aggregate host RAM read bandwidth (GB/s) across the physical cores (STREAM read).

    One worker per physical core (``resolve_threads_and_affinity``), each pinned,
    first-touching then streaming a private slice with a single-threaded reduction,
    all overlapped. The buffer is sized past the last-level cache so every pass reads
    DRAM, capped at half the available RAM (dropping worker count rather than under-
    sizing a slice) so the fill can't OOM the box.
    """
    n_threads, cores = resolve_threads_and_affinity(num_threads)

    total_bytes = max(min_total_bytes, n_threads * per_core_bytes)
    cap = _available_ram_bytes() // 2
    if cap > 0 and total_bytes > cap:
        max_fit = max(1, cap // per_core_bytes)
        if max_fit < n_threads:
            n_threads = max_fit
            cores = cores[:n_threads]
        total_bytes = min(max(min_total_bytes, n_threads * per_core_bytes), cap)

    n_elems = (total_bytes // 4) // n_threads * n_threads  # float32, evenly split
    per = n_elems // n_threads
    total_bytes = n_elems * 4

    buf = torch.empty(n_elems, dtype=torch.float32)  # workers first-touch their own slice
    results: list[float] = []  # keeps the reductions reachable (no dead-code elimination)

    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)  # each Python worker runs a single-threaded reduction
    release = threading.Barrier(n_threads + 1)
    done = threading.Barrier(n_threads + 1)

    def worker(idx: int, core: int) -> None:
        try:
            os.sched_setaffinity(0, {core})
        except (AttributeError, OSError):
            pass
        s = buf.narrow(0, idx * per, per)
        s.fill_(1.0)  # first-touch on this core's NUMA node + fault every page
        acc = float(s.sum())  # warmup
        release.wait()
        for _ in range(iters):
            acc += float(s.sum())
        done.wait()
        results.append(acc)

    threads = [threading.Thread(target=worker, args=(i, c)) for i, c in enumerate(cores)]
    for t in threads:
        t.start()
    try:
        release.wait()
        t0 = time.perf_counter()
        done.wait()
        wall = time.perf_counter() - t0
    finally:
        for t in threads:
            t.join()
        torch.set_num_threads(prev_threads)

    return {
        "bw_gbs": (total_bytes * iters) / wall / 1e9,
        "threads": n_threads,
        "buffer_bytes": total_bytes,
        "iters": iters,
    }


def measure_pcie_bw(device: torch.device, nbytes: int = 256 << 20, iters: int = 30) -> dict:
    """Linear pinned<->device copy bandwidth (GB/s) for H2D and D2H, via CUDA events."""
    n = max(1, nbytes // 2)  # bf16 elements
    moved = n * 2
    with torch.cuda.device(device):
        src = alloc_pinned_tensor(n, dtype=torch.bfloat16)
        dst = torch.empty(n, dtype=torch.bfloat16, device=device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        def _timed(copy) -> float:
            for _ in range(3):
                copy()
            torch.cuda.synchronize(device)
            start.record()
            for _ in range(iters):
                copy()
            end.record()
            end.synchronize()
            return (moved * iters) / (start.elapsed_time(end) / 1e3) / 1e9

        h2d = _timed(lambda: dst.copy_(src, non_blocking=True))
        d2h = _timed(lambda: src.copy_(dst, non_blocking=True))
    return {"h2d_gbs": h2d, "d2h_gbs": d2h, "bytes": moved, "iters": iters}


# ============================== bank geometry ==============================


def _offload_bank_specs(fmt: str, H: int, I: int) -> dict[str, tuple[int, torch.dtype]]:
    """Per-bank (elems_per_expert, dtype) for the flat offload host banks (the gather)."""
    u8, bf16, f16, f8 = torch.uint8, torch.bfloat16, torch.float16, torch.float8_e4m3fn
    if fmt == "bf16":
        return {"gate_up": (2 * I * H, bf16), "down": (H * I, bf16)}
    if fmt == "fp8_block":
        return {
            "gate_up": (2 * I * H, f8), "gate_up_scale": ((2 * I // 128) * (H // 128), bf16),
            "down": (H * I, f8), "down_scale": ((H // 128) * (I // 128), bf16),
        }
    if fmt == "nvfp4":
        return {
            "gate_up_packed": (2 * I * (H // 2), u8), "gate_up_scale": (2 * I * (H // 16), u8),
            "gate_up_global": (2 * I, f16), "down_packed": (H * (I // 2), u8),
            "down_scale": (H * (I // 16), u8), "down_global": (H, f16),
        }
    if fmt == "mxfp4_triton":
        return {
            "gate_up_blocks": (2 * I * (H // 2), u8), "gate_up_scales": (2 * I * (H // 32), u8),
            "gate_up_bias": (2 * I, bf16), "down_blocks": (H * (I // 2), u8),
            "down_scales": (H * (I // 32), u8), "down_bias": (H, bf16),
        }
    if fmt == "ds_fp4":
        return {
            "gate_up_packed": (2 * I * (H // 2), u8), "gate_up_scale": (2 * I * (H // 32), u8),
            "down_packed": (H * (I // 2), u8), "down_scale": (H * (I // 32), u8),
        }
    raise NotImplementedError(fmt)


def _expert_bytes(fmt: str, H: int, I: int) -> int:
    return sum(elems * dt.itemsize for elems, dt in _offload_bank_specs(fmt, H, I).values())


def _synth_experts(E: int, expert_bytes: int) -> int:
    """Expert count for the synthetic banks: as many as fit the memory budget (bandwidth is
    per-byte, so fewer experts don't change the GB/s), capped at the model's E. The budget
    (2 GiB) far exceeds any LLC, so the working set stays DRAM-sized whenever E is large; for
    huge-expert models each expert already exceeds the LLC, so a handful still defeats it."""
    return min(E, max(1, _SYNTH_BANK_BUDGET // max(1, expert_bytes)))


def _cpu_moe_bank_sources(fmt: str, H: int, I: int, E: int) -> dict:
    """3D pinned banks in the exact layout ``CpuMoeExecutor`` expects for ``fmt``.

    Scale banks that decode through an e8m0 exponent (mxfp4/ds_fp4) are set to a unit
    exponent so no weight lands in the float32 denormal range (which would slow the CPU
    GEMV and skew the timing); packed e2m1 codes are finite for any byte, so they're left
    as-is. Values otherwise don't affect the kernel's work.
    """
    def pin(*shape, dtype):
        return alloc_pinned_tensor(*shape, dtype=dtype)

    if fmt == "bf16":
        gate_up, down = pin(E, 2 * I, H, dtype=torch.bfloat16), pin(E, H, I, dtype=torch.bfloat16)
        gate_up.fill_(0.02)  # uninitialized bf16 can be denormal -> x86 FP slowdown
        down.fill_(0.02)
        return {"gate_up": gate_up, "down": down}
    if fmt == "nvfp4":
        b = {
            "gate_up_packed": pin(E, 2 * I, H // 2, dtype=torch.uint8),
            "gate_up_scale": pin(E, 2 * I, H // 16, dtype=torch.uint8),
            "gate_up_global": pin(E, 2 * I, dtype=torch.float16),
            "down_packed": pin(E, H, I // 2, dtype=torch.uint8),
            "down_scale": pin(E, H, I // 16, dtype=torch.uint8),
            "down_global": pin(E, H, dtype=torch.float16),
        }
        b["gate_up_global"].fill_(1.0)
        b["down_global"].fill_(1.0)  # e4m3 scales decode to normal float32; globals are fp16
        return b
    if fmt == "mxfp4_triton":  # transposed split-K layout
        b = {
            "gate_up_blocks": pin(E, H // 2, 2 * I, dtype=torch.uint8),
            "gate_up_scales": pin(E, H // 32, 2 * I, dtype=torch.uint8),
            "gate_up_bias": pin(E, 2 * I, dtype=torch.bfloat16),
            "down_blocks": pin(E, I // 2, H, dtype=torch.uint8),
            "down_scales": pin(E, I // 32, H, dtype=torch.uint8),
            "down_bias": pin(E, H, dtype=torch.bfloat16),
        }
        b["gate_up_scales"].fill_(127)  # e8m0 127 -> 2^0 = 1 (avoid denormal weights)
        b["down_scales"].fill_(127)
        b["gate_up_bias"].zero_()
        b["down_bias"].zero_()
        return b
    if fmt == "ds_fp4":
        b = {
            "gate_up_packed": pin(E, 2 * I, H // 2, dtype=torch.uint8),
            "gate_up_scale": pin(E, 2 * I, H // 32, dtype=torch.uint8),
            "down_packed": pin(E, H, I // 2, dtype=torch.uint8),
            "down_scale": pin(E, H, I // 32, dtype=torch.uint8),
        }
        b["gate_up_scale"].fill_(127)  # e8m0 unit exponent
        b["down_scale"].fill_(127)
        return b
    raise NotImplementedError(fmt)


# ========================= real kernels (per workload) =========================


def _build_gather_rig(fmt: str, wl: Workload, device: torch.device):
    """Production ``copy_missing`` rig: an ``OffloadMoeCache`` over synthetic pinned banks,
    staged so every ``copy_missing()`` refills a full layer (all synth experts, scattered
    sources). Returns ``(cache, bytes_per_copy)``; shared by the standalone PCIe gather
    bench and the overlapped CPU+PCIe bench."""
    from freetoken.moe.offload_cache import OffloadMoeCache

    H, I = wl.hidden, wl.inter
    E = _synth_experts(wl.experts, _expert_bytes(fmt, H, I))
    specs = _offload_bank_specs(fmt, H, I)
    cache = OffloadMoeCache(num_layers=1, num_experts=E, cache_size=E, device=device, quant_format=fmt)
    total_bytes = 0
    for name, (elems, dtype) in specs.items():
        src = alloc_pinned_tensor(E, elems, dtype=dtype)  # cudaHostAlloc (mapped) like the loaders
        dst = torch.empty(E, elems, dtype=dtype, device=device)
        cache.bank_sources[name] = [src]
        cache.bank_caches[name] = dst
        cache.banks.append(([src], dst))
        total_bytes += E * elems * dtype.itemsize
    cache._build_copy_plan()  # enable the fused path (production default) when aligned

    cache.evict_slots[:E].copy_(torch.arange(E, dtype=torch.int32, device=device))
    cache.src_indices[:E].copy_(torch.randperm(E, device=device).to(torch.int32))
    cache.num_indices.fill_(E)  # int64, allocated by the cache -> matches the kernel contract
    # copy_missing resolves the per-layer source through this (normally set by
    # ensure_experts/materialize_layer, bypassed here since this bench pokes the
    # evict/src/num_indices state directly).
    cache._pending_src_layer = 0
    return cache, total_bytes


def measure_pcie_gather_bw(fmt: str, wl: Workload, device: torch.device, iters: int = 20) -> dict:
    """Real PCIe gather bandwidth (GB/s): pinned host banks -> GPU slot cache.

    Drives the production ``OffloadMoeCache.copy_missing`` (fused multi-bank
    ``fast_index_copy`` when 16-byte aligned, else per-bank), refilling a full layer
    (all synth experts, scattered sources). Reusing the cache gives the kernel-correct
    int64 ``num_indices``. Timed with CUDA events.
    """
    eb = _expert_bytes(fmt, wl.hidden, wl.inter)
    cache, total_bytes = _build_gather_rig(fmt, wl, device)
    E = cache.num_experts

    for _ in range(3):
        cache.copy_missing()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if hasattr(torch.cuda, "_sleep"):
        torch.cuda._sleep(10**7)  # keep the CPU from running the launches ahead of the timer
    start.record()
    for _ in range(iters):
        cache.copy_missing()
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end) / iters
    return {
        "bw_gbs": total_bytes / (ms / 1e3) / 1e9,
        "expert_bytes": eb,
        "synth_experts": E,
        "fused": bool(cache._copy_fused_ok),
    }


def _build_cpu_moe_executor(fmt: str, wl: Workload, banks: dict, num_threads: int, E: int):
    """A production ``CpuMoeExecutor`` over synthetic banks (via a minimal cache stand-in)."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    cache = SimpleNamespace(
        quant_format=fmt,
        bank_sources={name: [t] for name, t in banks.items()},
        num_layers=1, num_experts=E,
        decode_target="cpu", cpu_executor=None,
    )
    return CpuMoeExecutor(
        cache, top_k=wl.top_k, activation=wl.activation,
        apply_router_weight_on_input=False, num_threads=num_threads, max_tokens=1,
        device=torch.device("cuda"), swiglu_alpha=wl.swiglu_alpha,
        swiglu_limit=wl.swiglu_limit,
    )


def _cpu_moe_step_fns(ex, wl: Workload, E: int):
    """``(set_ids, run)`` closures for bs=1 decode steps on a constructed executor.

    ``set_ids(i)`` routes step ``i`` to a rotating id window so consecutive steps touch
    disjoint experts (working set >> LLC) -- the DRAM-bound regime hybrid decode actually
    runs in; ``run()`` executes one step (releases the GIL inside the extension)."""
    bs, top_k = 1, wl.top_k
    io = ex._io_for(bs)
    io["x"].copy_(torch.randn(bs, wl.hidden, dtype=torch.bfloat16) * 0.1)
    io["w"].copy_(torch.rand(bs, top_k, dtype=torch.float32))
    task = ex._task_for(0, bs)
    base = torch.arange(top_k, dtype=torch.int32)

    def set_ids(i: int) -> None:
        io["ids"].copy_((((i * top_k) + base) % E).view(bs, top_k))

    def run() -> None:
        ex._ext.run_task(task)

    return set_ids, run


def _time_cpu_moe(ex, wl: Workload, eb: int, E: int, iters: int) -> float:
    """bs=1 CPU MoE decode bandwidth (GB/s) for a constructed executor, median-timed."""
    set_ids, run = _cpu_moe_step_fns(ex, wl, E)
    warmup = 8
    for i in range(warmup):
        set_ids(i)
        run()
    samples = []
    for i in range(iters):
        set_ids(warmup + i)  # continue the sweep so timed steps don't re-touch warmed experts
        t0 = time.perf_counter()
        run()
        samples.append(time.perf_counter() - t0)
    ms = statistics.median(samples) * 1e3
    return (wl.top_k * eb) / (ms / 1e3) / 1e9


def measure_cpu_moe_bw(fmt: str, wl: Workload, iters: int = 64, num_threads: int = 0,
                       isas: list[str] | None = None) -> dict:
    """Real CPU MoE GEMV bandwidth (GB/s) at bs=1, reading experts from pinned host banks.

    ``bw_gbs``/``isa`` are ALWAYS the kernel's auto-picked (best supported) tier -- what
    production actually runs -- so the recommendation never depends on a restricted ``--isa``
    subset. ``isas`` (a list) additionally forces each ``FREETOKEN_CPU_MOE_ISA`` tier over
    the SAME banks (built once) to populate the diagnostic ``isa_sweep`` breakdown; tiers
    above the CPU's support clamp down and dedupe by the resulting isa name.
    """
    H, I = wl.hidden, wl.inter
    eb = _expert_bytes(fmt, H, I)
    E = _synth_experts(wl.experts, eb)
    banks = _cpu_moe_bank_sources(fmt, H, I, E)

    def run(isa: str | None) -> tuple[str, float]:
        with _forced_isa(isa):
            ex = _build_cpu_moe_executor(fmt, wl, banks, num_threads, E)
        bw = round(_time_cpu_moe(ex, wl, eb, E, iters), 2)
        label = ex.isa
        del ex
        return label, bw

    auto_isa, auto_bw = run(None)  # recommendation basis: the tier production would pick
    sweep = {auto_isa: auto_bw}
    for isa in isas or []:
        label, bw = run(isa)
        sweep[label] = max(sweep.get(label, 0.0), bw)  # dedupe clamped tiers, keep best
    return {"bw_gbs": auto_bw, "isa": auto_isa, "isa_sweep": sweep,
            "expert_bytes": eb, "synth_experts": E}


def measure_overlap_bw(fmt: str, wl: Workload, device: torch.device,
                       num_threads: int = 0, seconds: float = 2.0) -> dict:
    """Concurrent achieved bandwidths (GB/s): the CPU MoE GEMV and the PCIe gather running
    at the same time -- the contention regime hybrid decode's overlap actually lives in.

    The standalone numbers cannot predict this split. Assuming full DRAM contention (CPU
    keeps ``cpu_bw - pcie_bw`` under DMA) over-penalizes a CPU kernel that never saturated
    DRAM to begin with -- the DMA then mostly rides the leftover bandwidth; assuming no
    contention ignores it entirely. Measuring the contended pair directly gives the hybrid
    backend its bandwidth-matched fetch split: fetched : cpu-computed = pcie_ov : cpu_ov.

    Both sides hammer their own pinned banks flat out for ``seconds`` from a shared
    barrier: a worker thread loops bs=1 CPU decode steps (``run_task`` releases the GIL)
    while the main thread loops full-layer ``copy_missing`` gathers (synchronized per copy
    so the DMA is really in flight, not just enqueued). Each side reports bytes / its own
    elapsed; the two windows differ by at most one CPU step + one gather.
    """
    H, I = wl.hidden, wl.inter
    eb = _expert_bytes(fmt, H, I)
    E = _synth_experts(wl.experts, eb)
    ex = _build_cpu_moe_executor(fmt, wl, _cpu_moe_bank_sources(fmt, H, I, E), num_threads, E)
    set_ids, run_step = _cpu_moe_step_fns(ex, wl, E)
    cache, gather_bytes = _build_gather_rig(fmt, wl, device)

    for i in range(8):  # warm both sides (JIT, page faults, clocks)
        set_ids(i)
        run_step()
        cache.copy_missing()
    torch.cuda.synchronize(device)

    start = threading.Barrier(2)
    stop = threading.Event()
    cpu_out: dict = {}

    def cpu_side() -> None:
        steps = 0
        start.wait()
        t0 = time.perf_counter()
        while not stop.is_set():
            set_ids(8 + steps)
            run_step()
            steps += 1
        cpu_out["gbs"] = steps * wl.top_k * eb / (time.perf_counter() - t0) / 1e9

    worker = threading.Thread(target=cpu_side)
    worker.start()
    start.wait()
    t0 = time.perf_counter()
    copies = 0
    while time.perf_counter() - t0 < seconds:
        cache.copy_missing()
        torch.cuda.synchronize(device)
        copies += 1
    pcie_gbs = copies * gather_bytes / (time.perf_counter() - t0) / 1e9
    stop.set()
    worker.join()
    del ex
    return {"cpu_gbs": cpu_out["gbs"], "pcie_gbs": pcie_gbs}


def recommend(cpu_bw_gbs: float, pcie_bw_gbs: float, threshold: float = 2.0) -> str:
    """``hybrid`` iff CPU bandwidth exceeds ``threshold`` x PCIe bandwidth, else ``offload``."""
    return "hybrid" if cpu_bw_gbs > threshold * pcie_bw_gbs else "offload"


# ================================ orchestration ================================


def _note(entry: dict, msg: str) -> None:
    entry["note"] = f"{entry['note']} | {msg}" if entry.get("note") else msg


def _bench_format(fmt: str, wl: Workload, device: torch.device, threshold: float,
                  cpu_threads: int, cpu_iters: int, pcie_iters: int,
                  isas: list[str] | None = None) -> dict:
    """Per (workload, format): real PCIe gather + real CPU MoE GEMV + hybrid/offload verdict.

    Expected degradations (extension not built, OOM, JIT failure) are caught per bench so
    one format can't kill the run; an unexpected exception is left to propagate.
    """
    entry: dict = {
        "expert_bytes": None, "synth_experts": None, "cpu_moe_gbs": None, "cpu_moe_isa": None,
        "isa_sweep": None, "pcie_gather_gbs": None,
        "cpu_moe_overlap_gbs": None, "pcie_gather_overlap_gbs": None,
        "ratio": None, "recommended": None,
        "note": None,
    }
    try:
        g = measure_pcie_gather_bw(fmt, wl, device, pcie_iters)
        entry["pcie_gather_gbs"] = round(g["bw_gbs"], 2)
        entry["expert_bytes"] = g["expert_bytes"]
        entry["synth_experts"] = g["synth_experts"]
    except (ImportError, RuntimeError) as e:
        logger.warning(f"benchbw: PCIe gather bench failed for {wl.name}/{fmt}: {e}")
        _note(entry, f"pcie gather unavailable ({e})")
    finally:
        torch.cuda.empty_cache()

    if fmt not in _CPU_MOE_FORMATS:
        _note(entry, f"CPU MoE has no {fmt} weight path; hybrid unavailable")
    else:
        try:
            c = measure_cpu_moe_bw(fmt, wl, cpu_iters, cpu_threads, isas=isas)
            entry["cpu_moe_gbs"] = c["bw_gbs"]  # auto/best tier (what production runs)
            entry["cpu_moe_isa"] = c["isa"]
            if isas:  # a sweep was requested -> keep the per-tier breakdown
                entry["isa_sweep"] = c["isa_sweep"]
                # The tier label comes from the bf16 dot chooser, so it can over-report tiers
                # the actual per-format kernel doesn't distinguish. Disclose the real story.
                if fmt == "nvfp4":  # WF_NVFP4 is the only use_vnni format
                    _note(entry, "isa tier labels are nominal: this W4A8 kernel rides AVX-VNNI "
                                 "regardless of the forced tier")
                elif fmt in ("mxfp4_triton", "ds_fp4"):
                    _note(entry, "isa tier labels are nominal: avx512bf16 == avx512f here "
                                 "(3 real kernels: scalar/avx2/avx512, no bf16/VNNI variant)")
            entry["expert_bytes"] = entry["expert_bytes"] or c["expert_bytes"]
            entry["synth_experts"] = entry["synth_experts"] or c["synth_experts"]
        except (ImportError, RuntimeError) as e:
            logger.warning(f"benchbw: CPU MoE bench failed for {wl.name}/{fmt}: {e}")
            _note(entry, f"cpu moe unavailable ({e})")

    cpu_g, pcie_g = entry["cpu_moe_gbs"], entry["pcie_gather_gbs"]
    if cpu_g is not None and pcie_g:  # pcie_g truthy also rules out a div-by-zero
        entry["ratio"] = round(cpu_g / pcie_g, 3)
        entry["recommended"] = recommend(cpu_g, pcie_g, threshold)
        # Both sides work standalone -> also measure them contended (concurrently). This
        # pair sets the hybrid backend's fetch split (load_hybrid_fetch_fraction); the
        # hybrid-vs-offload verdict above stays on the standalone numbers.
        try:
            o = measure_overlap_bw(fmt, wl, device, cpu_threads)
            entry["cpu_moe_overlap_gbs"] = round(o["cpu_gbs"], 2)
            entry["pcie_gather_overlap_gbs"] = round(o["pcie_gbs"], 2)
        except (ImportError, RuntimeError) as e:
            logger.warning(f"benchbw: overlap bench failed for {wl.name}/{fmt}: {e}")
            _note(entry, f"overlap bench unavailable ({e})")
        finally:
            torch.cuda.empty_cache()
    else:
        # No CPU-vs-PCIe pair to compare (no CPU path, or a bench failed): offload is the
        # backend that always works, so it's the safe call absent evidence for hybrid.
        entry["recommended"] = "offload"
    return entry


def _atomic_write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def run_benchbw(
    out_path: str | None = None,
    threshold: float = 2.0,
    device_index: int = 0,
    models: tuple[str, ...] = (),
    dtypes: tuple[str, ...] | None = None,
    formats: tuple[str, ...] | None = None,
    isas: list[str] | None = None,
    cpu_threads: int = 0,
    cpu_iters: int = 8,
    pcie_bytes: int = 256 << 20,
    pcie_iters: int = 30,
    kernel_cpu_iters: int = 64,
    kernel_pcie_iters: int = 20,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "benchbw needs a CUDA device to measure PCIe bandwidth (both offload and "
            "hybrid serve experts to the GPU)."
        )
    if torch.cuda.device_count() == 0:
        raise RuntimeError(
            f"no CUDA device visible (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})."
        )
    if not 0 <= device_index < torch.cuda.device_count():
        raise RuntimeError(
            f"device_index {device_index} out of range (found {torch.cuda.device_count()} CUDA devices)."
        )
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    gpu = gpu_identity(device_index)

    # Machine-parseable progress on stdout (opt-in), so the daemon/Desktop can stream feedback
    # while the bench runs -- mirrors ft checkpoint's FTCONVERT lines. `done`/`total` count the
    # measured units (ceilings + one per benched format); `label` is what's measuring now.
    prog_on = os.environ.get("FREETOKEN_BENCH_PROGRESS") == "1"
    n_model_units = sum(len(formats or WORKLOADS[m].formats) for m in models)
    total = 1 + len(dtypes or ()) + n_model_units
    done = 0

    def _prog(label: str) -> None:
        if prog_on:
            print(f"FTBENCH {done} {total} {label}", flush=True)

    _prog("ceilings")
    logger.info(f"benchbw: measuring hardware ceilings on cuda:{device_index} ...")
    cpu = measure_cpu_mem_bw(num_threads=cpu_threads, iters=cpu_iters)
    pcie = measure_pcie_bw(device, nbytes=pcie_bytes, iters=pcie_iters)
    done = 1

    # CpuMoeExecutor logs an INFO line per construction; with --isa that's several per
    # format and buries the report. Silence just that logger for the measurement loop.
    exec_log = logging.getLogger("freetoken.moe.cpu_executor")
    prev_level = exec_log.level
    exec_log.setLevel(logging.WARNING)
    dtypes_out: dict = {}
    workloads_out: dict = {}
    try:
        # dtype tuning: one bench per expert format (canonical geometry). This is what the
        # runtime backend pick keys off -- matched by dtype in bench_profile.py.
        for fmt in dtypes or ():
            wl = DTYPE_WORKLOADS[fmt]
            _prog(f"dtype:{_FORMAT_DISPLAY.get(fmt, fmt)}")
            logger.info(f"benchbw: dtype {_FORMAT_DISPLAY.get(fmt, fmt)} real kernels ...")
            dtypes_out[fmt] = _bench_format(
                fmt, wl, device, threshold, cpu_threads, kernel_cpu_iters, kernel_pcie_iters, isas
            )
            done += 1
        # optional per-model detail (--model): specific geometries, unchanged.
        for mname in models:
            wl = WORKLOADS[mname]
            fmts = formats or wl.formats
            kernels = {}
            for fmt in fmts:
                _prog(f"{mname}:{_FORMAT_DISPLAY.get(fmt, fmt)}")
                logger.info(f"benchbw: {mname}/{fmt} real kernels ...")
                kernels[fmt] = _bench_format(
                    fmt, wl, device, threshold, cpu_threads, kernel_cpu_iters, kernel_pcie_iters, isas
                )
                done += 1
            workloads_out[mname] = {
                "model": {"name": wl.name, "hidden": wl.hidden, "inter": wl.inter,
                          "experts": wl.experts, "top_k": wl.top_k},
                "kernels": kernels,
                "recommended_moe_backend": {f: kernels[f]["recommended"] for f in fmts},
            }
    finally:
        exec_log.setLevel(prev_level)
    _prog("done")

    result = {
        "version": 4,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "epoch": int(time.time()),
        "host": socket.gethostname(),
        # index is the CUDA ordinal the bench ran on; uuid keys the profile file
        "gpu": {"index": device_index, "name": gpu["name"], "uuid": gpu["uuid"]},
        "cpu": {"physical_cores": len(physical_core_cpus()), "threads_used": cpu["threads"]},
        "threshold": threshold,
        "ceilings": {
            "cpu_stream_read_gbs": round(cpu["bw_gbs"], 2),
            "pcie_linear_h2d_gbs": round(pcie["h2d_gbs"], 2),
            "pcie_linear_d2h_gbs": round(pcie["d2h_gbs"], 2),
        },
        # Per-dtype verdicts (the tuning axis) + optional per-model detail.
        "dtypes": {f: e["recommended"] for f, e in dtypes_out.items()},
        "dtype_kernels": dtypes_out,
        "workloads": workloads_out,
    }

    out_path = os.path.expanduser(out_path or default_out_path(gpu["uuid"]))
    _atomic_write_json(out_path, result)
    result["out_path"] = out_path
    if prog_on:
        print(f"FTBENCH_OUT {out_path}", flush=True)
    return result


# ================================== reporting ==================================


def _gbs(x) -> str:
    return f"{x:>8.1f} GB/s" if isinstance(x, (int, float)) else f"{'n/a':>13}"


def _print_kernels(kernels: dict, iw: int) -> None:
    print(f"    {'format':<8} {'expert':>9} {'CPU-MoE':>13} {'PCIe-gather':>13} "
          f"{'CPU/PCIe':>9}  backend")
    for fmt, e in kernels.items():
        disp = _FORMAT_DISPLAY.get(fmt, fmt)
        eb = f"{e['expert_bytes'] / 2**20:.2f} MB" if e["expert_bytes"] else "n/a"
        ratio = f"{e['ratio']:.2f}x" if e["ratio"] is not None else "—"
        print(f"    {disp:<8} {eb:>9} {_gbs(e['cpu_moe_gbs'])} {_gbs(e['pcie_gather_gbs'])} "
              f"{ratio:>9}  {e['recommended']}")
        c_ov, p_ov = e.get("cpu_moe_overlap_gbs"), e.get("pcie_gather_overlap_gbs")
        if c_ov and p_ov:
            print(f"       overlapped: CPU-MoE {c_ov:.1f} + PCIe {p_ov:.1f} GB/s "
                  f"-> hybrid fetches {p_ov / (p_ov + c_ov):.1%} of misses")
        if e.get("isa_sweep"):
            tiers = sorted(e["isa_sweep"].items(), key=lambda kv: -kv[1])
            for i, (k, v) in enumerate(tiers):
                label = "isa:" if i == 0 else "    "
                print(f"       {label} {k.split('(')[0]:<{iw}}  {v:>7.1f} GB/s")
        if e.get("note"):
            print(f"           └─ {e['note']}")


def _print_report(r: dict) -> None:
    c = r["ceilings"]
    print()
    print(f"  host {r['host']}   gpu cuda:{r['gpu']['index']} ({r['gpu']['name']})   "
          f"cpu {r['cpu']['physical_cores']}c/{r['cpu']['threads_used']}t")
    print(f"  ceilings: CPU STREAM read {c['cpu_stream_read_gbs']:.1f}  |  "
          f"PCIe linear H2D {c['pcie_linear_h2d_gbs']:.1f}  D2H {c['pcie_linear_d2h_gbs']:.1f}  GB/s"
          f"   (threshold {r['threshold']}x)")
    # one global ISA-name width so every sweep's GB/s lands in the same column; the
    # redundant "(<fmt>)" tag is dropped -- the format is already in the row header.
    all_kernels = list((r.get("dtype_kernels") or {}).values()) + [
        e for wl in r["workloads"].values() for e in wl["kernels"].values()
    ]
    iw = max((len(k.split("(")[0]) for e in all_kernels for k in (e.get("isa_sweep") or {})), default=0)
    if r.get("dtype_kernels"):
        print("\n  per-dtype (tuning — what the runtime backend pick matches on)")
        _print_kernels(r["dtype_kernels"], iw)
    for name, w in r["workloads"].items():
        m = w["model"]
        print(f"\n  {name}  H={m['hidden']} I={m['inter']} E={m['experts']} top_k={m['top_k']}")
        _print_kernels(w["kernels"], iw)
    print(f"\n  saved: {r['out_path']}\n")


# ================================ CLI plumbing ================================


def _positive_int(s: str) -> int:
    v = int(s)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return v


def _nonneg_int(s: str) -> int:
    v = int(s)
    if v < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return v


def _positive_float(s: str) -> float:
    v = float(s)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return v


def _model_list(s: str) -> tuple[str, ...]:
    if s.strip() == "all":
        return tuple(WORKLOADS)
    items = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in items if x not in WORKLOADS]
    if bad or not items:
        raise argparse.ArgumentTypeError(
            f"'all' or a comma-separated subset of {sorted(WORKLOADS)}, got {s!r}"
        )
    return tuple(dict.fromkeys(items))


def _dtype_list(s: str) -> tuple[str, ...]:
    if s.strip() == "all":
        return tuple(DTYPE_WORKLOADS)
    items = [_FORMAT_ALIASES.get(x.strip(), x.strip()) for x in s.split(",") if x.strip()]
    bad = [x for x in items if x not in DTYPE_WORKLOADS]
    if bad or not items:
        raise argparse.ArgumentTypeError(
            f"'all' or a comma-separated subset of {sorted(DTYPE_WORKLOADS)} "
            f"(aliases {sorted(_FORMAT_ALIASES)}), got {s!r}"
        )
    return tuple(dict.fromkeys(items))


def _format_list(s: str) -> tuple[str, ...]:
    items = [_FORMAT_ALIASES.get(x.strip(), x.strip()) for x in s.split(",") if x.strip()]
    bad = [x for x in items if x not in _BUILDABLE_FORMATS]
    if bad or not items:
        raise argparse.ArgumentTypeError(
            f"comma-separated subset of {sorted(_BUILDABLE_FORMATS)} (aliases "
            f"{sorted(_FORMAT_ALIASES)}), got {s!r}"
        )
    return tuple(dict.fromkeys(items))


def _isa_list(s: str) -> list[str] | None:
    if s.strip() == "auto":
        return None
    if s.strip() == "all":
        return list(_ISA_TIERS)
    items = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in items if x not in _ISA_TIERS]
    if bad or not items:
        raise argparse.ArgumentTypeError(f"'auto', 'all', or a subset of {list(_ISA_TIERS)}, got {s!r}")
    return list(dict.fromkeys(items))


def main(argv: list[str] | None = None, prog: str = "ft bench bw") -> int:
    p = argparse.ArgumentParser(prog=prog, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dtype", type=_dtype_list, default=None,
                   help=f"tuning bench: 'all' (default when no --model), or comma-separated from "
                        f"{sorted(DTYPE_WORKLOADS)} -- one bench per expert format (what the runtime "
                        f"backend pick matches on)")
    p.add_argument("--model", type=_model_list, default=None,
                   help=f"per-model detail instead of/in addition to --dtype: 'all', or comma-"
                        f"separated from {sorted(WORKLOADS)}")
    p.add_argument("--formats", type=_format_list, default=None,
                   help="override a --model's native expert formats (comma-separated)")
    p.add_argument("--isa", type=_isa_list, default=None, dest="isas",
                   help=f"CPU MoE ISA: 'auto' (default, best), 'all', or a subset of "
                        f"{list(_ISA_TIERS)} to sweep (kernel caps down to hw support)")
    p.add_argument("-o", "--out", default=None,
                   help=f"JSON output path (default {default_out_path('<gpu-uuid>')})")
    p.add_argument("--threshold", type=_positive_float, default=2.0,
                   help="recommend hybrid when CPU BW > threshold x PCIe BW (default 2.0)")
    p.add_argument("--gpu", type=single_gpu_arg, default=None,
                   help="GPU to bench: a GPU UUID (GPU-xxxx..., as nvidia-smi -L prints) or an "
                        "nvidia-smi index (default: the first visible GPU)")
    p.add_argument("--cpu-threads", type=_nonneg_int, default=0,
                   help="CPU worker threads (0 = one per physical core)")
    p.add_argument("--cpu-iters", type=_positive_int, default=8, help="STREAM read passes to time")
    p.add_argument("--pcie-mib", type=_positive_int, default=256, help="linear PCIe copy size in MiB")
    p.add_argument("--pcie-iters", type=_positive_int, default=30, help="linear PCIe copies to time")
    p.add_argument("--kernel-cpu-iters", type=_positive_int, default=64,
                   help="CPU MoE decode steps to time")
    p.add_argument("--kernel-pcie-iters", type=_positive_int, default=20,
                   help="fast_index_copy gather passes to time")
    ns = p.parse_args(argv)

    # same as ft serve --gpu: resolve, then bind by UUID at CUDA init
    try:
        assign_gpu(ns.gpu)
        device_index = bind_assigned_gpu().index
    except (ValueError, RuntimeError) as e:
        p.error(str(e))

    models = ns.model or ()
    dtypes = ns.dtype
    # Default (no selection): the full per-dtype tuning bench -- the minimal set the runtime
    # backend pick needs. --model alone skips it; --dtype + --model runs both.
    if not models and not dtypes:
        dtypes = tuple(DTYPE_WORKLOADS)

    try:
        result = run_benchbw(
            out_path=ns.out, threshold=ns.threshold, device_index=device_index, models=models,
            dtypes=dtypes, formats=ns.formats, isas=ns.isas, cpu_threads=ns.cpu_threads,
            cpu_iters=ns.cpu_iters, pcie_bytes=ns.pcie_mib << 20, pcie_iters=ns.pcie_iters,
            kernel_cpu_iters=ns.kernel_cpu_iters, kernel_pcie_iters=ns.kernel_pcie_iters,
        )
    except (RuntimeError, OSError) as e:
        print(f"error: {e}")
        return 1

    _print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
