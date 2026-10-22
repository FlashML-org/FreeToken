"""Python wrapper around the ``_cpu_moe`` C++ executor (--moe-strategy cpu).

Owns the persistent CPU worker pool and one pinned IO block of ``max_tokens`` rows that
every layer and batch size shares (one task is in flight at a time). ``decode`` issues
the whole CUDA-graph-capturable sequence on the current stream:

    D2H (hidden, topk_ids, topk_weights -> pinned IO rows [:bs])
      -> submit (doorbell naming (layer, bs), or a cudaLaunchHostFunc host node)
      -> sync   (wait for the CPU pool to finish)
      -> H2D (pinned expert output -> GPU)

With the doorbell on CUDA, a publish kernel does the D2H and the doorbell, and skips the step when no route goes to the CPU; a combine kernel does the H2D (csrc/cpu_moe/gpu_sync_io.h).
"""

from __future__ import annotations

import os
import threading
import time
import weakref

import torch

from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.utils import init_logger

logger = init_logger(__name__)

# Flag-based GPU<->CPU handshake for hybrid/cpu decode. The default host-func path
# (cudaLaunchHostFunc submit+sync per layer) pays ~30-50us of callback dispatch latency
# per call with the GPU stream idle -- 2 calls per MoE layer per decode step (~6 ms/step
# on a 75-layer model). Instead the GPU raises a mapped-pinned "ready" flag at submit; a
# persistent CPU coordinator (in _cpu_moe) polls it, runs the layer, and sets a "done"
# flag the GPU waits on at sync -- no host-func round-trip. Both GPU-side operations are
# STREAM MEMORY OPERATIONS (cuStreamWriteValue64 / cuStreamWaitValue64 on CUDA, see
# gpu_sync.h): they execute on the GPU front end with no SM-resident kernel, so
# GPU utilization stays truthful during the CPU compute window. (The first cut used a
# spin-wait kernel; that pinned reported utilization at 99% and laptop CPU/GPU dynamic
# power schedulers responded by clamping the CPU frequency -- a net decode regression on
# power-coupled edge devices.) The doorbell word names the (layer, decode batch size)
# task, so one ready/done pair serves every layer and every captured graph (gpu_sync.h).
# Where memops are unavailable (some driver models, vGPU, old drivers -- functionally probed at
# startup), decode keeps the host-func path (functional, slower). ROCm 7.14 cannot replay
# that path from a graph, so capture fails loudly there instead. A Python watchdog thread
# turns a wedged coordinator into a loud RuntimeError (via raise_if_unhealthy) instead of
# an indefinite stream stall.
# Caveat: the coordinator busy-polls one core while decode traffic flows (idle backoff
# otherwise); FREETOKEN_CPU_MOE_FLAG_SYNC=0 opts out entirely.
_FLAG_SYNC = os.getenv("FREETOKEN_CPU_MOE_FLAG_SYNC", "1") != "0"
# Pool threads spin up to 50 ms before they park. FREETOKEN_CPU_MOE_SPIN=0 opts out.
_SPIN_WAIT = os.getenv("FREETOKEN_CPU_MOE_SPIN", "1") != "0"
# ds_fp4 runs its input FP8 round-trip on the GPU when it can; tests turn this off to compare against the CPU round-trip.
_DSFP4_GPU_PREQUANT = True

# Activation ids must match ActKind in csrc/cpu_moe/common.h. Id 3 is the
# clamped (up + 1) swiglu: "swigluoai" runs it in the generic GEMV epilogue,
# "gpt_oss_swiglu" is the same math fused inside the mxfp4 kernel.
_ACT_IDS = {
    "silu": 0,
    "swish": 0,
    "gelu": 1,
    "gelu_tanh": 2,
    "gelu_pytorch_tanh": 2,
    "gpt_oss_swiglu": 3,
    "swigluoai": 3,
    "swiglu_clamp": 4,
}


def compiled_extension_supports(activation: str) -> bool:
    """Whether the compiled ``_cpu_moe`` extension can serve ``activation``
    through its generic epilogue. A stale prebuilt .so accepts newer act ids
    while silently computing the wrong math; the executor hard-errors on that,
    but the engine's auto offload->hybrid upgrade consults this first so a
    default boot degrades to offload instead of crashing after weight load."""
    if activation not in _ACT_IDS:
        return False
    if _ACT_IDS[activation] < 3:
        return True
    try:
        from freetoken.kernel import _cpu_moe
    except ImportError:
        return False
    return _ACT_IDS[activation] <= getattr(_cpu_moe, "max_generic_act_id", lambda: 2)()


def physical_core_cpus() -> list[int]:
    """One logical CPU per physical core, restricted to this process's affinity.

    MoE decode is memory-bandwidth-bound, so SMT siblings only contend for the
    same core's load ports without adding bandwidth. Picking one logical CPU per
    physical core (and pinning to it) gives the best, most stable bandwidth.
    Falls back to the full affinity set when sysfs topology is unavailable.
    """
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except AttributeError:
        allowed = list(range(os.cpu_count() or 1))
    reps: list[int] = []
    seen: set[str] = set()
    for cpu in allowed:
        try:
            with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as f:
                key = f.read().strip()
        except OSError:
            reps.append(cpu)
            continue
        if key not in seen:
            seen.add(key)
            reps.append(cpu)
    return reps or allowed or [0]


def _spin_wait_ok(core_ids: list[int], coord_core: int) -> bool:
    """Spin only if each pool thread owns a distinct CPU and two CPUs stay free for the main and CUDA callback threads."""
    if not _SPIN_WAIT or not hasattr(os, "sched_getaffinity"):
        return False
    pool = list(core_ids) + ([coord_core] if coord_core >= 0 else [])
    if len(set(pool)) != len(pool):
        return False
    spare = set(os.sched_getaffinity(0)) - set(pool)
    return len(spare) >= 2


def resolve_threads_and_affinity(requested: int) -> tuple[int, list[int]]:
    """Return (num_threads, core_ids) for the worker pool.

    ``requested == 0`` -> one thread per physical core, pinned to it (best for the
    bandwidth-bound GEMV: SMT siblings only contend for a core's load ports and the
    spin-barrier degrades badly when oversubscribed). An explicit count is honored,
    spreading first across physical cores, then across the remaining logical CPUs
    (so distinct hardware threads are used before any core is doubled up).
    """
    reps = physical_core_cpus()
    if requested and requested > 0:
        n = int(requested)
        try:
            allowed = sorted(os.sched_getaffinity(0))
        except AttributeError:
            allowed = list(range(os.cpu_count() or 1))
        # physical-core reps first, then the rest of the logical CPUs.
        order = reps + [c for c in allowed if c not in set(reps)]
        if not order:
            order = [0]
        core_ids = [order[i % len(order)] for i in range(n)]
        return n, core_ids
    return len(reps), list(reps)


def _format_label(name: str, opts: tuple) -> str:
    """One format instance as the startup log names it, e.g. ``gguf(gate_up=IQ2_XXS, down=IQ4_NL)``."""
    if name in ("gguf", "gguf_scaled"):
        from freetoken.layers.quantization.ggml_types import type_name

        gate_up, down = dict(opts)["gate_up_type"], dict(opts)["down_type"]
        return f"{name}(gate_up={type_name(gate_up)}, down={type_name(down)})"
    return name + (f"({', '.join(f'{k}={v}' for k, v in opts)})" if opts else "")


class CpuMoeExecutor:
    """Decode-time CPU expert compute over an ``OffloadMoeCache``'s host banks
    (the formats ``_cpu_moe.weight_formats()`` lists — see ``_resolve_banks``)."""

    def __init__(
        self,
        cache,
        *,
        top_k: int,
        activation: str,
        apply_router_weight_on_input: bool,
        num_threads: int,
        max_tokens: int,
        device: torch.device,
        swiglu_alpha: float = 1.702,
        swiglu_limit: float | None = None,
        layer_formats: list[tuple[str, dict[str, int]]] | None = None,
    ) -> None:
        from freetoken.kernel import _cpu_moe
        from freetoken.moe.legacy_format import canonical_role

        # layer_formats[l] = (format, options) of layer l; None runs every layer in cache.quant_format
        if layer_formats is None:
            layer_formats = [(cache.quant_format, {})] * int(cache.num_layers)
        assert len(layer_formats) == cache.num_layers, (len(layer_formats), cache.num_layers)
        keys = [(name, tuple(sorted(opts.items()))) for name, opts in layer_formats]
        specs = list(dict.fromkeys(keys))
        names = sorted({name for name, _ in specs})
        for f in names:
            if f not in _cpu_moe.weight_formats():
                raise NotImplementedError(
                    f"--moe-strategy cpu/hybrid computes experts on the CPU and supports "
                    f"{sorted(_cpu_moe.weight_formats())} formats, but this checkpoint's experts are "
                    f"{f!r}; use --moe-strategy offload (GPU-side dequant) instead."
                )
            if f in ("gguf", "gguf_scaled"):
                from freetoken.kernel import ggml_cpu

                if (variant := ggml_cpu.select()) is None:
                    raise NotImplementedError(
                        "--moe-strategy cpu/hybrid cannot compute GGUF experts here: this build has no ggml-cpu kernels (x86-64 only); "
                        "use --moe-strategy offload instead."
                    )
                _cpu_moe.set_ggml_cpu(variant.api)
        if activation not in _ACT_IDS:
            raise NotImplementedError(f"CPU MoE backend: unsupported activation {activation!r}")
        if apply_router_weight_on_input and (output_only := {"mxfp4_triton", "ds_fp4"} & set(names)):
            raise NotImplementedError(f"CPU MoE backend: {sorted(output_only)} experts apply the router weight on the output only")
        # ABI probe: a stale prebuilt _cpu_moe.so accepts newer act ids without
        # error and silently computes the wrong activation in the generic
        # epilogue -- fail loudly with the rebuild instruction instead. (mxfp4
        # handles its act inside the kernel and predates the marker.)
        if _ACT_IDS[activation] >= 3 and names != ["mxfp4_triton"]:
            supported = getattr(_cpu_moe, "max_generic_act_id", lambda: 2)()
            if _ACT_IDS[activation] > supported:
                raise RuntimeError(
                    f"the compiled _cpu_moe extension predates activation "
                    f"{activation!r} (max generic act id {supported}); rebuild it "
                    "with `python setup.py build_ext --inplace` (or reinstall the "
                    "wheel) before serving this model on the cpu/hybrid backend."
                )

        self.num_layers = int(cache.num_layers)
        self.num_experts = int(cache.num_experts)
        self.top_k = int(top_k)
        self.quant_format = ",".join(names)
        self.device = device
        self.max_tokens = int(max_tokens)
        self.apply_router_weight_on_input = bool(apply_router_weight_on_input)
        # The per-layer tensors and their pointer tables must outlive the executor
        # (C++ holds raw addresses into both).
        self._banks: list[torch.Tensor] = []
        by_role = {canonical_role(name): per_layer for name, per_layer in cache.bank_sources.items()}
        # every format's tables address every layer, but a format reads only its own layers' rows
        formats, shapes = [], set()
        for name, opts in specs:
            banks, options, shape = self._resolve_banks(by_role, name, keys.index((name, opts)), dict(opts))
            formats.append((name, banks, {**dict(opts), **options}))
            shapes.add(shape)
        if len(shapes) != 1:
            raise NotImplementedError(f"CPU MoE formats disagree on the expert shape (hidden, intermediate): {sorted(shapes)}")
        (self.H, self.I), = shapes
        layer_counts = {spec: keys.count(spec) for spec in specs}

        # ds_fp4: the reference FP8 activation round-trip is a scalar per-element chain
        # that the C++ side runs single-threaded on the CUDA host-callback thread --
        # straight on the decode critical path (~0.3ms/layer at H=4096, every worker and
        # the GPU waiting on it). When a GPU is present we run the numerically identical
        # round-trip as a captured GPU kernel BEFORE the D2H (see decode_submit) and tell
        # the C++ side to skip its own. Measured on DeepSeek-V4-Flash bs=1 decode:
        # 12.85 -> 15.65 tok/s, output bit-identical (tests/moe/test_dsfp4_prequant.py).
        # the W4A8 activation round-trip block follows the checkpoint's fp8 block (128 on V4, 32 on V4.1)
        blocks = {options.get("act_block") for name, _, options in formats if name == "ds_fp4"}
        assert None not in blocks, "ds_fp4 experts need the act_block option, the checkpoint's fp8 activation quant block"
        # the GPU round-trip rewrites the whole input, so only when every layer is ds_fp4 with one block
        self._gpu_prequant = names == ["ds_fp4"] and len(blocks) == 1 and device.type == "cuda" and _DSFP4_GPU_PREQUANT
        self._act_block = next(iter(blocks)) if self._gpu_prequant else None
        for name, _, options in formats:
            if name == "ds_fp4":
                options["input_prequant"] = int(self._gpu_prequant)
        if self._gpu_prequant:
            logger.info_rank0(
                "ds_fp4: input FP8 round-trip moved to the GPU "
                "(bit-identical grid; the CPU-side scalar round-trip is skipped)"
            )

        # Decide the flag handshake up front (env + device + a functional stream-memop
        # probe): its coordinator needs a core of its own, which the auto thread sizing
        # below reserves (a coordinator time-slicing against the GEMV workers measurably
        # destabilizes throughput on fully-subscribed boxes).
        self._flag_sync = _FLAG_SYNC and device.type == "cuda"
        if self._flag_sync and not _cpu_moe.memops_probe(torch.cuda.current_stream().cuda_stream):
            logger.info_rank0(
                "cpu-moe flag handshake unavailable: stream memory operations failed the "
                "startup probe; using the cudaLaunchHostFunc sync"
            )
            self._flag_sync = False
        # ROCm 7.14 returns wrong results when it replays host-function nodes, so there only the doorbell is graph-safe.
        self._graph_capture_safe = self._flag_sync or _cpu_moe.host_func_capture_safe()

        nthreads, core_ids = resolve_threads_and_affinity(num_threads)
        coord_core = -1
        if self._flag_sync and num_threads == 0 and nthreads > 2:
            # Auto sizing: give the coordinator the last physical core instead of
            # oversubscribing (workers drop from N to N-1).
            coord_core = core_ids[-1]
            nthreads -= 1
            core_ids = core_ids[:-1]
        self._coord_core = coord_core
        alloc_io = (
            (lambda *shape, dtype: torch.empty(*shape, dtype=dtype))
            if device.type == "cpu"
            else alloc_pinned_tensor
        )
        self._io = {
            "x": alloc_io(self.max_tokens, self.H, dtype=torch.bfloat16),
            "ids": alloc_io(self.max_tokens, self.top_k, dtype=torch.int32),
            "w": alloc_io(self.max_tokens, self.top_k, dtype=torch.float32),
            "y": alloc_io(self.max_tokens, self.H, dtype=torch.bfloat16),
        }
        self._ext = _cpu_moe.CpuMoeExecutor(
            num_threads=nthreads,
            num_layers=self.num_layers,
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.H,
            inter_size=self.I,
            max_tokens=self.max_tokens,
            activation_id=_ACT_IDS[activation],
            apply_router_weight_on_input=1 if apply_router_weight_on_input else 0,
            formats=formats,
            layer_formats=[specs.index(k) for k in keys],
            swiglu_alpha=float(swiglu_alpha),
            swiglu_limit=float(swiglu_limit) if swiglu_limit is not None else float("inf"),
            core_ids=core_ids,
            x_ptr=self._io["x"].data_ptr(),
            ids_ptr=self._io["ids"].data_ptr(),
            w_ptr=self._io["w"].data_ptr(),
            y_ptr=self._io["y"].data_ptr(),
        )
        self.num_threads = nthreads
        self.core_ids = core_ids
        self.isa = self._ext.isa_name()
        self.spin_wait = _spin_wait_ok(core_ids, coord_core)
        self._ext.set_spin_wait(self.spin_wait)

        spare = len(physical_core_cpus()) - nthreads - (1 if coord_core >= 0 else 0) - 1
        clamp = max(1, min(torch.get_num_threads(), spare))
        if clamp < torch.get_num_threads():
            logger.info_rank0(
                f"torch intra-op threads: {torch.get_num_threads()} -> {clamp} "
                "(cores reserved for the pinned CPU MoE pool)"
            )
            torch.set_num_threads(clamp)

        # Flag-based handshake: the extension owns the ready/done words (gpu_sync.h) and a
        # persistent CPU coordinator that polls the doorbell, runs the task it names on the
        # pool, and sets done for the GPU. The WATCHDOG thread marks the executor unhealthy
        # when a doorbell stays unanswered (dead coordinator): it poisons done to unblock the
        # stream and raise_if_unhealthy() turns the step into a loud error instead of silent
        # stale output (flag_sync itself was decided above, before thread sizing).
        self._unhealthy = False
        self._kernel_io = False
        if self._flag_sync:
            self._ext.start_flag_coordinator(self._coord_core)
            # ROCm keeps the copies until the kernel IO has been checked on AMD GPUs.
            if torch.version.hip is None:
                reason = _cpu_moe.kernel_io_probe(torch.cuda.current_stream().cuda_stream)
                if reason:
                    logger.warning_rank0(f"cpu-moe kernel IO unavailable ({reason}); using D2H/H2D copies around the doorbell")
                else:
                    self._ext.enable_kernel_io()
                    self._kernel_io = True
            self._watchdog_stop = False
            # The thread target holds a WEAKREF and re-derefs it each tick: a bound
            # method would strong-reference the executor forever (the loop never ends
            # on its own), pinning the C++ worker pool and the pinned banks against GC
            # in build-many-engines scenarios. NB: the stop flag / weakref death is
            # observed only between 2 s sleeps, so teardown of the THREAD can lag up to
            # ~2 s -- it is a daemon, so neither GC of the executor (weakref breaks the
            # cycle) nor process exit waits on it.
            self._watchdog = threading.Thread(
                target=_watchdog_main,
                args=(weakref.ref(self),),
                name="cpu-moe-flag-watchdog",
                daemon=True,
            )
            self._watchdog.start()


        fmt_label = "; ".join(f"{_format_label(name, opts)}: {n} layers" for (name, opts), n in layer_counts.items())
        logger.info_rank0(
            f"CPU MoE executor ready: threads={nthreads} (pinned to cores "
            f"{core_ids[0]}..{core_ids[-1]}) isa={self.isa} "
            f"fmt={fmt_label} "
            f"H={self.H} I={self.I} experts={self.num_experts} layers={self.num_layers} "
            f"top_k={self.top_k} act={activation} max_tokens={self.max_tokens} "
            f"spin_wait={self.spin_wait} "
            f"handshake={'kernel' if self._kernel_io else 'memop' if self._flag_sync else 'host-func'}"
        )

    def _make_table(self, layers: list[torch.Tensor]) -> torch.Tensor:
        """Build a CPU int64 tensor of per-layer base addresses for one bank.

        ``layers`` is one ``[num_experts, ...]`` tensor per layer (the per-layer host
        bank contract). The C++ side stores this table's pointer and indexes
        ``tbl[layer_id]`` at call time; both the table and the layer tensors are kept
        on ``self._banks`` (GC guard) so the raw pointers stay valid for the
        executor's lifetime.
        """
        assert len(layers) == self.num_layers, (len(layers), self.num_layers)
        table = torch.tensor([t.data_ptr() for t in layers], dtype=torch.int64)
        self._banks.append(table)
        self._banks.extend(layers)
        return table

    def _tables(self, **roles: list[torch.Tensor]) -> dict[str, int]:
        """Per-layer table addresses keyed by canonical bank role."""
        return {role: self._make_table(layers).data_ptr() for role, layers in roles.items()}

    def _resolve_fp8_block_banks(self, banks: dict, rep: int = 0) -> tuple[dict, dict, tuple[int, int]]:
        """Block-FP8 schema: row-major e4m3 + one bf16 scale per 128x128 block; the scale banks pad their last dim, so C++ gets the real row stride."""
        gu, gus = banks["gate_up"], banks["gate_up_scale"]
        dn, dns = banks["down"], banks["down_scale"]
        assert gu[rep].element_size() == 1 and dn[rep].element_size() == 1, (gu[rep].dtype, dn[rep].dtype)
        assert gus[rep].dtype == torch.bfloat16 and dns[rep].dtype == torch.bfloat16, (gus[rep].dtype, dns[rep].dtype)
        I = int(gu[rep].shape[1] // 2)
        H = int(gu[rep].shape[2])
        assert gu[rep].shape[1] == 2 * I
        assert tuple(dn[rep].shape[1:]) == (H, I), (dn[rep].shape, H, I)
        assert gus[rep].shape[1] == (2 * I + 127) // 128 and gus[rep].shape[2] >= (H + 127) // 128, (gus[rep].shape, H, I)
        assert dns[rep].shape[1] == (H + 127) // 128 and dns[rep].shape[2] >= (I + 127) // 128, (dns[rep].shape, H, I)
        banks = self._tables(gate_up=gu, down=dn, gate_up_scale=gus, down_scale=dns)
        options = dict(gu_scale_stride=int(gus[rep].stride(1)), dn_scale_stride=int(dns[rep].stride(1)))
        return banks, options, (H, I)

    def _resolve_banks(self, banks: dict, fmt: str, rep: int = 0, opts: dict | None = None) -> tuple[dict, dict, tuple[int, int]]:
        """Return (banks, options, (H, I)) for the C++ ctor: banks maps each canonical role
        the format reads to its per-layer table's address, options holds its integer settings.

        ``banks[name]`` is a list of ``num_layers`` ``[num_experts, ...]`` tensors
        (the per-layer host bank contract); shapes are read from the first layer so
        per-partition (TP) sizes are exact. Every bank address is a per-layer table
        (see ``_make_table``), not a single bank's -- the C++ side resolves
        ``tbl[layer_id]`` per task.
        """
        if fmt in ("gguf", "gguf_scaled"):
            return self._resolve_gguf_banks(banks, rep, int(opts["gate_up_type"]), int(opts["down_type"]), scaled=fmt == "gguf_scaled")
        if fmt == "bf16":
            gate_up = banks["gate_up"]
            down = banks["down"]
            if gate_up[rep].dtype != torch.bfloat16 or down[rep].dtype != torch.bfloat16:
                raise NotImplementedError(
                    f"bf16 CPU MoE requires bf16 banks, got {gate_up[rep].dtype}/{down[rep].dtype}"
                )
            H = int(gate_up[rep].shape[2])
            I = int(gate_up[rep].shape[1] // 2)
            assert gate_up[rep].shape[1] == 2 * I
            assert tuple(down[rep].shape[1:]) == (H, I), (down[rep].shape, H, I)
            banks = self._tables(gate_up=gate_up, down=down)
            return banks, {}, (H, I)

        if fmt == "mxfp4_triton":
            return self._resolve_mxfp4_banks(banks, rep)

        if fmt == "ds_fp4":
            return self._resolve_dsfp4_banks(banks, rep)

        if fmt == "fp8_block":
            return self._resolve_fp8_block_banks(banks, rep)

        # nvfp4: packed e2m1 (2/byte) + fp8-e4m3 per-16 block scales + fp32 row globals.
        gup, gus, gug = banks["gate_up"], banks["gate_up_scale"], banks["gate_up_global"]
        dnp, dns, dng = banks["down"], banks["down_scale"], banks["down_global"]
        assert gup[rep].dtype == torch.uint8 and dnp[rep].dtype == torch.uint8, (gup[rep].dtype, dnp[rep].dtype)
        assert gus[rep].element_size() == 1 and dns[rep].element_size() == 1, "block scales must be 1 byte"
        assert gug[rep].dtype == torch.float32 and dng[rep].dtype == torch.float32, (gug[rep].dtype, dng[rep].dtype)
        I = int(gup[rep].shape[1] // 2)
        H = int(gup[rep].shape[2] * 2)
        assert gup[rep].shape[1] == 2 * I
        assert H % 16 == 0 and I % 16 == 0, (H, I)
        assert tuple(dnp[rep].shape[1:]) == (H, I // 2), (dnp[rep].shape, H, I)
        assert tuple(gus[rep].shape[1:]) == (2 * I, H // 16), (gus[rep].shape, I, H)
        assert tuple(dns[rep].shape[1:]) == (H, I // 16), (dns[rep].shape, H, I)
        assert tuple(gug[rep].shape[1:]) == (2 * I,) and tuple(dng[rep].shape[1:]) == (H,)
        banks = self._tables(gate_up=gup, down=dnp, gate_up_scale=gus, gate_up_global=gug, down_scale=dns, down_global=dng)
        return banks, {}, (H, I)

    def _resolve_gguf_banks(self, banks: dict, rep: int, gate_up_type: int, down_type: int, scaled: bool = False) -> tuple[dict, dict, tuple[int, int]]:
        """GGUF rows read in place: gate_up [E, 2I, row_bytes(H)] (gate rows, then up rows) and down [E, H, row_bytes(I)], one ggml type each; float rows keep their dtype.

        ``scaled`` adds global [E, 4] fp32, each expert's (gate, up, down, 1) NVFP4 weight_scale_2."""
        from freetoken.layers.quantization.ggml_types import float_dtype, row_bytes, type_name

        gate_up, down = banks["gate_up"], banks["down"]
        H, I = int(down[rep].shape[1]), int(gate_up[rep].shape[1] // 2)
        for role, t, ggml_type, k in (("gate_up", gate_up[rep], gate_up_type, H), ("down", down[rep], down_type, I)):
            dtype = float_dtype(ggml_type) if float_dtype(ggml_type) is not None else torch.uint8
            if t.dtype != dtype or t.shape[-1] * t.element_size() != row_bytes(ggml_type, k):
                raise ValueError(f"MoE layer {rep} {role} bank {tuple(t.shape)} {t.dtype} is not {type_name(ggml_type)} rows of {k} values")
        types = {"gate_up_type": gate_up_type, "down_type": down_type}
        if not scaled:
            return self._tables(gate_up=gate_up, down=down), types, (H, I)
        scales = banks["global"]
        if scales[rep].dtype != torch.float32 or tuple(scales[rep].shape) != (gate_up[rep].shape[0], 4) or not scales[rep].is_contiguous():
            raise ValueError(f"MoE layer {rep} global bank {tuple(scales[rep].shape)} {scales[rep].dtype} is not contiguous [experts, 4] fp32")
        return self._tables(gate_up=gate_up, down=down, **{"global": scales}), types, (H, I)

    def _resolve_mxfp4_banks(self, banks: dict, rep: int = 0) -> tuple[dict, dict, tuple[int, int]]:
        """gpt-oss mxfp4 ``mxfp4_triton`` schema: transposed split-K blocks/scales
        (N innermost) + per-output-row biases. The C++ kernel streams K and
        accumulates a contiguous N-block, so the GPU-tiled layout is read in place
        (no repack, no extra host memory). Block scales are e8m0 (1 byte / 32 K)."""
        gub, gus, gob = banks["gate_up"], banks["gate_up_scale"], banks["gate_up_bias"]
        dnb, dns, dob = banks["down"], banks["down_scale"], banks["down_bias"]
        assert gub[rep].dtype == torch.uint8 and dnb[rep].dtype == torch.uint8, (gub[rep].dtype, dnb[rep].dtype)
        assert gus[rep].dtype == torch.uint8 and dns[rep].dtype == torch.uint8, (gus[rep].dtype, dns[rep].dtype)
        assert gob[rep].dtype == torch.bfloat16 and dob[rep].dtype == torch.bfloat16, (gob[rep].dtype, dob[rep].dtype)
        # gate_up_blocks [E, H//2, 2I]; down_blocks [E, I//2, H]
        H = int(gub[rep].shape[1] * 2)
        I = int(gub[rep].shape[2] // 2)
        assert gub[rep].shape[2] == 2 * I
        assert H % 32 == 0 and I % 32 == 0, (H, I)
        assert tuple(dnb[rep].shape[1:]) == (I // 2, H), (dnb[rep].shape, H, I)
        assert tuple(gus[rep].shape[1:]) == (H // 32, 2 * I), (gus[rep].shape, H, I)
        assert tuple(dns[rep].shape[1:]) == (I // 32, H), (dns[rep].shape, H, I)
        assert tuple(gob[rep].shape[1:]) == (2 * I,) and tuple(dob[rep].shape[1:]) == (H,)
        banks = self._tables(gate_up=gub, down=dnb, gate_up_scale=gus, down_scale=dns, gate_up_bias=gob, down_bias=dob)
        return banks, {}, (H, I)

    def _resolve_dsfp4_banks(self, banks: dict, rep: int = 0) -> tuple[dict, dict, tuple[int, int]]:
        """DeepSeek-V4 ``ds_fp4`` schema: row-major e2m1 (2/byte) + e8m0 per-32 block
        scales, no global, no bias. Layout matches nvfp4 (K contiguous per output row),
        so the C++ GEMV reads it in place. The kernel additionally FP8-round-trips the
        activations (the checkpoint's fp8 block) to match the DeepSeek W4A8 reference."""
        gup, gus = banks["gate_up"], banks["gate_up_scale"]
        dnp, dns = banks["down"], banks["down_scale"]
        assert gup[rep].dtype == torch.uint8 and dnp[rep].dtype == torch.uint8, (gup[rep].dtype, dnp[rep].dtype)
        assert gus[rep].element_size() == 1 and dns[rep].element_size() == 1, "block scales must be 1 byte"
        I = int(gup[rep].shape[1] // 2)
        H = int(gup[rep].shape[2] * 2)
        assert gup[rep].shape[1] == 2 * I
        assert H % 32 == 0 and I % 32 == 0, (H, I)  # e8m0 scales per 32 along K
        assert tuple(dnp[rep].shape[1:]) == (H, I // 2), (dnp[rep].shape, H, I)
        assert tuple(gus[rep].shape[1:]) == (2 * I, H // 32), (gus[rep].shape, I, H)
        assert tuple(dns[rep].shape[1:]) == (H, I // 32), (dns[rep].shape, H, I)
        banks = self._tables(gate_up=gup, down=dnp, gate_up_scale=gus, down_scale=dns)
        return banks, {}, (H, I)

    def decode(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """One MoE layer of decode on the CPU. Returns a GPU [bs, H] tensor.

        All ops go on the current CUDA stream so the whole thing is captured into
        the active CUDA graph (the two host nodes carry the data dependency on the
        pinned buffers, which hold this step's real routing on replay)."""
        pending = self.decode_submit(layer_id, hidden_states, topk_weights, topk_ids)
        return self.decode_sync(pending)

    def decode_submit(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple:
        """Issue the D2H copies + the CPU-pool submit host node (or the publish kernel), then return without
        waiting. Lets a caller (the hybrid backend) enqueue GPU work between this and
        :meth:`decode_sync` so the CPU compute overlaps the GPU GEMM / PCIe fetch.

        ``topk_ids`` may carry ``-1`` entries (the C++ kernel skips them), so the CPU
        computes only the routes assigned to it. Returns an opaque handle to pass to
        :meth:`decode_sync`. The output tensor is allocated here so it stays live (and
        distinct from the interleaved GPU work) across the overlap window."""
        bs = hidden_states.shape[0]
        if bs > self.max_tokens:
            raise ValueError(f"CPU MoE decode batch {bs} exceeds the executor's max_tokens {self.max_tokens}")
        if not self._graph_capture_safe and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "CPU MoE decode cannot be captured into a CUDA graph here: the stream-memop "
                "doorbell is off or failed its startup probe, and the host-function sync is not "
                "replay-safe on this runtime; run with --cuda-graph-max-bs 0"
            )

        if self._gpu_prequant:
            # DSV4: apply the reference FP8 round-trip on the GPU (the same kernel the
            # GPU W4A8 path uses -> bit-identical grid) so the CPU side reads
            # pre-quantized activations and skips its serial scalar pass.
            from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_roundtrip

            hidden_states = act_quant_fp8_roundtrip(hidden_states, block=self._act_block)

        stream = torch.cuda.current_stream().cuda_stream
        if self._kernel_io and hidden_states.dtype == torch.bfloat16:
            x = hidden_states.contiguous()
            ids = topk_ids.to(torch.int32).contiguous()
            w = topk_weights.to(torch.float32).contiguous()
            assert x.shape == (bs, self.H) and ids.shape == w.shape == (bs, self.top_k), (x.shape, ids.shape, w.shape)
            self._ext.flag_publish(stream, x.data_ptr(), ids.data_ptr(), w.data_ptr(), layer_id, bs)
            return (bs, torch.empty_like(x), True)

        # D2H: ship this step's activations + routing to pinned host memory.
        io = self._io
        io["x"][:bs].copy_(hidden_states, non_blocking=True)
        io["ids"][:bs].copy_(topk_ids.to(torch.int32), non_blocking=True)
        io["w"][:bs].copy_(topk_weights.to(torch.float32), non_blocking=True)

        out = torch.empty_like(hidden_states)
        if self._flag_sync:
            # Front-end memops: reset done, then ring the doorbell with (layer, bs).
            # No kernel launched; no host-func round trip.
            self._ext.flag_signal(stream, layer_id, bs)
        else:
            self._ext.submit_with_cuda_stream(stream, layer_id, bs)
        return (bs, out, False)

    def decode_sync(self, pending: tuple, add_to: torch.Tensor | None = None) -> torch.Tensor:
        """Issue the CPU-pool sync + the H2D result copy for a prior :meth:`decode_submit`,
        and return the GPU output tensor. With flag-sync the wait is a front-end stream
        memop on done (set by the CPU coordinator); otherwise a cudaLaunchHostFunc.
        With ``add_to``, return ``add_to`` + the CPU output; the kernel path adds in place into ``add_to``."""
        bs, out, kernel_io = pending
        stream = torch.cuda.current_stream().cuda_stream
        if self._flag_sync:
            # Front-end WAIT(done >= 1): blocks this stream's later nodes without
            # occupying an SM, so GPU utilization stays truthful during the CPU window.
            self._ext.flag_wait(stream)
        else:
            self._ext.sync_with_cuda_stream(stream)
        if kernel_io:
            if add_to is not None and add_to.dtype == torch.bfloat16 and add_to.is_contiguous() and add_to.shape == out.shape:
                self._ext.flag_combine(stream, add_to.data_ptr(), bs, True)
                return add_to
            self._ext.flag_combine(stream, out.data_ptr(), bs, False)
            return out if add_to is None else add_to + out
        out.copy_(self._io["y"][:bs], non_blocking=True)
        return out if add_to is None else add_to + out

    def _watchdog_tick(self, suspect: dict) -> None:
        """One watchdog sampling round (called every 2 s by ``_watchdog_main``).

        The doorbell is only declared dead when THREE things hold across >=10 s: it is
        still pending (ready set, done==0), it was already pending when first suspected, and
        the coordinator has served NOTHING since (flag_served unchanged). The served-count
        criterion kills the false-positive window: two point samples can land on
        (independent, us-scale) pending windows under heavy external load, but a
        coordinator that made progress in between is alive by definition. ``suspect``
        holds (first_seen, served_at_first_sight) and persists across ticks."""
        if not self._ext.flag_pending():
            suspect.clear()
            return
        now = time.monotonic()
        served = self._ext.flag_served()
        if not suspect or suspect["served"] != served:
            suspect.update(first_seen=now, served=served)  # new suspect, or alive-but-loaded: rearm
            return
        if now - suspect["first_seen"] < 10.0:
            return
        logger.error(
            "cpu-moe flag watchdog: doorbell unanswered for >10s with no coordinator "
            "progress (wedged/dead); poisoning done and failing the next step"
        )
        self._unhealthy = True
        self._ext.poison_flag()  # after the mark: unblock the stream into a checked failure
        suspect.clear()

    def raise_if_unhealthy(self) -> None:
        """Raise if the flag watchdog fired (a doorbell stayed unanswered because the
        coordinator never responded). Called by the engine once per forward -- a single
        flag read -- so a dead coordinator surfaces as a loud error on the next step
        instead of silently shipping stale expert outputs."""
        if self._unhealthy:
            raise RuntimeError(
                "CPU MoE flag-handshake watchdog fired: a decode step's doorbell was "
                "never answered by the coordinator thread (its outputs cannot be "
                "trusted). This indicates a wedged/killed coordinator; restart the "
                "engine, or set FREETOKEN_CPU_MOE_FLAG_SYNC=0 to use the "
                "cudaLaunchHostFunc sync."
            )


def _watchdog_main(executor_ref) -> None:
    """Watchdog daemon body: weakref-deref per tick so the thread never keeps a dead
    executor alive (see the start site in ``CpuMoeExecutor.__init__``)."""
    suspect: dict = {}
    while True:
        time.sleep(2.0)
        executor = executor_ref()
        if executor is None or executor._watchdog_stop:
            return
        try:
            executor._watchdog_tick(suspect)
        finally:
            del executor  # drop the strong ref before the next sleep
