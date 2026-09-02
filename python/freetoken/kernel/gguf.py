"""Borrowed llama.cpp GGUF dequant/GEMM CUDA kernels, JIT-compiled on first use.

The ``.cu``/``.cuh`` under ``csrc/gguf/`` are vendored verbatim from sgl-kernel
(``csrc/quantization/gguf/``), which are themselves ports of llama.cpp. We compile
them through ``torch.utils.cpp_extension.load`` (the same toolchain sglang/vllm use)
into a torch-op module and expose the handful of ops the GGUF path needs. This is a
separate, torch-native extension that sits alongside FreeToken's tvm-ffi kernels.

All ops keep the weight in its native GGUF block layout (packed ``uint8`` rows) and
dequantize *inside* the kernel -- no bf16 copy of the weight is ever materialized.
"""

from __future__ import annotations

import functools
import os
import pathlib
import shutil
import time

import torch

_CSRC = pathlib.Path(__file__).parent / "csrc" / "gguf"

_GGML_QUANT_NAMES = {
    2: "Q4_0", 8: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K"
}
_GGUF_QUANT_TYPES = frozenset(_GGML_QUANT_NAMES)
_DISPATCH_COUNTS: dict[tuple, int] = {}
# Algorithm subset pinned to llama.cpp b10434 (commit 7e4c0a968). The local HIP ABI
# wrapper remains separate, so provenance does not imply full source replacement.
_GGUF_SOURCE_VERSION = "llama.cpp-7e4c0a968-q4q5q6q8-subset-v1"
_GGUF_ABI_VERSION = "moe-abi-v2"
_MMVQ_BS1_ABI_VERSION = "llama.cpp-b10434-mmvq-bs1-v1"
_CANDIDATE_MODULE_NAME = "freetoken_gguf_moe_gfx1100_v6"
_GGUF_MOE_ABI: dict[str, str] = {
    "ggml_moe_a8_vec": _GGUF_ABI_VERSION,
    "ggml_moe_a8_vec_strided": _GGUF_ABI_VERSION,
}


def register_gguf_moe_abi(callables: dict[str, str], *, abi_version: str) -> None:
    """Register extension-bound MoE callables; metadata never advertises guesses."""
    if abi_version != _GGUF_ABI_VERSION:
        raise ValueError(f"unsupported GGUF MoE ABI {abi_version!r}")
    for name, version in callables.items():
        if version == _GGUF_ABI_VERSION:
            _GGUF_MOE_ABI[name] = version


def mmvq_bs1_workspace_bytes(hidden: int, rows: int, channels: int) -> int:
    """Exact caller-owned workspace formula for the b10434 single-token ABI."""
    hidden, rows, channels = int(hidden), int(rows), int(channels)
    if min(hidden, rows, channels) <= 0:
        raise ValueError("MMVQ b10434 shape must be positive")
    if hidden % 32:
        raise ValueError("MMVQ b10434 hidden dimension must be divisible by 32")
    align = lambda value: (value + 255) // 256 * 256
    return align(hidden // 32 * 36) + align(channels * rows * 4)


def validate_mmvq_bs1_workspace(hidden: int, rows: int, channels: int, workspace_bytes: int) -> None:
    required = mmvq_bs1_workspace_bytes(hidden, rows, channels)
    if int(workspace_bytes) < required:
        raise ValueError(f"MMVQ workspace too small: {workspace_bytes} < {required}")
_GGUF_BLOCK_SIZE = {2: 32, 8: 32, 12: 256, 13: 256, 14: 256}
_RDNA3_MOE_MMVQ_MAX = {2: 4, 8: 4, 12: 4, 13: 4, 14: 4}


def _runtime_backend() -> str:
    if torch.version.hip is not None:
        return "rocm"
    if torch.version.cuda is not None:
        return "cuda"
    return "cpu"


def _runtime_arch() -> str | None:
    forced = os.environ.get("FREETOKEN_KERNEL_CACHE_GFX")
    if forced:
        return forced
    if _runtime_backend() == "rocm" and torch.cuda.is_available():
        try:
            return torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
        except (AttributeError, RuntimeError):
            pass
    return None


def gguf_runtime_metadata() -> dict:
    """Report GGUF JIT/runtime selection without compiling or mutating state."""
    backend = _runtime_backend()
    arch = _runtime_arch()
    flags = ["-O3"]
    if backend == "rocm":
        flags.extend(["USE_HIP=1", "USE_ROCM=1", f"offload-arch={arch or 'unknown'}"])
    elif backend == "cuda":
        flags.append("expt-relaxed-constexpr")
    return {
        "backend": backend,
        "arch": arch,
        "device": torch.cuda.get_device_name(torch.cuda.current_device())
        if torch.cuda.is_available()
        else None,
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "compile_flags": flags,
        "source_version": os.environ.get("FREETOKEN_GGUF_SOURCE_VERSION", _GGUF_SOURCE_VERSION),
        "source": "gguf_kernel.cu",
    }


def _record_dispatch(report: dict) -> None:
    if os.environ.get("FREETOKEN_GGUF_DISPATCH_TRACE", "").lower() not in {"1", "true", "yes", "on"}:
        return
    key = tuple(
        report[field]
        for field in ("backend", "arch", "quant_type", "op", "implementation", "rows", "cols", "tokens")
    )
    _DISPATCH_COUNTS[key] = _DISPATCH_COUNTS.get(key, 0) + 1


def gguf_dispatch_report() -> list[dict]:
    """Return aggregate dispatch observations collected when trace is enabled."""
    fields = ("backend", "arch", "quant_type", "op", "implementation", "rows", "cols", "tokens")
    return [dict(zip(fields, key), calls=calls) for key, calls in _DISPATCH_COUNTS.items()]


def _arch_family(arch: str | None) -> str:
    """Map GPU target names to b10434's kernel tuning families."""
    value = (arch or "").lower()
    if value.startswith(("gfx11", "gfx12")):
        return "rdna3" if value.startswith("gfx11") else "rdna4"
    if value.startswith("gfx10"):
        return "rdna2"
    if value.startswith("gfx9"):
        return "cdna"
    if value.startswith("sm"):
        return "nvidia"
    return "generic"


def gguf_dispatch(
    op: str,
    quant_type: int,
    rows: int,
    cols: int,
    tokens: int,
    arch: str | None,
    impl: str | None = None,
) -> dict:
    """Resolve observable GGUF operation family without launching a kernel."""
    requested = (impl or os.environ.get("FREETOKEN_GGUF_MOE_IMPL", "legacy")).strip().lower()
    if requested == "gfx1100":
        requested = "rdna3_mmid"
    if requested not in {
        "auto", "legacy", "mmvq", "mmq", "rdna3_mmid", "rdna3_mmvdq", "grouped_mmq"
    }:
        raise ValueError(f"unsupported GGUF implementation {requested!r}")
    # FREETOKEN_GGUF_MOE_IMPL is a MoE-only selector. Dense GGUF projections
    # (including lm_head) must retain normal shape policy; otherwise forcing a
    # MoE candidate silently sends dense layers through full dequantization.
    if impl is None and op not in {"moe_decode", "moe", "moe_prefill", "grouped_prefill"}:
        requested = "auto"
    backend = _runtime_backend()
    runtime = gguf_runtime_metadata()
    report = {
        "backend": backend,
        "arch": arch,
        "quant_type": _GGML_QUANT_NAMES.get(quant_type, f"unknown:{quant_type}"),
        "op": op,
        "rows": rows,
        "cols": cols,
        "tokens": tokens,
        "implementation": "unsupported",
        "requested": requested,
        "compile_flags": runtime["compile_flags"],
        "source_version": runtime["source_version"],
        "abi_version": _GGUF_ABI_VERSION,
        "library": "gguf-jit",
        "stream": "current",
        "id_space": "declared-by-caller",
        "callable": None,
        "reason": None,
    }
    if rows <= 0 or cols <= 0 or tokens <= 0:
        report["reason"] = "non-positive shape"
    elif quant_type not in _GGUF_QUANT_TYPES:
        report["reason"] = "unsupported quantization type"
    elif cols % _GGUF_BLOCK_SIZE[quant_type]:
        report["reason"] = f"K dimension {cols} is not aligned to {_GGUF_BLOCK_SIZE[quant_type]}"
    elif backend not in {"cuda", "rocm"}:
        report["reason"] = "GPU backend unavailable"
    elif backend == "rocm" and arch and arch.startswith("sm"):
        report["reason"] = "NVIDIA architecture requested on ROCm"
    elif backend == "cuda" and arch and arch.startswith("gfx"):
        report["reason"] = "AMD architecture requested on CUDA"
    elif op == "moe_prefill":
        # b10434 uses the dedicated grouped MUL_MAT_ID path once the batch exceeds the
        # architecture/type MMVQ window. The old local HIP grouped ABI has produced
        # launch failures on gfx1100 for the Qwen3.6 prefill shape, so keep this opt-in
        # until its exact model-shape A/B and error-free replay gate pass. The vector path
        # is slower but proven and remains the fail-closed default.
        family = _arch_family(arch)
        limit = _RDNA3_MOE_MMVQ_MAX.get(quant_type, 8) if family == "rdna3" else 8
        grouped = os.environ.get("FREETOKEN_GGUF_GROUPED_PREFILL", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        report["implementation"] = "ggml_moe_a8" if tokens > limit and grouped else "ggml_moe_a8_vec"
        report["callable"] = report["implementation"]
        report["reason"] = (
            "multi-token grouped path"
            if tokens > limit and grouped
            else ("grouped path disabled after gfx1100 launch failure" if tokens > limit else None)
        )
    elif op == "grouped_prefill":
        report["implementation"] = "ggml_moe_a8"
        report["callable"] = report["implementation"]
    elif op in {"moe_decode", "moe"}:
        candidate_callable = {
            "rdna3_mmid": "ggml_moe_mmvq_id",
            "rdna3_mmvdq": "ggml_moe_mmvdq_id",
            "grouped_mmq": "ggml_moe_mmq_id_strided",
        }.get(requested)
        candidate_error = None
        if requested in {"rdna3_mmid", "rdna3_mmvdq"} and backend == "rocm" and arch == "gfx1100":
            try:
                ensure_gguf_moe_candidate_ready()
            except Exception as exc:
                candidate_error = exc
                report["reason"] = f"{requested} compile/self-test failed: {type(exc).__name__}"
        if candidate_error is not None and requested not in {"auto", "legacy"}:
            raise RuntimeError(report["reason"]) from candidate_error
        # Candidate bindings are registered only after extension load. Until
        # then explicit opt-in fails closed to the proven legacy callable.
        shape_supported = requested != "rdna3_mmvdq" or quant_type in {12, 13, 14}
        available = bool(
            shape_supported
            and candidate_callable
            and _GGUF_MOE_ABI.get(candidate_callable) == _GGUF_ABI_VERSION
        )
        if available:
            report["implementation"] = requested
            report["callable"] = candidate_callable
        else:
            report["implementation"] = "ggml_moe_a8_vec"
            report["callable"] = "ggml_moe_a8_vec"
            if requested not in {"auto", "legacy"}:
                raise RuntimeError(
                    f"{requested} ABI callable unavailable; forced mode refuses legacy fallback"
                )
    elif op in {"dense", "linear", "lm_head"}:
        family = "mmvq" if tokens <= 8 else "mmq"
        if requested not in {"auto", "legacy"} and requested != family:
            report["reason"] = f"forced {requested} conflicts with shape policy {family}"
        else:
            report["implementation"] = "ggml_mul_mat_vec_a8" if family == "mmvq" else "ggml_mul_mat_a8"
    else:
        report["reason"] = "unsupported operation"
    _record_dispatch(report)
    return report


def _host_compiler() -> str | None:
    """A host compiler nvcc + libtorch headers accept.

    The system default gcc can be too new for the torch headers (gcc 16 hard-errors),
    and on this toolchain even nvcc+gcc-13 trips a non-conformant ``typename
    decltype`` in ``List_inl.h`` once ``torch::Tensor`` is instantiated -- but nvcc
    with ``clang++`` as host compiles it cleanly. So prefer clang++, then fall back
    to an older gcc. Override with ``FREETOKEN_GGUF_HOST_CXX``.
    """
    override = os.environ.get("FREETOKEN_GGUF_HOST_CXX")
    if override:
        return override
    for cxx in ("clang++", "g++-13", "g++-14", "g++-15"):
        if shutil.which(cxx):
            return cxx
    return None


def _c_compiler_for(cxx: str) -> str:
    base = os.path.basename(cxx)
    if "clang" in base:
        return shutil.which("clang") or "clang"
    cc = base.replace("g++", "gcc")
    return shutil.which(cc) or cc


# A JIT-rebuild lock older than this is stale even while its owner lives: no honest
# rebuild of one extension takes hours (the slowest measured gguf_kernel build is
# ~13 min on the gfx1100 box).
_STALE_JIT_LOCK_AGE_S = 3 * 3600


def _clear_stale_jit_lock(module_name: str) -> None:
    """Remove a torch-extension ``lock`` whose owner died.

    torch's ``FileBaton`` waits forever when a previous compile was ``kill -9``-ed
    mid-rebuild (observed: every later serve hung in warmup at
    ``cpp_extension.py _jit_compile -> wait``; the lock file has no owner pid and
    no fd is held on it, so there is nothing to poll). Only the lock's age can
    distinguish stale from live: a live rebuild's lock is minutes old; no honest
    build of one extension takes hours (slowest measured gguf_kernel build ~13 min
    on the gfx1100 box), so _STALE_JIT_LOCK_AGE_S (3 h) is the staleness bar. A
    freshly-created lock is never touched, so a genuinely concurrent rebuild is
    not clobbered.
    """
    try:
        build_dir = pathlib.Path(
            torch.utils.cpp_extension._get_build_directory(module_name, False)
        )
        lock = build_dir / "lock"
        if not lock.exists():
            return
        age = time.time() - lock.stat().st_mtime
        if age < _STALE_JIT_LOCK_AGE_S:
            return
        lock.unlink()
    except Exception:  # noqa: BLE001 - hygiene must never break the build path
        pass


def _rocm_jit_source(module_name: str, source: pathlib.Path) -> pathlib.Path:
    """Prepare ROCm source in build dir without letting PyTorch rewrite repo files.

    PyTorch's HIP extension path runs hipify on CUDA sources and recursively rewrites
    included headers. That can overwrite checked-in ``*_hip`` files when an absolute
    repository source is supplied. A cache-local ``.cu`` copy keeps hipify's source
    and generated output in the extension cache; ``-I`` resolves the tracked HIP
    headers selected by the source.
    """
    from torch.utils.cpp_extension import _get_build_directory

    build_dir = pathlib.Path(_get_build_directory(module_name, False))
    build_dir.mkdir(parents=True, exist_ok=True)
    target = build_dir / source.name
    target.write_text(source.read_text())
    return target


@functools.cache
def _module():
    from torch.utils.cpp_extension import load

    if torch.version.hip is not None:
        # ROCm: hipcc (torch.utils.cpp_extension picks it up), pass the HIP defines so
        # the kernels compile their HIP branches; drop the CUDA-only -ccbin/flag logic.
        # Explicit --offload-arch (plus PYTORCH_ROCM_ARCH) prevents torch from auto-
        # emitting ~14 gfx arches, which would multiply build time per arch.
        gfx = os.getenv("FREETOKEN_KERNEL_CACHE_GFX", "gfx1100")
        os.environ.setdefault("PYTORCH_ROCM_ARCH", gfx)
        extra_cuda_cflags = [
            "-O3", f"--offload-arch={gfx}", "-DUSE_HIP=1", "-DUSE_ROCM=1",
            "-DFREETOKEN_GGUF_SOURCE_VERSION=2",
            f"-I{_CSRC}",
        ]
        os.environ.pop("CXX", None)
        os.environ.pop("CC", None)
        source = _rocm_jit_source("freetoken_gguf_kernels", _CSRC / "gguf_kernel.cu")
        include_paths = []
    else:
        extra_cuda_cflags = [
            "-O3",
            "--expt-relaxed-constexpr",
            "-DFREETOKEN_GGUF_SOURCE_VERSION=2",
        ]
        host_cxx = _host_compiler()
        if host_cxx is not None:
            # Point both nvcc's host pass (-ccbin) and torch's C++ compile (CXX) at a
            # libtorch/nvcc-compatible compiler. Force (not setdefault): the system
            # default (CXX unset -> g++) can be a gcc too new for the torch headers.
            cxx_path = shutil.which(host_cxx) or host_cxx
            extra_cuda_cflags += ["-ccbin", cxx_path]
            os.environ["CXX"] = cxx_path
            os.environ["CC"] = _c_compiler_for(cxx_path)
        source = _CSRC / "gguf_kernel.cu"
        include_paths = [str(_CSRC)]

    # Rows-per-warp for the MMVQ/MoE-vec launches (Inc 4, .plans/rocm-perf-parity).
    # Overridable for tuning/AB; the JIT cache keys on the cflags, so a changed value
    # rebuilds cleanly. CUDA-side the same -D reaches ggml-common.h's #ifndef guard.
    mmv_y = os.getenv("FREETOKEN_GGUF_MMV_Y", "").strip()
    if mmv_y:
        if not mmv_y.isdigit() or int(mmv_y) not in (1, 2, 4, 8):
            raise ValueError(
                f"FREETOKEN_GGUF_MMV_Y={mmv_y!r}: expected one of 1, 2, 4, 8"
            )
        extra_cuda_cflags.append(f"-DGGML_CUDA_MMV_Y={mmv_y}")

    # gguf_kernel.cu carries its own PYBIND11_MODULE (appended at the end), so a
    # plain `load` of the single source compiles + binds the ggml_* ops.
    _clear_stale_jit_lock("freetoken_gguf_kernels")
    module = load(
        name="freetoken_gguf_kernels",
        sources=[str(source)],
        extra_include_paths=include_paths,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
    )
    register_gguf_moe_abi(
        {
            name: _GGUF_ABI_VERSION
            for name in (
                "ggml_moe_a8_vec",
                "ggml_moe_a8_vec_strided",
                "ggml_moe_a8_vec_workspace",
                "ggml_moe_a8_vec_strided_workspace",
            )
            if hasattr(module, name)
        },
        abi_version=_GGUF_ABI_VERSION,
    )
    return module


_candidate_ready = False


def _gguf_moe_impl() -> str:
    impl = os.environ.get("FREETOKEN_GGUF_MOE_IMPL", "legacy").strip().lower()
    if impl not in {
        "auto", "legacy", "gfx1100", "rdna3_mmid", "rdna3_mmvdq", "grouped_mmq"
    }:
        raise ValueError(
            f"FREETOKEN_GGUF_MOE_IMPL={impl!r}: expected auto, legacy, rdna3_mmid, "
            "rdna3_mmvdq, grouped_mmq, or gfx1100"
        )
    return impl


def _gfx1100_supported() -> bool:
    if torch.version.hip is None or not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName == "gfx1100"
    except AttributeError:
        return False


@functools.cache
def _candidate_module():
    from torch.utils.cpp_extension import load

    gfx = os.getenv("FREETOKEN_KERNEL_CACHE_GFX", "gfx1100")
    if gfx != "gfx1100":
        raise RuntimeError(f"gfx1100 candidate requires FREETOKEN_KERNEL_CACHE_GFX=gfx1100, got {gfx!r}")
    os.environ.setdefault("PYTORCH_ROCM_ARCH", gfx)
    os.environ.pop("CXX", None)
    os.environ.pop("CC", None)
    extra_cuda_cflags = [
        "-O3",
        f"--offload-arch={gfx}",
        "-DUSE_HIP=1",
        "-DUSE_ROCM=1",
        "-DFREETOKEN_GGUF_MOE_SOURCE_VERSION=2",
        "-DFREETOKEN_GGUF_SOURCE_VERSION=2",
        f"-I{_CSRC}",
    ]
    mmv_y = os.getenv("FREETOKEN_GGUF_MMV_Y", "1").strip()
    if not mmv_y.isdigit() or int(mmv_y) not in (1, 2, 4, 8):
        raise ValueError(
            f"FREETOKEN_GGUF_MMV_Y={mmv_y!r}: expected one of 1, 2, 4, 8"
        )
    extra_cuda_cflags.append(f"-DGGML_CUDA_MMV_Y={mmv_y}")
    _clear_stale_jit_lock(_CANDIDATE_MODULE_NAME)
    source = _rocm_jit_source(
        _CANDIDATE_MODULE_NAME, _CSRC / "gguf_moe_gfx1100.cu"
    )
    abi_source = _rocm_jit_source(
        _CANDIDATE_MODULE_NAME, _CSRC / "gguf_b10434_kernel.cu"
    )
    extra_cuda_cflags.append("-DFREETOKEN_GGUF_NO_PYBIND=1")
    module = load(
        name=_CANDIDATE_MODULE_NAME,
        sources=[str(source), str(abi_source)],
        extra_include_paths=[],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
    )
    register_gguf_moe_abi(
        {
            name: _GGUF_ABI_VERSION
            for name in (
                "ggml_moe_mmvq_id", "ggml_moe_mmvq_id_workspace",
                "ggml_moe_mmvdq_id", "ggml_moe_mmvdq_id_workspace",
                "ggml_moe_gate_up_swiglu_id", "ggml_moe_gate_up_swiglu_id_workspace",
                "mmvq_bs1", "mmvq_bs1_workspace_bytes",
            )
            if hasattr(module, name)
        },
        abi_version=_GGUF_ABI_VERSION,
    )
    return module


def _candidate_self_test(module) -> None:
    """Exercise both candidate quant paths and synchronize before graph capture."""
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(1100)
    ids = torch.arange(8, dtype=torch.int32).reshape(1, 8).to(device)
    x = torch.randn(1, 256, generator=generator, dtype=torch.bfloat16).to(device)
    q4 = torch.zeros((8, 16, 144), dtype=torch.uint8, device=device)
    q4[..., :4] = torch.tensor([128, 63, 128, 63], dtype=torch.uint8, device=device)
    q4[..., 4:16] = torch.randint(1, 64, (8, 16, 12), generator=generator, dtype=torch.uint8).to(device)
    q4[..., 16:] = torch.randint(0, 255, (8, 16, 128), generator=generator, dtype=torch.uint8).to(device)
    q4_out = module.ggml_moe_a8_vec_gfx1100(x, q4, ids, 8, 12, 16, 1)
    inter = torch.randn(8, 128, generator=generator, dtype=torch.bfloat16).to(device)
    q8_blocks = torch.zeros((8, 16, 4, 34), dtype=torch.uint8, device=device)
    q8_blocks[..., :2] = torch.tensor([128, 63], dtype=torch.uint8, device=device)
    q8_blocks[..., 2:] = torch.randint(
        0, 255, (8, 16, 4, 32), generator=generator, dtype=torch.uint8
    ).to(device)
    q8 = q8_blocks.reshape(8, 16, 136)
    q8_out = module.ggml_moe_a8_vec_gfx1100(inter, q8, ids, 1, 8, 16, 8)
    route_ids = ids.reshape(-1, 1)
    id_out = torch.empty((8, 16), dtype=inter.dtype, device=device)
    id_qx = torch.empty((8, 144), dtype=torch.int32, device=device)
    id_out = module.ggml_moe_mmvq_id_workspace(
        inter, q8, route_ids, 1, 8, 16, 8,
        int(q8.stride(0)), int(q8.stride(1)), "slot", id_out, id_qx
    )
    mmvdq_out = module.ggml_moe_mmvdq_id(
        x, q4, ids, 8, 12, 16, 1,
        int(q4.stride(0)), int(q4.stride(1)), "slot"
    )
    fused_out = module.ggml_moe_gate_up_swiglu_id_workspace(
        x, q4, ids, 8, 8, 1,
        int(q4.stride(0)), int(q4.stride(1)), "slot",
        torch.empty((8, 8), dtype=x.dtype, device=device),
        torch.empty((1, 144), dtype=torch.int32, device=device),
    )
    # Exercise the pinned b10434 caller-owned ABI separately from the model-facing
    # BF16 candidate output. This catches module binding and workspace slicing before
    # graph capture, without allocating inside the measured model path.
    abi_x = torch.randn(1, 512, generator=generator, dtype=torch.bfloat16).to(device)
    abi_q4 = torch.zeros((8, 16, 288), dtype=torch.uint8, device=device)
    abi_q4[..., :4] = torch.tensor([128, 63, 128, 63], dtype=torch.uint8, device=device)
    abi_q4[..., 4:] = 1
    abi_workspace = torch.empty(
        mmvq_bs1_workspace_bytes(512, 16, 8), dtype=torch.uint8, device=device
    )
    abi_out = torch.empty((8, 16), dtype=torch.float32, device=device)
    abi_result = module.mmvq_bs1(abi_x, abi_q4, abi_out, abi_workspace, 12, 16, 8, ids)
    torch.cuda.synchronize(device)
    if (
        not torch.isfinite(q4_out).all()
        or not torch.isfinite(q8_out).all()
        or not torch.isfinite(id_out).all()
        or not torch.isfinite(mmvdq_out).all()
        or not torch.isfinite(fused_out).all()
        or not torch.isfinite(abi_result).all()
    ):
        raise RuntimeError("gfx1100 candidate self-test produced non-finite output")


def ensure_gguf_moe_candidate_ready() -> bool:
    """Compile and validate forced candidate before graph capture."""
    global _candidate_ready
    impl = _gguf_moe_impl()
    # Candidate remains forced-only until a later promotion is backed by measured evidence.
    if impl not in {"gfx1100", "rdna3_mmid", "rdna3_mmvdq"}:
        return False
    if _candidate_ready:
        return True
    try:
        module = _candidate_module()
        _candidate_self_test(module)
        _candidate_ready = True
        return True
    except Exception as exc:
        raise RuntimeError("forced gfx1100 GGUF MoE candidate failed compile/self-test") from exc


def _moe_module():
    impl = _gguf_moe_impl()
    if impl not in {"gfx1100", "rdna3_mmid", "rdna3_mmvdq"}:
        return None
    if not _gfx1100_supported():
        raise RuntimeError("forced gfx1100 GGUF MoE candidate requires ROCm gfx1100")
    if not ensure_gguf_moe_candidate_ready():
        return None
    return _candidate_module()


# ---- thin typed wrappers (signatures mirror sgl_kernel.quantization.gguf) ----


def ggml_dequantize(
    weight: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Dequantize a packed GGUF weight ``[m, row_bytes]`` to a dense ``[m, n]`` tensor."""
    return _module().ggml_dequantize(weight, quant_type, m, n, dtype)


def ggml_mul_mat_vec_a8(
    weight: torch.Tensor, x: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    """MMVQ: small-batch GEMV with on-the-fly dequant. ``row`` = output features."""
    return _module().ggml_mul_mat_vec_a8(weight, x, quant_type, row)


def ggml_mul_mat_a8(
    weight: torch.Tensor, x: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    """MMQ: large-batch quantized matmul. ``row`` = output features."""
    return _module().ggml_mul_mat_a8(weight, x, quant_type, row)


def ggml_moe_a8(
    x: torch.Tensor,
    weight: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    """MMQ grouped expert matmul over stacked experts ``weight[E, row, *]``."""
    return _module().ggml_moe_a8(
        x, weight, sorted_token_ids, expert_ids, num_tokens_post_padded,
        quant_type, row, top_k, tokens,
    )


def ggml_moe_a8_vec(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """MMVQ grouped expert GEMV over stacked experts ``weight[E, row, *]``."""
    candidate = _moe_module()
    if output is not None:
        if candidate is not None:
            raise ValueError("reusable output is unsupported by gfx1100 candidate")
        return _module().ggml_moe_a8_vec(
            x, weight, topk_ids, top_k, quant_type, row, tokens, output
        )
    if candidate is not None:
        if quant_type in (8, 12):
            return candidate.ggml_moe_a8_vec_gfx1100(
                x, weight, topk_ids, top_k, quant_type, row, tokens
            )
        # The candidate's ID-aware ABI owns native Q5_K/Q6_K rows and explicit
        # strides; never send those lanes through its older compact Q4/Q8 helper.
        return ggml_moe_mmvq_id(
            x, weight, topk_ids, top_k, quant_type, row, tokens,
            int(weight.stride(0)), int(weight.stride(1)), "slot",
        )
    return _module().ggml_moe_a8_vec(
        x, weight, topk_ids, top_k, quant_type, row, tokens
    )


def ggml_moe_a8_vec_strided(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
    expert_stride_bytes: int,
    row_stride_bytes: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Native Q5_K/Q6_K MoE GEMV over rows padded to a uniform Q6_K stride."""
    if output is None:
        return _module().ggml_moe_a8_vec_strided(
            x, weight, topk_ids, top_k, quant_type, row, tokens,
            expert_stride_bytes, row_stride_bytes
        )
    return _module().ggml_moe_a8_vec_strided(
        x, weight, topk_ids, top_k, quant_type, row, tokens,
        expert_stride_bytes, row_stride_bytes, output
    )


def ggml_moe_mmvq_id(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
    expert_stride_bytes: int,
    row_stride_bytes: int,
    id_space: str,
    output: torch.Tensor | None = None,
    quant_x: torch.Tensor | None = None,
) -> torch.Tensor:
    """Opt-in gfx1100 ID-aware MMVQ over raw or cache-slot IDs."""
    candidate = _moe_module()
    if candidate is None or not hasattr(candidate, "ggml_moe_mmvq_id"):
        raise RuntimeError("RDNA3 ID-aware GGUF MoE ABI is unavailable")
    args = (
        x, weight, topk_ids, top_k, quant_type, row, tokens,
        expert_stride_bytes, row_stride_bytes, id_space,
    )
    if output is not None and quant_x is not None:
        return candidate.ggml_moe_mmvq_id_workspace(*args, output, quant_x)
    if output is not None or quant_x is not None:
        raise ValueError("ggml_moe_mmvq_id requires output and quant_x together")
    return candidate.ggml_moe_mmvq_id(*args)


def ggml_moe_mmvdq_id(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
    expert_stride_bytes: int,
    row_stride_bytes: int,
    id_space: str,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Opt-in gfx1100 direct-float MMVDQ over native Q4_K/Q5_K/Q6_K rows."""
    candidate = _moe_module()
    if candidate is None or not hasattr(candidate, "ggml_moe_mmvdq_id"):
        raise RuntimeError("RDNA3 direct-float GGUF MoE ABI is unavailable")
    args = (
        x, weight, topk_ids, top_k, quant_type, row, tokens,
        expert_stride_bytes, row_stride_bytes, id_space,
    )
    if output is None:
        return candidate.ggml_moe_mmvdq_id(*args)
    return candidate.ggml_moe_mmvdq_id_workspace(*args, output)


def ggml_moe_gate_up_swiglu_id(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    nrows: int,
    tokens: int,
    expert_stride_bytes: int,
    row_stride_bytes: int,
    id_space: str,
    output: torch.Tensor | None = None,
    quant_x: torch.Tensor | None = None,
) -> torch.Tensor:
    """Opt-in gfx1100 fused Q4_K gate/up plus SwiGLU decode operation."""
    candidate = _moe_module()
    if candidate is None or not hasattr(candidate, "ggml_moe_gate_up_swiglu_id"):
        raise RuntimeError("RDNA3 fused gate/up GGUF MoE ABI is unavailable")
    args = (
        x, weight, topk_ids, top_k, nrows, tokens,
        expert_stride_bytes, row_stride_bytes, id_space,
    )
    if output is not None and quant_x is not None:
        return candidate.ggml_moe_gate_up_swiglu_id_workspace(*args, output, quant_x)
    if output is not None or quant_x is not None:
        raise ValueError("ggml_moe_gate_up_swiglu_id requires output and quant_x together")
    return candidate.ggml_moe_gate_up_swiglu_id(*args)


def ggml_moe_a8_vec_workspace(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
    output: torch.Tensor,
    quant_x: torch.Tensor,
) -> torch.Tensor:
    """Legacy MMVQ using caller-owned output and Q8_1 scratch tensors."""
    return _module().ggml_moe_a8_vec_workspace(
        x, weight, topk_ids, top_k, quant_type, row, tokens, output, quant_x
    )


def ggml_moe_a8_vec_strided_workspace(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
    expert_stride_bytes: int,
    row_stride_bytes: int,
    output: torch.Tensor,
    quant_x: torch.Tensor,
) -> torch.Tensor:
    """Strided native MMVQ using caller-owned output and Q8_1 scratch."""
    return _module().ggml_moe_a8_vec_strided_workspace(
        x, weight, topk_ids, top_k, quant_type, row, tokens,
        expert_stride_bytes, row_stride_bytes, output, quant_x
    )


def ggml_moe_get_block_size(quant_type: int) -> int:
    return _module().ggml_moe_get_block_size(quant_type)


__all__ = [
    "mmvq_bs1_workspace_bytes",
    "validate_mmvq_bs1_workspace",
    "gguf_runtime_metadata",
    "gguf_dispatch",
    "gguf_dispatch_report",
    "register_gguf_moe_abi",
    "ggml_dequantize",
    "ggml_mul_mat_vec_a8",
    "ggml_mul_mat_a8",
    "ggml_moe_a8",
    "ggml_moe_a8_vec",
    "ggml_moe_a8_vec_strided",
    "ggml_moe_a8_vec_workspace",
    "ggml_moe_a8_vec_strided_workspace",
    "ggml_moe_mmvq_id",
    "ggml_moe_mmvdq_id",
    "ggml_moe_gate_up_swiglu_id",
    "ggml_moe_get_block_size",
    "ensure_gguf_moe_candidate_ready",
]
