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
import sys
from contextlib import contextmanager

import torch

_CSRC = pathlib.Path(__file__).parent / "csrc" / "gguf"


def _hip_thrust_include() -> str | None:
    """Return a ROCm developer include directory that exposes ``thrust/complex.h``.

    The PyTorch ROCm wheel bundles hipcc but may omit the header-only Thrust
    dependency required by libtorch's HIP complex header.  Prefer explicitly
    configured ROCm homes, then inspect the standard versioned installation
    layout.  Returning ``None`` leaves hosts with a complete wheel toolchain
    unchanged.
    """
    candidates = [
        os.environ.get("ROCM_HOME"),
        os.environ.get("ROCM_PATH"),
        "/opt/rocm",
    ]
    candidates.extend(str(path) for path in sorted(pathlib.Path("/opt").glob("rocm-*"), reverse=True))
    for root in candidates:
        if not root:
            continue
        include = pathlib.Path(root) / "include"
        if (include / "thrust" / "complex.h").is_file():
            return str(include)
    return None


def _hip_runtime_library_dir() -> str | None:
    """Return a ROCm library directory that can satisfy ``-lamdhip64``.

    Some PyTorch ROCm wheels ship ``libamdhip64.so.7`` but not the unversioned
    linker name that ``torch.utils.cpp_extension`` emits.  A regular ROCm
    installation supplies that linker name under its ``lib`` directory.  Keep
    this discovery separate from the Thrust fallback so a host can provide one
    dependency through the wheel and the other through its ROCm installation.
    """
    candidates = [
        os.environ.get("ROCM_HOME"),
        os.environ.get("ROCM_PATH"),
        "/opt/rocm",
    ]
    candidates.extend(str(path) for path in sorted(pathlib.Path("/opt").glob("rocm-*"), reverse=True))
    for root in candidates:
        if not root:
            continue
        for lib_dir in (pathlib.Path(root) / "lib", pathlib.Path(root) / "lib64"):
            if os.name == "nt" and (lib_dir / "amdhip64.lib").is_file():
                return str(lib_dir)
            if (lib_dir / "libamdhip64.so").is_file():
                return str(lib_dir)
    return None


def _hip_device_library_dir() -> str | None:
    """Return a complete ROCm bitcode directory for HIP device linking."""
    candidates = [
        os.environ.get("ROCM_HOME"),
        os.environ.get("ROCM_PATH"),
        "/opt/rocm",
    ]
    candidates.extend(str(path) for path in sorted(pathlib.Path("/opt").glob("rocm-*"), reverse=True))
    for root in candidates:
        if not root:
            continue
        root_path = pathlib.Path(root)
        for relative in ("amdgcn/bitcode", "lib/amdgcn/bitcode", "lib/llvm/amdgcn/bitcode"):
            bitcode = root_path / relative
            if (bitcode / "ocml.bc").is_file() and (bitcode / "ockl.bc").is_file():
                return str(bitcode)
    return None


@contextmanager
def _windows_hip_link_flags():
    """Use HIP import libraries for PyTorch JIT extensions on native Windows.

    The official AMD torch 2.9.1 Windows wheel recognizes HIP sources and invokes
    hipcc, but its Windows linker helper still appends the CUDA-only
    ``c10_cuda.lib``, ``torch_cuda.lib`` and ``cudart.lib`` names.  Patch that
    helper only for the duration of this one HIP build; leave non-Windows,
    non-HIP and CPU-only extension behavior untouched.
    """
    if os.name != "nt" or torch.version.hip is None:
        yield
        return

    from torch.utils import cpp_extension
    from torch.utils.hipify import hipify_python

    if not cpp_extension.IS_HIP_EXTENSION:
        raise RuntimeError(
            "PyTorch did not recognize the configured ROCm SDK for a Windows HIP extension"
        )

    original = cpp_extension._prepare_ldflags
    original_hipify = hipify_python.hipify
    original_write_ninja = cpp_extension._write_ninja_file

    def prepare_ldflags(extra_ldflags, with_cuda, verbose, is_standalone):
        if not with_cuda:
            return original(extra_ldflags, with_cuda, verbose, is_standalone)

        flags = list(extra_ldflags)
        flags += [
            "c10.lib",
            "c10_hip.lib",
            "torch_cpu.lib",
            "torch_hip.lib",
            "torch.lib",
            f"/LIBPATH:{cpp_extension.TORCH_LIB_PATH}",
        ]
        if not is_standalone:
            flags += [
                "torch_python.lib",
                f"/LIBPATH:{os.path.join(sys.base_exec_prefix, 'libs')}",
            ]
        return flags

    def hipify_with_windows_paths(*args, **kwargs):
        # torch 2.9.1's hipify converts each path to POSIX separators before
        # checking membership, but leaves extra_files/header traversal results
        # with Windows separators.  Stage normalized copies under the requested
        # build directory so hipify neither ignores them nor writes generated
        # *_hip files beside an installed/source package.
        original_sources = [pathlib.Path(path).resolve() for path in kwargs.get("extra_files", ())]
        include_roots = [pathlib.Path(path).resolve() for path in kwargs.get("header_include_dirs", ())]
        supported_headers = {".cu", ".cuh", ".c", ".cc", ".cpp", ".h", ".in", ".hpp"}
        candidates = list(original_sources)
        for include_root in include_roots:
            if include_root.is_dir():
                candidates.extend(
                    path.resolve()
                    for path in include_root.rglob("*")
                    if path.is_file()
                    and path.suffix.lower() in supported_headers
                    and path.suffix.lower() != ".hip"
                    and not path.stem.endswith("_hip")
                )

        stage_root = pathlib.Path(kwargs["output_directory"]).resolve() / "hipify-src"
        source_map: dict[str, pathlib.Path] = {}
        for source in dict.fromkeys(candidates):
            relative = pathlib.Path(source.name)
            for include_root in include_roots:
                try:
                    relative = source.relative_to(include_root)
                    break
                except ValueError:
                    continue
            staged = stage_root / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, staged)
            source_map[os.path.abspath(source)] = staged.resolve()

        kwargs["extra_files"] = [path.as_posix() for path in source_map.values()]
        kwargs["header_include_dirs"] = [stage_root.as_posix()]
        result = original_hipify(*args, **kwargs)
        for original_path, staged in source_map.items():
            staged_key = os.path.abspath(staged)
            if staged_key in result:
                result[original_path] = result[staged_key]
        return result

    def write_ninja_with_hip_flags(*args, **kwargs):
        # The same wheel prepends MSVC's /std:c++17 to hipcc device flags.
        # hipcc/clang requires the portable -std=c++17 spelling.
        cuda_cflags = kwargs.get("cuda_cflags")
        if cuda_cflags is not None:
            kwargs["cuda_cflags"] = [
                "-std=c++17" if flag == "/std:c++17" else flag for flag in cuda_cflags
            ]
        return original_write_ninja(*args, **kwargs)

    cpp_extension._prepare_ldflags = prepare_ldflags
    cpp_extension._write_ninja_file = write_ninja_with_hip_flags
    hipify_python.hipify = hipify_with_windows_paths
    try:
        yield
    finally:
        hipify_python.hipify = original_hipify
        cpp_extension._write_ninja_file = original_write_ninja
        cpp_extension._prepare_ldflags = original


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

@functools.cache
def _module():
    from torch.utils.cpp_extension import load

    if torch.version.hip is not None:
        # Neither issue -ccbin works around applies under hipcc: it has no separate
        # nvcc-style host pass (its own bundled clang IS the host compiler), and
        # --expt-relaxed-constexpr is an nvcc-only flag hipcc/clang rejects outright.
        extra_cuda_cflags = ["-O3"]
        if os.name == "nt":
            extra_cuda_cflags += [
                "-DSTRIP_ERROR_MESSAGES",
                "-DNOMINMAX",
                "-fms-runtime-lib=dll",
            ]
        # The minimal PyTorch ROCm SDK can omit Thrust while libtorch's HIP
        # headers include it.  Add a real system ROCm developer include only
        # when present, retaining the wheel-only build on complete installs.
        # This must be a compiler flag, not ``extra_include_paths``: PyTorch's
        # hipify pass recursively rewrites every extension include path and
        # cannot write beneath the read-only system ROCm installation.
        hip_thrust_include = _hip_thrust_include()
        hip_runtime_library_dir = _hip_runtime_library_dir()
        hip_device_library_dir = _hip_device_library_dir()
        extra_include_paths = [str(_CSRC)]
        extra_ldflags: list[str] = []
        if hip_thrust_include is not None:
            extra_cuda_cflags += ["-isystem", hip_thrust_include]
        if hip_runtime_library_dir is not None:
            # The extension linker uses ``-lamdhip64``.  Add a real ROCm
            # library directory only when the wheel SDK lacks its unversioned
            # linker symlink, preserving self-contained wheel installations.
            if os.name == "nt":
                extra_ldflags += [f"/LIBPATH:{hip_runtime_library_dir}", "amdhip64.lib"]
            else:
                extra_ldflags += [f"-L{hip_runtime_library_dir}"]
        if hip_device_library_dir is not None:
            extra_cuda_cflags += [f"--rocm-device-lib-path={hip_device_library_dir}"]
    else:
        extra_cuda_cflags = ["-O3", "--expt-relaxed-constexpr"]
        host_cxx = _host_compiler()
        if host_cxx is not None:
            # Point both nvcc's host pass (-ccbin) and torch's C++ compile (CXX) at a
            # libtorch/nvcc-compatible compiler. Force (not setdefault): the system
            # default (CXX unset -> g++) can be a gcc too new for the torch headers.
            cxx_path = shutil.which(host_cxx) or host_cxx
            extra_cuda_cflags += ["-ccbin", cxx_path]
            os.environ["CXX"] = cxx_path
            os.environ["CC"] = _c_compiler_for(cxx_path)
        extra_include_paths = [str(_CSRC)]
        extra_ldflags = []

    # gguf_kernel.cu carries its own PYBIND11_MODULE (appended at the end), so a
    # plain `load` of the single source compiles + binds the ggml_* ops.
    with _windows_hip_link_flags():
        return load(
            name="freetoken_gguf_kernels",
            sources=[str(_CSRC / "gguf_kernel.cu")],
            extra_include_paths=extra_include_paths,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=extra_ldflags,
            verbose=True,
        )


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
) -> torch.Tensor:
    """MMVQ grouped expert GEMV over stacked experts ``weight[E, row, *]``."""
    return _module().ggml_moe_a8_vec(x, weight, topk_ids, top_k, quant_type, row, tokens)


def ggml_moe_get_block_size(quant_type: int) -> int:
    return _module().ggml_moe_get_block_size(quant_type)


__all__ = [
    "ggml_dequantize",
    "ggml_mul_mat_vec_a8",
    "ggml_mul_mat_a8",
    "ggml_moe_a8",
    "ggml_moe_a8_vec",
    "ggml_moe_get_block_size",
]
