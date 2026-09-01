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

import torch

_CSRC = pathlib.Path(__file__).parent / "csrc" / "gguf"


def _hip_target_arch() -> str | None:
    """Return the active AMD GPU target in ``gfxNNNN`` form when HIP exposes it.

    PyTorch's extension builder otherwise emits code for every visible AMD target.
    A one-GPU serving process only needs the active target, so preserving an explicit
    user selection or deriving the target from the active device avoids unnecessary
    JIT work and records the architecture in the extension build key.
    """
    explicit = os.environ.get("PYTORCH_ROCM_ARCH", "").strip()
    if explicit:
        return explicit.split(";", 1)[0].strip()
    if not torch.cuda.is_available():
        return None
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return str(arch).split(":", 1)[0] or None


def _hip_gguf_cflags() -> list[str]:
    """Build conservative HIP GGUF flags for the active AMD GPU target.

    The architecture environment variable is set before PyTorch asks hipcc to
    compile, which makes the cache target-specific without overriding a deployment's
    explicit multi-target configuration.  Keep floating-point flags conservative:
    the native GGUF kernels must preserve model output, and unsupported aggressive
    math flags belong only in isolated benchmark experiments.
    """
    target = _hip_target_arch()
    if target and not os.environ.get("PYTORCH_ROCM_ARCH"):
        os.environ["PYTORCH_ROCM_ARCH"] = target
    flags = ["-O3"]
    # This is deliberately opt-in because it allows the compiler to make
    # floating-point transformations that are unsuitable for the portable
    # default.  It is useful for a controlled ROCm performance candidate: the
    # value becomes part of PyTorch's extension cache key, so the candidate
    # cannot silently reuse a conservative binary.  ``-funsafe-math-
    # optimizations`` is narrower than ``-ffast-math`` and matches the HIP
    # optimization used by current llama.cpp builds without enabling its
    # additional finite-math assumptions.
    if os.environ.get("FREETOKEN_HIP_GGUF_FAST_MATH") == "1":
        flags.append("-funsafe-math-optimizations")
    return flags


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
            if (lib_dir / "libamdhip64.so").is_file():
                return str(lib_dir)
    return None


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
        extra_cuda_cflags = _hip_gguf_cflags()
        # The minimal PyTorch ROCm SDK can omit Thrust while libtorch's HIP
        # headers include it.  Add a real system ROCm developer include only
        # when present, retaining the wheel-only build on complete installs.
        # This must be a compiler flag, not ``extra_include_paths``: PyTorch's
        # hipify pass recursively rewrites every extension include path and
        # cannot write beneath the read-only system ROCm installation.
        hip_thrust_include = _hip_thrust_include()
        hip_runtime_library_dir = _hip_runtime_library_dir()
        extra_include_paths = [str(_CSRC)]
        extra_ldflags: list[str] = []
        if hip_thrust_include is not None:
            extra_cuda_cflags += ["-isystem", hip_thrust_include]
        if hip_runtime_library_dir is not None:
            # The extension linker uses ``-lamdhip64``.  Add a real ROCm
            # library directory only when the wheel SDK lacks its unversioned
            # linker symlink, preserving self-contained wheel installations.
            extra_ldflags += [f"-L{hip_runtime_library_dir}"]
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
