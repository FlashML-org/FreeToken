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
import subprocess
import time

import torch

_CSRC = pathlib.Path(__file__).parent / "csrc" / "gguf"


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
    no fd is held on it, so there is nothing to poll). Guard on BOTH the age and
    the absence of a live freetoken serve/worker process, so a legitimately
    concurrent rebuild is never clobbered.
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
        if _freetoken_processes_running():
            # Without the age check a mid-compile server would be clobbered; with it,
            # anything left after hours while no freetoken process lives is the corpse
            # of a killed run.
            return
        lock.unlink()
    except Exception:  # noqa: BLE001 - hygiene must never break the build path
        pass


def _freetoken_processes_running() -> bool:
    try:
        out = subprocess.run(
            ["pgrep", "-f", r"freetoke[n].cli serve|multiprocessing.s[p]awn"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return True  # cannot tell -> assume a live owner

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
        ]
        os.environ.pop("CXX", None)
        os.environ.pop("CC", None)
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

    # gguf_kernel.cu carries its own PYBIND11_MODULE (appended at the end), so a
    # plain `load` of the single source compiles + binds the ggml_* ops.
    _clear_stale_jit_lock("freetoken_gguf_kernels")
    return load(
        name="freetoken_gguf_kernels",
        sources=[str(_CSRC / "gguf_kernel.cu")],
        extra_include_paths=[str(_CSRC)],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
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
