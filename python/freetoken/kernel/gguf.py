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
import hashlib
import os
import pathlib
import shutil

import torch

_CSRC = pathlib.Path(__file__).parent / "csrc" / "gguf"


def _staged_rocm_sources() -> pathlib.Path:
    """Copy CUDA sources out of the checkout before PyTorch HIPifies them.

    ``torch.utils.cpp_extension.load`` writes generated ``*_hip`` sources next to
    the input file.  Keeping the staging directory under the extension cache makes
    the source checkout stay clean while still allowing normal incremental builds.
    """
    cache_root = pathlib.Path(
        os.environ.get("TORCH_EXTENSIONS_DIR", pathlib.Path.home() / ".cache" / "torch_extensions")
    )
    digest = hashlib.sha256()
    digest.update(f"torch={torch.__version__};hip={torch.version.hip}".encode())
    for source in sorted(_CSRC.iterdir()):
        if source.is_file() and "_hip." not in source.name and source.suffix != ".hip":
            digest.update(source.name.encode())
            digest.update(source.read_bytes())
    staged = cache_root / f"freetoken_gguf_sources_{digest.hexdigest()[:16]}"
    shutil.copytree(
        _CSRC,
        staged,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("*_hip.*", "*.hip", "__pycache__"),
    )
    return staged


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

    is_rocm = getattr(torch.version, "hip", None) is not None
    extra_cuda_cflags = ["-O3"]
    extra_ldflags: list[str] = []
    if is_rocm:
        from freetoken.kernel.utils import _rocm_link_flags

        extra_ldflags = _rocm_link_flags()
        # Ubuntu's generic Thrust headers otherwise select the CUDA backend and
        # try to include cuda_runtime_api.h. GGUF only reaches Thrust through a
        # libtorch complex-number header, so the backend-neutral C++ path is
        # sufficient for HIP compilation.
        extra_cuda_cflags.append("-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP")
        csrc = _staged_rocm_sources()
    else:
        extra_cuda_cflags.append("--expt-relaxed-constexpr")
        csrc = _CSRC

    host_cxx = None if is_rocm else _host_compiler()
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
    return load(
        name="freetoken_gguf_kernels",
        sources=[str(csrc / "gguf_kernel.cu")],
        extra_include_paths=[str(csrc)],
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
