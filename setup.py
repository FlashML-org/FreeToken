from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CppExtension, CUDAExtension


ROOT = Path(__file__).parent

# sm_75 / CUDA 12.8 support: include Turing (7.5) in the default arch list.
# Downstream users can override via TORCH_CUDA_ARCH_LIST as usual.
# We start at 7.5 (2080 Ti / sm_75) rather than 8.0 so the pinned_tensor and
# cpu_moe C++ extensions compile with sm_75 PTX fallback.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6;8.9;9.0")


def _fetch_marlin_sm75() -> bool:
    """Fetch the Marlin WNA16 sm_75 kernel sources from vLLM-2080Ti-Definitive.

    Returns True if the sources are ready (fetched or already present),
    False if the fetch failed (extension will be skipped with a warning).
    """
    marlin_dir = ROOT / "python" / "freetoken" / "kernel" / "csrc" / "marlin_wna16"
    selector = marlin_dir / "kernel_selector.h"
    if selector.exists() and list(marlin_dir.glob("sm75_kernel_*.cu")):
        return True  # already generated, skip re-fetch
    fetch_script = ROOT / "scripts" / "fetch_marlin_sm75.py"
    if not fetch_script.exists():
        print(
            "WARNING: scripts/fetch_marlin_sm75.py not found; "
            "skipping Marlin sm_75 extension build."
        )
        return False
    print("setup.py: fetching Marlin sm_75 sources...")
    result = subprocess.run(
        [sys.executable, str(fetch_script)],
        cwd=str(ROOT),
        capture_output=False,
    )
    if result.returncode != 0:
        print(
            "WARNING: fetch_marlin_sm75.py failed (network unavailable?); "
            "Marlin sm_75 extension will not be built. "
            "AWQ/GPTQ INT4 models will use the Triton dequant fallback."
        )
        return False
    return True


def _check_toolchain() -> None:
    path = ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_nvcc_matches_torch()


def _cuda_runtime_paths() -> tuple[list[str], list[str]]:
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is required to build freetoken.kernel._pinned_tensor "
            "because it links against the CUDA runtime API."
        )
    cuda_home = Path(CUDA_HOME)
    library_dirs = [str(cuda_home / "lib64")]
    if (cuda_home / "lib").exists():
        library_dirs.append(str(cuda_home / "lib"))
    return [str(cuda_home / "include")], library_dirs


cuda_include_dirs, cuda_library_dirs = _cuda_runtime_paths()
_check_toolchain()

# ---------------------------------------------------------------------------
# Optional: Marlin WNA16 sm_75 extension (AWQ/GPTQ INT4 on Turing GPUs)
#
# Fetches the kernel sources from weicj/vLLM-2080Ti-Definitive at build time,
# generates the sm_75-specific kernel instantiations (stages=2), and compiles
# them as freetoken.kernel._marlin_sm75.
#
# Skip by setting FREETOKEN_SKIP_MARLIN_SM75=1 or if fetch fails (network
# unavailable in offline environments). The runtime falls back to Triton
# dequant transparently when the extension is absent.
# ---------------------------------------------------------------------------
_marlin_sm75_ext_modules: list = []

_skip_marlin = os.environ.get("FREETOKEN_SKIP_MARLIN_SM75", "").strip().lower() in (
    "1", "true", "yes", "on"
)
if not _skip_marlin and _fetch_marlin_sm75():
    _marlin_dir = ROOT / "python" / "freetoken" / "kernel" / "csrc" / "marlin_wna16"
    _sm75_cu_files = sorted(str(p) for p in _marlin_dir.glob("sm75_kernel_*.cu"))
    if _sm75_cu_files:
        _marlin_include_dirs = [
            str(_marlin_dir),                        # kernel.h, kernel_selector.h
            str(_marlin_dir / "quantization"),       # quantization/marlin/
            str(_marlin_dir / "core"),               # scalar_type.hpp, registration.h
        ] + cuda_include_dirs
        _marlin_sm75_ext_modules.append(
            CUDAExtension(
                name="freetoken.kernel._marlin_sm75",
                sources=[
                    # pybind11 wrapper
                    "python/freetoken/kernel/csrc/marlin_wna16/marlin_sm75_ext.cpp",
                    # Marlin MoE dispatch + helper kernels
                    str(_marlin_dir / "ops.cu"),
                ] + _sm75_cu_files,
                include_dirs=_marlin_include_dirs,
                library_dirs=cuda_library_dirs,
                libraries=["cublas", "cudart"],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "nvcc": [
                        "-O3",
                        "--use_fast_math",
                        "-lineinfo",
                        # sm_75 only: the pipelining (stages) is already set to 2
                        # inside the generated sm75_kernel_*.cu files.
                        # Do NOT add sm_80+ here — wrong code would silently link.
                        "-gencode", "arch=compute_75,code=sm_75",
                        "-std=c++17",
                        # Suppress the "statement is unreachable" warning that
                        # Marlin's template metaprogramming generates on older nvcc.
                        "-Xcompiler", "-Wno-unused-function",
                        # The Marlin template uses __CUDA_ARCH__ guards; suppress
                        # the "declared but never referenced" notes for the sm<75 stub.
                        "--diag-suppress=177",
                    ],
                },
            )
        )
        print(
            f"setup.py: Marlin sm_75 extension configured "
            f"({len(_sm75_cu_files)} kernel files)"
        )
    else:
        print(
            "WARNING: fetch_marlin_sm75.py ran but produced no sm75_kernel_*.cu files; "
            "skipping _marlin_sm75 extension."
        )
elif _skip_marlin:
    print("setup.py: FREETOKEN_SKIP_MARLIN_SM75=1 — skipping Marlin sm_75 extension.")


setup(
    ext_modules=[
        CppExtension(
            name="freetoken.kernel._pinned_tensor",
            sources=[
                "python/freetoken/kernel/csrc/pinned_tensor.cpp",
            ],
            include_dirs=cuda_include_dirs,
            library_dirs=cuda_library_dirs,
            libraries=["cudart"],
            extra_compile_args=["-O3", "-std=c++17"],
        ),
        # CPU-compute MoE executor for --moe-backend cpu. Links cudart for the
        # cudaLaunchHostFunc submit/sync graph nodes; the bf16 GEMV microkernels
        # use per-function target attributes (avx512bf16/avx512f) + a runtime
        # __builtin_cpu_supports dispatch, so the single binary stays portable
        # (scalar fallback) -- no global -march is set.
        CppExtension(
            name="freetoken.kernel._cpu_moe",
            sources=[
                "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp",
            ],
            include_dirs=cuda_include_dirs,
            library_dirs=cuda_library_dirs,
            libraries=["cudart"],
            extra_compile_args=["-O3", "-std=c++17", "-pthread"],
        ),
    ] + _marlin_sm75_ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
