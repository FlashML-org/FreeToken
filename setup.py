from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CppExtension


ROOT = Path(__file__).parent
KERNEL_INCLUDE = ROOT / "python" / "freetoken" / "kernel" / "csrc" / "include"


def _check_toolchain() -> None:
    path = ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_nvcc_matches_torch()


def _is_rocm() -> bool:
    import torch
    return getattr(torch.version, "hip", None) is not None


def _rocm_paths() -> tuple[list[str], list[str]]:
    rocm_home = Path(os.getenv("ROCM_HOME", "/opt/rocm"))
    if not rocm_home.exists():
        raise RuntimeError(
            "ROCM_HOME is required to build on ROCm. Set ROCM_HOME to your ROCm install."
        )
    include_dirs = [str(rocm_home / "include")]
    library_dirs = [str(rocm_home / "lib")]
    return include_dirs, library_dirs


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


IS_ROCM = _is_rocm()

if IS_ROCM:
    runtime_include_dirs, runtime_library_dirs = _rocm_paths()
    runtime_lib = "amdhip64"
    # TODO(ROCm): allow override via FREETOKEN_ROCM_ARCH; default to all RDNA3.
    rocm_arch = os.getenv("FREETOKEN_ROCM_ARCH", "gfx1100;gfx1101;gfx1102;gfx1103")
    extra_compile = ["-O3", "-std=c++17", f"--offload-arch={rocm_arch}"]
else:
    runtime_include_dirs, runtime_library_dirs = _cuda_runtime_paths()
    runtime_lib = "cudart"
    extra_compile = ["-O3", "-std=c++17"]

runtime_include_dirs.append(str(KERNEL_INCLUDE))

_check_toolchain()


setup(
    ext_modules=[
        CppExtension(
            name="freetoken.kernel._pinned_tensor",
            sources=[
                "python/freetoken/kernel/csrc/pinned_tensor.cpp",
            ],
            include_dirs=runtime_include_dirs,
            library_dirs=runtime_library_dirs,
            libraries=[runtime_lib],
            extra_compile_args=extra_compile,
        ),
        # CPU-compute MoE executor for --moe-backend cpu. Links cudart/hip for the
        # cudaLaunchHostFunc submit/sync graph nodes; the bf16 GEMV microkernels
        # use per-function target attributes (avx512bf16/avx512f) + a runtime
        # __builtin_cpu_supports dispatch, so the single binary stays portable
        # (scalar fallback) -- no global -march is set.
        CppExtension(
            name="freetoken.kernel._cpu_moe",
            sources=[
                "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp",
            ],
            include_dirs=runtime_include_dirs,
            library_dirs=runtime_library_dirs,
            libraries=[runtime_lib],
            extra_compile_args=extra_compile + ["-pthread"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
