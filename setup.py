from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CppExtension


ROOT = Path(__file__).parent


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


def _hip_runtime_paths() -> tuple[list[str], list[str], list[str]]:
    # AMD ROCm port: build the same sources against HIP instead. The shim
    # header maps the CUDA runtime symbols csrc uses onto their HIP
    # equivalents, so the sources stay untouched.
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    if not rocm.exists():
        raise RuntimeError(f"ROCm not found at {rocm}; set ROCM_PATH")
    return (
        [str(ROOT / "csrc-hip-shim"), str(rocm / "include")],
        [str(rocm / "lib")],
        ["amdhip64"],
    )


if getattr(torch.version, "hip", None):
    on_hip = True
    include_dirs, library_dirs, libraries = _hip_runtime_paths()
elif getattr(torch.version, "cuda", None):
    on_hip = False
    include_dirs, library_dirs = _cuda_runtime_paths()
    _check_toolchain()
    libraries = ["cudart"]
else:
    raise RuntimeError(
        "freetoken requires a CUDA or ROCm build of torch; "
        f"found torch {torch.__version__}"
    )

common_compile_args = ["-O3", "-std=c++17"] + (
    ["-D__HIP_PLATFORM_AMD__"] if on_hip else []
)

setup(
    ext_modules=[
        CppExtension(
            name="freetoken.kernel._pinned_tensor",
            sources=[
                "python/freetoken/kernel/csrc/pinned_tensor.cpp",
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=common_compile_args,
        ),
        # CPU-compute MoE executor for --moe-backend cpu. Links the GPU runtime
        # library for the cudaLaunchHostFunc submit/sync graph nodes; the bf16 GEMV microkernels
        # use per-function target attributes (avx512bf16/avx512f) + a runtime
        # __builtin_cpu_supports dispatch, so the single binary stays portable
        # (scalar fallback) -- no global -march is set.
        CppExtension(
            name="freetoken.kernel._cpu_moe",
            sources=[
                "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp",
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=[*common_compile_args, "-pthread"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
