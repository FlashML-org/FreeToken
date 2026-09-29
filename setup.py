from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CppExtension, ROCM_HOME


ROOT = Path(__file__).parent
KERNEL_INCLUDE = str(ROOT / "python" / "freetoken" / "kernel" / "csrc" / "include")


def _check_toolchain() -> None:
    path = ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_nvcc_matches_torch()


def _is_rocm() -> bool:
    import torch
    return getattr(torch.version, "hip", None) is not None


def _rocm_paths() -> tuple[list[str], list[str], str]:
    candidates: list[Path] = []
    if os.getenv("ROCM_HOME"):
        candidates.append(Path(os.environ["ROCM_HOME"]))
    if ROCM_HOME:
        candidates.append(Path(ROCM_HOME))

    # ROCm 7.14 PyTorch images ship the SDK as a Python package instead of
    # installing it at /opt/rocm.
    spec = importlib.util.find_spec("_rocm_sdk_core")
    if spec and spec.submodule_search_locations:
        candidates.append(Path(next(iter(spec.submodule_search_locations))))
    candidates.append(Path("/opt/rocm"))

    for rocm_home in dict.fromkeys(candidates):
        include_dir = rocm_home / "include"
        library_dir = rocm_home / "lib"
        if not (include_dir / "hip" / "hip_runtime.h").exists():
            continue
        if (library_dir / "libamdhip64.so").exists():
            return [str(include_dir)], [str(library_dir)], "amdhip64"
        versioned = sorted(library_dir.glob("libamdhip64.so.*"))
        if versioned:
            return [str(include_dir)], [str(library_dir)], f":{versioned[-1].name}"

    searched = ", ".join(str(path) for path in dict.fromkeys(candidates))
    raise RuntimeError(
        "A ROCm SDK with HIP headers and libamdhip64 is required to build on ROCm; "
        f"searched: {searched}. Set ROCM_HOME to override."
    )


def _cuda_runtime_paths() -> tuple[list[str], list[str]]:
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is required to build freetoken.kernel._pinned_tensor "
            "because it links against the CUDA runtime API."
        )
    cuda_home = Path(CUDA_HOME)
    # Windows toolkits keep the import libraries in lib/x64.
    candidates = (cuda_home / "lib64", cuda_home / "lib", cuda_home / "lib" / "x64")
    library_dirs = [str(path) for path in candidates if path.exists()]
    return [str(cuda_home / "include")], library_dirs


IS_ROCM = _is_rocm()
IS_WINDOWS = sys.platform == "win32"

if IS_ROCM:
    runtime_include_dirs, runtime_library_dirs, runtime_lib = _rocm_paths()
    runtime_link_args = [f"-Wl,-rpath,{runtime_library_dirs[0]}"]
    # These extensions contain host code only. BuildExtension supplies the ROCm
    # platform defines to the C++ compiler; offload architecture flags belong on
    # HIP device sources and would be rejected by the host compiler here.
    extra_compile = ["-O3", "-std=c++17"]
else:
    runtime_include_dirs, runtime_library_dirs = _cuda_runtime_paths()
    runtime_lib = "cudart"
    runtime_link_args = []
    # cl ignores -O3 and -std with a warning (D9002): /O2 is its spelling, and BuildExtension adds /std:c++17.
    extra_compile = ["/O2"] if IS_WINDOWS else ["-O3", "-std=c++17"]

_check_toolchain()


setup(
    ext_modules=[
        CppExtension(
            name="freetoken.kernel._pinned_tensor",
            sources=[
                "python/freetoken/kernel/csrc/pinned_tensor.cpp",
            ],
            include_dirs=[KERNEL_INCLUDE, *runtime_include_dirs],
            library_dirs=runtime_library_dirs,
            libraries=[runtime_lib],
            extra_compile_args=extra_compile,
            extra_link_args=runtime_link_args,
        ),
        # CPU-compute MoE executor for --moe-backend cpu. Links cudart/amdhip64 for the
        # cudaLaunchHostFunc submit/sync graph nodes; the bf16 GEMV microkernels
        # use per-function target attributes (avx512bf16/avx512f) + a runtime
        # __builtin_cpu_supports dispatch, so the single binary stays portable
        # (scalar fallback) -- no global -march is set.
        # GCC target attributes and pthreads have no MSVC form; Windows serves the GPU offload path only.
        *([
            CppExtension(
                name="freetoken.kernel._cpu_moe",
                sources=[
                    "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp",
                ],
                include_dirs=[KERNEL_INCLUDE, *runtime_include_dirs],
                library_dirs=runtime_library_dirs,
                libraries=[runtime_lib],
                extra_compile_args=extra_compile + ["-pthread"],
                extra_link_args=runtime_link_args,
            )
        ] if not IS_WINDOWS else []),
        # disk-backed row store (PLE / Engram tables); Linux-only until the TableFile/BatchReader seams grow Windows bodies
        *([
            CppExtension(
                name="freetoken.kernel._row_store",
                sources=[
                    "python/freetoken/kernel/csrc/row_store/row_store_ext.cpp",
                ],
                extra_compile_args=["-O3", "-std=c++17"],
            )
        ] if sys.platform == "linux" else []),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
