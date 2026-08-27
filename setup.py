from __future__ import annotations

import importlib.util
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, ROCM_HOME, CppExtension


ROOT = Path(__file__).parent
IS_ROCM = CUDA_HOME is None and ROCM_HOME is not None


def _check_toolchain() -> None:
    if IS_ROCM:
        # nvcc/CUDA-major checks below are meaningless on a ROCm torch build
        # (torch.version.cuda is None there), so _toolchain.py's check is a no-op.
        return
    path = ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_nvcc_matches_torch()


def _gpu_runtime_paths() -> tuple[list[str], list[str], list[str], list[str]]:
    """Returns (include_dirs, library_dirs, libraries, extra_link_args)."""
    if IS_ROCM:
        rocm_home = Path(ROCM_HOME)
        library_dirs = [d for d in (rocm_home / "lib64", rocm_home / "lib") if d.exists()]
        # The pip-vendored rocm-sdk-core ships versioned sonames (libamdhip64.so.7)
        # without the bare .so dev symlink `-lamdhip64` needs, so link the exact
        # file. At runtime the dynamic linker dedupes on SONAME, so this resolves
        # to whichever libamdhip64 torch itself already loaded into the process.
        hip_lib = next(
            (f for d in library_dirs for f in sorted(d.glob("libamdhip64.so*"))), None
        )
        if hip_lib is None:
            raise RuntimeError(f"libamdhip64.so* not found under {library_dirs}")
        return (
            [str(rocm_home / "include")],
            [str(d) for d in library_dirs],
            [],
            [f"-l:{hip_lib.name}"],
        )
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME (or ROCM_HOME) is required to build freetoken.kernel._pinned_tensor "
            "because it links against the CUDA/HIP runtime API."
        )
    cuda_home = Path(CUDA_HOME)
    library_dirs = [str(cuda_home / "lib64")]
    if (cuda_home / "lib").exists():
        library_dirs.append(str(cuda_home / "lib"))
    return [str(cuda_home / "include")], library_dirs, ["cudart"], []


cuda_include_dirs, cuda_library_dirs, cuda_libraries, cuda_extra_link_args = _gpu_runtime_paths()
_check_toolchain()


setup(
    ext_modules=[
        CppExtension(
            name="freetoken.kernel._pinned_tensor",
            sources=[
                "python/freetoken/kernel/csrc/pinned_tensor.cpp",
            ],
            include_dirs=cuda_include_dirs,
            library_dirs=cuda_library_dirs,
            libraries=cuda_libraries,
            extra_link_args=cuda_extra_link_args,
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
            libraries=cuda_libraries,
            extra_link_args=cuda_extra_link_args,
            extra_compile_args=["-O3", "-std=c++17", "-pthread"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
