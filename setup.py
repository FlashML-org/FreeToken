from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from setuptools import setup

_IS_MACOS = sys.platform == "darwin"


ROOT = Path(__file__).parent


def _check_toolchain() -> None:
    path = ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_nvcc_matches_torch()


def _cuda_runtime_paths(cuda_home: Path) -> tuple[list[str], list[str]]:
    library_dirs = [str(cuda_home / "lib64")]
    if (cuda_home / "lib").exists():
        library_dirs.append(str(cuda_home / "lib"))
    return [str(cuda_home / "include")], library_dirs


cuda_include_dirs: list[str] = []
cuda_library_dirs: list[str] = []
ext_modules: list = []
cmdclass: dict = {}

if not _IS_MACOS:
    # The two C++ extensions below link the CUDA runtime; macOS (Metal) builds
    # have no compiled extensions at all (the Metal path runs Apple's mlx /
    # llama.cpp runtimes as upstream engines, see server/metal.py).
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDA_HOME

    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is required to build freetoken.kernel._pinned_tensor "
            "because it links against the CUDA runtime API."
        )
    cuda_include_dirs, cuda_library_dirs = _cuda_runtime_paths(Path(CUDA_HOME))
    _check_toolchain()
    ext_modules = [
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
    ]
    cmdclass = {"build_ext": BuildExtension.with_options(use_ninja=True)}


setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
