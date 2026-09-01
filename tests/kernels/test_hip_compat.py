"""HIP header / JIT flag surface — compile-free, no AMD GPU."""

from pathlib import Path

from freetoken.kernel import utils as kutils

ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / "python" / "freetoken" / "kernel" / "csrc"


def test_hip_compat_header_exists():
    text = (CSRC / "include" / "freetoken" / "hip_compat.cuh").read_text()
    assert "hipLaunchKernel" in text
    assert "cudaLaunchKernelEx" in text
    assert "__HIP_PLATFORM_AMD__" in (CSRC / "include" / "freetoken" / "utils.cuh").read_text()


def test_fast_index_copy_gates_ptx():
    text = (CSRC / "jit" / "fast_index_copy.cuh").read_text()
    assert "__HIP_PLATFORM_AMD__" in text
    assert "ld.global.L1::no_allocate" in text


def test_hip_cflags_emit_offload_arch(monkeypatch):
    monkeypatch.setenv("TVM_FFI_ROCM_ARCH_LIST", "gfx1201")
    flags = kutils._hip_cflags([])
    assert "--offload-arch=gfx1201" in flags
    assert "-DUSE_ROCM" in flags
    assert "-D__HIP_PLATFORM_AMD__=1" in flags
    assert "--expt-relaxed-constexpr" not in flags


def test_cuda_cflags_stay_nvcc_on_nvidia(monkeypatch):
    monkeypatch.delenv("HIP_PATH", raising=False)
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.delenv("ROCM_HOME", raising=False)
    monkeypatch.setattr(kutils, "_is_hip_build", lambda: False)
    flags = kutils._cuda_cflags([])
    assert "--expt-relaxed-constexpr" in flags
    assert "--offload-arch=gfx1201" not in flags


def test_is_hip_build_prefers_cuda_torch_over_rocm_home(monkeypatch):
    """A leftover ROCM_HOME on an NVIDIA wheel must not flip JIT to hipcc."""
    from types import SimpleNamespace

    monkeypatch.setenv("ROCM_HOME", "/opt/rocm")
    monkeypatch.setattr(
        kutils,
        "torch",
        SimpleNamespace(version=SimpleNamespace(hip=None, cuda="13.0")),
        raising=False,
    )

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            return SimpleNamespace(version=SimpleNamespace(hip=None, cuda="13.0"))
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert kutils._is_hip_build() is False


def test_thrust_shim_present():
    shim = CSRC / "gguf" / "jit_shim" / "thrust" / "complex.h"
    assert shim.is_file()
    assert "namespace thrust" in shim.read_text()
