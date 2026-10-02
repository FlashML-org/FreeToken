"""Executable coverage for accelerator-specific clangd configuration generation."""

from __future__ import annotations

import subprocess  # Patch NVIDIA discovery without invoking host tooling.

import torch  # Supply the CUDA capability fallback used when nvidia-smi is absent.

from freetoken.kernel.__main__ import generate_clangd  # Exercise the production entry-point function.


def test_generate_clangd_uses_rocm_arch_flags(monkeypatch, tmp_path):  # Verify the AMD branch writes HIP tooling flags.
    import freetoken.utils as utils  # Patch the runtime selector imported inside the function.
    import freetoken.utils.arch as arch  # Patch the architecture flag provider imported inside the function.
    import tvm_ffi.libinfo as libinfo  # Patch include discovery away from the installed environment.

    monkeypatch.chdir(tmp_path)  # Keep the generated tooling file inside the disposable test directory.
    monkeypatch.setattr(utils, "is_rocm", lambda: True)  # Select the ROCm branch explicitly.
    monkeypatch.setattr(arch, "rocm_clang_flags", lambda: ["-xhip", "--offload-arch=gfx1151"])  # Supply measured target flags.
    monkeypatch.setattr(libinfo, "find_include_path", lambda: "/include/tvm")  # Provide deterministic TVM includes.
    monkeypatch.setattr(libinfo, "find_dlpack_include_path", lambda: "/include/dlpack")  # Provide deterministic DLPack includes.

    generate_clangd()  # Generate the actual YAML configuration through the ROCm path.

    content = (tmp_path / ".clangd").read_text(encoding="utf-8")  # Inspect the persisted result.
    assert "-xhip" in content  # Require HIP language mode.
    assert "--offload-arch=gfx1151" in content  # Require the exact architecture binding.
    assert "--cuda-gpu-arch" not in content  # Prevent CUDA flags from leaking into ROCm tooling.


def test_generate_clangd_falls_back_to_torch_cuda_capability(monkeypatch, tmp_path):  # Verify NVIDIA fallback without nvidia-smi.
    import freetoken.utils as utils  # Patch the runtime selector imported inside the function.
    import tvm_ffi.libinfo as libinfo  # Patch include discovery away from host installation details.

    monkeypatch.chdir(tmp_path)  # Keep the generated file isolated.
    monkeypatch.setattr(utils, "is_rocm", lambda: False)  # Select the NVIDIA branch.
    monkeypatch.setattr(subprocess, "run", lambda **_kwargs: (_ for _ in ()).throw(FileNotFoundError()))  # Simulate absent nvidia-smi.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))  # Supply the fallback compute capability.
    monkeypatch.setattr(libinfo, "find_include_path", lambda: "/include/tvm")  # Provide deterministic TVM includes.
    monkeypatch.setattr(libinfo, "find_dlpack_include_path", lambda: "/include/dlpack")  # Provide deterministic DLPack includes.

    generate_clangd()  # Generate the actual YAML through the Torch capability fallback.

    content = (tmp_path / ".clangd").read_text(encoding="utf-8")  # Inspect the generated NVIDIA configuration.
    assert "-xcuda" in content  # Require CUDA language mode.
    assert "--cuda-gpu-arch=sm_120" in content  # Require the Torch-derived architecture.
    assert "--offload-arch" not in content  # Prevent HIP flags from leaking into NVIDIA tooling.
