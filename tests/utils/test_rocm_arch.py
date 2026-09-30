import importlib
import os
import pathlib
import sys
import types
from types import SimpleNamespace

import torch

from freetoken.utils import arch


def _clear_arch_caches() -> None:
    arch.get_rocm_gfx_arch.cache_clear()


def test_rocm_arch_prefers_visible_device_over_multi_arch_build_env(monkeypatch):
    monkeypatch.setattr(arch, "is_rocm", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(gcnArchName="gfx1201:sramecc-:xnack-"),
    )
    monkeypatch.setenv("FREETOKEN_ROCM_ARCH", "gfx1100;gfx1200")
    _clear_arch_caches()

    assert arch.get_rocm_gfx_arch() == "gfx1201"

    _clear_arch_caches()


def test_rocm_arch_falls_back_to_cross_compile_env(monkeypatch):
    monkeypatch.setattr(arch, "is_rocm", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("FREETOKEN_ROCM_ARCH", "gfx1200;gfx1201")
    _clear_arch_caches()

    assert arch.get_rocm_gfx_arch() == "gfx1200"

    _clear_arch_caches()


def test_rocm_clang_flags_do_not_guess_unknown_arch(monkeypatch):  # Protect nonstandard and headless AMD systems.
    monkeypatch.setattr(arch, "get_rocm_gfx_arch", lambda: None)  # Simulate missing device and build target.
    assert arch.rocm_clang_flags() == ["-xhip"]  # Keep HIP mode without asserting a wrong GPU target.


def test_rocm_clang_flags_include_detected_arch(monkeypatch):  # Keep a known gfx target explicit for tooling.
    monkeypatch.setattr(arch, "get_rocm_gfx_arch", lambda: "gfx1151")  # Use the lab's AMD target.
    assert arch.rocm_clang_flags() == ["-xhip", "--offload-arch=gfx1151"]  # Preserve exact architecture binding.


def test_hip_cflags_target_only_resolved_arch(monkeypatch):
    from freetoken.kernel import utils

    monkeypatch.setenv("FREETOKEN_ROCM_ARCH", "gfx1200;gfx1201")
    monkeypatch.setattr(arch, "get_rocm_gfx_arch", lambda: "gfx1201")

    arch_list = utils._rocm_arch_list()
    flags = utils._hip_cflags(["-Wno-unused-command-line-argument"], arch_list)

    assert arch_list == ["gfx1201"]
    assert "--offload-arch=gfx1201" in flags
    assert "--offload-arch=gfx1200" not in flags


def test_rocm_loaders_pin_tvm_ffi_to_resolved_arch(monkeypatch):
    from freetoken.kernel import utils

    builds = []

    def record_build(*_args, **kwargs):
        builds.append(
            (
                os.environ[utils.ROCM_ARCH_LIST_ENV],
                kwargs["extra_cuda_cflags"],
            )
        )
        return object()

    tvm_ffi = types.ModuleType("tvm_ffi")
    tvm_ffi.__path__ = []
    tvm_ffi_cpp = types.ModuleType("tvm_ffi.cpp")
    tvm_ffi_cpp.load = record_build
    tvm_ffi_cpp.load_inline = record_build
    monkeypatch.setitem(sys.modules, "tvm_ffi", tvm_ffi)
    monkeypatch.setitem(sys.modules, "tvm_ffi.cpp", tvm_ffi_cpp)
    monkeypatch.setenv(utils.DISABLE_KERNEL_CACHE_ENV, "1")
    monkeypatch.setenv(utils.ROCM_ARCH_LIST_ENV, "gfx1100 gfx1200")
    monkeypatch.setattr(utils, "_is_rocm", lambda: True)
    monkeypatch.setattr(utils, "_rocm_arch_list", lambda: ["gfx1201"])
    monkeypatch.setattr(utils, "_rocm_link_flags", lambda: [])

    utils.load_aot("test_rocm_aot_arch", cuda_files=["unused.cu"])
    utils.load_jit("test_rocm_jit_arch", cuda_files=["unused.cu"])

    assert builds == [
        ("gfx1201", [*utils.DEFAULT_HIP_CFLAGS, "--offload-arch=gfx1201"]),
        ("gfx1201", [*utils.DEFAULT_HIP_CFLAGS, "--offload-arch=gfx1201"]),
    ]
    assert os.environ[utils.ROCM_ARCH_LIST_ENV] == "gfx1100 gfx1200"


def test_rocm_link_flags_support_versioned_modular_sdk(monkeypatch, tmp_path):
    import torch.utils.cpp_extension as cpp_extension

    from freetoken.kernel import utils

    sdk = tmp_path / "sdk"
    library_dir = sdk / "lib"
    library_dir.mkdir(parents=True)
    versioned_runtime = library_dir / "libamdhip64.so.7"
    versioned_runtime.write_bytes(b"")
    real_find_spec = importlib.util.find_spec

    def find_spec(name: str):
        if name == "_rocm_sdk_core":
            return SimpleNamespace(submodule_search_locations=[str(sdk)])
        return real_find_spec(name)

    monkeypatch.delenv("ROCM_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)  # Exercise the default home-cache fallback independently of the caller.
    monkeypatch.setattr(cpp_extension, "ROCM_HOME", None)
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    utils._rocm_link_flags.cache_clear()

    stale_runtime = tmp_path / "retired-rocm" / "lib" / "libamdhip64.so.6"  # Represent a runtime from the replaced ROCm stack.
    stale_runtime.parent.mkdir(parents=True)  # Create only the isolated test fixture hierarchy.
    stale_runtime.write_bytes(b"")  # Keep the old compatibility target valid so staleness, not breakage, drives replacement.
    compat_dir = tmp_path / ".cache" / "freetoken" / "rocm-lib"  # Match the production compatibility-cache location.
    compat_dir.mkdir(parents=True)  # Prepare the preexisting cache as an earlier installation would.
    compat_link = compat_dir / "libamdhip64.so"  # Address the shared linker compatibility name.
    compat_link.symlink_to(stale_runtime)  # Seed the exact stale-link regression.

    flags = utils._rocm_link_flags()

    assert f"-L{compat_dir}" in flags
    assert f"-Wl,-rpath,{library_dir}" in flags
    assert compat_link.resolve() == versioned_runtime.resolve()

    utils._rocm_link_flags.cache_clear()


def test_rocm_link_flags_support_lib64_and_numeric_sonames(monkeypatch, tmp_path):  # Cover both corrected discovery decisions together.
    import torch.utils.cpp_extension as cpp_extension  # Patch Torch's fallback root without loading a real SDK.

    from freetoken.kernel import utils  # Import the cached linker helper under test.

    sdk = tmp_path / "sdk"  # Build a disposable modular ROCm root.
    library_dir = sdk / "lib64"  # Exercise the layout omitted by the previous implementation.
    library_dir.mkdir(parents=True)  # Create only the isolated fixture directory.
    older_runtime = library_dir / "libamdhip64.so.9"  # Provide a lexically larger but numerically older candidate.
    newer_runtime = library_dir / "libamdhip64.so.10"  # Provide the runtime numeric ordering must select.
    older_runtime.write_bytes(b"")  # Materialize the older SONAME.
    newer_runtime.write_bytes(b"")  # Materialize the newer SONAME.
    monkeypatch.setenv("ROCM_HOME", str(sdk))  # Make the fixture the first explicit discovery root.
    monkeypatch.setattr(cpp_extension, "ROCM_HOME", None)  # Prevent a host Torch root from influencing selection.
    xdg_cache = tmp_path / "xdg-cache"  # Model a writable cache redirected away from a full home filesystem.
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_cache))  # Require the linker helper to honor the standard cache override.
    utils._rocm_link_flags.cache_clear()  # Remove results from earlier tests before discovery.

    flags = utils._rocm_link_flags()  # Resolve the runtime and publish the compatibility symlink.

    compat_link = xdg_cache / "freetoken" / "rocm-lib" / "libamdhip64.so"  # Address the redirected linker name.
    assert f"-Wl,-rpath,{library_dir}" in flags  # Require the selected lib64 directory at runtime.
    assert compat_link.resolve() == newer_runtime.resolve()  # Require numeric 10 to outrank lexical 9.
    utils._rocm_link_flags.cache_clear()  # Avoid leaking the fixture result to later tests.


def test_rocm_cpp_only_loaders_skip_runtime_link_discovery(monkeypatch):  # Keep host-only extensions usable under HIP Torch.
    from freetoken.kernel import utils  # Import the loader helpers after test dependencies are available.

    recorded = []  # Capture every synthetic TVM-FFI invocation.

    def record_build(*_args, **kwargs):  # Replace compilation with an argument recorder.
        recorded.append(kwargs)  # Preserve the linker flags selected by the loader.
        return object()  # Satisfy the loader's module return contract.

    tvm_ffi = types.ModuleType("tvm_ffi")  # Provide the package imported by the loader.
    tvm_ffi.__path__ = []  # Mark the synthetic package as package-like.
    tvm_ffi_cpp = types.ModuleType("tvm_ffi.cpp")  # Provide the compilation submodule.
    tvm_ffi_cpp.load = record_build  # Record AOT calls.
    tvm_ffi_cpp.load_inline = record_build  # Record JIT calls.
    monkeypatch.setitem(sys.modules, "tvm_ffi", tvm_ffi)  # Route the package import to the fixture.
    monkeypatch.setitem(sys.modules, "tvm_ffi.cpp", tvm_ffi_cpp)  # Route compilation imports to the recorder.
    monkeypatch.setenv(utils.DISABLE_KERNEL_CACHE_ENV, "1")  # Force both calls through the compilation path.
    monkeypatch.setattr(utils, "_is_rocm", lambda: True)  # Model a ROCm PyTorch environment.

    def reject_runtime_lookup():  # Make an unnecessary HIP runtime lookup fail the test immediately.
        raise AssertionError("C++-only builds must not discover libamdhip64")  # Explain the violated boundary.

    monkeypatch.setattr(utils, "_rocm_link_flags", reject_runtime_lookup)  # Guard the corrected no-GPU path.
    utils.load_aot("test_cpp_only_aot", cpp_files=["unused.cpp"])  # Exercise an AOT C++-only extension.
    utils.load_jit("test_cpp_only_jit", cpp_files=["unused.cpp"])  # Exercise a JIT C++-only extension.

    assert [call["extra_ldflags"] for call in recorded] == [[], []]  # Require no implicit HIP link flags.
