import importlib
import inspect
import os
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

from freetoken.utils import arch


def _clear_arch_caches() -> None:
    arch.get_rocm_gfx_arch.cache_clear()
    arch.is_gfx11xx_family.cache_clear()
    arch.is_gfx12xx_family.cache_clear()


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
    assert arch.is_gfx12xx_family()
    assert not arch.is_gfx11xx_family()

    _clear_arch_caches()


def test_rocm_arch_falls_back_to_cross_compile_env(monkeypatch):
    monkeypatch.setattr(arch, "is_rocm", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("FREETOKEN_ROCM_ARCH", "gfx1200;gfx1201")
    _clear_arch_caches()

    assert arch.get_rocm_gfx_arch() == "gfx1200"

    _clear_arch_caches()


def test_rocm_compile_flags_emit_one_offload_flag_per_arch(monkeypatch):
    from freetoken.kernel.utils import rocm_compile_flags

    monkeypatch.setenv("FREETOKEN_ROCM_ARCH", "gfx1200;gfx1201")

    flags = rocm_compile_flags(["-Wno-unused-command-line-argument"])

    assert "--offload-arch=gfx1200" in flags
    assert "--offload-arch=gfx1201" in flags
    assert not any(";" in flag for flag in flags)


def test_ensure_rocm_env_discovers_modular_sdk_and_visible_arch(monkeypatch, tmp_path):
    from freetoken.kernel import _toolchain

    sdk = tmp_path / "_rocm_sdk_core"
    clang = sdk / "lib" / "llvm" / "bin" / "clang.exe"
    clang.parent.mkdir(parents=True)
    clang.write_bytes(b"")
    module = SimpleNamespace(__file__=str(sdk / "__init__.py"))

    monkeypatch.setitem(sys.modules, "_rocm_sdk_core", module)
    fake_os = SimpleNamespace(environ={}, path=os.path)
    monkeypatch.setattr(_toolchain, "os", fake_os)
    monkeypatch.setattr(torch.version, "hip", "test-rocm")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(gcnArchName="gfx1201:sramecc-:xnack-"),
    )
    _toolchain.ensure_rocm_env.cache_clear()

    try:
        assert _toolchain.ensure_rocm_env() == str(sdk)
        assert fake_os.environ["HIP_PATH"] == str(sdk)
        assert fake_os.environ["ROCM_HOME"] == str(sdk)
        assert "ROCM_PATH" not in fake_os.environ
        for variable in (
            "TVM_FFI_ROCM_ARCH_LIST",
            "PYTORCH_ROCM_ARCH",
            "TRITON_OVERRIDE_ARCH",
            "ROCM_SDK_TARGET_FAMILY",
        ):
            assert fake_os.environ[variable] == "gfx1201"
    finally:
        _toolchain.ensure_rocm_env.cache_clear()


@pytest.mark.parametrize(
    ("rows", "expected"),
    ((1, 1), (2, 2), (3, 4), (4, 4), (17, 32), (32, 32), (33, 32)),
)
def test_triton_gguf_row_tile_tracks_small_decode_batches(rows, expected):
    from freetoken.kernel.triton.gguf_gemm.utils import select_row_tile

    assert select_row_tile(rows) == expected


def test_triton_gguf_row_tile_rejects_empty_batches():
    from freetoken.kernel.triton.gguf_gemm.utils import select_row_tile

    with pytest.raises(ValueError, match="positive"):
        select_row_tile(0)


def test_triton_gguf_launchers_apply_row_aware_tiles():
    from freetoken.kernel.triton.gguf_gemm import utils
    from freetoken.kernel.triton.gguf_gemm.standard_quant import q4_0

    assert "block_m = select_row_tile(X_2d.shape[0])" in inspect.getsource(
        utils.run_triton_kernel
    )
    assert "BLOCK_M=block_m" in inspect.getsource(utils.run_triton_kernel)
    assert "block_m = select_row_tile(X_2d.shape[0])" in inspect.getsource(
        q4_0.ggml_gemm_q4_0_triton
    )
    assert "BLOCK_M=block_m" in inspect.getsource(q4_0.ggml_gemm_q4_0_triton)


@pytest.mark.skipif(os.name == "nt", reason="POSIX .so/rpath compatibility contract")
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

    for variable in ("ROCM_HOME", "ROCM_PATH", "HIP_PATH"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(cpp_extension, "ROCM_HOME", None)
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    utils._rocm_link_flags.cache_clear()

    try:
        flags = utils._rocm_link_flags()

        compat_dir = tmp_path / ".cache" / "freetoken" / "rocm-lib"
        compat_link = compat_dir / "libamdhip64.so"
        assert f"-L{compat_dir}" in flags
        assert f"-Wl,-rpath,{library_dir}" in flags
        assert compat_link.resolve() == versioned_runtime.resolve()
    finally:
        utils._rocm_link_flags.cache_clear()


@pytest.mark.skipif(os.name != "nt", reason="native Windows import-library contract")
def test_rocm_link_flags_support_windows_modular_sdk(monkeypatch, tmp_path):
    import torch.utils.cpp_extension as cpp_extension

    from freetoken.kernel import utils

    sdk = tmp_path / "sdk"
    library_dir = sdk / "lib"
    library_dir.mkdir(parents=True)
    (library_dir / "amdhip64.lib").write_bytes(b"")
    real_find_spec = importlib.util.find_spec

    def find_spec(name: str):
        if name == "_rocm_sdk_core":
            return SimpleNamespace(submodule_search_locations=[str(sdk)])
        return real_find_spec(name)

    for variable in ("ROCM_HOME", "ROCM_PATH", "HIP_PATH"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(cpp_extension, "ROCM_HOME", None)
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    utils._rocm_link_flags.cache_clear()

    try:
        assert utils._rocm_link_flags() == [
            f"/LIBPATH:{library_dir}",
            "amdhip64.lib",
        ]
    finally:
        utils._rocm_link_flags.cache_clear()
