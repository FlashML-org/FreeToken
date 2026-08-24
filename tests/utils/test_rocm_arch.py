import importlib
import pathlib
from types import SimpleNamespace

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


def test_hip_cflags_emit_one_offload_flag_per_arch(monkeypatch):
    from freetoken.kernel.utils import _hip_cflags

    monkeypatch.setenv("FREETOKEN_ROCM_ARCH", "gfx1200;gfx1201")

    flags = _hip_cflags(["-Wno-unused-command-line-argument"])

    assert "--offload-arch=gfx1200" in flags
    assert "--offload-arch=gfx1201" in flags
    assert not any(";" in flag for flag in flags)


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
    monkeypatch.setattr(cpp_extension, "ROCM_HOME", None)
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    utils._rocm_link_flags.cache_clear()

    flags = utils._rocm_link_flags()

    compat_dir = tmp_path / ".cache" / "freetoken" / "rocm-lib"
    compat_link = compat_dir / "libamdhip64.so"
    assert f"-L{compat_dir}" in flags
    assert f"-Wl,-rpath,{library_dir}" in flags
    assert compat_link.resolve() == versioned_runtime.resolve()

    utils._rocm_link_flags.cache_clear()
