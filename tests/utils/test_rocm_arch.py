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
