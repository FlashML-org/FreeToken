import importlib
import importlib.util
import pathlib
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

_ARCH_PATH = Path(__file__).resolve().parents[2] / "python" / "freetoken" / "utils" / "arch.py"
_ARCH_SPEC = importlib.util.spec_from_file_location("freetoken_test_arch", _ARCH_PATH)
arch = importlib.util.module_from_spec(_ARCH_SPEC)
assert _ARCH_SPEC.loader is not None
sys.modules[_ARCH_SPEC.name] = arch
_ARCH_SPEC.loader.exec_module(arch)


def _clear_arch_caches() -> None:
    arch.get_rocm_gfx_arch.cache_clear()
    arch.current_gpu_name.cache_clear()
    arch._get_gfx_arch.cache_clear()


@pytest.fixture
def torch_module():
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"torch unavailable: {exc}")
    return torch


def test_rocm_arch_prefers_visible_device_over_multi_arch_build_env(monkeypatch, torch_module):
    monkeypatch.setattr(arch, "is_rocm", lambda: True)
    monkeypatch.setattr(torch_module.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch_module.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch_module.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(gcnArchName="gfx1201:sramecc-:xnack-"),
    )
    monkeypatch.setenv("FREETOKEN_ROCM_ARCH", "gfx1100;gfx1200")
    _clear_arch_caches()

    assert arch.get_rocm_gfx_arch() == "gfx1201"

    _clear_arch_caches()


def test_rocm_arch_falls_back_to_cross_compile_env(monkeypatch, torch_module):
    monkeypatch.setattr(arch, "is_rocm", lambda: True)
    monkeypatch.setattr(torch_module.cuda, "is_available", lambda: False)
    monkeypatch.setenv("FREETOKEN_ROCM_ARCH", "gfx1200;gfx1201")
    _clear_arch_caches()

    assert arch.get_rocm_gfx_arch() == "gfx1200"

    _clear_arch_caches()


def test_rocm_target_matrix_has_closed_family_records():
    matrix = arch.rocm_arch_matrix()
    assert set(matrix) == {
        "gfx1100", "gfx1101", "gfx1102", "gfx1103",
        "gfx1150", "gfx1151", "gfx1200", "gfx1201",
    }
    assert {item.family for item in matrix.values()} == {"rdna3", "rdna3.5", "rdna4"}
    assert all(item.wave_size == 32 for item in matrix.values())
    assert all(item.status == "compile-only" for item in matrix.values())
    assert all(item.generic_gguf_types for item in matrix.values())
    assert arch.rocm_arch_capability("gfx1201:sramecc-:xnack-").target == "gfx1201"


def test_rocm_candidate_matrix_is_exact_and_not_generic_support():
    candidates = arch.rocm_candidate_matrix()
    assert set(candidates) == {"gfx1100"}
    assert candidates["gfx1100"].status == "compile-only"
    assert arch.rocm_candidate_capability("gfx1101") is None
    assert arch.rocm_candidate_capability("gfx1100:sramecc-:xnack-").target == "gfx1100"


def test_rocm_target_matrix_rejects_nearest_arch_substitution():
    with pytest.raises(ValueError, match="not in FreeToken target matrix"):
        arch.rocm_arch_capability("gfx9999")


def test_hip_cflags_emit_one_offload_flag_per_arch(monkeypatch, torch_module):
    from freetoken.kernel.utils import _hip_cflags

    monkeypatch.setenv("FREETOKEN_ROCM_ARCH", "gfx1200;gfx1201")

    flags = _hip_cflags(["-Wno-unused-command-line-argument"])

    assert "--offload-arch=gfx1200" in flags
    assert "--offload-arch=gfx1201" in flags
    assert not any(";" in flag for flag in flags)


def test_hip_cflags_honor_kernel_cache_target(monkeypatch, torch_module):
    monkeypatch.delenv("FREETOKEN_ROCM_ARCH", raising=False)
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    monkeypatch.setenv("FREETOKEN_KERNEL_CACHE_GFX", "gfx1100")

    from freetoken.kernel.utils import _hip_cflags

    flags = _hip_cflags([])

    assert "--offload-arch=gfx1100" in flags


def test_jit_diagnostics_uses_rocm_namespace(monkeypatch, tmp_path, torch_module):
    import torch.utils.cpp_extension as cpp_extension

    from freetoken.kernel import utils

    monkeypatch.delenv("TORCH_EXTENSIONS_DIR", raising=False)
    monkeypatch.setattr(utils, "_is_rocm", lambda: True)
    monkeypatch.setattr(
        cpp_extension,
        "get_default_build_root",
        lambda: str(tmp_path),
    )
    diagnostics = utils.jit_cache_diagnostics("gguf")
    assert diagnostics["build_directory"].endswith("_rocm/gguf")


@pytest.mark.parametrize("value", ["gfx9999", "gfx1201;bad", "gfx1201:xnack-"])
def test_rocm_arch_override_rejects_unknown_tokens(monkeypatch, value, torch_module):
    monkeypatch.setenv("FREETOKEN_ROCM_ARCH", value)
    monkeypatch.setattr(arch, "is_rocm", lambda: True)
    monkeypatch.setattr(torch_module.cuda, "is_available", lambda: False)
    _clear_arch_caches()

    with pytest.raises(ValueError, match="unsupported target"):
        arch.get_rocm_gfx_arch()

    _clear_arch_caches()


def test_rocm_link_flags_support_versioned_modular_sdk(monkeypatch, tmp_path, torch_module):
    import torch.utils.cpp_extension as cpp_extension

    from freetoken.kernel import utils

    sdk = tmp_path / "sdk"
    library_dir = sdk / "lib"
    library_dir.mkdir(parents=True)
    older_runtime = library_dir / "libamdhip64.so.9"
    versioned_runtime = library_dir / "libamdhip64.so.10"
    older_runtime.write_bytes(b"")
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


def test_device_kind_uses_torch_build_without_device_probe(monkeypatch):
    import sys
    import types

    torch_version = types.ModuleType("torch.version")
    torch_version.hip = "6.2"
    torch_version.cuda = None
    torch = types.ModuleType("torch")
    torch.version = torch_version
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.version", torch_version)

    assert arch.device_kind() == "rocm"
    assert arch.is_rocm() is True
    assert arch.is_cuda() is False
    arch.is_rocm.cache_clear()


def test_cuda_build_does_not_report_rocm(monkeypatch):
    import sys
    import types

    torch_version = types.ModuleType("torch.version")
    torch_version.hip = None
    torch_version.cuda = "13.0"
    torch = types.ModuleType("torch")
    torch.version = torch_version
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.version", torch_version)

    assert arch.device_kind() == "cuda"
    assert arch.is_rocm() is False
    assert arch.is_cuda() is True
    arch.is_rocm.cache_clear()
