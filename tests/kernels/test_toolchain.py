from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from freetoken.kernel import _toolchain


def _load_kernel_cache_backend():
    path = Path(__file__).resolve().parents[2] / "freetoken-kernel-cache" / "build_backend.py"
    spec = importlib.util.spec_from_file_location("_freetoken_kernel_cache_backend", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _isolate_environment(
    monkeypatch: pytest.MonkeyPatch,
    cuda: str | None = "13.0",
) -> None:
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv(_toolchain.ALLOW_MISMATCH_ENV, raising=False)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(
        _toolchain,
        "_torch_cuda_release",
        lambda: None if cuda is None else tuple(map(int, cuda.split("."))),
    )
    monkeypatch.setattr(_toolchain.shutil, "which", lambda _name: None)


def test_cpu_or_missing_torch_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_environment(monkeypatch, cuda=None)

    assert _toolchain.configure_cuda_toolchain() is None
    assert "CUDA_HOME" not in os.environ
    assert os.environ["PATH"] == "/usr/bin:/bin"


def test_jit_check_requires_a_valid_exact_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_environment(monkeypatch)
    monkeypatch.setattr(_toolchain, "nvcc_release", lambda _path: None)

    with pytest.raises(RuntimeError, match=r"no nvcc matching torch CUDA 13\.0"):
        _toolchain.check_nvcc_matches_torch()


def test_discovers_exact_versioned_usr_local_toolkit(monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_environment(monkeypatch)
    monkeypatch.setattr(
        _toolchain,
        "nvcc_release",
        lambda path: (13, 0) if path == "/usr/local/cuda-13.0/bin/nvcc" else None,
    )

    assert _toolchain.configure_cuda_toolchain() == "/usr/local/cuda-13.0"
    assert os.environ["CUDA_HOME"] == "/usr/local/cuda-13.0"
    assert os.environ["PATH"].split(os.pathsep)[0] == "/usr/local/cuda-13.0/bin"


def test_explicit_exact_cuda_home_remains_authoritative(monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_environment(monkeypatch, cuda="12.8")
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda-12.8")
    monkeypatch.setattr(_toolchain, "nvcc_release", lambda _path: (12, 8))

    assert _toolchain.configure_cuda_toolchain() == "/usr/local/cuda-12.8"
    assert os.environ["PATH"].split(os.pathsep)[0] == "/usr/local/cuda-12.8/bin"


def test_explicit_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_environment(monkeypatch)
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda-12.8")
    monkeypatch.setattr(_toolchain, "nvcc_release", lambda _path: (12, 8))

    with pytest.raises(RuntimeError, match=r"nvcc 12\.8.*torch CUDA 13\.0"):
        _toolchain.configure_cuda_toolchain()
    assert os.environ["CUDA_HOME"] == "/usr/local/cuda-12.8"


def test_explicit_mismatch_override_preserves_selected_home(monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_environment(monkeypatch)
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda-12.8")
    monkeypatch.setenv(_toolchain.ALLOW_MISMATCH_ENV, "1")
    monkeypatch.setattr(_toolchain, "nvcc_release", lambda _path: (12, 8))

    assert _toolchain.configure_cuda_toolchain() == "/usr/local/cuda-12.8"


def test_auto_discovery_rejects_wrong_path_nvcc(monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_environment(monkeypatch)
    monkeypatch.setattr(_toolchain.shutil, "which", lambda _name: "/usr/bin/nvcc")
    monkeypatch.setattr(
        _toolchain,
        "nvcc_release",
        lambda path: None if path == "/usr/local/cuda-13.0/bin/nvcc" else (12, 0),
    )

    with pytest.raises(RuntimeError, match=r"nvcc 12\.0.*torch CUDA 13\.0"):
        _toolchain.configure_cuda_toolchain()
    assert "CUDA_HOME" not in os.environ


def test_prebuilt_server_ignores_mismatched_path_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from freetoken.server.launch import launch_server

    _isolate_environment(monkeypatch)
    monkeypatch.setattr(_toolchain.shutil, "which", lambda _name: "/usr/bin/nvcc")
    monkeypatch.setattr(
        _toolchain,
        "nvcc_release",
        lambda path: None if path == "/usr/local/cuda-13.0/bin/nvcc" else (12, 8),
    )

    server_args = SimpleNamespace(gpu=())
    observed: list[tuple[object, bool]] = []
    fake_args_module = SimpleNamespace(
        parse_args=lambda _argv, run_shell, prog=None: (server_args, run_shell)
    )
    fake_api_module = SimpleNamespace(
        run_api_server=lambda args, _start, run_shell: observed.append((args, run_shell))
    )
    monkeypatch.setitem(sys.modules, "freetoken.server.args", fake_args_module)
    monkeypatch.setitem(sys.modules, "freetoken.server.api_server", fake_api_module)

    launch_server(argv=[], prog="ft serve")

    assert observed == [(server_args, False)]
    assert "CUDA_HOME" not in os.environ
    with pytest.raises(RuntimeError, match=r"nvcc 12\.8.*torch CUDA 13\.0"):
        _toolchain.check_nvcc_matches_torch()


def test_updates_already_imported_cpp_extension_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_environment(monkeypatch)
    monkeypatch.setattr(_toolchain, "nvcc_release", lambda _path: (13, 0))

    class FakeCppExtension:
        CUDA_HOME = None

    fake = FakeCppExtension()
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", fake)

    _toolchain.configure_cuda_toolchain()
    assert fake.CUDA_HOME == "/usr/local/cuda-13.0"


def test_path_exact_toolkit_is_published_from_its_own_nvcc(monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_environment(monkeypatch)
    path_nvcc = Path("/opt/cuda-13.0/bin/nvcc")
    monkeypatch.setattr(_toolchain.shutil, "which", lambda _name: str(path_nvcc))
    monkeypatch.setattr(
        _toolchain,
        "nvcc_release",
        lambda path: None if path == "/usr/local/cuda-13.0/bin/nvcc" else (13, 0),
    )

    assert _toolchain.configure_cuda_toolchain() == "/opt/cuda-13.0"


def test_path_symlink_uses_public_entry_toolkit_root(monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_environment(monkeypatch)
    monkeypatch.setattr(_toolchain.shutil, "which", lambda _name: "/usr/bin/nvcc")
    monkeypatch.setattr(
        _toolchain,
        "nvcc_release",
        lambda path: None if path == "/usr/local/cuda-13.0/bin/nvcc" else (13, 0),
    )

    assert _toolchain.configure_cuda_toolchain() == "/usr"
    assert os.environ["CUDA_HOME"] == "/usr"


def test_kernel_cache_metadata_does_not_require_nvcc(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _load_kernel_cache_backend()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(cuda="12.8")))

    def fail_if_called() -> None:
        raise AssertionError("metadata generation must not require a compiler")

    monkeypatch.setattr(backend, "_check_toolchain", fail_if_called)
    assert backend._cuda_version_suffix() == "+cu128"


def test_kernel_cache_build_checks_toolchain_before_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _load_kernel_cache_backend()

    def reject_toolchain() -> None:
        raise RuntimeError("toolchain rejected")

    monkeypatch.setattr(backend, "_check_toolchain", reject_toolchain)
    with pytest.raises(RuntimeError, match="toolchain rejected"):
        backend._build_jit_cache()


def test_server_help_does_not_require_toolchain_resolution(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from freetoken.server.launch import launch_server

    def fail_if_called() -> None:
        raise AssertionError("toolchain resolution must follow argument parsing")

    monkeypatch.setattr(_toolchain, "configure_cuda_toolchain", fail_if_called)
    with pytest.raises(SystemExit) as exc_info:
        launch_server(argv=["--help"], prog="ft serve")

    assert exc_info.value.code == 0
    assert "usage: ft serve" in capsys.readouterr().out
