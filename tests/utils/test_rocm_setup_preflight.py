"""No-hardware regression coverage for the ROCm native-build preflight."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_setup_with_fake_torch(
    monkeypatch,
    *,
    hip_version: str | None,
    rocm_home: str | None,
    cuda_home: str | None = None,
):
    """Execute ``setup.py`` without loading real Torch or compiling an extension."""
    fake_torch = types.ModuleType("torch")
    fake_torch.version = types.SimpleNamespace(hip=hip_version)
    fake_torch_utils = types.ModuleType("torch.utils")
    fake_cpp_extension = types.ModuleType("torch.utils.cpp_extension")

    class FakeBuildExtension:
        @staticmethod
        def with_options(**_kwargs):
            return object

    fake_cpp_extension.BuildExtension = FakeBuildExtension
    fake_cpp_extension.CUDA_HOME = cuda_home
    fake_cpp_extension.ROCM_HOME = rocm_home
    fake_cpp_extension.CppExtension = lambda **kwargs: kwargs
    fake_torch.utils = fake_torch_utils

    fake_setuptools = types.ModuleType("setuptools")

    def capture_setup(**kwargs):
        fake_setuptools.setup_kwargs = kwargs

    fake_setuptools.setup = capture_setup

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", fake_torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", fake_cpp_extension)
    monkeypatch.setitem(sys.modules, "setuptools", fake_setuptools)

    spec = importlib.util.spec_from_file_location("_freetoken_test_setup", ROOT / "setup.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.setup_kwargs = fake_setuptools.setup_kwargs
    return module


def _gpu_extensions(module):
    expected = {
        "freetoken.kernel._pinned_tensor",
        "freetoken.kernel._cpu_moe",
    }
    extensions = {
        extension["name"]: extension for extension in module.setup_kwargs["ext_modules"]
    }
    assert expected <= extensions.keys()
    return [extensions[name] for name in sorted(expected)]


def test_hip_build_without_rocm_root_reports_actionable_preflight(monkeypatch):
    """A HIP build must not turn a missing root into an opaque ``Path(None)`` error."""
    with pytest.raises(RuntimeError) as error:
        _load_setup_with_fake_torch(monkeypatch, hip_version="7.2", rocm_home=None)

    message = str(error.value)
    assert "ROCM_HOME" in message
    assert "ROCM_PATH" in message
    assert "HIP_PATH" in message
    assert "Path(None)" not in message


def test_hip_build_links_versioned_runtime_from_rocm_home(monkeypatch, tmp_path):
    """The HIP extension must link the runtime soname when no unversioned .so exists."""
    rocm_home = tmp_path / "rocm"
    include_dir = rocm_home / "include"
    library_dir = rocm_home / "lib64"
    include_dir.mkdir(parents=True)
    library_dir.mkdir()
    (library_dir / "libamdhip64.so.7").touch()

    module = _load_setup_with_fake_torch(
        monkeypatch,
        hip_version="7.2",
        rocm_home=str(rocm_home),
    )

    assert module._gpu_runtime_paths() == (
        [str(include_dir)],
        [str(library_dir)],
        [],
        ["-l:libamdhip64.so.7"],
    )
    for extension in _gpu_extensions(module):
        assert extension["libraries"] == []
        assert extension["extra_link_args"] == ["-l:libamdhip64.so.7"]
        assert extension["define_macros"] == [("FREETOKEN_USE_ROCM", "1")]


def test_cuda_build_keeps_cudart_link_and_no_hip_macro(monkeypatch, tmp_path):
    """Selecting a CUDA PyTorch runtime must preserve the existing CUDA linkage."""
    cuda_home = tmp_path / "cuda"
    include_dir = cuda_home / "include"
    library_dir = cuda_home / "lib64"
    include_dir.mkdir(parents=True)
    library_dir.mkdir()

    module = _load_setup_with_fake_torch(
        monkeypatch,
        hip_version=None,
        rocm_home=None,
        cuda_home=str(cuda_home),
    )

    assert module.GPU_RUNTIME_MACROS == []
    assert module._gpu_runtime_paths() == (
        [str(include_dir)],
        [str(library_dir)],
        ["cudart"],
        [],
    )
    for extension in _gpu_extensions(module):
        assert extension["libraries"] == ["cudart"]
        assert extension["extra_link_args"] == []
        assert extension["define_macros"] == []


def test_hip_build_prefers_rocm_when_cuda_toolkit_is_also_available(
    monkeypatch, tmp_path
):
    """The active HIP Torch ABI must win over a discoverable CUDA toolkit."""
    rocm_home = tmp_path / "rocm"
    (rocm_home / "include").mkdir(parents=True)
    rocm_lib = rocm_home / "lib64"
    rocm_lib.mkdir()
    (rocm_lib / "libamdhip64.so.7").touch()

    cuda_home = tmp_path / "cuda"
    (cuda_home / "include").mkdir(parents=True)
    (cuda_home / "lib64").mkdir()

    module = _load_setup_with_fake_torch(
        monkeypatch,
        hip_version="7.2",
        rocm_home=str(rocm_home),
        cuda_home=str(cuda_home),
    )

    assert module.IS_ROCM
    assert module._gpu_runtime_paths() == (
        [str(rocm_home / "include")],
        [str(rocm_lib)],
        [],
        ["-l:libamdhip64.so.7"],
    )
    for extension in _gpu_extensions(module):
        assert extension["libraries"] == []
        assert extension["extra_link_args"] == ["-l:libamdhip64.so.7"]
        assert extension["define_macros"] == [("FREETOKEN_USE_ROCM", "1")]


def test_linux_row_store_extension_does_not_link_gpu_runtime(monkeypatch, tmp_path):
    rocm_home = tmp_path / "rocm"
    (rocm_home / "include").mkdir(parents=True)
    library_dir = rocm_home / "lib64"
    library_dir.mkdir()
    (library_dir / "libamdhip64.so.7").touch()

    module = _load_setup_with_fake_torch(
        monkeypatch,
        hip_version="7.2",
        rocm_home=str(rocm_home),
    )

    row_store = [
        extension
        for extension in module.setup_kwargs["ext_modules"]
        if extension["name"] == "freetoken.kernel._row_store"
    ]
    if sys.platform == "linux":
        assert len(row_store) == 1
        assert "libraries" not in row_store[0]
        assert "extra_link_args" not in row_store[0]
        assert "define_macros" not in row_store[0]
    else:
        assert row_store == []
