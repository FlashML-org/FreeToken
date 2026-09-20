"""No-hardware regression coverage for the ROCm native-build preflight."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_setup_with_fake_torch(monkeypatch, *, hip_version: str | None, rocm_home: str | None):
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
    fake_cpp_extension.CUDA_HOME = None
    fake_cpp_extension.ROCM_HOME = rocm_home
    fake_cpp_extension.CppExtension = lambda **kwargs: kwargs
    fake_torch.utils = fake_torch_utils

    fake_setuptools = types.ModuleType("setuptools")
    fake_setuptools.setup = lambda **_kwargs: None

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", fake_torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", fake_cpp_extension)
    monkeypatch.setitem(sys.modules, "setuptools", fake_setuptools)

    spec = importlib.util.spec_from_file_location("_freetoken_test_setup", ROOT / "setup.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hip_build_without_rocm_root_reports_actionable_preflight(monkeypatch):
    """A HIP build must not turn a missing root into an opaque ``Path(None)`` error."""
    with pytest.raises(RuntimeError) as error:
        _load_setup_with_fake_torch(monkeypatch, hip_version="7.2", rocm_home=None)

    message = str(error.value)
    assert "ROCM_HOME" in message
    assert "ROCM_PATH" in message
    assert "HIP_PATH" in message
    assert "Path(None)" not in message
