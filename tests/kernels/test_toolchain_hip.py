"""HIP (ROCm) toolchain helper tests.

These load ``kernel/_toolchain.py`` by file path (as setup.py and the kernel-cache
build backend do) so they need no torch import and no ROCm toolkit to exercise the
pure parsing/detection logic.
"""

import importlib.util
import os
from pathlib import Path

import pytest

_TOOLCHAIN_PATH = (
    Path(__file__).resolve().parents[2] / "python" / "freetoken" / "kernel" / "_toolchain.py"
)


def _load_toolchain():
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", _TOOLCHAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tc():
    return _load_toolchain()


def _write_fake_hipcc(tmp_path, version: str):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    hipcc = bin_dir / "hipcc"
    hipcc.write_text(
        "#!/bin/sh\n"
        f'echo "HIP version: {version}"\n',
        encoding="utf-8",
    )
    hipcc.chmod(0o755)
    return str(hipcc)


def test_hip_hip_version(tc, tmp_path):
    hipcc = _write_fake_hipcc(tmp_path, "6.2.41000")
    assert tc.hip_hip_version(hipcc) == (6, 2)


def test_hip_hip_version_missing(tc):
    assert tc.hip_hip_version("/nonexistent/hipcc") is None


def test_hipcc_path_rocm_home(tc, tmp_path, monkeypatch):
    hipcc = _write_fake_hipcc(tmp_path, "6.2.41000")
    monkeypatch.setenv("ROCM_HOME", str(tmp_path))
    monkeypatch.delenv("HIP_PATH", raising=False)
    assert tc._hipcc_path() == hipcc


def test_hipcc_path_default_opt_rocm(tc, monkeypatch):
    monkeypatch.delenv("ROCM_HOME", raising=False)
    monkeypatch.delenv("HIP_PATH", raising=False)
    # If a real /opt/rocm/bin/hipcc exists it wins; otherwise we expect None (no PATH hit
    # guaranteed in the test sandbox, so only assert it returns None or a path string).
    result = tc._hipcc_path()
    if os.path.exists("/opt/rocm/bin/hipcc"):
        assert result is not None
    else:
        assert result is None or result.endswith("hipcc")


def test_torch_hip_major_rocm(tc, monkeypatch):
    # Simulate ROCm torch via a fake torch.version with hip set.
    import sys
    import types

    fake_torch = types.ModuleType("torch")
    fake_version = types.ModuleType("torch.version")
    fake_version.hip = "6.2.4100000"
    fake_version.cuda = None
    fake_torch.version = fake_version
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert tc.is_rocm_torch() is True
    assert tc.torch_hip_major() == 6
