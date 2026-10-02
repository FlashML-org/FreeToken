"""No-hardware regression coverage for the ROCm native-build preflight."""

from __future__ import annotations

import importlib.util
from importlib import metadata
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
    torch_version: str = "2.13.0+rocm10.0.0",
    triton_version: str = "3.8.0+git4cff872c.rocm10.0.0",
    rocm_sdk_version: str = "10.0.0",
    env_rocm_home: str | None = None,
):
    """Execute ``setup.py`` without loading real Torch or compiling an extension."""
    for variable in ("ROCM_HOME", "ROCM_PATH", "HIP_PATH"):  # Keep real toolkit selectors out of the synthetic fixture.
        monkeypatch.delenv(variable, raising=False)  # Ensure each test controls every accepted SDK selector.
    if env_rocm_home is not None:  # Model an explicitly selected AMD build root when requested.
        monkeypatch.setenv("ROCM_HOME", env_rocm_home)  # Declare ROCm intent before setup.py evaluates module constants.
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = torch_version  # Supply the wheel provenance checked by the ROCm 10 preflight.
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

    def fake_distribution_version(name: str) -> str:  # Supply deterministic accelerator package metadata.
        versions = {"triton": triton_version, "rocm-sdk-core": rocm_sdk_version}  # Model the qualified ROCm 10 environment.
        if name in versions:  # Return only the packages intentionally installed in the fixture.
            return versions[name]  # Preserve local ROCm suffixes for provenance checks.
        raise metadata.PackageNotFoundError(name)  # Model absent CUDA-only optional packages.

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", fake_torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", fake_cpp_extension)
    monkeypatch.setitem(sys.modules, "setuptools", fake_setuptools)
    monkeypatch.setattr(metadata, "version", fake_distribution_version)  # Isolate setup preflight from the test runner's packages.

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
    (include_dir / "hip").mkdir(parents=True)
    (include_dir / "hip" / "hip_runtime.h").touch()
    library_dir.mkdir()
    (library_dir / "libamdhip64.so.7").touch()

    module = _load_setup_with_fake_torch(
        monkeypatch,
        hip_version="7.2",
        rocm_home=str(rocm_home),
    )

    assert module._gpu_runtime_paths() == (
        [str(ROOT / "python" / "freetoken" / "kernel" / "csrc" / "include"), str(include_dir)],
        [str(library_dir)],
        [":libamdhip64.so.7"],
        [f"-Wl,-rpath,{library_dir}"],
    )
    for extension in _gpu_extensions(module):
        assert extension["libraries"] == [":libamdhip64.so.7"]
        assert extension["extra_link_args"] == [f"-Wl,-rpath,{library_dir}"]
        assert extension["define_macros"] == [("FREETOKEN_USE_ROCM", "1")]


def test_hip_build_rejects_non_rocm10_torch(monkeypatch, tmp_path):  # Enforce the selected externally provisioned runtime major.
    rocm_home = tmp_path / "rocm"  # Provide a syntactically complete SDK root.
    (rocm_home / "include" / "hip").mkdir(parents=True)  # Create the expected HIP include hierarchy.
    (rocm_home / "include" / "hip" / "hip_runtime.h").touch()  # Materialize the required header.
    (rocm_home / "lib").mkdir()  # Create the runtime library directory.
    (rocm_home / "lib" / "libamdhip64.so.10").touch()  # Provide a valid ROCm 10 runtime SONAME.

    with pytest.raises(RuntimeError, match="torch .* is not a ROCm 10 build"):  # Require wheel provenance to fail closed.
        _load_setup_with_fake_torch(  # Execute the real setup preflight with an older ROCm wheel identity.
            monkeypatch,  # Keep the synthetic modules scoped to this test.
            hip_version="7.15",  # Preserve the valid HIP component version that must not decide the ROCm release.
            rocm_home=str(rocm_home),  # Point setup at the valid SDK fixture.
            torch_version="2.13.0+rocm7.14.0",  # Reproduce the prohibited wheel stack.
        )  # The preflight must stop before extension definitions are accepted.


def test_hip_build_rejects_generic_triton(monkeypatch, tmp_path):  # Prevent a generic Triton wheel from entering the ROCm 10 ABI.
    rocm_home = tmp_path / "rocm"  # Provide the selected SDK root.
    (rocm_home / "include" / "hip").mkdir(parents=True)  # Create the HIP include hierarchy.
    (rocm_home / "include" / "hip" / "hip_runtime.h").touch()  # Materialize the required header.
    (rocm_home / "lib").mkdir()  # Create the runtime library directory.
    (rocm_home / "lib" / "libamdhip64.so.10").touch()  # Provide a valid runtime library.

    with pytest.raises(RuntimeError, match="triton 3.8.0 is not a ROCm 10 build"):  # Require local ROCm provenance in Triton metadata.
        _load_setup_with_fake_torch(  # Execute the preflight against a generic wheel identity.
            monkeypatch,  # Scope synthetic package metadata to this test.
            hip_version="7.15",  # Keep HIP component evidence valid.
            rocm_home=str(rocm_home),  # Use the complete SDK fixture.
            triton_version="3.8.0",  # Remove the required ROCm 10 local-version marker.
        )  # The preflight must reject the mixed accelerator environment.


def test_rocm_root_rejects_cuda_torch(monkeypatch, tmp_path):  # Catch a combined ROCm marker and CUDA accelerator extra.
    rocm_home = tmp_path / "rocm"  # Represent an explicitly selected AMD toolkit.
    with pytest.raises(RuntimeError, match="ROCm environment variables require a ROCm 10 PyTorch build"):  # Reject CUDA selection early.
        _load_setup_with_fake_torch(  # Execute setup with CUDA Torch plus an AMD root selector.
            monkeypatch,  # Isolate the synthetic toolchain.
            hip_version=None,  # Model CUDA or CPU-only Torch.
            rocm_home=None,  # Keep Torch's detected ROCm root empty.
            cuda_home=None,  # Prevent a valid CUDA toolkit from obscuring the contract error.
            torch_version="2.11.0+cu130",  # Reproduce the CUDA extra's wheel identity.
            env_rocm_home=str(rocm_home),  # Declare AMD build intent independently of Torch.
        )  # Setup must not silently reinterpret the build as CUDA.


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
        [str(ROOT / "python" / "freetoken" / "kernel" / "csrc" / "include"), str(include_dir)],
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
    (rocm_home / "include" / "hip").mkdir(parents=True)
    (rocm_home / "include" / "hip" / "hip_runtime.h").touch()
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
        [str(ROOT / "python" / "freetoken" / "kernel" / "csrc" / "include"), str(rocm_home / "include")],
        [str(rocm_lib)],
        [":libamdhip64.so.7"],
        [f"-Wl,-rpath,{rocm_lib}"],
    )
    for extension in _gpu_extensions(module):
        assert extension["libraries"] == [":libamdhip64.so.7"]
        assert extension["extra_link_args"] == [f"-Wl,-rpath,{rocm_lib}"]
        assert extension["define_macros"] == [("FREETOKEN_USE_ROCM", "1")]


def test_linux_row_store_extension_does_not_link_gpu_runtime(monkeypatch, tmp_path):
    rocm_home = tmp_path / "rocm"
    (rocm_home / "include" / "hip").mkdir(parents=True)
    (rocm_home / "include" / "hip" / "hip_runtime.h").touch()
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
