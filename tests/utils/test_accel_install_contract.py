"""Regression coverage for explicit CUDA versus ROCm dependency selection."""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the supported Python 3.10 floor.
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"


def _project_config() -> dict:
    with PYPROJECT.open("rb") as pyproject_file:
        return tomllib.load(pyproject_file)


def _source_for_extra(config: dict, package: str, extra: str) -> dict:
    sources = config["tool"]["uv"]["sources"][package]
    assert isinstance(sources, list)
    return next(source for source in sources if source.get("extra") == extra)


def _has_conflict(config: dict, left: str, right: str) -> bool:
    for conflict in config["tool"]["uv"]["conflicts"]:
        names = {entry.get("extra") for entry in conflict}
        if {left, right}.issubset(names):
            return True
    return False


def test_accelerator_extras_are_explicit_and_match_torchvision():
    """The root package cannot silently choose an accelerator backend."""
    config = _project_config()
    dependencies = config["project"]["dependencies"]
    extras = config["project"]["optional-dependencies"]

    assert not any(dependency.startswith("torch") for dependency in dependencies)
    assert "torch==2.11.0" in extras["cuda"]
    assert "torchvision==0.26.0" in extras["cuda"]
    assert "torch[device-gfx1151]==2.11.0+rocm7.14.0" in extras["rocm"]
    assert "torchvision[device-gfx1151]==0.26.0+rocm7.14.0" in extras["rocm"]
    assert extras["accel"] == extras["cuda"]
    flashinfer = "flashinfer-python[cu13]==0.6.18.post1"
    assert flashinfer in extras["cuda"]
    assert flashinfer in extras["accel"]
    assert flashinfer in extras["fi"]
    assert not any(dep.startswith("flashinfer-python") for dep in extras["rocm"])
    assert "sglang-kernel==0.4.5" not in extras["rocm"]
    assert "triton==3.7.1+git0263a6a6.rocm7.14.0; sys_platform == 'linux'" in extras["rocm"]
    assert "rocm[libraries]==7.14.0; sys_platform == 'linux'" in extras["rocm"]
    assert "rocm[devel]==7.14.0; sys_platform == 'linux'" in extras["rocm"]
    assert "rocm-sdk-core==7.14.0; sys_platform == 'linux'" in extras["rocm"]
    assert "rocm-sdk-libraries==7.14.0; sys_platform == 'linux'" in extras["rocm"]
    assert "rocm-sdk-devel==7.14.0; sys_platform == 'linux'" in extras["rocm"]
    assert "rocm-sdk-device-gfx1151==7.14.0; sys_platform == 'linux'" in extras["rocm"]


def test_uv_sources_and_conflicts_keep_cuda_out_of_rocm():
    """uv gets an explicit source for every HIP runtime package it resolves."""
    config = _project_config()
    indexes = {index["name"]: index for index in config["tool"]["uv"]["index"]}

    assert indexes["pytorch-cu130"]["url"].endswith("/cu130")
    assert indexes["amd-rocm714"]["url"] == "https://repo.amd.com/rocm/whl-multi-arch/"
    assert indexes["pytorch-cu130"]["explicit"] is True
    assert indexes["amd-rocm714"]["explicit"] is True
    assert config["tool"]["uv"]["no-build-isolation-package"] == ["freetoken"]
    for package in ("torch", "torchvision"):
        for cuda_extra in ("cuda", "accel"):
            assert _source_for_extra(config, package, cuda_extra)["index"] == "pytorch-cu130"
    for package in (
        "triton",
        "rocm",
        "rocm-sdk-core",
        "rocm-sdk-libraries",
        "rocm-sdk-devel",
        "rocm-sdk-device-gfx1151",
        "amd-torch-device-gfx1151",
        "amd-torch-device-gfx115x",
        "amd-torchvision-device-gfx1151",
    ):
        assert _source_for_extra(config, package, "rocm")["index"] == "amd-rocm714"

    for cuda_extra in ("cuda", "accel", "fi", "sgl"):
        assert _has_conflict(config, cuda_extra, "rocm")


def test_rocm_guide_uses_the_isolated_development_sdk():
    guide = (ROOT / "docs" / "amd-rocm-gfx1151.md").read_text()

    assert '"rocm[libraries,devel,device-gfx1151]==7.14.0"' in guide
    assert 'export ROCM_HOME="$(rocm-sdk path --root)"' in guide
    assert 'export PATH="$(rocm-sdk path --bin):$PATH"' in guide
    assert "/opt/rocm-7.14" not in guide


def test_resolver_helper_covers_explicit_and_legacy_cuda_selection():
    """The disposable integration check covers every advertised accelerator path."""
    helper = (ROOT / "scripts" / "verify-accel-resolver.sh").read_text(encoding="utf-8")

    for extra in ("rocm", "cuda", "accel"):
        assert f"resolve {extra}" in helper
    assert "uv pip compile" in helper
    assert '--python-version "${PYTHON_VERSION}"' in helper
    assert '--python-platform "${PYTHON_PLATFORM}"' in helper
    assert '--output-file "${output}"' in helper
    assert "3.10 3.11 3.12 3.13 3.14" in helper
    assert 'resolve rocm "${version}"' in helper
    assert "rocm-sdk-(core|libraries|devel|device-gfx1151)==" in helper
    assert "did not resolve the 7.14.0 development SDK" in helper
    assert "--extra rocm" in helper
    assert "--extra accel" in helper
    assert "--no-cache" in helper
    assert "did not resolve AMD PyTorch 7.14.0 Triton" in helper
