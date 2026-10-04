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


def test_accelerator_extras_are_explicit_and_rocm_is_externally_provisioned():
    """The root package cannot silently choose an accelerator backend."""
    config = _project_config()
    dependencies = config["project"]["dependencies"]
    extras = config["project"]["optional-dependencies"]

    assert not any(dependency.startswith(("torch", "torchvision", "triton")) for dependency in dependencies)
    assert "torch==2.11.0" in extras["cuda"]
    assert "torchvision==0.26.0" in extras["cuda"]
    assert extras["rocm"] == []
    assert extras["accel"] == extras["cuda"]
    flashinfer = "flashinfer-python[cu13]==0.6.18.post1"
    assert flashinfer in extras["cuda"]
    assert flashinfer in extras["accel"]
    assert flashinfer in extras["fi"]
    assert not any("rocm7." in dependency.lower() or "7.14" in dependency for dependency in dependencies)
    assert not any("rocm7." in dependency.lower() or "7.14" in dependency for dependency in extras["rocm"])


def test_uv_sources_and_conflicts_keep_cuda_out_of_rocm():
    """uv maps CUDA wheels explicitly and never selects a ROCm 7.x source."""
    config = _project_config()
    uv_config = config["tool"]["uv"]
    indexes = {index["name"]: index for index in uv_config["index"]}

    assert indexes["pytorch-cu130"]["url"].endswith("/cu130")
    assert indexes["pytorch-cu130"]["explicit"] is True
    assert "amd-rocm714" not in indexes
    assert uv_config["no-build-isolation-package"] == ["freetoken"]
    for package in ("torch", "torchvision"):
        for cuda_extra in ("cuda", "accel"):
            assert _source_for_extra(config, package, cuda_extra)["index"] == "pytorch-cu130"
    assert all(
        source.get("index") != "amd-rocm714"
        for package_sources in uv_config["sources"].values()
        for source in (package_sources if isinstance(package_sources, list) else [package_sources])
    )

    for cuda_extra in ("cuda", "accel", "fi", "sgl"):
        assert _has_conflict(config, cuda_extra, "rocm")


def test_rocm_guide_uses_the_provisioned_rocm10_runtime():
    guide = (ROOT / "docs" / "amd-rocm-gfx1151.md").read_text()

    assert "ROCm `10.0.0`" in guide
    assert "`/opt/rocm-10.0`" in guide
    assert "PyTorch `2.13.0+rocm10.0.0`" in guide
    assert "rocm7.14.0" not in guide.lower()
    assert "rocm-sdk init" not in guide


def test_resolver_helper_covers_explicit_and_legacy_cuda_selection():
    """The disposable integration check covers every advertised accelerator path."""
    helper = (ROOT / "scripts" / "verify-accel-resolver.sh").read_text(encoding="utf-8")

    for extra in ("rocm", "cuda", "accel"):
        assert f"resolve {extra}" in helper
    assert "uv pip compile" in helper
    assert '--python-version "${python_version}"' in helper
    assert '--python-platform "${PYTHON_PLATFORM}"' in helper
    assert '--output-file "${output}"' in helper
    assert "3.10 3.11 3.12 3.13 3.14" in helper
    assert 'resolve rocm "${version}"' in helper
    assert "ROCM_RUNTIME_PACKAGE_PATTERN" in helper
    assert "rocm-sdk-(core|libraries|devel|device-gfx1151)==" not in helper
    assert "--no-cache" in helper
    assert "ROCm marker extra selected an accelerator runtime package" in helper
    assert "ROCm remains externally provisioned" in helper
    assert "uv pip compile ignores tool.uv.conflicts" in helper
