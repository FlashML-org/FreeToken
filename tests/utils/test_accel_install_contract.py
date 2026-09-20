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
    assert "torch==2.11.0" in extras["rocm"]
    assert "torchvision==0.26.0" in extras["rocm"]
    assert extras["accel"] == extras["cuda"]
    assert "flashinfer-python[cu13]>=0.6,<0.7" not in extras["rocm"]
    assert "sglang-kernel==0.4.5" not in extras["rocm"]
    assert not any(dependency.startswith("triton==") for dependency in extras["rocm"])


def test_uv_sources_and_conflicts_keep_cuda_out_of_rocm():
    """uv gets an explicit source for every HIP runtime package it resolves."""
    config = _project_config()
    indexes = {index["name"]: index for index in config["tool"]["uv"]["index"]}

    assert indexes["pytorch-cu130"]["url"].endswith("/cu130")
    assert indexes["pytorch-rocm72"]["url"].endswith("/rocm7.2")
    assert indexes["pytorch-cu130"]["explicit"] is True
    assert indexes["pytorch-rocm72"]["explicit"] is True
    assert config["tool"]["uv"]["no-build-isolation-package"] == ["freetoken"]
    for package in ("torch", "torchvision"):
        for cuda_extra in ("cuda", "accel"):
            assert _source_for_extra(config, package, cuda_extra)["index"] == "pytorch-cu130"
        assert _source_for_extra(config, package, "rocm")["index"] == "pytorch-rocm72"
    for package in ("pytorch-triton-rocm", "triton-rocm"):
        assert _source_for_extra(config, package, "rocm")["index"] == "pytorch-rocm72"

    for cuda_extra in ("cuda", "accel", "fi", "sgl"):
        assert _has_conflict(config, cuda_extra, "rocm")


def test_resolver_helper_covers_explicit_and_legacy_cuda_selection():
    """The disposable integration check covers every advertised accelerator path."""
    helper = (ROOT / "scripts" / "verify-accel-resolver.sh").read_text(encoding="utf-8")

    for extra in ("rocm", "cuda", "accel"):
        assert f"resolve {extra}" in helper
    assert "--no-install-project" in helper
    assert "--dry-run" in helper
