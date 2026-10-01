from __future__ import annotations

import importlib
import logging
from types import SimpleNamespace

import pytest


@pytest.fixture
def backend_log_capture():
    from freetoken.kernel import backend

    records = []

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    capture_handler = CaptureHandler()
    backend.logger.addHandler(capture_handler)
    try:
        yield backend, records
    finally:
        backend.logger.removeHandler(capture_handler)


def test_flashinfer_missing_logs_fallback_once(monkeypatch, backend_log_capture):
    backend, records = backend_log_capture

    def missing(_name):
        raise ModuleNotFoundError("No module named 'flashinfer'", name="flashinfer")

    monkeypatch.setattr(importlib, "import_module", missing)
    backend.is_flashinfer_installed.cache_clear()
    try:
        assert backend.is_flashinfer_installed() is False
        assert backend.is_flashinfer_installed() is False
        output = "\n".join(record.getMessage() for record in records)
        assert output.count("flashinfer is not installed") == 1
        assert "Triton kernel fallback" in output
        assert "failed to import" not in output
    finally:
        backend.is_flashinfer_installed.cache_clear()


def test_flashinfer_missing_transitive_dependency_is_diagnosed(monkeypatch, backend_log_capture):
    backend, records = backend_log_capture

    def missing_dependency(_name):
        raise ModuleNotFoundError(
            "No module named 'flashinfer._kernels'", name="flashinfer._kernels"
        )

    monkeypatch.setattr(importlib, "import_module", missing_dependency)
    backend.is_flashinfer_installed.cache_clear()
    try:
        assert backend.is_flashinfer_installed() is False
        output = "\n".join(record.getMessage() for record in records)
        assert "flashinfer is installed but failed to import" in output
        assert "flashinfer._kernels" in output
        assert "Triton kernel fallback" in output
    finally:
        backend.is_flashinfer_installed.cache_clear()


def test_flashinfer_success_is_available_without_fallback_log(monkeypatch, backend_log_capture):
    backend, records = backend_log_capture
    monkeypatch.setattr(importlib, "import_module", lambda _name: object())
    backend.is_flashinfer_installed.cache_clear()
    try:
        assert backend.is_flashinfer_installed() is True
        assert records == []
    finally:
        backend.is_flashinfer_installed.cache_clear()


def test_broken_flashinfer_import_logs_reason_and_keeps_fallback(monkeypatch, backend_log_capture):
    backend, records = backend_log_capture

    def broken(_name):
        raise RuntimeError("incompatible native extension")

    monkeypatch.setattr(importlib, "import_module", broken)
    backend.is_flashinfer_installed.cache_clear()
    try:
        assert backend.is_flashinfer_installed() is False
        assert backend.is_flashinfer_installed() is False
        output = "\n".join(record.getMessage() for record in records)
        assert output.count("flashinfer is installed but failed to import") == 1
        assert "incompatible native extension" in output
        assert "Triton kernel fallback" in output
    finally:
        backend.is_flashinfer_installed.cache_clear()


def test_explicit_flashinfer_backend_error_explains_unavailable_import(monkeypatch):
    from freetoken.attention import AttnType
    from freetoken.engine import engine

    monkeypatch.setattr(engine, "_flashinfer_available", lambda: False)
    config = SimpleNamespace(
        attention_backend="fi",
        model_config=SimpleNamespace(model_type="test"),
    )
    with pytest.raises(RuntimeError, match=r"unavailable \(not installed or failed to import"):
        engine._validate_attention_backend_choice(config, None, frozenset({AttnType.FULL}))
