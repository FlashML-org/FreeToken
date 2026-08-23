"""The daemon's serve argv is platform-specific: Metal on Darwin, CUDA elsewhere."""

from __future__ import annotations

import sys

from freetoken.daemon.serve_manager import build_serve_command


def test_build_serve_command_uses_serve_metal_on_darwin(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "darwin")
    argv, log_path = build_serve_command(
        "mlx-community/Qwen3-0.6B-4bit",
        1919,
        ["--tp-size", "1"],
        python="/usr/bin/python3",
        log_dir=str(tmp_path),
    )
    assert argv[0] == "/usr/bin/python3"
    assert argv[argv.index("-m") + 2] == "serve-metal"
    assert "--model" in argv and "mlx-community/Qwen3-0.6B-4bit" in argv
    assert log_path.endswith("serve-1919.log")


def test_build_serve_command_uses_serve_on_linux(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    argv, _ = build_serve_command(
        "Qwen",
        1919,
        [],
        python="/usr/bin/python3",
        log_dir=str(tmp_path),
    )
    assert argv[argv.index("-m") + 2] == "serve"
    assert "serve-metal" not in argv
