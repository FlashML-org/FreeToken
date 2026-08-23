"""macOS serve path: ``ft serve`` must not import the CUDA/torch stack.

The native launcher (``freetoken.server.launch``) imports torch at module
level. On Apple Silicon there is no CUDA torch wheel in the Metal venv, so
``ft serve`` has to route to ``serve-metal`` *before* that import.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def test_run_serve_on_darwin_routes_to_metal_without_cuda_launcher(monkeypatch):
    """``ft serve`` on Darwin calls metal_main and never launch_server."""
    import freetoken.cli as cli

    monkeypatch.setattr(cli.sys, "platform", "darwin")
    seen: dict = {}

    def fake_metal(argv):
        seen["argv"] = list(argv)
        return 0

    monkeypatch.setattr("freetoken.server.metal_main.main", fake_metal)

    rc = cli._run_serve(["--model", "mlx-community/Qwen3-0.6B-4bit", "--port", "1919"])
    assert rc == 0
    assert seen["argv"] == ["--model", "mlx-community/Qwen3-0.6B-4bit", "--port", "1919"]


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal venv has no torch")
def test_ft_serve_help_does_not_need_torch():
    """``ft serve --help`` must succeed on a Metal install (no torch installed)."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pkg = os.path.join(root, "python")
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = pkg + (os.pathsep + existing if existing else "")
    proc = subprocess.run(
        [sys.executable, "-m", "freetoken.cli", "serve", "--help"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"ft serve --help failed (likely imported torch)\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "No module named 'torch'" not in proc.stderr
    assert "--model" in proc.stdout
