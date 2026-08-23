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


def test_metal_parser_accepts_shared_shell_and_model_name_flags():
    from freetoken.server.metal_main import _parse

    args = _parse(
        [
            "--model",
            "/models/example",
            "--shell-mode",
            "--served-model-name",
            "public-model",
            "--cors-origins",
            "http://localhost:3000",
        ]
    )

    assert args.shell is True
    assert args.served_model_name == "public-model"
    assert args.cors_origins == "http://localhost:3000"


def test_shell_metal_filter_does_not_consume_backend_after_boolean_flag():
    from freetoken.shell import _split_engine_args

    model, passthrough = _split_engine_args(
        ["--model", "M", "--moe-cache-auto", "--backend", "llama"]
    )

    assert model == "M"
    assert passthrough == ["--backend", "llama"]


def test_metal_cors_uses_same_browser_allow_list():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from freetoken.server.cors import install_cors

    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    install_cors(app, "http://localhost:1420")
    response = TestClient(app).options(
        "/health",
        headers={
            "Origin": "http://localhost:1420",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.headers["access-control-allow-origin"] == "http://localhost:1420"


def test_serve_metal_help_uses_lightweight_import_path():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pkg = os.path.join(root, "python")
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = pkg + (os.pathsep + existing if existing else "")

    proc = subprocess.run(
        [sys.executable, "-m", "freetoken.cli", "serve-metal", "--help"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    assert "PyTorch was not found" not in proc.stderr
    assert "--served-model-name" in proc.stdout


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
