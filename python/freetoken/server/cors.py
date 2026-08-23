"""Torch-free CORS setup shared by the native and Metal API servers."""

from __future__ import annotations

from fastapi import FastAPI


DEFAULT_CORS_ORIGINS = "tauri://localhost,http://tauri.localhost,http://localhost:1420"


def install_cors(app: FastAPI, origins_csv: str) -> None:
    """Attach CORS headers for browser/webview clients.

    An empty allow-list disables CORS; ``*`` allows every origin.
    """
    origins = [origin.strip() for origin in origins_csv.split(",") if origin.strip()]
    if not origins:
        return

    from starlette.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if "*" in origins else origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
