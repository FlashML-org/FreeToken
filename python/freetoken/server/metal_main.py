"""Standalone entrypoint for ``ft serve-metal`` (Apple Silicon, no CUDA).

This module deliberately does NOT import FreeToken's CUDA engine stack. The
classic ``ft serve`` path (``server/args.parse_args`` / ``SchedulerConfig`` / the
MoE + layer graph) transitively imports ``flashlib`` and the CUDA kernels, which
have no macOS build. On Apple Silicon that whole chain cannot even be imported.

``serve-metal`` therefore parses its own minimal arguments, launches the chosen
Apple Metal runtime (mlx or llama.cpp) as an upstream OpenAI/Anthropic-compatible
engine, and serves a thin HTTP proxy on the configured host/port. This gives the
exact OpenAI/Anthropic/Responses wire surface with nothing imported from the
CUDA scheduler.

Built on the reusable pieces in :mod:`freetoken.server.metal`.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from freetoken.server.metal import (
    MetalBackendHandle,
    launch_metal_backend,
    register_metal_proxy_routes,
    resolve_backend,
)


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ft serve-metal",
        description="Serve a model on an Apple Silicon Metal backend (mlx or llama.cpp).",
    )
    p.add_argument("--model", required=True, help="Model path or HF id for the Metal backend.")
    p.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "mlx", "llama"],
        help="Metal engine: mlx (mlx_lm.server) or llama (llama.cpp). auto prefers mlx.",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=1919)
    p.add_argument("--metal-port", type=int, default=0, help="Upstream port (0 = auto).")
    return p.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    backend = resolve_backend(args.backend)
    handle: MetalBackendHandle = launch_metal_backend(
        backend, args.model, args.metal_port
    )

    _HANDLE = {"handle": handle}
    app = FastAPI(title="FreeToken Metal API Server")

    def get_backend():
        h = _HANDLE["handle"]
        return h if h is not None and h.is_alive() else None

    register_metal_proxy_routes(app, get_backend)

    @app.get("/health")
    async def health(request: Request):
        h = _HANDLE["handle"]
        if h is None or not h.is_alive():
            return JSONResponse({"status": "down"}, status_code=503)
        return {"status": "ok", "backend": backend, "upstream": handle.upstream_base_url}

    try:
        uvicorn.run(app, host=args.host, port=args.port)
    except KeyboardInterrupt:
        pass
    finally:
        handle.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
