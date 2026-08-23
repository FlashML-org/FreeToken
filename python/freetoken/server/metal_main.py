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
import asyncio
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
    p.add_argument(
        "--shell",
        action="store_true",
        help="Attach the interactive ft shell to this server (serve+chat in one process).",
    )
    args, unknown = p.parse_known_args(list(argv))
    if unknown:
        # Callers built for the CUDA engine (the daemon, benchmarks, `ft serve`
        # flag passthrough) legitimately carry CUDA-only flags; the Metal path
        # has no such engine knobs, so drop them loudly rather than refusing.
        print(
            f"ft serve-metal: ignoring CUDA-only engine flags: {' '.join(unknown)}",
            file=sys.stderr,
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    backend = resolve_backend(args.backend)
    # Non-blocking: spawns the upstream and returns while its watcher thread
    # supervises the load (see _watch_mlx_load). uvicorn binds immediately, so
    # /health reports live load progress and ft shell can attach and render it
    # instead of the terminal sitting silent through a 50 GiB download.
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
        doc = h.health_doc()
        if doc.get("status") == "error":
            return JSONResponse(doc, status_code=503)
        return doc

    # The shell/desktop poll /health, /v1/stats and /v1/cache/status every
    # second; hide those from the access log so they don't bury real requests.
    # Same filter the CUDA api_server installs (access_log_filter.py).
    from freetoken.server.access_log_filter import install_polling_access_log_filter

    install_polling_access_log_filter()

    try:
        if args.shell:
            import threading

            origin = f"http://{args.host}:{args.port}"
            server = uvicorn.Server(
                uvicorn.Config(app, host=args.host, port=args.port, access_log=False)
            )
            thread = threading.Thread(
                target=server.run, name="freetoken-uvicorn", daemon=True
            )
            thread.start()
            from freetoken.shell.tui import run_shell

            return asyncio.run(run_shell(origin, connect_grace=30.0))
        uvicorn.run(app, host=args.host, port=args.port)
    except KeyboardInterrupt:
        pass
    finally:
        handle.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
