"""``ft shell`` -- an interactive terminal chat that drives a FreeToken server over its API.

Two ways in, one code path:

* ``ft shell`` attaches to a server that is already running (``--server``, ``$FREETOKEN_HOST``,
  else ``http://127.0.0.1:1919``), exactly like ``ft launch`` attaches an agent. Nothing is
  loaded locally -- no torch import, no GPU -- so it works against a remote box too.
* ``ft shell --model <path> [engine flags]`` starts the engine here first (the same thing
  ``ft serve --shell-mode`` does) and then attaches to it over the loopback.

Either way the conversation travels over ``POST /v1/chat/completions``, so the shell gets the
prompt rendering, sampling defaults, reasoning split and accounting every other client gets.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence

# Flags that mean "start an engine here" rather than "attach to one". Anything else engine-
# side (--moe-cache-auto, --attn, ...) only makes sense alongside these.
_ENGINE_FLAGS = ("--model", "--model-path")

#: Flags the Metal path does not understand (CUDA/engine-specific); dropped when
#: `ft shell --model` routes to `ft serve-metal` on macOS.
_METAL_UNKNOWN_VALUE_FLAGS = (
    "--dtype",
    "--moe-backend",
    "--moe-cache-size",
    "--moe-cache-rate",
    "--attn",
    "--attention-backend",
    "--tool-call-parser",
    "--reasoning-parser",
    "--sampling-defaults",
    "--cuda-graph-max-bs",
    "--max-running-req",
)
_METAL_UNKNOWN_BOOLEAN_FLAGS = (
    "--moe-cache-auto",
    "--use-dummy-weight",
    "--silent-output",
    "--shell-mode",
)


def _wants_local_engine(argv: Sequence[str]) -> bool:
    return any(arg in _ENGINE_FLAGS or arg.startswith(tuple(f + "=" for f in _ENGINE_FLAGS))
               for arg in argv)


def _build_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Chat with a running FreeToken server in the terminal.",
        epilog=(
            "Pass --model <path> (plus any ft serve flag) to start an engine here instead of "
            "attaching to one."
        ),
    )
    parser.add_argument(
        "--server",
        "--base-url",
        dest="server",
        default=None,
        help="FreeToken server URL (default: $FREETOKEN_HOST, else http://127.0.0.1:1919)",
    )
    return parser


def _split_engine_args(argv: Sequence[str]) -> tuple[str | None, list[str]]:
    """Extract the (last) --model/--model-path value and the remaining args.

    Understands both ``--model X`` and ``--model=X`` forms, and skips over the
    value token of engine flags that take one, so it never mistakes a value for
    a flag or leaks it into the passthrough list.
    """
    value_flags = set(_ENGINE_FLAGS + _METAL_UNKNOWN_VALUE_FLAGS)
    boolean_flags = set(_METAL_UNKNOWN_BOOLEAN_FLAGS)
    model = None
    passthrough: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--") and "=" in arg:
            name, _, value = arg.partition("=")
            if name in value_flags:
                if name in _ENGINE_FLAGS:
                    model = value
            else:
                passthrough.append(arg)
            i += 1
        elif arg in boolean_flags:
            i += 1
        elif arg in value_flags:
            value = argv[i + 1] if i + 1 < len(argv) else None
            if arg in _ENGINE_FLAGS:
                model = value
            i += 2 if value is not None else 1
        else:
            passthrough.append(arg)
            i += 1
    return model, passthrough


def main(argv: Sequence[str] | None = None, *, prog: str = "ft shell") -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if _wants_local_engine(args):
        # On Apple Silicon the CUDA launcher cannot even import (torch/flashlib have
        # no macOS build); route the same flags to the Metal backend instead, so
        # `ft shell --model <mlx/hf id>` works there too. On a CUDA box the native
        # launcher takes over, unchanged.
        if sys.platform == "darwin":
            from freetoken.server.metal import resolve_backend

            try:
                backend = resolve_backend("auto")
            except RuntimeError:
                backend = None  # fall through to the native launcher's own error
            if backend in ("mlx", "llama"):
                from freetoken.server.metal_main import main as metal_main

                model, passthrough = _split_engine_args(args)
                if model is None:
                    print("--model is required", file=sys.stderr)
                    return 2
                return metal_main(["--shell", "--model", model, *passthrough])
        from freetoken.server import launch_server

        launch_server(run_shell=True, argv=args, prog=prog)
        return 0

    parser = _build_parser(prog)
    try:
        parsed = parser.parse_args(args)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    from freetoken.launch import resolve_server_url

    try:
        server = resolve_server_url(parsed.server)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    from .tui import run_shell

    try:
        return asyncio.run(run_shell(server.origin))
    except KeyboardInterrupt:
        return 130


__all__ = ["main"]
