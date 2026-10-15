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
    parser.add_argument(
        "--api-key",
        help="API key of a server started with --api-key (default: $FREETOKEN_API_KEY)",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, prog: str = "ft shell") -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if _wants_local_engine(args):
        from freetoken.server import launch_server

        launch_server(run_shell=True, argv=args, prog=prog)
        return 0

    parser = _build_parser(prog)
    try:
        parsed = parser.parse_args(args)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    from freetoken.launch import resolve_api_key, resolve_server_url

    try:
        server = resolve_server_url(parsed.server)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    from .tui import run_shell

    try:
        return asyncio.run(run_shell(server.origin, api_key=resolve_api_key(parsed.api_key)))
    except KeyboardInterrupt:
        return 130


__all__ = ["main"]
