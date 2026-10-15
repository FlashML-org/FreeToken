"""``ft daemon <verb>`` — the client half of the daemon's own control entry point.

Kept separate from ``ft ctl`` (which targets a running *serve*): the daemon has its own dedicated
CLI. Bare ``ft daemon`` (or ``ft daemon --host/--port …``) runs the server; ``ft daemon <verb>``
below controls a running daemon over HTTP. Torch-free — stdlib ``urllib`` only."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from typing import Any

from freetoken.launch import resolve_api_key

DEFAULT_URL = "http://127.0.0.1:1900"
DEFAULT_TIMEOUT = 10.0
# prepare-stop (15s transport budget) + default SIGTERM grace (10s) + reap wait (10s),
# with enough HTTP scheduling slack that a valid lifecycle transaction does not look failed.
DEFAULT_LIFECYCLE_TIMEOUT = 40.0

# Positional verbs that mean "act as a client"; anything else (bare, or a flag like --host) runs
# the server. Kept in one place so the server dispatcher and this parser agree.
CLIENT_VERBS = ("self", "status", "health", "metrics", "stats", "start", "stop", "switch", "logs")


class ClientError(Exception):
    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _effective_timeout(verb: str, configured: float | None) -> float:
    if configured is not None:
        return configured
    return (
        DEFAULT_LIFECYCLE_TIMEOUT
        if verb in {"stop", "switch"}
        else DEFAULT_TIMEOUT
    )


def _request(method: str, url: str, path: str, *, body=None, query=None, token=None, timeout=10.0, accept="application/json"):
    full = f"{url.rstrip('/')}{path}"
    if query:
        full = f"{full}?{urllib.parse.urlencode(query)}"
    headers = {"Accept": accept}
    if token:
        headers["X-FT-Token"] = token
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(full, data=data, headers=headers, method=method)
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise ClientError(f"HTTP {exc.code}: {_err_body(exc.read()) or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise ClientError(f"failed to reach {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ClientError(f"timed out connecting to {url}") from exc


def _err_body(raw: bytes) -> str:
    if not raw:
        return ""
    text = raw.decode("utf-8", "replace")
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(doc, dict):
        for key in ("error", "detail", "message"):
            if doc.get(key):
                return str(doc[key])
    return text


def _request_json(method, url, path, *, body=None, query=None, token=None, timeout=10.0) -> dict:
    with _request(method, url, path, body=body, query=query, token=token, timeout=timeout) as resp:
        raw = resp.read()
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ClientError("daemon returned invalid JSON") from exc


def _stream_logs(url, since, token, timeout) -> None:
    # The stream is endless; print each line until interrupted. A long read timeout survives idle
    # heartbeats without hanging forever on a dead socket.
    with _request("GET", url, "/engine/logs", query={"since": since}, token=token,
                  timeout=max(timeout, 3600.0), accept="text/event-stream") as resp:
        try:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    rec = json.loads(line[len("data:"):].strip())
                except json.JSONDecodeError:
                    continue
                if rec.get("kind") == "gap":
                    print(f"... [{rec.get('dropped')} log lines dropped]")
                else:
                    print(rec.get("text", ""))
        except KeyboardInterrupt:
            pass


def _build_parser(prog: str) -> argparse.ArgumentParser:
    # The package dispatcher routes to the client only when a verb is argv[0], so --url/--token/
    # --timeout live on each verb (`ft daemon status --url X`), not before it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", default=os.environ.get("FREETOKEN_DAEMON_URL", DEFAULT_URL),
                        help=f"daemon URL (default {DEFAULT_URL})")
    common.add_argument("--token", default=os.environ.get("FREETOKEN_DAEMON_TOKEN"), help="X-FT-Token shared secret")
    common.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="HTTP timeout (default 10s; stop/switch 40s)",
    )

    p = argparse.ArgumentParser(prog=prog, description="Control a running ft daemon")
    sub = p.add_subparsers(dest="verb", required=True)
    sub.add_parser("self", parents=[common], help="Daemon self-health (GET /health)")
    sub.add_parser("status", parents=[common], help="Engine status (GET /engine/status)")
    sub.add_parser("health", parents=[common], help="Proxied serve health (GET /engine/health)")
    sub.add_parser("metrics", parents=[common], help="Engine footprint (GET /engine/metrics)")
    sub.add_parser("stats", parents=[common], help="Proxied serve stats (GET /engine/stats)")
    stop = sub.add_parser("stop", parents=[common], help="Stop the serve (POST /engine/stop)")
    stop.add_argument(
        "--force",
        action="store_true",
        help="stop even if final accounting cannot be sealed (may lose the unobserved token tail)",
    )
    for name in ("start", "switch"):
        sp = sub.add_parser(name, parents=[common], help=f"POST /engine/{name}")
        sp.add_argument("model", help="Model path/id")
        sp.add_argument("--port", type=int, default=None, help="Serve port")
        sp.add_argument(
            "--api-key",
            help="API key the serve requires, set in its environment (default: $FREETOKEN_API_KEY)",
        )
        if name == "switch":
            sp.add_argument(
                "--force",
                action="store_true",
                help="replace even if final accounting cannot be sealed (may lose the unobserved token tail)",
            )
        # Everything after `--` is forwarded verbatim to ft serve (opaque passthrough):
        #   ft daemon start MODEL --port 1919 -- --moe-cache-auto --graph 256
        sp.add_argument("serve_args", nargs="*", default=[], help="Extra ft serve args (after --)")
    lg = sub.add_parser("logs", parents=[common], help="Stream engine logs (SSE, GET /engine/logs)")
    lg.add_argument("--since", type=int, default=0, help="Replay from this seq cursor")
    return p


def main(argv: Sequence[str] | None = None, *, prog: str = "ft daemon") -> int:
    args = _build_parser(prog).parse_args(list(argv) if argv is not None else None)
    timeout = _effective_timeout(args.verb, args.timeout)
    try:
        if args.verb == "logs":
            _stream_logs(args.url, args.since, args.token, timeout)
            return 0
        table = {
            "self": ("GET", "/health", None),
            "status": ("GET", "/engine/status", None),
            "health": ("GET", "/engine/health", None),
            "metrics": ("GET", "/engine/metrics", None),
            "stats": ("GET", "/engine/stats", None),
            "stop": (
                "POST",
                "/engine/stop",
                {"force": True} if getattr(args, "force", False) else {},
            ),
        }
        if args.verb in ("start", "switch"):
            body: dict[str, Any] = {"model": args.model, "args": list(args.serve_args)}
            if args.port is not None:
                body["port"] = args.port
            if api_key := resolve_api_key(args.api_key):
                body["apiKey"] = api_key
            if args.verb == "switch" and args.force:
                body["force"] = True
            method, path = "POST", f"/engine/{args.verb}"
        else:
            method, path, body = table[args.verb]
        doc = _request_json(method, args.url, path, body=body, token=args.token, timeout=timeout)
        print(json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except ClientError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
