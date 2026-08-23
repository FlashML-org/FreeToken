from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:1919"
# (CLI dest, pool, rebuild body key). moe/mamba are slot counts and pass straight through;
# kv/swa are typed in TOKENS -- the unit `cache status` reports them in -- and are rounded up
# to the pool's own page unit against the live geometry before they go on the wire.
CACHE_TARGETS = (
    ("moe_cache_size", "moe", "moe_cache_size"),
    ("kv_tokens", "kv", "num_pages"),
    ("num_mamba_slots", "mamba", "num_mamba_slots"),
    ("swa_tokens", "swa", "num_swa_pages"),
)


class ControlCliError(Exception):
    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _prefill_summary(doc: dict[str, Any]) -> str:
    requested = doc.get("requested_prefill_tokens", "unknown")
    pool_cap = doc.get("pool_prefill_cap_tokens", "unknown")
    effective = doc.get("effective_prefill_tokens", "unknown")
    source = doc.get("swa_capacity_source", "unknown")
    reason = doc.get("prefill_limiting_reason", "unknown")
    return (
        f"prefill requested={requested} pool_cap={pool_cap} effective={effective} "
        f"source={source} reason={reason}"
    )


def parse_count(token: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*", token)
    if not match:
        raise ValueError(f"invalid count: {token!r}")
    multiplier = {"": 1, "k": 1024, "m": 1024 * 1024}[match.group(2).lower()]
    value = float(match.group(1)) * multiplier
    if value <= 0 or value != int(value):
        raise ValueError(f"invalid count: {token!r}")
    return int(value)


def _build_url(base_url: str, path: str, query: dict[str, Any] | None = None) -> str:
    url = f"{base_url.rstrip('/')}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"
    return url


def _decode_error_body(raw: bytes) -> str:
    if not raw:
        return ""
    text = raw.decode("utf-8", errors="replace")
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(doc, dict):
        for key in ("error", "detail", "message"):
            value = doc.get(key)
            if value:
                suffix = (
                    f" ({_prefill_summary(doc)})"
                    if "effective_prefill_tokens" in doc else ""
                )
                return f"{value}{suffix}"
        status = doc.get("status")
        if status:
            return str(status)
    return text


def _request_json(
    method: str,
    base_url: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    query: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    with _open_request(
        method,
        base_url,
        path,
        body=body,
        query=query,
        accept="application/json",
        timeout=timeout,
    ) as response:
        raw = response.read()

    if not raw:
        return {}
    try:
        doc = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ControlCliError("server returned invalid JSON", exit_code=1) from exc
    if not isinstance(doc, dict):
        raise ControlCliError("server returned a non-object JSON document", exit_code=1)
    return doc


def _request_sse_text(
    method: str,
    base_url: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> str:
    with _open_request(
        method,
        base_url,
        path,
        body=body,
        accept="text/event-stream",
        timeout=timeout,
    ) as response:
        return _read_sse_text(response)


def _open_request(
    method: str,
    base_url: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    query: dict[str, Any] | None = None,
    accept: str,
    timeout: float,
) -> Any:
    data = None
    headers = {"Accept": accept}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        _build_url(base_url, path, query),
        data=data,
        headers=headers,
        method=method,
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        message = _decode_error_body(exc.read()) or exc.reason
        raise ControlCliError(f"HTTP {exc.code}: {message}", exit_code=1) from exc
    except urllib.error.URLError as exc:
        raise ControlCliError(f"failed to reach {base_url}: {exc.reason}", exit_code=1) from exc
    except TimeoutError as exc:
        raise ControlCliError(f"timed out connecting to {base_url}", exit_code=1) from exc


def _read_sse_text(response: Any) -> str:
    chunks: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            break
        try:
            doc = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(doc, dict) and isinstance(doc.get("text"), str):
            chunks.append(doc["text"])
    return "".join(chunks)


def _fmt_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _format_health(doc: dict[str, Any]) -> str:
    status = doc.get("status", "unknown")
    parts = [f"status={status}"]
    for key in ("model", "maintenance", "phase", "uptime_s", "version"):
        value = doc.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    progress = doc.get("progress")
    if isinstance(progress, dict):
        done = progress.get("done_bytes", 0)
        total = progress.get("total_bytes", 0)
        parts.append(f"progress={done}/{total} bytes")
    message = doc.get("message")
    if message:
        parts.append(f"message={message}")
    return " ".join(parts)


def _format_stats(doc: dict[str, Any]) -> str:
    model = doc.get("model") if isinstance(doc.get("model"), dict) else {}
    throughput = doc.get("throughput") if isinstance(doc.get("throughput"), dict) else {}
    requests = doc.get("requests") if isinstance(doc.get("requests"), dict) else {}
    lines = [
        " ".join(
            [
                f"model={model.get('id', 'unknown')}",
                f"ctx={model.get('ctx', 'unknown')}",
                f"attn={model.get('attn', 'unknown')}",
                f"moe={_fmt_bool(model.get('moe', 'unknown'))}",
                f"uptime_s={doc.get('uptime_s', 0)}",
            ]
        ),
        (
            f"throughput decode_tps={throughput.get('decode_tps', 0)} "
            f"prefill_tps={throughput.get('prefill_tps', 0)}"
        ),
        (
            f"requests active={requests.get('active', 0)} "
            f"completed={requests.get('completed', 0)} "
            f"p95_ms={requests.get('p95_ms', 0)} "
            f"ttft_mean_ms={requests.get('ttft_mean_ms', 0)}"
        ),
        f"vram_bytes={doc.get('vram_bytes', 0)}",
    ]
    kv = doc.get("kv")
    if isinstance(kv, dict):
        lines.append(
            f"kv={kv.get('used_pages', 0)}/{kv.get('total_pages', 0)} pages "
            f"page_size={kv.get('page_size', 0)}"
        )
    else:
        lines.append("kv=none")
    mamba = doc.get("mamba")
    if isinstance(mamba, dict):
        lines.append(f"mamba={mamba.get('used_slots', 0)}/{mamba.get('total_slots', 0)} slots")
    else:
        lines.append("mamba=none")
    return "\n".join(lines)


def _format_cache_status(doc: dict[str, Any]) -> str:
    """The pool table (shared with the shell's ``/cache`` -- see freetoken.cache_report), plus
    the one thing only this CLI reports: the outcome of the last rebuild."""
    from freetoken.cache_report import format_cache_status

    lines = [format_cache_status(doc, prefix="")]
    last = doc.get("last_rebuild")
    if isinstance(last, dict):
        lines.append(f"  last rebuild  {last.get('status', 'unknown')}")
    return "\n".join(lines)


def _format_rebuild(doc: dict[str, Any]) -> str:
    parts = [
        f"status={doc.get('status', 'unknown')}",
        f"moe={doc.get('moe_cache_size', 'unknown')}",
        f"kv={doc.get('num_pages', 'unknown')}",
        f"mamba={doc.get('mamba_slots', 'unknown')}",
        f"swa={doc.get('num_swa_pages', 'unknown')}",
        _prefill_summary(doc),
    ]
    if doc.get("error"):
        parts.append(f"error={doc['error']}")
    return " ".join(parts)


def _format_requests(doc: dict[str, Any]) -> str:
    entries = doc.get("entries") if isinstance(doc.get("entries"), list) else []
    lines = [f"{'time':20} {'method':6} {'status':6} {'duration':9} {'stream':6} path"]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        stream = "yes" if entry.get("stream") else "no"
        lines.append(
            f"{str(entry.get('ts', '')):20} "
            f"{str(entry.get('method', '')):6} "
            f"{str(entry.get('status', '')):6} "
            f"{str(entry.get('duration_ms', '')) + 'ms':9} "
            f"{stream:6} "
            f"{entry.get('path', '')}"
        )
    lines.append(f"next_cursor={doc.get('next_cursor', 0)}")
    return "\n".join(lines)


def _build_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"FreeToken server URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds")
    parser.add_argument("--json", action="store_true", help="Print raw JSON response")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("health", help="Show /health")
    sub.add_parser("stats", help="Show /v1/stats")

    generate = sub.add_parser("generate", help="Run one prompt through /generate")
    generate.add_argument("prompt", nargs="?", default="Hello", help="Prompt text")
    generate.add_argument("--max-tokens", type=int, default=16)
    generate.add_argument("--ignore-eos", action="store_true")

    cache = sub.add_parser("cache", help="Inspect or rebuild cache pools")
    _add_rebuild_args(cache)
    cache_sub = cache.add_subparsers(dest="cache_command")
    cache_sub.add_parser("status", help="Show /v1/cache/status")
    rebuild = cache_sub.add_parser("rebuild", help="POST /v1/cache/rebuild")
    _add_rebuild_args(rebuild)

    requests = sub.add_parser("requests", help="Show /v1/requests")
    requests.add_argument("--since", type=int, default=0)
    requests.add_argument("--limit", type=int, default=100)
    return parser


def _add_rebuild_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--moe",
        type=parse_count,
        dest="moe_cache_size",
        help="MoE cache slots (k=1024, m=1024*1024)",
    )
    parser.add_argument(
        "--kv",
        type=parse_count,
        dest="kv_tokens",
        help="KV cache size in TOKENS, rounded up to the page size (k=1024, m=1024*1024)",
    )
    parser.add_argument(
        "--mamba",
        type=parse_count,
        dest="num_mamba_slots",
        help="Mamba/GDN state slots (k=1024, m=1024*1024)",
    )
    parser.add_argument(
        "--swa",
        type=parse_count,
        dest="swa_tokens",
        help="Window (SWA) pool size in TOKENS, rounded up to its page size (k=1024, m=1024*1024)",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=300.0,
        help="Server-side rebuild wait timeout",
    )


def _print_doc(doc: dict[str, Any], formatter, *, raw_json: bool) -> None:
    if raw_json:
        print(json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(formatter(doc))


def main(argv: Sequence[str] | None = None, *, prog: str = "ft ctl") -> int:
    parser = _build_parser(prog)
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    try:
        if args.command == "health":
            doc = _request_json("GET", args.base_url, "/health", timeout=args.timeout)
            _print_doc(doc, _format_health, raw_json=args.json)
            return 0
        if args.command == "stats":
            doc = _request_json("GET", args.base_url, "/v1/stats", timeout=args.timeout)
            _print_doc(doc, _format_stats, raw_json=args.json)
            return 0
        if args.command == "generate":
            text = _request_sse_text(
                "POST",
                args.base_url,
                "/generate",
                body={
                    "prompt": args.prompt,
                    "max_tokens": args.max_tokens,
                    "ignore_eos": args.ignore_eos,
                },
                timeout=args.timeout,
            )
            if args.json:
                print(json.dumps({"text": text}, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                print(text)
            return 0
        if args.command == "requests":
            query = {"since": args.since, "limit": args.limit}
            doc = _request_json(
                "GET",
                args.base_url,
                "/v1/requests",
                query=query,
                timeout=args.timeout,
            )
            _print_doc(doc, _format_requests, raw_json=args.json)
            return 0
        if args.command == "cache" and args.cache_command in (None, "status"):
            if args.cache_command is None and _cache_targets(args):
                return _run_cache_rebuild(args)
            doc = _request_json("GET", args.base_url, "/v1/cache/status", timeout=args.timeout)
            _print_doc(doc, _format_cache_status, raw_json=args.json)
            return 0
        if args.command == "cache" and args.cache_command == "rebuild":
            return _run_cache_rebuild(args)
    except ControlCliError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code

    print(f"error: unsupported command: {args.command}", file=sys.stderr)
    return 2


def _rebuild_body(targets: dict[str, int], geometry: dict[str, Any]) -> dict[str, Any]:
    """Typed targets -> the rebuild body, converting the token counts to the page unit each
    pool speaks. Raises ControlCliError for a pool this model does not have -- better than
    letting the engine reject a request that was never meaningful."""
    from freetoken.cache_report import CachePools, pages_for_tokens

    pools = CachePools.from_geometry(geometry)
    page_size = max(1, int(geometry.get("page_size", 1) or 1))
    swa_page_size = int(geometry.get("swa_page_size", 0) or 0)

    body: dict[str, Any] = {}
    for pool, value in targets.items():
        if pool not in pools.targets:
            raise ControlCliError(
                f"this server's model has no {pool} pool "
                f"(it has: {', '.join(pools.targets)})",
                exit_code=2,
            )
        if pool == "kv":
            body["num_pages"] = pages_for_tokens(value, page_size)
        elif pool == "swa":
            body["num_swa_pages"] = pages_for_tokens(value, swa_page_size)
        elif pool == "moe":
            body["moe_cache_size"] = value
        else:
            body["num_mamba_slots"] = value
    return body


def _run_cache_rebuild(args: argparse.Namespace) -> int:
    targets = _cache_targets(args)
    if not targets:
        print("error: cache rebuild requires at least one cache target", file=sys.stderr)
        return 2
    # The token targets can only be converted against the live geometry, so read it first.
    status = _request_json("GET", args.base_url, "/v1/cache/status", timeout=args.timeout)
    body = _rebuild_body(targets, status.get("geometry") or {})
    body["timeout"] = args.wait
    doc = _request_json(
        "POST",
        args.base_url,
        "/v1/cache/rebuild",
        body=body,
        timeout=args.wait + args.timeout,
    )
    if args.json or doc.get("status") != "ok":
        _print_doc(doc, _format_rebuild, raw_json=args.json)
        return 0 if doc.get("status") == "ok" else 1
    # Report the geometry the engine landed on rather than an echo of the request: it resolves
    # the sizes itself. Same answer `ft ctl cache status` gives, and the same table the shell's
    # /cache prints. Best-effort -- the rebuild already succeeded.
    print("status=ok")
    try:
        print(_format_cache_status(_request_json("GET", args.base_url, "/v1/cache/status")))
    except ControlCliError:
        pass
    return 0


def _cache_targets(args: argparse.Namespace) -> dict[str, int]:
    """The pools the user asked for, keyed by pool name, in the units they typed."""
    return {
        pool: value
        for attr, pool, _body_key in CACHE_TARGETS
        if (value := getattr(args, attr, None)) is not None
    }


if __name__ == "__main__":
    raise SystemExit(main())
