"""Ollama base-decode adapter for the FreeToken Qwen MoE comparison.

The adapter uses Ollama's native ``/api/chat`` NDJSON stream and keeps two metrics
separate:

* ``client_arrival_tok_s`` matches FreeToken's SSE arrival window and is the only
  cross-runtime metric;
* ``native_decode_tok_s`` comes from Ollama's ``eval_count/eval_duration`` fields and
  is diagnostic only.

Ollama's public API exposes draft-token configuration through ``/api/show``'s generated
Modelfile. Rows carry ``mtp=off`` only when ``draft_num_predict=0`` is explicit; otherwise
they remain unknown and must not be presented as the user's base-mode reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from bench_decode_moe import (
    git_revision,
    load_problem_details,
    runtime_metadata,
    sha256_file,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--origin", default="http://127.0.0.1:11434")
    p.add_argument("--model", required=True, help="Ollama model name/tag")
    p.add_argument("--decode", type=int, default=512, help="num_predict and exact eval_count")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--aime", default=os.environ.get("FREETOKEN_AIME25_JSONL"))
    p.add_argument("--aime-revision", default=os.environ.get("FREETOKEN_AIME25_REVISION"))
    p.add_argument("--aime-sha256", default=os.environ.get("FREETOKEN_AIME25_SHA256"))
    p.add_argument("--problem", type=int, default=0)
    p.add_argument("--greedy", action="store_true")
    p.add_argument(
        "--ollama-gguf",
        default=os.environ.get("OLLAMA_GGUF_PATH"),
        help="local GGUF file verified as Ollama's loaded model blob",
    )
    p.add_argument(
        "--reference-gguf",
        default=os.environ.get("FREETOKEN_REFERENCE_GGUF"),
        help="FreeToken GGUF file used for byte-identity comparison",
    )
    p.add_argument(
        "--spawn-hip",
        action="store_true",
        help="spawn one ollama serve with a HIP-forced environment for one "
        "directional row (never edits the installed unit); failure blocks nothing",
    )
    p.add_argument("--json", dest="json_out", default=None)
    return p.parse_args(argv)


def worker_backend_evidence(origin: str) -> dict:
    """Best-effort worker backend evidence; unknown stays unknown, never a guess."""
    evidence: dict = {"backend": None, "source": None, "gpu_share": None}
    try:
        ps = request_json(origin, "/api/ps")
    except Exception:
        return evidence
    models = ps.get("models") or []
    if not models:
        return evidence
    entry = models[0]
    size_vram = entry.get("size_vram") or 0
    size = entry.get("size") or 0
    evidence["source"] = "api_ps"
    evidence["gpu_share"] = (size_vram / size) if size else None
    # Ollama does not expose its loaded library over the HTTP API; the backend
    # stays unknown unless the row was captured from an explicitly HIP-forced
    # spawned instance. Unknown backend downgrades the row to directional-only.
    evidence["worker_backend"] = "unknown"
    return evidence


def spawn_hip_forced_server(port: int, timeout: float = 120):
    """Spawn ``ollama serve`` with a HIP-forced library override.

    Directional evidence only, per the plan: the installed unit is never edited,
    and failure to obtain the row blocks nothing."""
    env = dict(os.environ)
    env["OLLAMA_LLM_LIBRARY"] = "rocm"
    env.setdefault("OLLAMA_HOST", f"127.0.0.1:{port}")
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    proc = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return None
        try:
            request_json(f"http://127.0.0.1:{port}", "/api/version")
            return proc
        except (OSError, RuntimeError, ValueError, urllib.error.URLError):
            time.sleep(1)
    return None


def request_json(origin: str, path: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(f"{origin.rstrip('/')}{path}", data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:500]
        raise RuntimeError(f"Ollama {path} failed: HTTP {exc.code}: {detail!r}") from exc


def ollama_model_info(origin: str, model: str) -> tuple[dict, dict, str | None]:
    version = request_json(origin, "/api/version")
    shown = request_json(origin, "/api/show", {"name": model})
    digest = next(
        (
            row.get("digest")
            for row in request_json(origin, "/api/tags").get("models", [])
            if row.get("name") == model or row.get("model") == model
        ),
        None,
    )
    return version, shown, digest


def sampling(greedy: bool) -> dict:
    if greedy:
        return {"temperature": 0.0, "top_p": 1.0, "top_k": -1}
    return {"temperature": 1.0, "top_p": 0.95, "top_k": 64}


def _mtp_status(model_info: dict) -> str:
    """Return proof state from Ollama's explicit draft-token runtime parameter."""
    parameters = str(model_info.get("parameters", ""))
    draft = re.search(r"(?m)^\s*draft_num_predict\s+(\d+)\s*$", parameters)
    if draft and int(draft.group(1)) == 0:
        return "off"
    if draft:
        return "unknown"
    return "unknown"


def ollama_blob_identity(
    model_digest: str | None,
    *,
    ollama_gguf: str | None = None,
    reference_gguf: str | None = None,
) -> dict:
    """Separate Ollama manifest digest from verified GGUF byte identity."""
    identity = {
        "manifest_digest": model_digest,
        "manifest_digest_kind": "ollama-model-manifest" if model_digest else "missing",
        "same_blob": False,
        "status": "unproven",
        "ollama_gguf": None,
        "reference_gguf": None,
    }
    if ollama_gguf:
        path = Path(ollama_gguf).expanduser()
        if not path.is_file():
            identity["status"] = "invalid_ollama_gguf_path"
            return identity
        identity["ollama_gguf"] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(str(path)),
        }
    if reference_gguf:
        path = Path(reference_gguf).expanduser()
        if not path.is_file():
            identity["status"] = "invalid_reference_gguf_path"
            return identity
        identity["reference_gguf"] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(str(path)),
        }
    if identity["ollama_gguf"] and identity["reference_gguf"]:
        ollama_sha = identity["ollama_gguf"]["sha256"]
        reference_sha = identity["reference_gguf"]["sha256"]
        identity["same_blob"] = ollama_sha == reference_sha
        identity["status"] = "verified" if identity["same_blob"] else "mismatch"
    elif identity["ollama_gguf"] or identity["reference_gguf"]:
        identity["status"] = "one_side_only"
    return identity


def stream_chat(
    origin: str,
    model: str,
    problem: str,
    options: dict,
    decode: int,
) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": problem}],
        "stream": True,
        "think": True,
        "options": {**options, "num_predict": decode},
    }
    req = urllib.request.Request(
        f"{origin.rstrip('/')}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    stamps: list[float] = []
    pieces: list[str] = []
    done: dict | None = None
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=1800)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Ollama /api/chat failed: HTTP {exc.code}: {exc.read()[:500]!r}") from exc
    with resp:
        for raw in resp:
            line = raw.strip()
            if not line:
                continue
            chunk = json.loads(line)
            message = chunk.get("message") or {}
            fragments = [message.get("thinking"), message.get("content")]
            text = "".join(fragment for fragment in fragments if fragment)
            if text:
                stamps.append(time.perf_counter())
                pieces.append(text)
            if chunk.get("done"):
                done = chunk
                break
    if done is None:
        raise RuntimeError("Ollama stream ended without done=true")
    return {
        "t0": t0,
        "stamps": stamps,
        "text": "".join(pieces),
        "done": done,
        "prompt": problem,
    }


def make_row(
    args: argparse.Namespace,
    repeat: int,
    result: dict,
    version: dict,
    model_info: dict,
    model_digest: str | None,
    dataset: dict,
    mtp: str,
    options: dict,
    blob_identity: dict,
) -> dict:
    done = result["done"]
    eval_count = done.get("eval_count")
    if eval_count is None:
        raise RuntimeError("rejected Ollama run: done message missing eval_count")
    if eval_count != args.decode:
        raise RuntimeError(
            f"rejected Ollama run: eval_count={eval_count} != --decode {args.decode}"
        )
    stamps = result["stamps"]
    if len(stamps) < 2:
        raise RuntimeError(f"rejected Ollama run: need >=2 output events, got {len(stamps)}")
    if not result["text"]:
        raise RuntimeError("rejected Ollama run: server returned no generated text")
    client_window = stamps[-1] - stamps[0]
    native_duration_ns = done.get("eval_duration")
    if not isinstance(native_duration_ns, (int, float)) or native_duration_ns <= 0:
        raise RuntimeError("rejected Ollama run: done message missing positive eval_duration")
    gaps = sorted((b - a) * 1e3 for a, b in zip(stamps, stamps[1:]))
    return {
        "schema": "ollama-base-decode-v1",
        "status": "accepted",
        "acceptance": {
            "status": "accepted",
            "accepted": True,
            "checks": {
                "exact_completion_count": True,
                "finite_output": True,
                "finite_logits": "unavailable_api",
                "mtp_off": mtp == "off",
                "same_blob": blob_identity["same_blob"],
            },
            "reasons": [],
        },
        "origin": args.origin,
        "model": args.model,
        "model_sha256": (
            blob_identity.get("ollama_gguf") or {}
        ).get("sha256"),
        "reference_identity": blob_identity,
        "repeat": repeat,
        "git_revision": git_revision(),
        "dataset": dataset,
        "prompt_sha256": hashlib.sha256(result["prompt"].encode("utf-8")).hexdigest(),
        "decode_requested": args.decode,
        "eval_count": eval_count,
        "prompt_eval_count": done.get("prompt_eval_count"),
        "client_arrival_tok_s": (eval_count - 1) / client_window if client_window > 0 else 0.0,
        "client_arrival_window_s": client_window,
        "native_decode_tok_s": eval_count / (native_duration_ns / 1e9),
        "eval_duration_ns": native_duration_ns,
        "prompt_eval_duration_ns": done.get("prompt_eval_duration"),
        "total_duration_ns": done.get("total_duration"),
        "load_duration_ns": done.get("load_duration"),
        "event_ms_p50": gaps[len(gaps) // 2],
        "event_ms_p99": gaps[min(len(gaps) - 1, int(len(gaps) * 0.99))],
        "ttft_ms": (stamps[0] - result["t0"]) * 1e3,
        "events": len(stamps),
        "output_sha1": hashlib.sha1(result["text"].encode()).hexdigest()[:12],
        "output_sample": result["text"][:240],
        "options": options,
        "think": True,
        "speculative": "off" if mtp == "off" else "unknown",
        "mtp": mtp,
        "comparable_to_freetoken_base": mtp in ("off", "unsupported"),
        "ollama_version": version,
        "model_info": {
            "digest": model_digest,
            **{
                key: model_info.get(key)
                for key in ("modified_at", "details", "parameters", "template")
                if key in model_info
            },
        },
        "runtime": {"python": sys.version, "platform": platform.platform(), **runtime_metadata()},
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.decode < 2:
        raise SystemExit("--decode must be >= 2")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    hip_proc = None
    hip_forced = False
    if args.spawn_hip:
        hip_port = 11500 + (int(time.time()) % 4000)
        hip_proc = spawn_hip_forced_server(hip_port)
        if hip_proc is None:
            print(
                "WARNING: HIP-forced ollama spawn failed; keeping the running "
                "service row directional-only (plan: failure blocks nothing)",
                file=sys.stderr,
            )
        else:
            args.origin = f"http://127.0.0.1:{hip_port}"
            hip_forced = True
    problem, _, dataset = load_problem_details(
        args.aime, args.problem, args.aime_revision, args.aime_sha256
    )
    version, model_info, model_digest = ollama_model_info(args.origin, args.model)
    blob_identity = ollama_blob_identity(
        model_digest,
        ollama_gguf=args.ollama_gguf,
        reference_gguf=args.reference_gguf,
    )
    options = sampling(args.greedy)
    mtp = _mtp_status(model_info)
    backend = worker_backend_evidence(args.origin)
    if hip_forced:
        backend = {**backend, "worker_backend": "hip-forced", "requested_backend": "hip"}
    # Warm Ollama's loaded model before collecting measured rows.
    stream_chat(args.origin, args.model, problem, options, args.decode)
    rows = []
    for repeat in range(args.repeats):
        row = make_row(
            args,
            repeat,
            stream_chat(args.origin, args.model, problem, options, args.decode),
            version,
            model_info,
            model_digest,
            dataset,
            mtp,
            options,
            blob_identity,
        )
        row["worker_backend"] = backend
        row["backend_evidence"] = backend
        row["directional_only"] = not hip_forced
        rows.append(row)
        print(
            f"repeat {repeat + 1}/{args.repeats}: client {row['client_arrival_tok_s']:.2f} "
            f"tok/s, native {row['native_decode_tok_s']:.2f} tok/s, mtp={mtp}",
            flush=True,
        )
    if mtp == "unknown":
        print("WARNING: Ollama MTP status unknown; rows are non-comparable base reference", file=sys.stderr)
    if args.json_out:
        with open(args.json_out, "a") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    if hip_proc is not None:
        try:
            hip_proc.terminate()
            hip_proc.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            hip_proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
