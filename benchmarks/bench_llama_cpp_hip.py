"""Run separate llama.cpp HIP reference lanes for decode parity diagnostics.

``llama-cli`` owns raw-prompt/output/sampling evidence. ``llama-bench`` owns native
decode throughput only; its rows never close FreeToken's serving parity gate.
Missing binaries are recorded as ``unavailable`` rows when ``--json`` is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from bench_decode_moe import load_problem_details, model_fingerprint


PINNED_COMMIT = "7e4c0a968"
COMPARATOR_CONTEXT = 9216
COMPARATOR_BATCH = 512
COMPARATOR_UBATCH = 512
COMPARATOR_KV_TYPE = "q8_0"


_EVAL_RE = re.compile(
    r"eval\s+time\s*=.*?/\s*(?P<count>\d+)\s+runs?\s*\([^)]*?"
    r"(?P<tps>[0-9]+(?:\.[0-9]+)?)\s+tokens?\s+per\s+second",
    re.IGNORECASE | re.DOTALL,
)
_JSON_TPS_KEYS = (
    "eval_tokens_per_second",
    "tokens_per_second",
    "eval_t_s",
    "eval_t/s",
    "t/s",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="exact local GGUF file")
    parser.add_argument(
        "--cli",
        default=os.environ.get("LLAMA_CLI"),
        help="llama-cli HIP binary; absent = unavailable lane",
    )
    parser.add_argument(
        "--bench",
        default=os.environ.get("LLAMA_BENCH"),
        help="llama-bench HIP binary; absent = unavailable lane",
    )
    parser.add_argument(
        "--server-rocm",
        default=os.environ.get("LLAMA_SERVER_ROCM"),
        help="llama-server ROCm binary; absent = unavailable lane",
    )
    parser.add_argument(
        "--server-vulkan",
        default=os.environ.get("LLAMA_SERVER_VULKAN"),
        help="llama-server Vulkan binary; absent = unavailable lane",
    )
    parser.add_argument("--server-context", type=int, default=9216)
    parser.add_argument("--server-batch", type=int, default=512)
    parser.add_argument("--server-ubatch", type=int, default=512)
    parser.add_argument("--aime", default=os.environ.get("FREETOKEN_AIME25_JSONL"))
    parser.add_argument("--aime-revision", default=os.environ.get("FREETOKEN_AIME25_REVISION"))
    parser.add_argument("--aime-sha256", default=os.environ.get("FREETOKEN_AIME25_SHA256"))
    parser.add_argument("--problem", type=int, default=0)
    parser.add_argument("--decode", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--commit",
        default=os.environ.get("LLAMA_CPP_COMMIT", PINNED_COMMIT),
        help="expected llama.cpp source commit; reference builder defaults to b10434",
    )
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--replay-manifest",
        default=None,
        help="run the fixed-token teacher-forced replay lane from this manifest "
        "against the supplied ROCm server binary instead of the decode lanes",
    )
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--json", dest="json_out", default=None)
    return parser.parse_args(argv)


def _version(binary: str) -> str | None:
    try:
        result = subprocess.run(
            [binary, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (result.stdout + result.stderr).strip()
    return text[:1000] or None


def _binary_commit(version: str | None) -> str | None:
    if not version:
        return os.environ.get("LLAMA_CPP_COMMIT")
    match = re.search(r"\b(?:commit|rev(?:ision)?)\s*[:=]?\s*([0-9a-f]{7,40})\b", version, re.I)
    return match.group(1) if match else os.environ.get("LLAMA_CPP_COMMIT")


def _base_row(
    *,
    lane: str,
    args: argparse.Namespace,
    model_sha256: str | None,
    prompt_sha256: str,
    dataset: dict,
    binary: str | None,
    binary_version: str | None,
    backend: str = "hip",
    schema: str = "llama-cpp-hip-decode-v1",
) -> dict:
    return {
        "schema": schema,
        "status": "unavailable",
        "lane": lane,
        "binary": binary,
        "binary_version": binary_version,
        "binary_commit": _binary_commit(binary_version),
        "expected_commit": args.commit,
        "backend": backend,
        "model": str(Path(args.model).expanduser().resolve()),
        "model_sha256": model_sha256,
        "model_identity": "full-file-sha256" if model_sha256 else "unverified",
        "dataset": dataset,
        "prompt_sha256": prompt_sha256,
        "prompt_protocol": "raw_user_prompt",
        "sampling": {
            "temperature": 0.0 if args.greedy else 1.0,
            "top_p": 1.0 if args.greedy else 0.95,
            "top_k": -1 if args.greedy else 64,
            "seed": 0,
        },
        "decode_requested": args.decode,
        "context": args.server_context,
        "batch": args.server_batch,
        "ubatch": args.server_ubatch,
        "kv_type": COMPARATOR_KV_TYPE,
        "fixture_sha256": dataset.get("sha256"),
        "placement": {
            "ngl": 99,
            "flash_attention": True,
            "extra_args": os.environ.get("LLAMA_SERVER_EXTRA_ARGS", "").strip(),
        },
        "mtp": "off",
        "speculative": False,
        "eval_count": None,
        "native_decode_tok_s": None,
        "client_arrival_tok_s": None,
        "output_sha256": None,
        "reference_status": "reference_only",
        "acceptance": {
            "status": "unavailable",
            "accepted": False,
            "checks": {
                "exact_completion_count": False,
                "model_sha256": bool(model_sha256),
                "finite_output": False,
                "prompt_options_match": False,
                "fixture_sha256": bool(dataset.get("sha256")),
                "exact_comparator_config": (
                    args.decode == 512
                    and args.server_context == COMPARATOR_CONTEXT
                    and args.server_batch == COMPARATOR_BATCH
                    and args.server_ubatch == COMPARATOR_UBATCH
                ),
                "mtp_off": True,
                "speculative_off": True,
                "native_only_for_llama_bench": lane == "llama-bench",
            },
            "reasons": [],
        },
    }


def _parse_cli_timing(output: str) -> tuple[int, float] | None:
    matches = list(_EVAL_RE.finditer(output))
    if not matches:
        return None
    match = matches[-1]
    return int(match.group("count")), float(match.group("tps"))


def _run_cli(
    binary: str,
    *,
    args: argparse.Namespace,
    prompt: str,
    model_sha256: str | None,
    prompt_sha256: str,
    dataset: dict,
    repeat: int,
) -> dict:
    row = _base_row(
        lane="llama-cli",
        args=args,
        model_sha256=model_sha256,
        prompt_sha256=prompt_sha256,
        dataset=dataset,
        binary=binary,
        binary_version=_version(binary),
    )
    command = [
        binary,
        "-m",
        args.model,
        "-p",
        prompt,
        "-n",
        str(args.decode),
        "--temp",
        str(row["sampling"]["temperature"]),
        "--top-p",
        str(row["sampling"]["top_p"]),
        "--top-k",
        str(row["sampling"]["top_k"]),
        "--seed",
        str(row["sampling"]["seed"]),
        "--no-display-prompt",
        "--no-conversation",
        "-c", str(args.server_context),
        "-b", str(args.server_batch),
        "-ub", str(args.server_ubatch),
        "-ngl", "99",
        "-fa", "on",
        "-ctk", "q8_0",
        "-ctv", "q8_0",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        row["status"] = "rejected"
        row["acceptance"]["status"] = "rejected"
        row["acceptance"]["reasons"].append(f"binary execution failed: {type(exc).__name__}")
        return row
    output = result.stdout + result.stderr
    timing = _parse_cli_timing(output)
    row["repeat"] = repeat
    row["command"] = command
    row["returncode"] = result.returncode
    row["output_sha256"] = hashlib.sha256(result.stdout.encode()).hexdigest()
    if result.returncode != 0:
        row["status"] = "rejected"
        row["acceptance"]["status"] = "rejected"
        row["acceptance"]["reasons"].append(f"binary exited {result.returncode}")
        return row
    if timing is None:
        row["status"] = "rejected"
        row["acceptance"]["status"] = "rejected"
        row["acceptance"]["reasons"].append("llama-cli eval timing/count not found")
        return row
    eval_count, tok_s = timing
    row["eval_count"] = eval_count
    row["native_decode_tok_s"] = tok_s
    checks = row["acceptance"]["checks"]
    checks["exact_completion_count"] = eval_count == args.decode
    checks["model_sha256"] = bool(model_sha256)
    checks["finite_output"] = bool(result.stdout)
    checks["prompt_options_match"] = True
    reasons = row["acceptance"]["reasons"]
    if eval_count != args.decode:
        reasons.append(f"eval_count={eval_count} != --decode {args.decode}")
    if not model_sha256:
        reasons.append("model full SHA-256 unavailable")
    if dataset.get("sha256") is None:
        reasons.append("fixture full SHA-256 unavailable")
    if not checks["exact_comparator_config"]:
        reasons.append("comparator requires decode=512 context=9216 batch=512 ubatch=512")
    if row["binary_commit"] != args.commit:
        reasons.append(
            f"binary commit={row['binary_commit']!r} != expected {args.commit!r}"
        )
    if not result.stdout:
        reasons.append("llama-cli emitted no stdout")
    row["status"] = "accepted" if not reasons else "rejected"
    row["acceptance"]["status"] = row["status"]
    row["acceptance"]["accepted"] = not reasons
    return row


def _find_bench_result(stdout: str) -> tuple[int | None, float | None]:
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        count = next(
            (value[key] for key in ("n_gen", "eval_count", "tokens") if key in value),
            None,
        )
        tps = next((value[key] for key in _JSON_TPS_KEYS if key in value), None)
        if isinstance(tps, (int, float)):
            return int(count) if isinstance(count, (int, float)) else None, float(tps)
    match = re.search(
        r"(?P<count>\d+)\s+tokens?\s+per\s+second.*?(?P<tps>[0-9]+(?:\.[0-9]+)?)",
        stdout,
        re.IGNORECASE,
    )
    return (
        (int(match.group("count")) if match else None),
        (float(match.group("tps")) if match else None),
    )


def _run_bench(
    binary: str,
    *,
    args: argparse.Namespace,
    model_sha256: str | None,
    prompt_sha256: str,
    dataset: dict,
) -> dict:
    row = _base_row(
        lane="llama-bench",
        args=args,
        model_sha256=model_sha256,
        prompt_sha256=prompt_sha256,
        dataset=dataset,
        binary=binary,
        binary_version=_version(binary),
    )
    command = [binary, "-m", args.model, "-n", str(args.decode), "-r", str(args.repeats), "-o", "json"]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        row["status"] = "rejected"
        row["acceptance"]["status"] = "rejected"
        row["acceptance"]["reasons"].append(f"binary execution failed: {type(exc).__name__}")
        return row
    count, tok_s = _find_bench_result(result.stdout + result.stderr)
    row["command"] = command
    row["returncode"] = result.returncode
    row["eval_count"] = count
    row["native_decode_tok_s"] = tok_s
    checks = row["acceptance"]["checks"]
    checks["exact_completion_count"] = count == args.decode
    checks["model_sha256"] = bool(model_sha256)
    checks["finite_output"] = True
    checks["prompt_options_match"] = False
    reasons = row["acceptance"]["reasons"]
    if result.returncode != 0:
        reasons.append(f"binary exited {result.returncode}")
    if count != args.decode:
        reasons.append(f"eval_count={count!r} != --decode {args.decode}")
    if tok_s is None:
        reasons.append("llama-bench JSON eval throughput not found")
    if not model_sha256:
        reasons.append("model full SHA-256 unavailable")
    if dataset.get("sha256") is None:
        reasons.append("fixture full SHA-256 unavailable")
    if not checks["exact_comparator_config"]:
        reasons.append("comparator requires decode=512 context=9216 batch=512 ubatch=512")
    if row["binary_commit"] != args.commit:
        reasons.append(
            f"binary commit={row['binary_commit']!r} != expected {args.commit!r}"
        )
    row["status"] = "accepted" if not reasons else "rejected"
    row["acceptance"]["status"] = row["status"]
    row["acceptance"]["accepted"] = not reasons
    row["acceptance"]["reasons"] = reasons
    return row


def _unavailable(lane: str, args: argparse.Namespace, model_sha256: str | None, prompt_sha256: str, dataset: dict) -> dict:
    row = _base_row(
        lane=lane,
        args=args,
        model_sha256=model_sha256,
        prompt_sha256=prompt_sha256,
        dataset=dataset,
        binary=None,
        binary_version=None,
    )
    row["acceptance"]["reasons"].append("binary not supplied")
    return row


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_command(binary: str, args: argparse.Namespace, port: int) -> list[str]:
    command = [
        binary,
        "-m", args.model,
        "--host", "127.0.0.1",
        "--port", str(port),
        "-c", str(args.server_context),
        "-b", str(args.server_batch),
        "-ub", str(args.server_ubatch),
        "-np", "1",
        "-ngl", "99",
        "-fa", "on",
        "-ctk", "q8_0",
        "-ctv", "q8_0",
    ]
    extra = os.environ.get("LLAMA_SERVER_EXTRA_ARGS", "").strip()
    if extra:
        command.extend(shlex.split(extra))
    return command


def _server_json(origin: str, path: str, *, timeout: float = 10) -> dict:
    with urllib.request.urlopen(f"{origin}{path}", timeout=timeout) as response:
        return json.load(response)


def _wait_server(origin: str, proc: subprocess.Popen, log_path: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited {proc.returncode}; log={log_path}")
        try:
            health = _server_json(origin, "/health", timeout=5)
            if health.get("status") in ("ok", "ready"):
                return
        except (OSError, ValueError, urllib.error.HTTPError):
            pass
        time.sleep(1)
    raise RuntimeError(f"llama-server not ready after {timeout:.0f}s; log={log_path}")


def _stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            proc.kill()
            proc.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass


def _stream_server(origin: str, model: str, prompt: str, args: argparse.Namespace) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.decode,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0 if args.greedy else 1.0,
        "top_p": 1.0 if args.greedy else 0.95,
        "top_k": -1 if args.greedy else 64,
        "seed": 0,
    }
    request = urllib.request.Request(
        f"{origin}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    stamps: list[float] = []
    pieces: list[str] = []
    usage: dict = {}
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for raw in response:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:"):].strip()
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                text = "".join(
                    part for part in (delta.get("reasoning_content"), delta.get("content")) if part
                )
                if text:
                    stamps.append(time.perf_counter())
                    pieces.append(text)
    return {
        "t0": started,
        "stamps": stamps,
        "text": "".join(pieces),
        "usage": usage,
    }


def _server_native_timing(log_path: str) -> tuple[int, float] | None:
    text = Path(log_path).read_text(errors="replace")
    return _parse_cli_timing(text)


def _run_server(
    binary: str,
    *,
    backend: str,
    args: argparse.Namespace,
    prompt: str,
    model_sha256: str | None,
    prompt_sha256: str,
    dataset: dict,
) -> list[dict]:
    port = _free_port()
    origin = f"http://127.0.0.1:{port}"
    lane = f"llama-server-{backend}"
    version = _version(binary)
    rows: list[dict] = []
    with tempfile.NamedTemporaryFile(prefix=f"{lane}-", suffix=".log", delete=False) as log:
        log_path = log.name
    command = _server_command(binary, args, port)
    log_handle = open(log_path, "wb")
    proc = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        _wait_server(origin, proc, log_path, args.timeout)
        model_id = _server_json(origin, "/v1/models").get("data", [{}])[0].get("id", args.model)
        _stream_server(origin, model_id, prompt, args)
        for repeat in range(args.repeats):
            row = _base_row(
                lane=lane,
                args=args,
                model_sha256=model_sha256,
                prompt_sha256=prompt_sha256,
                dataset=dataset,
                binary=binary,
                binary_version=version,
                backend=backend,
                schema="llama-cpp-server-decode-v1",
            )
            row["repeat"] = repeat
            row["command"] = command
            try:
                result = _stream_server(origin, model_id, prompt, args)
            except (OSError, ValueError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
                row["status"] = "rejected"
                row["acceptance"]["status"] = "rejected"
                row["acceptance"]["reasons"].append(f"server request failed: {type(exc).__name__}")
                rows.append(row)
                continue
            usage = result["usage"]
            count = usage.get("completion_tokens")
            stamps = result["stamps"]
            client_window = stamps[-1] - stamps[0] if len(stamps) >= 2 else 0.0
            row["eval_count"] = count
            row["client_arrival_tok_s"] = (
                (count - 1) / client_window if isinstance(count, int) and client_window > 0 else None
            )
            row["client_arrival_window_s"] = client_window
            row["output_sha256"] = hashlib.sha256(result["text"].encode()).hexdigest()
            native = _server_native_timing(log_path)
            if native:
                row["native_eval_count"], row["native_decode_tok_s"] = native
            checks = row["acceptance"]["checks"]
            checks["exact_completion_count"] = count == args.decode
            checks["model_sha256"] = bool(model_sha256)
            checks["finite_output"] = bool(result["text"])
            checks["prompt_options_match"] = True
            reasons = row["acceptance"]["reasons"]
            if count != args.decode:
                reasons.append(f"completion_tokens={count!r} != --decode {args.decode}")
            if len(stamps) < 2:
                reasons.append(f"need >=2 output events, got {len(stamps)}")
            if not result["text"]:
                reasons.append("llama-server emitted no output")
            if not model_sha256:
                reasons.append("model full SHA-256 unavailable")
            if dataset.get("sha256") is None:
                reasons.append("fixture full SHA-256 unavailable")
            if not checks["exact_comparator_config"]:
                reasons.append("comparator requires decode=512 context=9216 batch=512 ubatch=512")
            if row["binary_commit"] != args.commit:
                reasons.append(
                    f"binary commit={row['binary_commit']!r} != expected {args.commit!r}"
                )
            row["status"] = "accepted" if not reasons else "rejected"
            row["acceptance"]["status"] = row["status"]
            row["acceptance"]["accepted"] = not reasons
            rows.append(row)
    except (OSError, ValueError, RuntimeError, urllib.error.HTTPError) as exc:
        row = _unavailable(lane, args, model_sha256, prompt_sha256, dataset)
        row["binary"] = binary
        row["binary_version"] = version
        row["backend"] = backend
        row["status"] = "rejected"
        row["acceptance"]["status"] = "rejected"
        row["acceptance"]["reasons"] = [f"server startup failed: {type(exc).__name__}: {exc}"]
        rows.append(row)
    finally:
        _stop_server(proc)
        log_handle.close()
    for row in rows:
        row["server_log"] = log_path
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.decode < 2 or args.repeats < 1:
        raise SystemExit("--decode must be >= 2 and --repeats must be >= 1")
    if args.replay_manifest:
        # Delegated fixed-token replay lane: one common token sequence per
        # runtime, timing logits without sampling (rocm-ollama-gap Inc 0).
        from bench_decode_replay import load_manifest, replay_llama

        manifest = load_manifest(args.replay_manifest)
        rows = []
        for backend, binary in (("rocm", args.server_rocm), ("vulkan", args.server_vulkan)):
            if not binary:
                continue
            replay_args = argparse.Namespace(**vars(args))
            replay_args.server = binary
            replay_args.backend = backend
            replay_args.context = args.server_context
            replay_args.batch = args.server_batch
            replay_args.ubatch = args.server_ubatch
            replay_args.kv_type = COMPARATOR_KV_TYPE
            rows.extend(
                replay_llama(replay_args, manifest, repeat)
                for repeat in range(args.repeats)
            )
        if args.json_out:
            with open(args.json_out, "a") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        for row in rows:
            print(
                f"{row['runtime']}: steps={row['steps'].get('steps')} "
                f"prompt_ids_match={row['prompt_ids_match']}",
                flush=True,
            )
        return 0 if rows and all(
            row["prompt_ids_match"]
            and row["steps"].get("steps") == manifest["measured_tokens"]
            for row in rows
        ) else 1
    model = model_fingerprint(args.model)
    model_sha256 = model.get("sha256")
    prompt, _, dataset = load_problem_details(
        args.aime, args.problem, args.aime_revision, args.aime_sha256
    )
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    rows = []
    if args.cli:
        rows.extend(
            _run_cli(
                args.cli,
                args=args,
                prompt=prompt,
                model_sha256=model_sha256,
                prompt_sha256=prompt_sha256,
                dataset=dataset,
                repeat=repeat,
            )
            for repeat in range(args.repeats)
        )
    else:
        rows.append(_unavailable("llama-cli", args, model_sha256, prompt_sha256, dataset))
    if args.bench:
        rows.append(
            _run_bench(
                args.bench,
                args=args,
                model_sha256=model_sha256,
                prompt_sha256=prompt_sha256,
                dataset=dataset,
            )
        )
    else:
        rows.append(_unavailable("llama-bench", args, model_sha256, prompt_sha256, dataset))
    for backend, binary in (("rocm", args.server_rocm), ("vulkan", args.server_vulkan)):
        if binary:
            rows.extend(
                _run_server(
                    binary,
                    backend=backend,
                    args=args,
                    prompt=prompt,
                    model_sha256=model_sha256,
                    prompt_sha256=prompt_sha256,
                    dataset=dataset,
                )
            )
        else:
            rows.append(_unavailable(
                f"llama-server-{backend}", args, model_sha256, prompt_sha256, dataset
            ))
    for row in rows:
        print(
            f"{row['lane']}: {row['status']} native={row['native_decode_tok_s']} "
            f"count={row['eval_count']}",
            flush=True,
        )
    if args.json_out:
        with open(args.json_out, "a") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    if any(row["status"] == "accepted" for row in rows):
        return 0
    if all(row["status"] == "unavailable" for row in rows):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
