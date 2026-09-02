"""Single-stream (bs=1) decode benchmark for any MoE model on any offload backend.

Measures through the real serving path: for each backend the bench spawns ``ft serve``,
sends a warmed chat request over /v1/chat/completions with ``stream=true``, and
timestamps every SSE event as it arrives. Numbers therefore include the scheduler,
detokenizer, and HTTP/SSE hop -- what a client actually sees -- not bare engine forwards.

Method -- at bs=1 the server emits one delta event per decode step, and the final chunk
(``stream_options.include_usage``) reports exact token counts, so

    decode_tok_s = (completion_tokens - 1) / (t_last_event - t_first_event)

which stays correct even when the detokenizer coalesces a few tokens into one event
(multibyte characters): the window is still anchored on the first and last token's
arrival. ``ignore_eos`` keeps the step count at exactly ``D`` regardless of sampling.
TTFT is the measured run's warm first-token latency (template rendering + prefill
included). Engine-internal diagnostics (expert-cache miss rate, hybrid fetch split) are
not exposed over the API and are not reported; VRAM is the server's live /v1/stats figure.

Prompt: an AIME-25 problem sent as a chat message with thinking enabled -- a real
reasoning workload, so expert routing is representative. The server renders the chat
template (including checkpoint-shipped encoders like DSV4's ``encoding_dsv4.py``). The
problems come from a local immutable jsonl fixture; Hub input is accepted only with an
explicit revision and SHA-256, then cached by the Hub client.

Sampling: the checkpoint's recommended params (``generation_config.json``), falling back
to temperature 1.0 / top_p 0.95 / top_k 64 for fields the checkpoint does not specify --
resolved here and sent explicitly, because the server's own unspecified-field defaults
are greedy and would silently degrade the routing workload for checkpoints without a
full sampling recommendation. The generated text is per-server-process deterministic
(fresh server, fixed request sequence), so one text sha1 per backend is a real
cross-backend check; token ids are not visible over the API, so this is a weaker
invariant than the old in-process id hash. ``--greedy`` sends temperature 0 for the
stricter comparison.

Run (one backend):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python benchmarks/bench_decode_moe.py \
        --model /path/to/model

Run (all three backends, one server per backend):
    ... --model /path/to/model --backend offload,cpu,hybrid --json out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# Applied for every field the checkpoint's generation_config.json does not specify.
FALLBACK_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}

# Primary comparator contract. Keep these values in metadata even when a local
# runtime cannot prove one of them; an unproven row must be rejected by the gate.
COMPARATOR_CONTEXT = 9216
COMPARATOR_BATCH = 512
COMPARATOR_UBATCH = 512
COMPARATOR_KV_TYPE = "q8_0"

# AIME-25 problems, pulled from the Hub into the usual HF cache on first run.
AIME_REPO = "math-ai/aime25"
AIME_FILE = "test.jsonl"
# Reasoning models need the answer format spelled out; the boxed answer is also what makes
# a run spot-checkable by eye.
BOXED_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="checkpoint dir (or .ftw)")
    p.add_argument(
        "--backend",
        default="offload",
        help="comma list of fused|offload|cpu|hybrid; one server per backend",
    )
    p.add_argument(
        "--aime",
        default=os.environ.get("FREETOKEN_AIME25_JSONL"),
        help=f"local jsonl instead of downloading pinned {AIME_REPO}; default $FREETOKEN_AIME25_JSONL",
    )
    p.add_argument(
        "--aime-revision",
        default=os.environ.get("FREETOKEN_AIME25_REVISION"),
        help="Hub dataset commit/revision; required with --aime-sha256 when --aime is absent",
    )
    p.add_argument(
        "--aime-sha256",
        default=os.environ.get("FREETOKEN_AIME25_SHA256"),
        help="expected SHA-256 for local or pinned Hub AIME JSONL",
    )
    p.add_argument("--problem", type=int, default=0, help="0-based AIME problem index")
    p.add_argument("--decode", type=int, default=512, help="decode tokens to measure (D)")
    p.add_argument("--repeats", type=int, default=10, help="measured requests per server")
    p.add_argument("--context", type=int, default=COMPARATOR_CONTEXT)
    p.add_argument("--batch", type=int, default=COMPARATOR_BATCH)
    p.add_argument("--ubatch", type=int, default=COMPARATOR_UBATCH)
    p.add_argument("--kv-type", default=COMPARATOR_KV_TYPE)
    p.add_argument(
        "--cache",
        type=int,
        default=0,
        help="GPU expert cache slots; 0 = auto-size from free VRAM",
    )
    p.add_argument("--cache-rate", type=float, default=None, help="cache slots as a fraction of L*E")
    p.add_argument(
        "--kv-reserve-tokens",
        type=int,
        default=8192,
        help="KV tokens reserved while comparing MoE cache sizes",
    )
    p.add_argument(
        "--hybrid-fetch",
        type=int,
        default=-1,
        help="hybrid: max PCIe fetches/layer; -1 = auto (benched pcie/cpu bandwidth fraction)",
    )
    p.add_argument("--mem-ratio", type=float, default=0.9, help="target VRAM utilization")
    p.add_argument(
        "--attention-backend",
        default="triton",
        help="attention backend passed to ft serve (default: triton)",
    )
    p.add_argument("--gpu", default=None,
                   help="GPU for the serve: a UUID or nvidia-smi index (as ft serve --gpu)")
    p.add_argument("--no-graph", action="store_true", help="eager decode instead of CUDA graph")
    p.add_argument(
        "--greedy",
        action="store_true",
        help="force temperature 0 (ignore the checkpoint's sampling) so ids are comparable",
    )
    p.add_argument(
        "--server-timeout",
        type=float,
        default=1800,
        help="seconds to wait for the spawned server to become ready",
    )
    p.add_argument("--json", dest="json_out", default=None, help="append the result rows here")
    return p.parse_args(argv)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_path(
    path: str | None, revision: str | None, expected_sha256: str | None
) -> tuple[str, dict[str, str | None]]:
    """Resolve an immutable local or pinned Hub dataset and return provenance."""
    if path:
        dataset_path = Path(path).expanduser()
        if not dataset_path.is_file():
            sys.exit(f"AIME fixture not found: {dataset_path}")
        provenance = {
            "source": "local",
            "path": str(dataset_path.resolve()),
            "revision": revision,
            "sha256": sha256_file(dataset_path),
        }
    else:
        if not revision or not expected_sha256:
            sys.exit(
                "AIME dataset must be local (--aime) or pinned with both "
                "--aime-revision and --aime-sha256"
            )
        from huggingface_hub import hf_hub_download

        try:
            dataset_path = Path(
                hf_hub_download(AIME_REPO, AIME_FILE, repo_type="dataset", revision=revision)
            )
        except Exception as e:  # offline, rate-limited, repo moved
            sys.exit(
                f"could not fetch pinned {AIME_REPO}/{AIME_FILE}@{revision} ({e}); "
                "pass --aime <local jsonl>"
            )
        provenance = {
            "source": "huggingface",
            "path": str(dataset_path),
            "revision": revision,
            "sha256": sha256_file(dataset_path),
        }
    actual = str(provenance["sha256"])
    if expected_sha256 and actual.lower() != expected_sha256.lower():
        sys.exit(f"AIME fixture SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    return str(dataset_path), provenance


def load_problem_details(
    path: str | None,
    index: int,
    revision: str | None = None,
    expected_sha256: str | None = None,
) -> tuple[str, str, dict[str, str | None]]:
    """One immutable AIME-25 problem plus answer and dataset provenance."""
    path, provenance = _dataset_path(path, revision, expected_sha256)
    if not path:
        raise AssertionError("_dataset_path returned an empty path")
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not 0 <= index < len(rows):
        sys.exit(f"--problem {index} out of range ({len(rows)} problems available)")
    row = rows[index]
    text = row.get("problem") or row["prompt"]
    if "boxed" not in text:
        text = f"{text}\n{BOXED_INSTRUCTION}"
    return text, str(row.get("answer", "")), provenance


def load_problem(
    path: str | None,
    index: int,
    revision: str | None = None,
    expected_sha256: str | None = None,
) -> tuple[str, str]:
    """Compatibility wrapper for callers that only need problem text and answer."""
    text, answer, _ = load_problem_details(path, index, revision, expected_sha256)
    return text, answer


def resolve_sampling(model_path: str, greedy: bool) -> tuple[dict, str]:
    """Checkpoint-recommended sampling with per-field fallback; returns (params, source).

    Resolved client-side and sent explicitly: the server fills unspecified fields with
    its framework defaults (temperature 0 / no filtering), not with these fallbacks."""
    if greedy:
        return {"temperature": 0.0, "top_p": 1.0, "top_k": -1}, "greedy (--greedy)"
    recommended: dict = {}
    cfg = Path(model_path) / "generation_config.json"
    if cfg.is_file():
        raw = json.loads(cfg.read_text())
        recommended = {k: raw[k] for k in FALLBACK_SAMPLING if raw.get(k) is not None}
        if raw.get("do_sample") is False or recommended.get("temperature") == 0.0:
            return {"temperature": 0.0, "top_p": 1.0, "top_k": -1}, "checkpoint (greedy)"
    params = {**FALLBACK_SAMPLING, **recommended}
    if params["top_k"] == 0:
        params["top_k"] = -1  # HF spells "no top-k filtering" as 0; the API as -1
    taken = sorted(recommended)
    source = f"checkpoint{taken} + fallback" if taken else "fallback (no generation_config)"
    return params, source


def get_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve_cmd(args: argparse.Namespace, backend: str, port: int) -> list[str]:
    cmd = [
        sys.executable, "-m", "freetoken.cli", "serve",
        "--model", args.model,
        "--host", "127.0.0.1", "--port", str(port),
        "--moe-backend", backend,
        "--attention-backend", args.attention_backend,
        "--max-running-requests", "1",
        "--max-seq-len-override", str(args.context),
        "--kv-type", args.kv_type,
        "--kv-reserve-tokens", str(args.kv_reserve_tokens),
        "--memory-ratio", str(args.mem_ratio),
        "--cuda-graph-max-bs", "0" if args.no_graph else "1",
        "--moe-hybrid-max-fetch", str(args.hybrid_fetch),
        "--moe-collect-stats",
    ]
    if args.gpu:
        cmd += ["--gpu", args.gpu]
    if args.cache > 0:
        cmd += ["--moe-cache-size", str(args.cache)]
    elif args.cache_rate is not None:
        cmd += ["--moe-cache-rate", str(args.cache_rate)]
    else:
        cmd.append("--moe-cache-auto")
    return cmd


def die_with_log(msg: str, log_path: str) -> None:
    tail = "".join(Path(log_path).read_text().splitlines(keepends=True)[-30:])
    sys.exit(f"[bench] {msg}\n[bench] server log tail ({log_path}):\n{tail}")


def wait_ready(origin: str, proc: subprocess.Popen, log_path: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            die_with_log(f"server exited with code {proc.returncode} during startup", log_path)
        try:
            health = get_json(f"{origin}/health", timeout=5)
        except (OSError, ValueError):  # not bound yet / reset / partial response
            time.sleep(1.0)
            continue
        if health.get("status") == "error":
            die_with_log(f"server reported startup error: {health}", log_path)
        if health.get("maintenance") == "serving":
            return
        time.sleep(1.0)
    die_with_log(f"server not ready after {timeout:.0f}s", log_path)


def pump_output(src, log_f) -> None:
    """Mirror the server's output to our terminal while keeping the log file complete.

    Raw byte chunks (read1, not line-buffered) so \\r progress bars render live."""
    for chunk in iter(lambda: src.read1(65536), b""):
        log_f.write(chunk)
        log_f.flush()
        sys.stdout.buffer.write(chunk)
        sys.stdout.flush()


def stop_server(proc: subprocess.Popen) -> None:
    """SIGTERM the whole session (frontend + scheduler/tokenizer workers), escalate.

    Best-effort by design: it runs in ``finally`` and must not mask the real error.
    killpg runs even when the frontend already exited -- a crashed frontend leaves live
    non-daemon workers in the group, and they hold the GPU."""
    for sig, wait_s in ((signal.SIGTERM, 90), (signal.SIGKILL, 30)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:  # whole group already gone
            pass
        try:
            proc.wait(timeout=wait_s)
            break
        except subprocess.TimeoutExpired:
            continue
    time.sleep(3)  # let the driver reclaim VRAM before the next backend's server


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def model_fingerprint(model_path: str) -> dict:
    """Full content identity plus stable source and GGUF metadata."""
    root = Path(model_path).expanduser()
    if not root.exists():
        return {"path": str(root), "error": "model path does not exist"}
    if root.is_file():
        stat = root.stat()
        sample = 1 << 20
        digest = hashlib.sha256()
        with root.open("rb") as f:
            digest.update(f.read(sample))
            if stat.st_size > sample:
                f.seek(max(0, stat.st_size - sample))
                digest.update(f.read(sample))
        result = {
            "path": str(root.resolve()),
            "kind": "file",
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(str(root)),
            "identity": "full-file-sha256",
            "fingerprint": f"head-tail-sha256:{digest.hexdigest()}",
        }
        if root.suffix == ".gguf":
            try:
                from freetoken.models.gguf.reader import load_gguf_metadata

                metadata = load_gguf_metadata(str(root))
                result["gguf_metadata"] = _jsonable(
                    {
                        key: value
                        for key, value in metadata.items()
                        if key.startswith("general.")
                        or "quantization" in key
                        or key.endswith(".file_type")
                    }
                )
            except Exception as exc:
                result["gguf_metadata_error"] = f"{type(exc).__name__}: {exc}"
        return result
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": sha256_file(str(path)),
                }
            )
    content_manifest = [
        {key: entry[key] for key in ("path", "size_bytes", "sha256")}
        for entry in files
    ]
    digest = hashlib.sha256(json.dumps(content_manifest, sort_keys=True).encode()).hexdigest()
    return {
        "path": str(root.resolve()),
        "kind": "directory",
        "files": files,
        "sha256": digest,
        "identity": "directory-content-sha256",
        "fingerprint": f"manifest-sha256:{digest}",
    }


def runtime_metadata() -> dict:
    result = {"python": sys.version, "platform": platform.platform()}
    try:
        import torch
        from freetoken.utils.graph_gate import rocm_blas_report

        result["torch"] = torch.__version__
        result["cuda"] = torch.version.cuda
        result["rocm"] = torch.version.hip
        # Benchmark metadata must report current worker policy without launching a graph
        # probe in the client process; graph gate remains a server-start concern.
        result["blas"] = rocm_blas_report(gate={})
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            result.update(
                {
                    "gpu_index": index,
                    "gpu_name": props.name,
                    "gpu_capability": f"{props.major}.{props.minor}",
                    "gpu_total_memory_bytes": props.total_memory,
                    "gpu_arch": getattr(props, "gcnArchName", None)
                    or os.environ.get("FREETOKEN_GPU_ARCH"),
                }
            )
    except Exception as exc:
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
    result["gpu_telemetry"] = gpu_telemetry()
    # FreeToken currently exposes no server flag for quantized KV storage. Keep
    # observation explicit so q8_0 rows cannot be mistaken for BF16/default KV.
    result.setdefault("kv_type", None)
    result["dirty_diff_hash"] = dirty_diff_hash()
    result["env_selector"] = environment_selector_digest()
    result["jit_binary"] = jit_binary_sha()
    return result


def gpu_telemetry() -> dict:
    """Best-effort clocks/thermal/power snapshot; unavailable fields stay null."""
    telemetry = {
        "core_clock_mhz": None,
        "memory_clock_mhz": None,
        "hotspot_c": None,
        "power_w": None,
        "source": None,
    }
    try:
        result = subprocess.run(
            ["rocm-smi", "--showclocks", "--showtemp", "--showpower", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return telemetry
    if result.returncode != 0:
        return telemetry
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {}
    values = {}
    if isinstance(payload, dict):
        values = next((v for v in payload.values() if isinstance(v, dict)), payload)
    if isinstance(values, dict):
        flat = {str(k).lower().replace(" ", "_"): v for k, v in values.items()}

        def as_number(raw):
            """rocm-smi JSON emits strings like "51.0" or "(42Mhz)"; strip decor."""
            if isinstance(raw, (int, float)):
                return float(raw)
            if isinstance(raw, str):
                digits = re.sub(r"[^0-9.\-]", "", raw)
                if digits:
                    try:
                        return float(digits)
                    except ValueError:
                        return None
            return None

        for key, aliases in {
            "core_clock_mhz": (
                "current_gpu_clk", "gpu_clk", "gpu_clock",
                "sclk_clock_speed:", "sclk_clock_speed", "sclk",
            ),
            "memory_clock_mhz": (
                "current_mem_clk", "mem_clk", "memory_clock",
                "mclk_clock_speed:", "mclk_clock_speed", "mclk",
            ),
            "hotspot_c": (
                "junction_temperature", "hotspot_temperature",
                "temperature_(sensor_junction)_(c)",
            ),
            "power_w": (
                "average_graphics_package_power_(w)", "average_power",
                "power_avg", "power",
            ),
        }.items():
            for alias in aliases:
                value = as_number(flat.get(alias))
                if value is not None:
                    telemetry[key] = value
                    break
    telemetry["source"] = "rocm-smi-json"
    return telemetry


def run_metadata(
    *,
    args: argparse.Namespace,
    backend: str,
    model: dict,
    runtime: dict,
    graph: dict,
    dataset: dict,
    sampling: dict,
    sampling_src: str,
    revision: str,
    prompt_sha256: str,
    model_id: str,
) -> dict:
    """Machine-readable provenance shared by every accepted or rejected row."""
    gguf_metadata = model.get("gguf_metadata") or {}
    gguf_type = next(
        (
            gguf_metadata[key]
            for key in ("general.file_type", "general.quantization_version")
            if key in gguf_metadata
        ),
        None,
    )
    return {
        "model_sha256": model.get("sha256"),
        "model_identity": model.get("identity", "unverified"),
        "gguf_type": gguf_type,
        "device": runtime.get("gpu_name") or runtime.get("device", "unknown"),
        "arch": runtime.get("gpu_arch") or runtime.get("gpu_capability"),
        "torch": runtime.get("torch"),
        "rocm": runtime.get("rocm"),
        "blas": runtime.get("blas"),
        "backend": backend,
        "context": args.context,
        "batch": args.batch,
        "ubatch": args.ubatch,
        "kv_type": args.kv_type,
        "kv_type_source": "requested_cli",
        "fixture_sha256": dataset.get("sha256"),
        "graph": graph,
        "attention": args.attention_backend,
        "moe": backend,
        "mtp": "off",
        "speculative": False,
        "decode_batch_size": 1,
        "prompt_sha256": prompt_sha256,
        "prompt_problem": args.problem,
        "prompt_tokens_expected": None,
        "completion_tokens_expected": args.decode,
        "sampling": sampling,
        "sampling_source": sampling_src,
        "model_id": model_id,
        "dataset": dataset,
        "git_revision": revision,
        "dirty_diff_hash": runtime.get("dirty_diff_hash"),
        "env_selector": runtime.get("env_selector"),
        "jit_binary": runtime.get("jit_binary"),
        "freetoken_commit": revision,
        "lane": "greedy_correctness" if getattr(args, "greedy", False) else "sampled_absolute",
    }


def execution_metadata(
    *, args: argparse.Namespace, backend: str, graph: dict, cache_status: dict
) -> dict:
    """Normalize observed runtime facts into one promotion-gate record."""
    geometry = cache_status.get("geometry") if isinstance(cache_status, dict) else None
    observed = geometry.get("execution") if isinstance(geometry, dict) else None
    observed = observed if isinstance(observed, dict) else {}
    kv_storage = observed.get("kv_storage")
    kv_storage = kv_storage if isinstance(kv_storage, dict) else {}
    return {
        "requested_moe_backend": backend,
        "effective_moe_backend": observed.get("effective_moe_backend"),
        "expert_storage": observed.get("expert_storage"),
        "resident_gguf": observed.get("resident_gguf"),
        "expert_fetches": observed.get("expert_fetches"),
        "expert_remaps": observed.get("expert_remaps"),
        "attention_backend": args.attention_backend,
        "graph_state": graph.get("state"),
        "graph_gate": graph.get("gate"),
        "decode_batch_size": 1,
        "mtp": "off",
        "speculative": False,
        "execution_class": observed.get("execution_class"),
        "kv_type": observed.get("kv_type", kv_storage.get("storage_type")),
        "kv_contract_id": kv_storage.get("contract_id"),
        "kv_pointer_generation": kv_storage.get("pointer_generation"),
        "kv_unit_bytes": kv_storage.get("unit_bytes"),
        "memory_phases": observed.get("memory_phases", []),
    }


def acceptance_status(
    *,
    args: argparse.Namespace,
    result: dict,
    graph: dict,
    usage: dict,
    model: dict,
    runtime: dict | None = None,
    dataset: dict | None = None,
) -> dict:
    """Return explicit row status; rejected rows never contribute to medians."""
    completion = usage.get("completion_tokens")
    context = getattr(args, "context", COMPARATOR_CONTEXT)
    batch = getattr(args, "batch", COMPARATOR_BATCH)
    ubatch = getattr(args, "ubatch", COMPARATOR_UBATCH)
    kv_type = getattr(args, "kv_type", COMPARATOR_KV_TYPE)
    stamps = result.get("stamps") or []
    reasons = []
    checks = {
        "exact_completion_count": completion == args.decode,
        "finite_output": bool(result.get("text")),
        "graph_state_known": graph.get("state") != "unknown",
        "mtp_off": True,
        "speculative_off": True,
        "model_sha256": bool(model.get("sha256")),
        "fixture_sha256": bool((dataset or {}).get("sha256")),
        "kv_type": (runtime or {}).get("kv_type") == COMPARATOR_KV_TYPE,
        "thermal_clock_valid": True,
        "exact_comparator_config": (
            args.decode == 512
            and context == COMPARATOR_CONTEXT
            and batch == COMPARATOR_BATCH
            and ubatch == COMPARATOR_UBATCH
            and kv_type == COMPARATOR_KV_TYPE
        ),
        # API streaming does not expose logits. Probe artifacts own this gate.
        "finite_logits": "unavailable_api",
    }
    if len(stamps) < 2:
        reasons.append(f"need >=2 token events, got {len(stamps)}")
    if completion != args.decode:
        reasons.append(f"completion_tokens={completion!r} != --decode {args.decode}")
    if not checks["exact_comparator_config"]:
        reasons.append(
            "comparator requires decode=512 context=9216 batch=512 ubatch=512 kv_type=q8_0"
        )
    if not result.get("text"):
        reasons.append("server returned no generated text")
    if not model.get("sha256"):
        reasons.append("model full SHA-256 unavailable")
    if dataset is not None and not dataset.get("sha256"):
        reasons.append("fixture full SHA-256 unavailable")
    if runtime is not None:
        if runtime.get("kv_type") != COMPARATOR_KV_TYPE:
            reasons.append(
                f"observed KV type={runtime.get('kv_type')!r} != {COMPARATOR_KV_TYPE!r}"
            )
        telemetry = runtime.get("gpu_telemetry") or {}
        required_telemetry = ("core_clock_mhz", "memory_clock_mhz", "hotspot_c")
        if not all(isinstance(telemetry.get(key), (int, float)) for key in required_telemetry):
            checks["thermal_clock_valid"] = False
            reasons.append("GPU clock/thermal telemetry unavailable")
        end_telemetry = runtime.get("gpu_telemetry_end") or {}
        if all(isinstance(end_telemetry.get(key), (int, float)) for key in required_telemetry):
            hotspot_delta = abs(
                float(end_telemetry["hotspot_c"]) - float(telemetry["hotspot_c"])
            )
            if hotspot_delta > 8.0:
                checks["thermal_clock_valid"] = False
                reasons.append(f"GPU hotspot drift={hotspot_delta:.1f}C exceeds 8C")
    if result.get("stream_error"):
        reasons.append(str(result["stream_error"]))
    if graph.get("state") == "unknown":
        reasons.append("actual graph state unavailable")
    return {
        "status": "accepted" if not reasons else "rejected",
        "accepted": not reasons,
        "checks": checks,
        "reasons": reasons,
    }


def graph_metadata(log_path: str, requested: bool) -> dict[str, str | bool]:
    """Infer actual graph path only from engine log evidence."""
    text = Path(log_path).read_text(errors="replace")
    if not requested:
        return {"requested": False, "state": "disabled", "gate": "not_requested"}
    if "HIP graph capture gate FAILED" in text:
        return {"requested": True, "state": "eager", "gate": "fail"}
    if "CUDA graph is disabled." in text:
        return {"requested": True, "state": "eager", "gate": "disabled"}
    if "Start capturing CUDA graphs" in text and "Free GPU memory after capturing CUDA graphs" in text:
        return {"requested": True, "state": "replay", "gate": "pass"}
    return {"requested": True, "state": "unknown", "gate": "unknown"}


def decode_cache_stats(log_path: str) -> dict[str, float | str]:
    """Read latest engine decode cache counters from the mirrored server log."""
    text = Path(log_path).read_text(errors="replace")
    matches = re.findall(
        r"Decode batch.*?moe hit: ([0-9.]+), miss: ([0-9.]+), "
        r"fetch: ([0-9.]+), cpu: ([0-9.]+)",
        text,
    )
    if not matches:
        return {}
    hit, miss, fetch, cpu = (float(value) for value in matches[-1])
    return {
        "hit_rate": hit,
        "miss_rate": miss,
        "fetch_rate": fetch,
        "cpu_per_layer": cpu,
        "source": "server_log_latest_decode_batch",
    }


def native_decode_stats(log_path: str) -> dict[str, float | int | str]:
    """Read scheduler's device-side decode rate, separate from API arrival timing."""
    text = Path(log_path).read_text(errors="replace")
    matches = re.findall(
        r"Decode batch.*?gen throughput \(token/s\): ([0-9.]+)", text
    )
    if not matches:
        return {}
    return {
        "native_decode_tok_s": float(matches[-1]),
        "native_timing_source": "scheduler_decode_log",
        "native_timing_samples": len(matches),
    }


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


_RELEVANT_ENV_PREFIXES = (
    "FREETOKEN_", "LLAMA_", "HIP", "ROCR", "ROCM", "HSA_", "GPU_MAX_",
    "PYTORCH_ROCM", "TORCH_ROCM", "CUDA_VISIBLE", "ROCM_PATH", "MIOPEN_",
)


def dirty_diff_hash() -> str | None:
    """Content hash of the exact dirty state the measured code came from.

    A dirty tree is legitimate evidence as long as its content identity is pinned;
    the hash is what makes the pinned state reproducible later."""
    repo = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "-uall"], cwd=repo, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    if not status.strip():
        return None
    digest = hashlib.sha256()
    digest.update(f"head={head}\n".encode())
    digest.update(status.encode())
    try:
        diff = subprocess.check_output(
            ["git", "diff", "HEAD"], cwd=repo, text=True, timeout=30
        )
    except (OSError, subprocess.CalledProcessError):
        diff = ""
    digest.update(f"diff_bytes={len(diff)}\n".encode())
    digest.update(diff.encode("utf-8", errors="replace"))
    return digest.hexdigest()


def environment_selector_digest(env: dict | None = None) -> dict:
    """Digest the runtime-selection environment the spawned server inherits.

    Covers FREETOKEN_* feature selectors plus the HIP/ROCm runtime knobs that can
    change kernel selection or code generation. Values are included (they are
    config, not secrets); unknown/absent keys contribute nothing."""
    relevant = {
        key: value
        for key, value in (env if env is not None else os.environ).items()
        if key.startswith(("FREETOKEN_", "HIP", "ROCR", "ROCM", "HSA", "PYTORCH_ROCM"))
        or key in {"LD_LIBRARY_PATH", "PYTHONPATH"}
    }
    canonical = "\n".join(
        f"{key}={relevant[key]}" for key in sorted(relevant)
    )
    return {
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "keys": sorted(relevant),
        "count": len(relevant),
    }


def jit_binary_sha() -> dict:
    """Best-effort SHA-256 of the JIT-compiled GGUF kernel module binary.

    The server compiles ``freetoken_gguf_kernels`` into torch's extension cache;
    the newest ``.so`` in that build directory is what actually loads. Absent or
    unreadable cache is reported as null with a reason, never as a fake hash."""
    info: dict = {"sha256": None, "path": None, "reason": None}
    try:
        from torch.utils.cpp_extension import _get_build_directory

        build_dir = Path(_get_build_directory("freetoken_gguf_kernels", False))
        binaries = sorted(build_dir.glob("*.so"), key=lambda p: p.stat().st_mtime)
        if not binaries:
            info["reason"] = "no built .so in extension cache (module not compiled yet)"
            return info
        newest = binaries[-1]
        info["path"] = str(newest)
        info["sha256"] = sha256_file(str(newest))
    except Exception as exc:  # extension cache layout changed / torch absent
        info["reason"] = f"{type(exc).__name__}: {exc}"
    return info


def kernel_implementation_from_log(log_path: str) -> dict:
    """Observed kernel implementation/fallback evidence from the server log.

    The engine owns implementation selection; the log is the only client-visible
    record of what actually loaded, mirroring how graph state is already read."""
    text = Path(log_path).read_text(errors="replace")
    match = re.search(r"Auto-selected MoE backend: (\S+)", text)
    fallbacks = re.findall(r"fallback\b", text, re.IGNORECASE)
    return {
        "auto_selected_moe_backend": match.group(1) if match else None,
        "fallback_markers": len(fallbacks),
        "source": "server_log",
    }


def stream_generate(origin: str, model_id: str, problem: str, sampling: dict,
                    args: argparse.Namespace) -> dict:
    """One streamed chat completion; returns per-token arrival stamps, text, and usage."""
    # On current FreeToken, a warmed full-prefix request consumes one output budget slot
    # before its first streamed completion ack. Ask for one extra engine slot so strict
    # validation still produces exactly D completion tokens instead of accepting D-1.
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": problem}],
        "max_tokens": args.decode + 1,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": True},
        **sampling,
    }
    req = urllib.request.Request(
        f"{origin}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    stamps: list[float] = []
    pieces: list[str] = []
    usage: dict | None = None
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=1800)
    except urllib.error.HTTPError as e:
        sys.exit(f"[bench] request failed: HTTP {e.code}: {e.read()[:500]!r}")
    # Iterate the SSE stream line by line as bytes; json.loads decodes UTF-8 itself.
    # (A text-mode reader keyed off the content-type would decode latin-1: the server
    # sends ensure_ascii=False JSON with no charset on text/event-stream.)
    with resp:
        for raw in resp:
            line = raw.strip()
            if not line or not line.startswith(b"data:"):
                continue  # blank separators between events
            payload = line[len(b"data:"):].strip()
            if payload == b"[DONE]":
                break
            now = time.perf_counter()
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                text = "".join(
                    part for part in (delta.get("reasoning_content"), delta.get("content")) if part
                )
                if text:
                    stamps.append(now)
                    pieces.append(text)
    return {
        "t0": t0,
        "stamps": stamps,
        "text": "".join(pieces),
        "usage": usage or {},
        "stream_error": None if usage is not None else "missing_usage_chunk",
        "prompt_sha256": hashlib.sha256(problem.encode("utf-8")).hexdigest(),
    }


def run_one(args: argparse.Namespace, backend: str) -> list[dict]:
    problem, answer, dataset = load_problem_details(
        args.aime, args.problem, args.aime_revision, args.aime_sha256
    )
    sampling, sampling_src = resolve_sampling(args.model, args.greedy)
    model = model_fingerprint(args.model)
    runtime = runtime_metadata()
    revision = git_revision()
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    fd, log_path = tempfile.mkstemp(prefix=f"bench-serve-{backend}-", suffix=".log")
    cmd = serve_cmd(args, backend, port)

    if args.repeats < 1:
        sys.exit("--repeats must be >= 1")
    print(
        f"[bench] model={args.model}\n"
        f"[bench] backend={backend} cache={args.cache or args.cache_rate or 'auto'} "
        f"mem_ratio={args.mem_ratio} decode={args.decode} graph={not args.no_graph}\n"
        f"[bench] sampling={sampling} <- {sampling_src}\n"
        f"[bench] server log: {log_path}",
        flush=True,
    )

    with os.fdopen(fd, "wb") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
        )
        pump = threading.Thread(target=pump_output, args=(proc.stdout, log_f), daemon=True)
        pump.start()
        try:
            wait_ready(origin, proc, log_path, args.server_timeout)
            model_id = get_json(f"{origin}/v1/models")["data"][0]["id"]
            print(f"[bench] model_id={model_id}", flush=True)
            print(f"[bench] AIME25 #{args.problem} (answer {answer})", flush=True)

            # Warm the expert cache to a steady-state decode working set.
            stream_generate(origin, model_id, problem, sampling, args)
            rows = []
            for repeat in range(args.repeats):
                # Per-repeat start telemetry: a like-for-like loaded-vs-loaded
                # window. The pre-spawn snapshot is idle and would make every
                # row fail the hotspot-delta check by construction.
                runtime_start = {**runtime, "gpu_telemetry": gpu_telemetry()}
                r = stream_generate(origin, model_id, problem, sampling, args)
                stats = get_json(f"{origin}/v1/stats")
                try:
                    cache_status = get_json(f"{origin}/v1/cache/status")
                except (OSError, ValueError):
                    cache_status = {}
                rows.append(
                    make_row(
                        args,
                        backend,
                        repeat,
                        r,
                        stats,
                        cache_status,
                        log_path,
                        model,
                        runtime_start,
                        revision,
                        dataset,
                        sampling,
                        sampling_src,
                        model_id,
                    )
                )
        finally:
            stop_server(proc)
            pump.join(timeout=10)

    for row in rows:
        print_row(row, args, answer, rows)
    return rows


def make_row(
    args: argparse.Namespace,
    backend: str,
    repeat: int,
    result: dict,
    stats: dict,
    cache_status: dict,
    log_path: str,
    model: dict,
    runtime: dict,
    revision: str,
    dataset: dict,
    sampling: dict,
    sampling_src: str,
    model_id: str,
) -> dict:
    stamps, usage = result["stamps"], result["usage"]
    runtime = dict(runtime)
    runtime["gpu_telemetry_end"] = gpu_telemetry()
    completion = usage.get("completion_tokens")
    steps = completion - 1 if isinstance(completion, int) and completion > 0 else 0
    decode_time = stamps[-1] - stamps[0] if len(stamps) >= 2 else 0.0
    gaps = sorted((b - a) * 1e3 for a, b in zip(stamps, stamps[1:]))
    graph = graph_metadata(log_path, not args.no_graph)
    execution = execution_metadata(
        args=args, backend=backend, graph=graph, cache_status=cache_status
    )
    # CLI request is not observation. The readiness/status response is the first
    # server-owned source proving physical KV allocation and contract.
    observed_kv = execution.get("kv_type")
    if observed_kv is not None:
        runtime["kv_type"] = observed_kv
        runtime["kv_storage"] = {
            "contract_id": execution.get("kv_contract_id"),
            "pointer_generation": execution.get("kv_pointer_generation"),
            "unit_bytes": execution.get("kv_unit_bytes"),
        }
    acceptance = acceptance_status(
        args=args,
        result=result,
        graph=graph,
        usage=usage,
        model=model,
        runtime=runtime,
        dataset=dataset,
    )
    metadata = run_metadata(
        args=args,
        backend=backend,
        model=model,
        runtime=runtime,
        graph=graph,
        dataset=dataset,
        sampling=sampling,
        sampling_src=sampling_src,
        revision=revision,
        prompt_sha256=result.get("prompt_sha256", "unknown"),
        model_id=model_id,
    )
    metadata["execution"] = execution
    metadata["prompt_tokens_expected"] = usage.get("prompt_tokens")
    metadata["kernel_observed"] = kernel_implementation_from_log(log_path)
    metadata["observed_kv_type"] = execution.get("kv_type")
    metadata["kv_type_source"] = "engine_cache_metadata" if execution.get("kv_type") else "unobserved"
    row = {
        "schema": "freetoken-base-decode-v2",
        "status": acceptance["status"],
        "acceptance": acceptance,
        "metadata": metadata,
        "git_revision": revision,
        "lane": metadata["lane"],
        "model": args.model,
        "model_id": model_id,
        "model_fingerprint": model,
        "backend": backend,
        "repeat": repeat,
        "problem": args.problem,
        "dataset": dataset,
        "prompt_tokens": usage.get("prompt_tokens"),
        "decode_steps": steps,
        "decode_tok_s": steps / decode_time if decode_time > 0 and steps > 0 else None,
        "ms_per_token": decode_time / steps * 1e3 if decode_time > 0 and steps > 0 else None,
        "event_ms_p50": gaps[len(gaps) // 2] if gaps else None,
        "event_ms_p99": gaps[min(len(gaps) - 1, int(len(gaps) * 0.99))] if gaps else None,
        "ttft_ms": (stamps[0] - result["t0"]) * 1e3 if stamps else None,
        "events": len(stamps),
        "completion_tokens": completion,
        "decode_requested": args.decode,
        "context": args.context,
        "batch": args.batch,
        "ubatch": args.ubatch,
        "kv_type": args.kv_type,
        "fixture_sha256": dataset.get("sha256"),
        "vram_gib": stats.get("vram_bytes", 0) / 2**30,
        "sampling": sampling,
        "sampling_source": sampling_src,
        "mtp": "off",
        "speculative": False,
        "decode_batch_size": 1,
        "attention_backend": args.attention_backend,
        "graph": graph,
        "cache_request": {
            "slots": args.cache,
            "rate": args.cache_rate,
            "auto": args.cache <= 0 and args.cache_rate is None,
            "memory_ratio": args.mem_ratio,
        },
        "cache_status": cache_status,
        "cache_decode_stats": decode_cache_stats(log_path),
        "runtime": runtime,
        "execution": execution,
        "decode_window_s": decode_time,
        "timing_window": "first_generated_token_to_last_generated_token",
        **native_decode_stats(log_path),
        "output_sha1": hashlib.sha1(result.get("text", "").encode()).hexdigest()[:12],
        "output_sha256": hashlib.sha256(result.get("text", "").encode()).hexdigest(),
        "output_sample": result.get("text", "")[:240],
        "stream_error": result.get("stream_error"),
        "server_log": log_path,
        "stage_summary": {
            "status": "unavailable",
            "reason": "launch timeline is collected by scripts/profile-rocm-decode.sh",
        },
    }
    return row


def print_row(row: dict, args: argparse.Namespace, answer: str, rows: list[dict]) -> None:
    throughput = row["decode_tok_s"]
    ms_per_token = row["ms_per_token"]
    ttft = row["ttft_ms"]
    p50 = row["event_ms_p50"]
    p99 = row["event_ms_p99"]
    throughput_text = "rejected" if throughput is None else f"{throughput:8.2f} tok/s"
    ms_text = "n/a" if ms_per_token is None else f"{ms_per_token:.3f} ms/token"
    ttft_text = "n/a" if ttft is None else f"{ttft:8.1f} ms"
    p50_text = "n/a" if p50 is None else f"{p50:.3f}"
    p99_text = "n/a" if p99 is None else f"{p99:.3f}"
    print(f"\n==== decode bs=1 [{row['backend']}] via /v1/chat/completions ====", flush=True)
    print(f"  repeat            : {row['repeat'] + 1}/{len(rows)}")
    print(f"  decode throughput : {throughput_text}  ({ms_text})")
    print(f"  TTFT (warm)       : {ttft_text}  (prompt {row['prompt_tokens']} tok)")
    print(f"  decode measured   : {row['decode_steps']} steps in {row['decode_window_s']:.3f} s  "
          f"(event p50 {p50_text} / p99 {p99_text} ms, "
          f"{row['events']} events)")
    print(f"  vram (server)     : {row['vram_gib']:8.2f} GiB")
    sha_note = "greedy" if args.greedy else "sampled, per-server deterministic"
    print(f"  output sha1       : {row['output_sha1']}  ({sha_note}; compare across backends)")
    print(f"  graph             : {row['graph']['state']} (gate={row['graph']['gate']})")
    print(f"  status            : {row['status']} ({'; '.join(row['acceptance']['reasons']) or 'ok'})")
    print(f"  output sample     : {row['output_sample']!r}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    backends = [b.strip() for b in args.backend.split(",") if b.strip()]
    unknown = [b for b in backends if b not in ("fused", "offload", "cpu", "hybrid")]
    if unknown:
        sys.exit(f"unknown backend(s): {unknown}")

    failed = []
    rejected = []
    for backend in backends:
        try:
            rows = run_one(args, backend)
        # SystemExit inherits BaseException, not Exception, so name both: a mid-decode
        # connection drop (server crash) must not abort the remaining backends either.
        except (SystemExit, Exception) as e:
            if len(backends) == 1:
                raise
            print(f"\n[bench] backend {backend} failed: {e!r}", flush=True)
            failed.append(backend)
            continue
        if args.json_out:
            with open(args.json_out, "a") as f:
                for row in rows:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
        rejected.extend(row for row in rows if row["status"] != "accepted")
    if failed:
        print(f"\n[bench] backends that failed: {failed}", flush=True)
        return 1
    if rejected:
        print(f"\n[bench] rejected measured rows: {len(rejected)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
