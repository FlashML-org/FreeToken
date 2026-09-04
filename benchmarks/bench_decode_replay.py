"""Teacher-forced replay lanes and golden-output pinning (rocm-ollama-gap Inc 0).

This module owns the replay contract shared by every runtime comparator:

* **Golden capture** -- one in-process FreeToken greedy run pins the legacy
  512-token ID sequence, per-run logit finiteness, and the boxed answer. The
  golden must reproduce from fresh processes; it is the correctness anchor for
  every later fused-kernel increment.
* **Teacher-forced replay** -- every runtime is fed the *same* token sequence.
  FreeToken is driven in-process with the sampler overridden to emit the pinned
  next token, so logits are timed without sampling feedback and routes cannot
  diverge after the first logit difference. llama.cpp is driven through
  ``llama-server`` with ``cache_prompt`` and ``n_predict=1`` so each step
  evaluates exactly one forced token per request over identical inputs.
* **Correctness replay** -- a separate untimed pass that captures per-(token,
  layer) top-k route-ID hashes. Route capture adds host synchronization, so it
  must never run inside timed passes; performance replay is primary only when
  route hashes match exactly across runtimes. llama.cpp route capture requires
  an instrumented build and is recorded as unavailable until then.

Lane discipline: every row produced here carries ``lane=teacher_forced_replay``
and is disjoint from ``sampled_absolute`` and ``greedy_correctness`` rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from bench_decode_moe import (
    git_revision,
    load_problem_details,
    model_fingerprint,
    runtime_metadata,
)

MANIFEST_SCHEMA = "freetoken-replay-manifest-v1"
LANE = "teacher_forced_replay"


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def route_hash(topk_ids) -> str:
    """Stable hash of one (token, layer) top-k route-ID tuple."""
    return _sha256_bytes(json.dumps([int(v) for v in topk_ids]).encode())[:16]


def continuation_hash(ids: list[int]) -> str:
    return _sha256_bytes(json.dumps([int(v) for v in ids]).encode())


def validate_manifest(manifest: object) -> list[str]:
    """Structural + contract validation; empty list means acceptable."""
    problems: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest is not an object"]
    if manifest.get("schema") != MANIFEST_SCHEMA:
        problems.append(f"schema={manifest.get('schema')!r} != {MANIFEST_SCHEMA!r}")
    if manifest.get("lane") != LANE:
        problems.append(f"lane={manifest.get('lane')!r} != {LANE!r}")
    prompt_ids = manifest.get("prompt_ids")
    continuation = manifest.get("continuation_ids")
    if not isinstance(prompt_ids, list) or not prompt_ids:
        problems.append("prompt_ids missing or empty")
    elif any(not isinstance(v, int) or v < 0 for v in prompt_ids):
        problems.append("prompt_ids must be non-negative token IDs")
    if not isinstance(continuation, list) or len(continuation) < 2:
        problems.append("continuation_ids missing or shorter than two pinned IDs")
    else:
        if any(not isinstance(v, int) or v < 0 for v in continuation):
            problems.append("continuation_ids must be non-negative token IDs")
        if manifest.get("measured_tokens") != len(continuation):
            problems.append("measured_tokens != len(continuation_ids)")
        warmup = manifest.get("warmup_tokens", 0)
        measured = manifest.get("measured_tokens", 0)
        if not isinstance(warmup, int) or warmup < 0:
            problems.append("warmup_tokens must be a non-negative integer")
        if not isinstance(measured, int) or measured < 1:
            problems.append("measured_tokens must be a positive integer")
    for key in ("model_sha256", "fixture_sha256", "tokenizer_sha256"):
        value = manifest.get(key)
        if not (isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)):
            problems.append(f"{key} missing or not a full SHA-256")
    if not isinstance(manifest.get("route_top_k"), int) or manifest["route_top_k"] < 1:
        problems.append("route_top_k missing or not positive")
    golden = manifest.get("golden")
    if not isinstance(golden, dict) or not (
        isinstance(golden.get("ids_sha256"), str)
        and len(golden["ids_sha256"]) == 64
        and all(c in "0123456789abcdefABCDEF" for c in golden["ids_sha256"])
    ):
        problems.append("golden.ids_sha256 missing")
    prompt_text = manifest.get("prompt_text")
    prompt_text_sha = manifest.get("prompt_text_sha256")
    if prompt_text is not None:
        if not isinstance(prompt_text, str):
            problems.append("prompt_text must be a string when present")
        elif not (
            isinstance(prompt_text_sha, str)
            and len(prompt_text_sha) == 64
            and all(c in "0123456789abcdefABCDEF" for c in prompt_text_sha)
        ):
            problems.append("prompt_text_sha256 missing or not a full SHA-256")
        elif _sha256_bytes(prompt_text.encode()) != prompt_text_sha:
            problems.append("prompt_text_sha256 does not match prompt_text")
    return problems


def load_manifest(path: str) -> dict:
    manifest = json.loads(Path(path).read_text())
    problems = validate_manifest(manifest)
    if problems:
        raise ValueError(f"replay manifest rejected ({path}): {'; '.join(problems)}")
    return manifest


def summarize_steps(steps_ms: list[float], warmup_steps: int) -> dict:
    """Warmup-excluded step statistics; the raw per-step list is always kept."""
    warmup_steps = max(0, min(int(warmup_steps), len(steps_ms)))
    measured = list(steps_ms[warmup_steps:])
    if not measured:
        return {
            "steps": 0,
            "warmup_steps": warmup_steps,
            "ms_per_token_median": None,
            "raw_steps_ms": list(steps_ms),
        }
    return {
        "steps": len(measured),
        "warmup_steps": len(steps_ms) - len(measured),
        "ms_per_token_median": statistics.median(measured),
        "ms_per_token_mean": statistics.fmean(measured),
        "ms_per_token_min": min(measured),
        "ms_per_token_max": max(measured),
        "raw_steps_ms": list(steps_ms),
    }


# ---------------------------------------------------------------------------
# FreeToken in-process adapter
# ---------------------------------------------------------------------------


def _build_llm(args: argparse.Namespace):
    import torch

    from freetoken.llm import LLM

    kwargs = {
        "attention_backend": args.attention_backend,
        "max_running_req": 1,
        "max_extend_tokens": 8192,
        "max_seq_len_override": args.context,
        "moe_backend": args.moe_backend,
        "memory_ratio": args.memory_ratio,
        "moe_cache_auto": args.cache == 0,
        "cuda_graph_max_bs": 1 if args.graph else 0,
        "kv_storage_type": args.kv_type,
    }
    if args.cache > 0:
        kwargs["moe_cache_size"] = args.cache
    return LLM(args.model, dtype=torch.bfloat16, **kwargs)


def _tokenizer_sha(llm) -> str:
    tok = llm.tokenizer
    payload = {
        "class": type(tok).__name__,
        "vocab_size": getattr(tok, "vocab_size", None),
        "eos_token_id": getattr(tok, "eos_token_id", None),
        "bos_token_id": getattr(tok, "bos_token_id", None),
        "pad_token_id": getattr(tok, "pad_token_id", None),
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True).encode())


class _RouteCapture:
    """Record per-(token, layer) top-k route-ID hashes from the router logits.

    Routes are re-derived with ``torch.topk`` over the float router logits --
    the same decision the layer's own router makes (softmax is order-preserving,
    ``torch.topk`` tie-breaking is index-stable). Every call appends host-side
    data, so capture belongs ONLY in the untimed correctness replay."""

    def __init__(self, llm):
        import torch

        self._torch = torch
        self.records: list[dict] = []
        self._restores: list = []
        self._offsets: dict[int, int] = {}
        for index, block in _routed_blocks(llm):
            top_k = int(getattr(block, "top_k", 8))
            original = block.experts.forward

            def make_wrapper(fn, layer_index, k):
                def wrapped(*call_args, **call_kwargs):
                    out = fn(*call_args, **call_kwargs)
                    try:
                        logits = call_kwargs.get("router_logits")
                        if logits is None and len(call_args) >= 2:
                            candidate = call_args[1]
                            if hasattr(candidate, "dim") and candidate.dim() == 2:
                                logits = candidate
                        if logits is not None:
                            ids = self._torch.topk(
                                logits.detach().float(), k=k, dim=-1
                            ).indices
                            start = self._offsets.get(layer_index, 0)
                            hashes = [route_hash(row) for row in ids.tolist()]
                            self.records.append(
                                {
                                    "layer": layer_index,
                                    "start": start,
                                    "hashes": hashes,
                                }
                            )
                            self._offsets[layer_index] = start + len(hashes)
                    except Exception:  # capture must never break the run
                        pass
                    return out

                return wrapped

            block.experts.forward = make_wrapper(original, index, top_k)
            self._restores.append((block.experts, original))

    def restore(self) -> None:
        for obj, original in self._restores:
            try:
                obj.forward = original
            except Exception:
                pass


def _routed_blocks(llm) -> list[tuple[int, object]]:
    """(model-layer-index, MoE block) for every layer with routed experts."""
    layers = getattr(llm.engine.model, "model", None)
    layers = getattr(layers, "layers", None) or []
    # FreeToken's execution model stores transformer blocks in OPList.op_list;
    # ordinary ModuleList containers remain directly iterable.
    layers = getattr(layers, "op_list", layers)
    found = []
    for index, layer in enumerate(layers):
        for attr in ("moe", "mlp"):
            block = getattr(layer, attr, None)
            if block is not None and hasattr(block, "experts"):
                found.append((index, block))
                break
    return found


def build_manifest(args: argparse.Namespace) -> dict:
    """Golden run + tokenizer/model/fixture identity, ready for replay lanes.

    Golden means the current default FreeToken path -- no candidate kernels, no
    forced switches. This output is the fusion golden for Inc 9/12."""
    import torch

    from freetoken.core import SamplingParams

    problem, answer, dataset = load_problem_details(
        args.aime, args.problem, args.aime_revision, args.aime_sha256
    )
    llm = _build_llm(args)
    captured: list = []
    sampler = llm.engine.sampler
    original_sample = sampler.sample
    original_sample_into_device = sampler.sample_into_device

    def capture_sample(logits, sample_args, batch):
        captured.append(logits.detach().float().cpu())
        return original_sample(logits, sample_args, batch)

    def capture_sample_into_device(logits, sample_args, batch, out, scratch):
        captured.append(logits.detach().float().cpu())
        return original_sample_into_device(logits, sample_args, batch, out, scratch)

    try:
        encoded = llm.tokenizer.encode(problem)
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        prompt_ids = [int(v) for v in encoded]
        tokenizer_sha = _tokenizer_sha(llm)
        sampler.sample = capture_sample
        sampler.sample_into_device = capture_sample_into_device
        try:
            output = llm.generate(
                [problem],
                SamplingParams(
                    temperature=0.0, top_p=1.0, top_k=-1,
                    # Offline scheduler currently consumes one fewer output
                    # slot at exact max length than HTTP generation. Request
                    # one sentinel step, then pin only requested decode IDs.
                    max_tokens=args.decode + 1, ignore_eos=True,
                ),
            )[0]
        finally:
            sampler.sample = original_sample
            sampler.sample_into_device = original_sample_into_device
    finally:
        llm.shutdown()

    token_ids = [int(v) for v in output["token_ids"]][: args.decode]
    if len(token_ids) != args.decode:
        raise RuntimeError(f"golden produced {len(token_ids)} usable ids, expected {args.decode}")
    rows = torch.cat(captured, dim=0)[: args.decode] if captured else torch.empty(0)
    if rows.shape[0] != args.decode:
        raise RuntimeError(f"golden captured {rows.shape[0]} usable logit rows, expected {args.decode}")
    if not bool(torch.isfinite(rows).all()):
        raise RuntimeError("golden run produced non-finite logits")
    text = output["text"]
    boxed = text.split("\\boxed{")[-1].split("}")[0] if "\\boxed{" in text else None
    model = model_fingerprint(args.model)
    runtime = runtime_metadata()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "lane": LANE,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_path": str(Path(args.model).expanduser().resolve()),
        "model_sha256": model.get("sha256"),
        "model_size_bytes": model.get("size_bytes"),
        "model_identity": model.get("identity", "unverified"),
        "fixture_sha256": dataset.get("sha256"),
        "fixture_revision": dataset.get("revision"),
        "tokenizer_sha256": tokenizer_sha,
        "prompt_text": problem,
        "prompt_text_sha256": _sha256_bytes(problem.encode()),
        "prompt_ids": prompt_ids,
        "continuation_ids": token_ids,
        "warmup_tokens": args.warmup,
        "measured_tokens": len(token_ids),
        "route_top_k": args.route_top_k,
        "golden": {
            "source": "freetoken-legacy-greedy",
            "ids_sha256": continuation_hash(token_ids),
            "text_sha256": _sha256_bytes(text.encode()),
            "answer": boxed,
            "expected_answer": answer,
            "answer_match": boxed == answer,
            "finite_logits": True,
            "logit_rows": int(rows.shape[0]),
            "logit_sha256": _sha256_bytes(rows.numpy().tobytes()),
            "decode": args.decode,
        },
        "runtime": {
            key: runtime.get(key)
            for key in ("torch", "rocm", "gpu_capability", "gpu_arch", "gpu_name")
        },
        "git_revision": git_revision(),
    }
    problems = validate_manifest(manifest)
    if problems:
        raise RuntimeError(f"built manifest is invalid: {'; '.join(problems)}")
    return manifest


def replay_freetoken(
    args: argparse.Namespace, manifest: dict, *, capture_routes: bool = False
) -> dict:
    """Teacher-forced replay against FreeToken on the pinned token sequence.

    The measured window forces ``measured_tokens`` pinned IDs and reports host
    submission gaps plus a synchronized end-to-end window. With
    ``capture_routes`` the run becomes the untimed correctness replay that
    additionally hashes per-(token, layer) top-k route IDs."""
    import torch

    from freetoken.core import SamplingParams

    prompt_ids = list(manifest["prompt_ids"])
    pinned = list(manifest["continuation_ids"])
    warmup = int(manifest.get("warmup_tokens", 0))
    measured = int(manifest["measured_tokens"])

    llm = _build_llm(args)
    stamps: list[tuple[float, int]] = []
    timed_ids: list[int] = []
    sampler = None
    original_sample = None
    original_sample_into_device = None
    route_digest = None
    route_hash_status = "not_requested"
    try:
        # Warmup pass: identical forced inputs, discarded (JIT/graph amortized here).
        if warmup:
            llm.generate(
                [prompt_ids],
                SamplingParams(
                    temperature=0.0, top_p=1.0, top_k=-1,
                    max_tokens=warmup, ignore_eos=True,
                ),
            )
        sampler = llm.engine.sampler
        original_sample = sampler.sample
        original_sample_into_device = sampler.sample_into_device

        def forced_sample(logits, sample_args, batch):
            out = original_sample(logits, sample_args, batch)
            if len(timed_ids) < measured:
                index = len(timed_ids)
                stamps.append((time.perf_counter(), int(pinned[index])))
                timed_ids.append(int(pinned[index]))
                out.copy_(torch.full_like(out, pinned[index]))
            return out

        def forced_sample_into_device(logits, sample_args, batch, out, scratch):
            sampled = original_sample_into_device(logits, sample_args, batch, out, scratch)
            if len(timed_ids) < measured:
                index = len(timed_ids)
                stamps.append((time.perf_counter(), int(pinned[index])))
                timed_ids.append(int(pinned[index]))
                out.copy_(torch.full_like(out, pinned[index]))
                return out
            return sampled

        sampler.sample = forced_sample
        sampler.sample_into_device = forced_sample_into_device
        torch.cuda.synchronize()
        window_start = time.perf_counter()
        result = llm.generate(
                [prompt_ids],
                SamplingParams(
                    temperature=0.0, top_p=1.0, top_k=-1,
                    max_tokens=measured + 1, ignore_eos=True,
                ),
            )[0]
        torch.cuda.synchronize()
        window_s = time.perf_counter() - window_start
        sampler.sample = original_sample
        sampler.sample_into_device = original_sample_into_device
        if capture_routes:
            route_digest, route_hash_status = _capture_freetoken_routes_on_llm(
                args, manifest, llm
            )
    finally:
        if sampler is not None and original_sample is not None:
            sampler.sample = original_sample
        if sampler is not None and original_sample_into_device is not None:
            sampler.sample_into_device = original_sample_into_device
        llm.shutdown()

    token_ids = [int(v) for v in result.get("token_ids") or []]
    forced_ids = token_ids[:measured]
    expected_ids = pinned[:measured]
    steps_ms = [
        (b - ta) * 1e3 for (ta, _), (b, _) in zip(stamps, stamps[1:])
    ][:measured]
    return {
        "schema": "freetoken-replay-v1",
        "lane": LANE,
        "runtime": "freetoken",
        "status": "accepted" if forced_ids == expected_ids and len(token_ids) == measured else "rejected",
        "execution": "graph_replay" if args.graph else "eager",
        "repeat": getattr(args, "repeat", None),
        "model_sha256": manifest["model_sha256"],
        "fixture_sha256": manifest["fixture_sha256"],
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "manifest_ids_sha256": manifest["golden"]["ids_sha256"],
        "forced": True,
        "ids_match": forced_ids == expected_ids and len(token_ids) == measured,
        "forced_ids_sha256": continuation_hash(forced_ids),
        "steps": summarize_steps(steps_ms, 0),
        "window_ms": window_s * 1e3,
        "ms_per_token_window": window_s * 1e3 / measured if measured else None,
        "timing_domain": "host_window_synchronized",
        "route_digest": route_digest,
        "route_hash_status": route_hash_status,
        "route_capture_timing": "untimed_separate_pass" if capture_routes else None,
        "mtp": "off",
        "speculative": False,
        "decode_batch_size": 1,
        "context": args.context,
        "batch": args.batch,
        "ubatch": args.ubatch,
        "kv_type": args.kv_type,
        "decode_ms_per_token_median": window_s * 1e3 / measured if measured else None,
        "acceptance": {
            "accepted": forced_ids == expected_ids and len(token_ids) == measured,
            "ids_match": forced_ids == expected_ids and len(token_ids) == measured,
        },
    }


def _route_digest(records: list[dict]) -> dict[str, str]:
    """Combine per-layer route hashes into deterministic per-token digests."""
    per_step: dict[int, list[tuple[int, str]]] = {}
    for record in records:
        layer = int(record["layer"])
        start = int(record.get("start", 0))
        for offset, digest in enumerate(record.get("hashes", [])):
            per_step.setdefault(start + offset, []).append((layer, digest))
    return {
        str(step): hashlib.sha256(
            json.dumps(sorted(values), separators=(",", ":")).encode()
        ).hexdigest()
        for step, values in sorted(per_step.items())
    }


def _capture_freetoken_routes_on_llm(
    args: argparse.Namespace, manifest: dict, llm
) -> tuple[dict | None, str]:
    """Run untimed route capture on already initialized FreeToken state."""
    import torch

    from freetoken.core import SamplingParams

    prompt_ids = list(manifest["prompt_ids"])
    pinned = list(manifest["continuation_ids"])
    measured = int(manifest["measured_tokens"])
    expected_layers = {index for index, _ in _routed_blocks(llm)}
    capture = _RouteCapture(llm)
    sampler = llm.engine.sampler
    original_sample = sampler.sample
    original_sample_into_device = sampler.sample_into_device
    forced: list[int] = []
    ids_match = False
    try:
        def forced_sample(logits, sample_args, batch):
            out = original_sample(logits, sample_args, batch)
            if len(forced) < measured:
                token = int(pinned[len(forced)])
                forced.append(token)
                out.copy_(torch.full_like(out, token))
            return out

        def forced_sample_into_device(logits, sample_args, batch, out, scratch):
            sampled = original_sample_into_device(logits, sample_args, batch, out, scratch)
            if len(forced) < measured:
                token = int(pinned[len(forced)])
                forced.append(token)
                out.copy_(torch.full_like(out, token))
                return out
            return sampled

        sampler.sample = forced_sample
        sampler.sample_into_device = forced_sample_into_device
        result = llm.generate(
            [prompt_ids],
            SamplingParams(
                temperature=0.0, top_p=1.0, top_k=-1,
                max_tokens=measured + 1, ignore_eos=True,
            ),
        )[0]
        token_ids = [int(v) for v in result.get("token_ids") or []]
        ids_match = token_ids == pinned[:measured]
        records = [dict(record) for record in capture.records]
    finally:
        sampler.sample = original_sample
        sampler.sample_into_device = original_sample_into_device
        capture.restore()
    if not ids_match or len(forced) != measured:
        return None, "ids_mismatch"
    if not records:
        return None, "unavailable"
    observed_layers = {int(record["layer"]) for record in records}
    if observed_layers != expected_layers:
        return None, "incomplete_layer_capture"
    return _route_digest(records), "captured"


def capture_freetoken_routes(
    args: argparse.Namespace, manifest: dict
) -> tuple[dict | None, str]:
    """Run untimed forced replay solely for route capture.

    Host synchronization and route copies stay outside measured replay timing.
    """
    llm = _build_llm(args)
    try:
        return _capture_freetoken_routes_on_llm(args, manifest, llm)
    finally:
        llm.shutdown()


# ---------------------------------------------------------------------------
# llama.cpp adapter
# ---------------------------------------------------------------------------


def _server_json(origin: str, path: str, body: dict | None = None, timeout: float = 30):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"{origin.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _stop_llama_server(proc: subprocess.Popen) -> None:
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


def replay_llama(args: argparse.Namespace, manifest: dict, repeat: int) -> dict:
    """Teacher-forced replay against a pinned ``llama-server``.

    Each step sends ``prompt = prompt_ids + pinned[:k]`` with ``cache_prompt``
    and ``n_predict=1``. The sampled output is discarded. Because each forced
    token is appended to cached prompt, llama-server reports its forward pass
    under ``timings.prompt_ms``; ``predicted_ms`` is only an additional sampled
    token and is retained as a diagnostic.
    The input context stays byte-identical to every other runtime's."""
    binary = Path(args.server).expanduser()
    if not binary.is_file():
        raise SystemExit(f"llama-server binary not found: {binary}")
    port = _free_port()
    origin = f"http://127.0.0.1:{port}"
    command = [
        str(binary),
        "-m", str(Path(args.model).expanduser()),
        "--host", "127.0.0.1", "--port", str(port),
        "-c", str(getattr(args, "context", 9216)),
        "-b", str(getattr(args, "batch", 512)),
        "-ub", str(getattr(args, "ubatch", 512)), "-np", "1",
        "-ngl", "99", "-fa", "on",
        "-ctk", getattr(args, "kv_type", "q8_0"),
        "-ctv", getattr(args, "kv_type", "q8_0"),
    ]
    extra = os.environ.get("LLAMA_SERVER_EXTRA_ARGS", "").strip()
    if extra:
        command = command + shlex.split(extra)
    with tempfile.NamedTemporaryFile(prefix="replay-llama-", suffix=".log", delete=False) as log:
        log_path = log.name
    log_handle = open(log_path, "wb")
    proc = subprocess.Popen(
        command, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True
    )
    try:
        deadline = time.monotonic() + args.timeout
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited {proc.returncode}; log={log_path}")
            try:
                if _server_json(origin, "/health", timeout=5).get("status") == "ok":
                    break
            except (OSError, ValueError, urllib.error.HTTPError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(f"llama-server not ready after {args.timeout:.0f}s")
            time.sleep(1)
        tokens = _server_json(origin, "/tokenize", {"content": prompt_text_of(manifest)})
        prompt_ids_match = tokens_list(tokens) == manifest["prompt_ids"]
        pinned = manifest["continuation_ids"]
        warmup = int(manifest.get("warmup_tokens", 0))
        measured = int(manifest["measured_tokens"])
        steps_ms: list[float] = []
        prompt_costs: list[float] = []
        decode_costs: list[float] = []
        base = manifest["prompt_ids"]
        for _ in range(warmup):
            _server_json(
                origin,
                "/completion",
                {
                    "prompt": base,
                    "n_predict": 1,
                    "cache_prompt": True,
                    "temperature": 0.0,
                    "top_k": 1,
                    "seed": 0,
                },
                timeout=600,
            )
        for step in range(measured):
            count = step + 1
            body = {
                "prompt": base + pinned[:count],
                "n_predict": 1,
                "cache_prompt": True,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "seed": 0,
            }
            response = _server_json(origin, "/completion", body, timeout=600)
            timings = response.get("timings") or {}
            prompt_cost = float(timings.get("prompt_ms") or 0.0)
            decode_cost = float(timings.get("predicted_ms") or 0.0)
            prompt_costs.append(prompt_cost)
            decode_costs.append(decode_cost)
            # The forced token is appended to the cached prompt. llama-server
            # therefore accounts its forward pass as one prompt-eval token;
            # predicted_ms covers only an additional sampled token and is near
            # zero for this request shape.
            steps_ms.append(prompt_cost)
    finally:
        _stop_llama_server(proc)
        log_handle.close()
    timing_reasons = []
    if not prompt_ids_match:
        timing_reasons.append("llama-server tokenizer did not reproduce pinned prompt IDs")
    if len(steps_ms) != measured:
        timing_reasons.append(
            f"forced-token timing count={len(steps_ms)} != measured={measured}"
        )
    if any(value <= 0 for value in steps_ms):
        timing_reasons.append("llama-server returned missing/non-positive forced-token timing")
    return {
        "schema": "freetoken-replay-v1",
        "lane": LANE,
        "runtime": "llama-cpp-hip",
        "backend": getattr(args, "backend", "hip"),
        "repeat": repeat,
        "status": "accepted" if not timing_reasons else "rejected",
        "acceptance": {
            "accepted": not timing_reasons,
            "reasons": timing_reasons,
            "prompt_ids_match": prompt_ids_match,
            "decode_timing_complete": len(steps_ms) == measured,
        },
        "execution": "kernel",
        "model_sha256": manifest["model_sha256"],
        "fixture_sha256": manifest["fixture_sha256"],
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "manifest_ids_sha256": manifest["golden"]["ids_sha256"],
        "forced": True,
        "ids_match": prompt_ids_match,
        "prompt_ids_match": prompt_ids_match,
        "steps": summarize_steps(steps_ms, 0),
        "prompt_ms_per_token_median": (
            statistics.median(prompt_costs) if prompt_costs else None
        ),
        "decode_ms_per_token_median": (
            statistics.median(steps_ms) if steps_ms else None
        ),
        "route_digest": None,
        "route_hash_status": "unavailable_without_instrumentation",
        "timing_domain": "llama_server_prompt_eval_ms_for_forced_token",
        "mtp": "off",
        "speculative": False,
        "decode_batch_size": 1,
        "context": getattr(args, "context", 9216),
        "batch": getattr(args, "batch", 512),
        "ubatch": getattr(args, "ubatch", 512),
        "kv_type": getattr(args, "kv_type", "q8_0"),
        "server_log": log_path,
    }


def prompt_text_of(manifest: dict) -> str:
    """Return raw prompt text matching pinned FreeToken prompt IDs.

    The golden manifest stores raw prompt text and its hash. This avoids trying to
    load a Transformers tokenizer from a GGUF file, which is not a valid tokenizer
    source. Legacy manifests may provide ``tokenizer_path`` explicitly."""
    prompt = manifest.get("prompt_text")
    if isinstance(prompt, str):
        expected = manifest.get("prompt_text_sha256")
        if expected and _sha256_bytes(prompt.encode()) != expected:
            raise ValueError("prompt_text does not match prompt_text_sha256")
        return prompt
    tokenizer_path = manifest.get("tokenizer_path")
    if not isinstance(tokenizer_path, str):
        raise ValueError("manifest lacks prompt_text; regenerate golden manifest with current tool")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return tokenizer.decode(manifest["prompt_ids"])


def tokens_list(tokens) -> list[int] | None:
    if isinstance(tokens, dict):
        tokens = tokens.get("tokens")
    if not isinstance(tokens, list):
        return None
    try:
        return [int(v) for v in tokens]
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", required=True)
        p.add_argument("--aime", default=os.environ.get("FREETOKEN_AIME25_JSONL"))
        p.add_argument("--aime-revision", default=os.environ.get("FREETOKEN_AIME25_REVISION"))
        p.add_argument("--aime-sha256", default=os.environ.get("FREETOKEN_AIME25_SHA256"))
        p.add_argument("--problem", type=int, default=0)
        p.add_argument("--context", type=int, default=9216)
        p.add_argument("--batch", type=int, default=512)
        p.add_argument("--ubatch", type=int, default=512)
        p.add_argument("--attention-backend", default="triton")
        p.add_argument("--moe-backend", default="offload")
        p.add_argument("--memory-ratio", type=float, default=0.9)
        p.add_argument("--cache", type=int, default=0)
        p.add_argument("--graph", action="store_true", help="capture/replay graphs")
        p.add_argument(
            "--kv-type", default="q8_0",
            help="llama.cpp cache type flags for the replay server (llama side only)",
        )

    golden = sub.add_parser("golden", help="pin legacy greedy golden IDs/hash")
    common(golden)
    golden.add_argument("--decode", type=int, default=512)
    golden.add_argument(
        "--warmup", type=int, default=8, help="untimed forced steps before the measured window"
    )
    golden.add_argument("--route-top-k", type=int, default=8)
    golden.add_argument("--out", required=True, help="manifest JSON to create")

    replay = sub.add_parser("replay-freetoken", help="timed teacher-forced replay")
    common(replay)
    replay.add_argument("--manifest", required=True)
    replay.add_argument("--routes", action="store_true", help="add the untimed route pass")
    replay.add_argument("--repeats", type=int, default=10)
    replay.add_argument("--json", dest="json_out", default=None)

    rllama = sub.add_parser("replay-llama", help="timed replay against llama-server")
    common(rllama)
    rllama.add_argument("--manifest", required=True)
    rllama.add_argument("--server", required=True, help="llama-server HIP binary")
    rllama.add_argument("--repeats", type=int, default=1, help="server restarts (each reruns all steps)")
    rllama.add_argument("--timeout", type=float, default=1800)
    rllama.add_argument("--json", dest="json_out", default=None)
    return parser.parse_args(argv)


def _write_rows(rows: list[dict], json_out: str | None) -> None:
    if not json_out:
        return
    with open(json_out, "a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "golden":
        manifest = build_manifest(args)
        Path(args.out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        golden = manifest["golden"]
        print(
            json.dumps(
                {
                    "ids_sha256": golden["ids_sha256"],
                    "answer": golden["answer"],
                    "answer_match": golden["answer_match"],
                    "finite_logits": golden["finite_logits"],
                    "logit_sha256": golden["logit_sha256"],
                },
                indent=2,
            )
        )
        return 0
    if args.command == "replay-freetoken":
        manifest = load_manifest(args.manifest)
        rows = []
        for repeat in range(args.repeats):
            args.repeat = repeat
            row = replay_freetoken(args, manifest, capture_routes=args.routes)
            row["repeat"] = repeat
            rows.append(row)
            stats = row["steps"]
            median = stats.get("ms_per_token_median") if stats else None
            median_text = "n/a" if median is None else f"{median:.3f} ms/token"
            print(
                f"repeat {repeat + 1}/{args.repeats}: {median_text}, "
                f"ids_match={row['ids_match']}",
                flush=True,
            )
        _write_rows(rows, args.json_out)
        return 0 if rows and all(row["ids_match"] for row in rows) else 1
    if args.command == "replay-llama":
        manifest = load_manifest(args.manifest)
        rows = [replay_llama(args, manifest, repeat) for repeat in range(args.repeats)]
        _write_rows(rows, args.json_out)
        for row in rows:
            median = (row["steps"] or {}).get("ms_per_token_median")
            print(
                f"{row['runtime']}: {median if median is None else f'{median:.3f} ms/token'} "
                f"prompt_ids_match={row['prompt_ids_match']}",
                flush=True,
            )
        return 0 if rows and all(row["status"] == "accepted" for row in rows) else 1
    raise RuntimeError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
