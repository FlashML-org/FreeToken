"""Finite-logit and eager/graph parity probe for real Qwen MoE checkpoints.

This is intentionally separate from the throughput benchmark. It drives FreeToken's
offline scheduler, captures logits immediately before sampling, and then checks the
greedy token IDs. HTTP readiness or generated text alone cannot detect NaN logits or
captured-graph corruption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from bench_decode_moe import load_problem_details, model_fingerprint, runtime_metadata


@dataclass
class ProbeResult:
    requested: str
    actual: str
    token_ids: list[int]
    logits: list[torch.Tensor]

    def artifact(self) -> dict:
        rows = torch.cat(self.logits, dim=0) if self.logits else torch.empty(0)
        digest = hashlib.sha256(rows.numpy().tobytes()).hexdigest()[:16]
        finite = bool(torch.isfinite(rows).all())
        return {
            "requested": self.requested,
            "actual": self.actual,
            "decode_rows": len(self.logits),
            "logit_shape": list(rows.shape),
            "finite_logits": finite,
            "logit_sha256": digest,
            "token_ids": self.token_ids,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--aime", default=os.environ.get("FREETOKEN_AIME25_JSONL"))
    p.add_argument("--aime-revision", default=os.environ.get("FREETOKEN_AIME25_REVISION"))
    p.add_argument("--aime-sha256", default=os.environ.get("FREETOKEN_AIME25_SHA256"))
    p.add_argument("--problem", type=int, default=0)
    p.add_argument("--decode", type=int, default=8)
    p.add_argument("--memory-ratio", type=float, default=0.9)
    p.add_argument("--cache", type=int, default=0)
    p.add_argument("--json", dest="json_out", default=None)
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_mode", choices=("eager", "graph"), help=argparse.SUPPRESS)
    p.add_argument("--_graph-gate", default="unknown", help=argparse.SUPPRESS)
    p.add_argument("--_prompt-file", help=argparse.SUPPRESS)
    p.add_argument("--_artifact-file", help=argparse.SUPPRESS)
    p.add_argument("--_logits-file", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def run_probe(
    args: argparse.Namespace,
    prompt: str,
    requested: str,
    graph_gate: str,
) -> ProbeResult:
    from freetoken.core import SamplingParams
    from freetoken.llm import LLM

    graph_requested = requested == "graph"
    kwargs = {
        "attention_backend": "triton",
        "max_running_req": 1,
        "max_extend_tokens": max(8192, args.decode + 128),
        "max_seq_len_override": 8192 + args.decode,
        "moe_backend": "offload",
        "memory_ratio": args.memory_ratio,
        "moe_cache_auto": args.cache == 0,
        "cuda_graph_max_bs": 1 if graph_requested else 0,
    }
    if args.cache > 0:
        kwargs["moe_cache_size"] = args.cache
    llm = LLM(args.model, dtype=torch.bfloat16, **kwargs)
    captured: list[torch.Tensor] = []
    original_sample = llm.engine.sampler.sample

    def capture_sample(logits, sample_args, batch):
        if batch.is_decode:
            captured.append(logits.detach().float().cpu())
        return original_sample(logits, sample_args, batch)

    llm.engine.sampler.sample = capture_sample
    try:
        actual = "replay" if llm.engine.graph_runner.graph_map else "eager"
        if graph_requested and graph_gate == "pass" and actual != "replay":
            raise RuntimeError("graph gate passed but no graph was captured")
        output = llm.generate(
            [prompt],
            SamplingParams(
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                # Match the serving path: current full-prefix scheduling consumes one
                # engine output budget slot before the first returned completion token.
                max_tokens=args.decode + 1,
                ignore_eos=True,
            ),
        )[0]
        token_ids = list(output["token_ids"])
    finally:
        llm.shutdown()
    if len(token_ids) != args.decode:
        raise RuntimeError(
            f"{requested} probe generated {len(token_ids)} tokens, expected {args.decode}"
        )
    result = ProbeResult(requested, actual, token_ids, captured)
    artifact = result.artifact()
    if not artifact["finite_logits"]:
        raise RuntimeError(f"{requested} probe found non-finite logits: {artifact}")
    if artifact["decode_rows"] != args.decode:
        raise RuntimeError(
            f"{requested} probe captured {artifact['decode_rows']} decode logit rows, "
            f"expected {args.decode}"
        )
    return result


def run_probe_worker(args: argparse.Namespace) -> int:
    """Run one LLM lifetime in a pristine child process.

    Engine construction intentionally rejects an already initialized CUDA runtime, so eager
    and graph probes cannot share this interpreter. Keep this boundary explicit rather than
    weakening that runtime invariant.
    """
    prompt = Path(args._prompt_file).read_text()
    result = run_probe(args, prompt, args._mode, args._graph_gate)
    artifact = result.artifact()
    torch.save(torch.cat(result.logits, dim=0), args._logits_file)
    Path(args._artifact_file).write_text(json.dumps(artifact, sort_keys=True))
    return 0


def run_probe_child(
    args: argparse.Namespace,
    prompt: str,
    requested: str,
    graph_gate: str,
    workdir: str,
) -> ProbeResult:
    prompt_file = Path(workdir) / f"{requested}-prompt.txt"
    artifact_file = Path(workdir) / f"{requested}-artifact.json"
    logits_file = Path(workdir) / f"{requested}-logits.pt"
    prompt_file.write_text(prompt)
    cmd = [
        sys.executable,
        __file__,
        "--_worker",
        "--model", args.model,
        "--problem", str(args.problem),
        "--decode", str(args.decode),
        "--memory-ratio", str(args.memory_ratio),
        "--cache", str(args.cache),
        "--_mode", requested,
        "--_graph-gate", graph_gate,
        "--_prompt-file", str(prompt_file),
        "--_artifact-file", str(artifact_file),
        "--_logits-file", str(logits_file),
    ]
    if args.aime:
        cmd += ["--aime", args.aime]
    if args.aime_revision:
        cmd += ["--aime-revision", args.aime_revision]
    if args.aime_sha256:
        cmd += ["--aime-sha256", args.aime_sha256]
    child = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if child.returncode != 0:
        tail = "\n".join((child.stdout + child.stderr).splitlines()[-30:])
        raise RuntimeError(f"{requested} probe worker failed (rc={child.returncode}):\n{tail}")
    if not artifact_file.is_file() or not logits_file.is_file():
        raise RuntimeError(f"{requested} probe worker produced no artifacts")
    artifact = json.loads(artifact_file.read_text())
    rows = torch.load(logits_file, map_location="cpu", weights_only=True)
    if rows.ndim != 2 or rows.shape[0] != args.decode:
        raise RuntimeError(f"{requested} probe worker returned bad logits shape {tuple(rows.shape)}")
    return ProbeResult(
        requested=artifact["requested"],
        actual=artifact["actual"],
        token_ids=artifact["token_ids"],
        logits=list(rows.split(1, dim=0)),
    )


def compare(eager: ProbeResult, graph: ProbeResult) -> dict:
    if eager.token_ids != graph.token_ids:
        raise RuntimeError("eager/graph greedy token IDs differ")
    if len(eager.logits) != len(graph.logits):
        raise RuntimeError("eager/graph decode logit row counts differ")
    max_abs = 0.0
    max_rel = 0.0
    for left, right in zip(eager.logits, graph.logits):
        if left.shape != right.shape:
            raise RuntimeError(f"eager/graph logit shapes differ: {left.shape} vs {right.shape}")
        delta = (left - right).abs()
        max_abs = max(max_abs, float(delta.max()))
        max_rel = max(max_rel, float((delta / right.abs().clamp_min(1e-6)).max()))
        torch.testing.assert_close(left, right, rtol=2e-2, atol=2e-2)
    return {"token_ids_equal": True, "max_abs": max_abs, "max_rel": max_rel}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args._worker:
        if not all((args._mode, args._prompt_file, args._artifact_file, args._logits_file)):
            raise SystemExit("probe worker arguments incomplete")
        return run_probe_worker(args)
    if args.decode < 1:
        raise SystemExit("--decode must be >= 1")
    prompt, answer, dataset = load_problem_details(
        args.aime, args.problem, args.aime_revision, args.aime_sha256
    )
    from freetoken.utils.graph_gate import graph_capture_status

    gate = graph_capture_status()
    with tempfile.TemporaryDirectory(prefix="freetoken-qwen-probe-") as workdir:
        eager = run_probe_child(args, prompt, "eager", gate, workdir)
        graph = None
        comparison = {"status": "skipped", "reason": f"graph gate={gate}"}
        if gate == "pass":
            graph = run_probe_child(args, prompt, "graph", gate, workdir)
            comparison = compare(eager, graph)
    artifact = {
        "schema": "qwen-moe-base-probe-v2",
        "model": model_fingerprint(args.model),
        "dataset": dataset,
        "problem": args.problem,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "answer": answer,
        "decode": args.decode,
        "mtp": "off",
        "speculative": False,
        "graph_gate": gate,
        "eager": eager.artifact(),
        "graph": graph.artifact() if graph else None,
        "comparison": comparison,
        "runtime": runtime_metadata(),
    }
    print(json.dumps(artifact, indent=2, sort_keys=True))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(artifact, f, indent=2, sort_keys=True)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
