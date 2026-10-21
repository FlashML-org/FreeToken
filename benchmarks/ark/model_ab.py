"""Fresh-process off/on Qwen3 A/B; instrumentation runs separately via --probe."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time

from . import http_client as client


@contextmanager
def service(args, directory, enabled):
    directory.mkdir(parents=True, exist_ok=False)
    if int((client.DEVICE / "mem_info_vram_used").read_text()) > 256 * 2**20:
        raise RuntimeError("Target GPU occupied; no service launched")
    module = "benchmarks.ark.probe" if args.probe else "freetoken.cli"
    argv = [
        "numactl",
        "--interleave=" + args.memory_nodes,
        sys.executable,
        "-m",
        module,
        "serve",
        "--model-path",
        args.model_path,
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "triton",
        "--moe-strategy",
        "offload",
        "--moe-cpu-threads",
        "17",
        "--moe-cache-mem",
        "20GiB",
        "--moe-cache-scope",
        "unified",
        "--moe-cache-split",
        "equal",
        "--num-tokens",
        "33058",
        "--page-size",
        "1",
        "--max-seq-len-override",
        "33024",
        "--max-prefill-length",
        "8192",
        "--max-running-requests",
        "1",
        "--cuda-graph-max-bs",
        str(args.graph),
        "--max-output-tokens",
        "256",
        "--memory-ratio",
        "0.9",
        "--expert-load",
        "serial",
        "--cache-type",
        "naive",
        "--enable-cache-report",
        "--sampling-defaults",
        "none",
        "--reasoning-parser",
        "off",
        "--host",
        "127.0.0.1",
        "--port",
        "19673",
    ]
    env = dict(
        os.environ,
        FREETOKEN_USE_ARK=str(enabled),
        FREETOKEN_ARK_POLICY="validated",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    for key in ("PYTHONPATH", "FREETOKEN_ARK_TRACE", "FREETOKEN_ARK_PROFILE"):
        env.pop(key, None)
    if args.profile:
        env["FREETOKEN_ARK_PROFILE"] = str(args.profile.resolve())
    if args.probe:
        env["FREETOKEN_ARK_TRACE"] = str(directory / "dispatch-{pid}.json")
        env["ARK_MODEL_PROBE"] = str(directory)
    client.save(directory / "command.json", dict(argv=argv, env={k: v for k, v in env.items() if k.startswith(("FREETOKEN_", "ROCR_"))}))
    stop = threading.Event()
    with (directory / "server.log").open("w") as out:
        proc = subprocess.Popen(argv, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        monitor = threading.Thread(target=client.monitor, args=(proc.pid, stop, directory / "resources.jsonl"), daemon=True)
        monitor.start()
        start = time.monotonic()
        try:
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited {proc.returncode}: {directory}")
                if time.monotonic() - start > 600:
                    raise TimeoutError("startup timeout")
                try:
                    health = client.get("/health")
                except OSError:
                    time.sleep(1)
                    continue
                if health.get("status") == "error":
                    raise RuntimeError(health)
                if health.get("status") == "ok":
                    break
                time.sleep(1)
            print("READY", directory.name, flush=True)
            yield
            client.save(directory / "complete.json", dict(success=True, stats=client.get("/v1/stats")))
        finally:
            owned = client.tree(proc.pid)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(10)
            for pid in owned - {proc.pid}:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            stop.set()
            monitor.join(5)
            deadline = time.monotonic() + 180
            while int((client.DEVICE / "mem_info_vram_used").read_text()) > 256 * 2**20:
                if time.monotonic() > deadline:
                    raise RuntimeError("GPU did not return to idle")
                time.sleep(0.5)
            client.save(
                directory / "shutdown.json", dict(returncode=proc.returncode, vram=int((client.DEVICE / "mem_info_vram_used").read_text()))
            )


def make_prompts(root, model):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    head = "<|im_start|>user\nRead the following archive.\n"
    tail = (
        "\nExplain how inspectors compare transport, energy and weather records. Give a detailed answer.<|im_end|>\n<|im_start|>assistant\n"
    )
    body = (
        "The archive tracks transport, energy, weather and equipment inspections. Inspectors compare observations from different stations.\n"
        * 2000
    )
    head_ids, tail_ids = (tok.encode(x, add_special_tokens=False) for x in (head, tail))
    body_ids = tok.encode(body, add_special_tokens=False)
    for n in (512, 32768):
        path = root / f"prompt-{n}.txt"
        if not path.exists():
            budget = n - len(head_ids) - len(tail_ids)
            for _ in range(32):
                text = tok.decode(head_ids + body_ids[:budget] + tail_ids)
                count = len(tok.encode(text, add_special_tokens=False))
                if count == n:
                    break
                budget += n - count
            else:
                raise ValueError("prompt token roundtrip failed")
            path.write_text(text)
        assert len(tok.encode(path.read_text(), add_special_tokens=False)) == n
    client.save(root / "prompts.json", {str(n): hashlib.sha256((root / f"prompt-{n}.txt").read_bytes()).hexdigest() for n in (512, 32768)})


def summarize(root):
    samples = []
    for path in sorted(root.glob("round-*/measured-*.json")):
        row = json.loads(path.read_text())
        assert row["success"]
        samples.append({k: v for k, v in row.items() if k not in ("events", "text", "request")})
    groups = []
    for n in (512, 32768):
        for enabled in (0, 1):
            rows = [r for r in samples if r["context"] == n and r["enabled"] == enabled]
            if rows:
                groups.append(
                    dict(
                        context=n,
                        enabled=enabled,
                        n=len(rows),
                        decode_tok_s=statistics.median(r["decode_tok_s"] for r in rows),
                        ttft_s=statistics.median(r["ttft_s"] for r in rows),
                        e2e_s=statistics.median(r["e2e_s"] for r in rows),
                    )
                )
    client.save(root / "summary.json", dict(samples=samples, groups=groups))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu-pci", required=True)
    p.add_argument("--memory-nodes", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--start-round", type=int, default=1)
    p.add_argument("--graph", type=int, default=0)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--quality-only", action="store_true", help="run corrected short generation checks without context probes")
    p.add_argument("--profile", type=Path, help="explicit candidate profile for separate experiments")
    a = p.parse_args()
    if a.quality_only and not a.probe:
        p.error("--quality-only requires --probe")
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    client.ROOT = a.output
    client.MODEL = a.model_path
    client.DEVICE = Path("/sys/bus/pci/devices") / a.gpu_pci
    make_prompts(a.output, a.model_path)
    for r in range(a.start_round, a.start_round + a.rounds):
        for enabled in (0, 1) if r % 2 else (1, 0):
            directory = a.output / f"round-{r}-ark{enabled}"
            with service(a, directory, enabled):
                for n in () if a.quality_only else ((512, 32768) if r % 2 else (32768, 512)):
                    if not a.probe:
                        client.request(directory, n, "warmup")
                    row = client.request(directory, n, "probe" if a.probe else "measured", output_tokens=8 if a.probe else 256)
                    row.update(round=r, enabled=enabled)
                    client.save(directory / f"{row['phase']}-{n}.json", row)
                    if a.probe:
                        counts = [row for path in directory.glob("dispatch-*.json") for row in json.loads(path.read_text())["counts"]]
                        client.save(directory / f"hits-{n}.json", counts)
                    if not a.probe:
                        summarize(a.output)
                    print("RESULT", directory.name, n, row["decode_tok_s"], row["ttft_s"], flush=True)
                if a.probe:
                    from transformers import AutoTokenizer
                    from .quality_prompts import CASES, first_answer

                    tokenizer = AutoTokenizer.from_pretrained(a.model_path)
                    quality = []
                    for name, question, predicate in CASES:
                        prompt = tokenizer.apply_chat_template(
                            [{"role": "user", "content": question}],
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                        n = len(tokenizer.encode(prompt, add_special_tokens=False))
                        result = client.request(directory, n, "quality-" + name, output_tokens=128, prompt=prompt)
                        answer = first_answer(result["text"])
                        quality.append(dict(name=name, answer=answer, passed=bool(predicate(answer))))
                        client.save(directory / "quality.json", quality)


if __name__ == "__main__":
    main()
