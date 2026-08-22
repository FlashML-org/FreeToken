"""``ft bench decode``: end-to-end decode throughput, and A/B across engine flags.

``ft bench bw`` measures one kernel in isolation over a small synthetic bank. That is
the right shape for calibrating a bandwidth ratio and the wrong shape for answering
"is this configuration faster", because a serving step is the kernel *plus* the PCIe
gather it contends with, the KV traffic, the GPU<->CPU handshake and the scheduler --
on the real model, at the real cache size. The two can disagree completely: on one
2-socket box a change measured +30% on the microbenchmark and -6.7% on tokens/s.

So this loads the actual model and generates.

    ft bench decode --model DIR
    ft bench decode --model DIR --compare moe-backend=hybrid,offload
    ft bench decode --model DIR --compare moe-cache-rate=0.1,0.25,0.5 --cycles 3

Two things it does that a hand-rolled loop usually does not:

* **Decode is isolated from prefill** by timing the same prompt twice, once with
  ``max_tokens=1`` and once with ``max_tokens=n``, and taking ``(n-1)/(t_n - t_1)``.
  Prefill cost cancels, so a long prompt does not quietly flatter the result.
* **Variants alternate**, one full pass per cycle rather than all runs of A then all
  of B. Thermal drift, page-cache state and whatever else the box is doing move over
  minutes; blocked runs attribute that drift to the variant.

Each measurement runs in a fresh subprocess. Expert banks are pinned and cannot be
unregistered, so tearing an engine down in-process does not reliably give the memory
back -- the second variant would be measuring the first one's leftovers.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

DEFAULT_PROMPT = (
    "Write a detailed technical explanation of how a mixture-of-experts transformer "
    "routes tokens to experts, and why that makes memory bandwidth the bottleneck."
)


def _coerce(v: str):
    """CLI strings to the types SchedulerConfig fields expect."""
    low = v.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def measure_decode_tps(model_path: str, engine_kwargs: dict, prompt: str,
                       tokens: int, samples: int) -> dict:
    """Decode tokens/s for one engine configuration, median over ``samples``."""
    import torch

    from freetoken.core import SamplingParams
    from freetoken.llm import LLM

    # `ft serve` defaults the offload-family backends to --moe-cache-auto when no
    # cache-sizing flag is given (prepare_server_args); the offline LLM path does not,
    # so a bare `--compare moe-backend=hybrid,offload` would die on moe_cache_size=0.
    if not any(k in engine_kwargs
               for k in ("moe_cache_size", "moe_cache_rate", "moe_cache_auto")):
        engine_kwargs = {**engine_kwargs, "moe_cache_auto": True}

    llm = LLM(model_path, dtype=torch.bfloat16, **engine_kwargs)
    # `ignore_eos` so every run generates exactly `tokens` -- otherwise an early stop
    # silently shortens the measurement and inflates the rate.
    one = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    many = SamplingParams(max_tokens=tokens, temperature=0.0, ignore_eos=True)

    llm.generate([prompt], SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True))

    rates = []
    for _ in range(samples):
        t0 = time.perf_counter()
        llm.generate([prompt], one)
        t_prefill = time.perf_counter() - t0
        t0 = time.perf_counter()
        llm.generate([prompt], many)
        t_total = time.perf_counter() - t0
        decode_s = t_total - t_prefill
        if decode_s <= 0:
            continue
        rates.append((tokens - 1) / decode_s)
    if not rates:
        raise RuntimeError("no usable timing samples")
    return {
        "tps": statistics.median(rates),
        "samples": [round(r, 2) for r in rates],
        "prefill_s": round(t_prefill, 3),
    }


def _run_worker(argv: list[str]) -> int:
    """Hidden per-measurement subprocess: emit one JSON line on stdout."""
    spec = json.loads(argv[0])
    try:
        out = measure_decode_tps(spec["model"], spec["kwargs"], spec["prompt"],
                                 spec["tokens"], spec["samples"])
    except Exception as e:  # noqa: BLE001 - reported to the parent, not swallowed
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}), flush=True)
        return 1
    print(json.dumps(out), flush=True)
    return 0


def _measure_in_subprocess(spec: dict, quiet: bool) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "freetoken.moe.bench_decode", "--_worker", json.dumps(spec)],
        capture_output=True, text=True, check=False,
        env={**os.environ, **spec.get("env", {})},
    )
    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{")), None)
    if line is None:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
        return {"error": "no result from worker:\n    " + "\n    ".join(tail)}
    out = json.loads(line)
    if "error" in out and not quiet:
        print(f"      worker: {out['error']}", file=sys.stderr)
    return out


def _variants(compare: str | None, extra: list[str]) -> list[tuple[str, dict]]:
    base = {}
    for kv in extra:
        k, _, v = kv.partition("=")
        base[k.strip().lstrip("-").replace("-", "_")] = _coerce(v)
    if not compare:
        return [("baseline", base)]
    flag, _, values = compare.partition("=")
    if not values:
        raise SystemExit("--compare wants FLAG=value1,value2")
    key = flag.strip().lstrip("-").replace("-", "_")
    return [(f"{flag}={v}", {**base, key: _coerce(v)}) for v in values.split(",")]


def main(argv: list[str] | None = None, prog: str = "ft bench decode") -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--_worker":
        return _run_worker(argv[1:])

    p = argparse.ArgumentParser(prog=prog, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="local checkpoint directory")
    p.add_argument("--compare", default=None, metavar="FLAG=A,B",
                   help="engine flag to vary, e.g. 'moe-backend=hybrid,offload'")
    p.add_argument("--set", action="append", default=[], metavar="FLAG=VALUE",
                   help="engine flag held fixed across variants (repeatable)")
    p.add_argument("--cycles", type=int, default=2,
                   help="alternating passes over the variants (default 2)")
    p.add_argument("--samples", type=int, default=3,
                   help="timed generations per load (default 3)")
    p.add_argument("--tokens", type=int, default=128, help="tokens per generation")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("-o", "--out", default=None, help="write results as JSON")
    p.add_argument("-q", "--quiet", action="store_true")
    ns = p.parse_args(argv)

    if not os.path.isdir(ns.model):
        raise SystemExit(f"--model must be a local directory (got {ns.model!r})")

    variants = _variants(ns.compare, ns.set)
    results: dict[str, list[float]] = {name: [] for name, _ in variants}
    print(f"  {len(variants)} variant(s) x {ns.cycles} cycle(s), "
          f"{ns.samples} timed generations of {ns.tokens} tokens each")
    print("  each measurement reloads the model in a fresh process\n")

    for cycle in range(1, ns.cycles + 1):
        for name, kwargs in variants:
            spec = {"model": ns.model, "kwargs": kwargs, "prompt": ns.prompt,
                    "tokens": ns.tokens, "samples": ns.samples}
            t0 = time.perf_counter()
            out = _measure_in_subprocess(spec, ns.quiet)
            dt = time.perf_counter() - t0
            if "error" in out:
                print(f"  cycle {cycle}  {name:<28} FAILED ({dt:.0f}s)")
                continue
            results[name].append(out["tps"])
            print(f"  cycle {cycle}  {name:<28} {out['tps']:7.2f} tok/s  ({dt:.0f}s)")

    print()
    rows = [(n, v) for n, v in results.items() if v]
    if not rows:
        print("  no successful measurements")
        return 1
    best = max(statistics.median(v) for _, v in rows)
    print(f"  {'variant':<28} {'median':>9} {'spread':>17}   vs best")
    for name, vals in rows:
        med = statistics.median(vals)
        spread = f"{min(vals):.1f}-{max(vals):.1f}" if len(vals) > 1 else "-"
        rel = "best" if med == best else f"{(med / best - 1) * 100:+.1f}%"
        print(f"  {name:<28} {med:8.2f}  {spread:>17}   {rel}")
    if any(len(v) < 2 for _, v in rows):
        print("\n  Only one cycle per variant: nothing separates a real difference from "
              "drift.\n  Use --cycles 2 or more before believing a small gap.")

    if ns.out:
        with open(ns.out, "w") as f:
            json.dump({"model": ns.model, "tokens": ns.tokens,
                       "results": {n: v for n, v in results.items()}}, f, indent=2)
            f.write("\n")
        print(f"\n  saved: {ns.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
