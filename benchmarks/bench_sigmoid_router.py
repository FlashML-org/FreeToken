"""Checkpoint-free eager/fused router A/B, runnable with only torch and triton.

Run: python benchmarks/bench_sigmoid_router.py
Loads the kernel file directly to avoid initializing the serving engine.
Times only routing after logits; not GEMV, weight casts or end-to-end generation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from functools import partial
from pathlib import Path

import torch


def reference(logits, bias, k, groups, keep, eps, scale):
    scores = logits.float().sigmoid()
    choice = scores + bias.float()
    if groups > 1:
        grouped = choice.reshape(len(logits), groups, logits.shape[1] // groups)
        group_scores = grouped.topk(2, dim=-1).values.sum(-1)
        group_ids = group_scores.topk(keep, dim=-1, sorted=False).indices
        mask = torch.zeros_like(group_scores).scatter_(1, group_ids, 1.0)
        choice = grouped.masked_fill(~mask.bool()[..., None], -float("inf")).reshape_as(choice)
    ids = choice.topk(k, dim=-1).indices
    w = scores.gather(1, ids)
    denom = w.sum(-1, keepdim=True)
    if eps:
        denom = denom + eps
    w = w / denom
    if scale != 1.0:
        w = w * scale
    return w.float().contiguous(), ids.to(torch.int32).contiguous()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 256])
    parser.add_argument("--output", type=Path, help="Save settings and all timing samples as JSON")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if any(n < 1 for n in args.tokens):
        parser.error("--tokens must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("A working CUDA PyTorch GPU environment is required; no benchmark was run.")
    import triton
    from triton.testing import do_bench_cudagraph

    path = Path(__file__).resolve().parents[1] / "python/freetoken/kernel/triton/sigmoid_router.py"
    spec = importlib.util.spec_from_file_location("sigmoid_router_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    fused = module.fused_topk_sigmoid
    print(f"GPU={torch.cuda.get_device_name()} torch={torch.__version__} CUDA={torch.version.cuda} triton={triton.__version__}")
    print("Synthetic post-logit routing, CUDA Graph median; no model checkpoint loaded.")
    print("tokens experts topk groups keep eager_us fused_us speedup")
    torch.manual_seed(123)
    report = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                  "cuda": torch.version.cuda, "triton": triton.__version__, "seed": 123,
                  "scope": "post-logit routing only, CUDA Graph", "repeats": args.repeats, "results": []}
    for tokens in args.tokens:
        for experts, k, groups, keep, eps, scale in (
            (128, 4, 1, 1, 0.0, 1.0),
            (128, 4, 1, 1, 1e-20, 2.0),
            (192, 8, 8, 4, 1e-20, 2.5),
            (288, 8, 1, 1, 1e-20, 2.5),
        ):
            x = torch.randn(tokens, experts, device="cuda")
            b = torch.randn(experts, device="cuda") * 0.1
            old = partial(reference, x, b, k, groups, keep, eps, scale)
            new = partial(fused, x, b, k, renormalize=True, n_group=groups,
                          topk_group=keep, eps=eps, routed_scale=scale)
            wr, ir = old()
            w, ids = new()
            torch.testing.assert_close(ids, ir, rtol=0, atol=0)
            torch.testing.assert_close(w, wr, rtol=1e-5, atol=1e-6)
            samples = [[], []]
            for repeat in range(args.repeats):
                for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                    samples[index].append(do_bench_cudagraph((old, new)[index], rep=100) * 1000)
            before, after = map(statistics.median, samples)
            print(f"{tokens:6d} {experts:7d} {k:4d} {groups:6d} {keep:4d} {before:8.3f} {after:8.3f} {before/after:7.3f}x")
            report["results"].append({"tokens": tokens, "experts": experts, "topk": k,
                "groups": groups, "keep": keep, "eps": eps, "scale": scale,
                "eager_samples_us": samples[0], "fused_samples_us": samples[1],
                "eager_median_us": before, "fused_median_us": after, "speedup": before/after})
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
