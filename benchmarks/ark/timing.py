"""Correctness-gated, paired eager and HIP graph operator experiments."""

import statistics
import time
import torch


OPS = ["moe_decode_fp8", "dense_fp8_decode"]


def metrics(actual, reference):
    a, r = actual.float(), reference.float()
    diff = (a - r).abs()
    norm = diff.square().mean().sqrt() / (r.square().mean().sqrt() + 1e-12)
    cos = torch.nn.functional.cosine_similarity(a.flatten(), r.flatten(), dim=0, eps=1e-30).item()
    if r.abs().max().item() == 0:
        cos = 1.0 if a.abs().max().item() == 0 else 0.0
    return {
        "normalized_rmse": norm.item(),
        "cosine": cos,
        "max_abs": diff.max().item(),
        "max_ref": r.abs().max().item(),
        "finite": bool(torch.isfinite(a).all().item()),
    }


def assert_close(m, op):
    ok = m["finite"] and m["cosine"] > 0.999 and m["normalized_rmse"] < 0.03 and m["max_abs"] <= 0.04 * m["max_ref"] + 1e-3
    if not ok:
        raise AssertionError(m)


def timing_pair(base, candidate, reset, restore, iters, repeats=5):
    def invoke(fn):
        if restore:
            reset()
        return fn()

    for f in (base, candidate):
        for _ in range(4):
            invoke(f)
    torch.cuda.synchronize()
    samples = {"baseline": [], "candidate": []}
    for r in range(repeats):
        order = [("baseline", base), ("candidate", candidate)]
        if r % 2:
            order.reverse()
        for name, fn in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iters):
                invoke(fn)
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - start) * 1e6 / iters)
    return summarize(samples)


def summarize(samples):
    out = {k + "_us": statistics.median(v) for k, v in samples.items()}
    out["speedup"] = out["baseline_us"] / out["candidate_us"]
    out["samples_us"] = samples
    return out


def graph_pair(base, candidate, reset, restore, iters, repeats=5):
    graphs = []
    for fn in (base, candidate):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(4):
                if restore:
                    reset()
                fn()
        s.synchronize()
        graph = torch.cuda.CUDAGraph()
        outputs = []
        with torch.cuda.graph(graph, stream=s):
            for _ in range(4):
                if restore:
                    reset()
                out = fn()
            outputs.append(out)
        graphs.append((graph, outputs))
    samples = {"baseline": [], "candidate": []}
    for r in range(repeats):
        order = [0, 1] if r % 2 == 0 else [1, 0]
        for idx in order:
            g = graphs[idx][0]
            g.replay()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                g.replay()
            end.record()
            end.synchronize()
            samples["baseline" if idx == 0 else "candidate"].append(start.elapsed_time(end) * 1000 / (iters * 4))
    return summarize(samples)
