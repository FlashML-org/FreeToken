"""A/B microbenchmark for the shared block-scaled fp8 linear (``kernel/triton/dsv4/fp8_linear``).

The kernel serves two checkpoint dialects -- DeepSeek-V4's 128x128 blocks and DeepSeek-V4.1's
32x32 -- from one source, so a change for one must not tax the other. This script times the
current kernel against a baseline copy of the file (default: the same file at ``flashml/main``)
on the DeepSeek projection shapes under CUDA-graph replay, and checks the outputs stay bit-identical.

    python benchmarks/bench_fp8_block_linear.py                 # baseline = git show flashml/main:<file>
    python benchmarks/bench_fp8_block_linear.py --base <path>   # baseline = a saved copy of fp8_linear.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import time

import torch

FILE = "python/freetoken/kernel/triton/dsv4/fp8_linear.py"
# (M, N, K, block): DeepSeek-V4 wq_b / wo_b / shared w1 at prefill and decode, DeepSeek-V4.1 wq_b / wo_b
SHAPES = [
    (128, 32768, 1024, 128), (128, 4096, 8192, 128), (128, 2048, 4096, 128), (1, 4096, 8192, 128),
    (128, 32768, 1280, 32), (128, 5120, 8192, 32), (1, 5120, 8192, 32),
]


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bench(fn, iters: int = 200) -> float:
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(20):
                fn()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters // 20):
        graph.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", help="baseline fp8_linear.py (default: flashml/main's copy)")
    parser.add_argument("--rev", default="flashml/main", help="git revision for the baseline copy")
    args = parser.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.base is None:
        blob = subprocess.check_output(["git", "-C", root, "show", f"{args.rev}:{FILE}"])
        tmp = tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False)
        tmp.write(blob)
        tmp.close()
        args.base = tmp.name
    base = _load(args.base, "fp8_linear_base")
    new = _load(os.path.join(root, FILE), "fp8_linear_new")
    base_takes_block = "block" in base.block_fp8_linear.__code__.co_varnames

    torch.manual_seed(0)
    print(f"{'M x N x K':>18} {'blk':>4} {'base us':>9} {'new us':>9} {'change':>8}  identical")
    for M, N, K, block in SHAPES:
        if block != 128 and not base_takes_block:
            continue  # the baseline predates the block parameter
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = (torch.randn(N, K, device="cuda") * 0.05).to(torch.float8_e4m3fn)
        scale = torch.randint(120, 130, (N // block, K // block), device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)
        run_base = (lambda: base.block_fp8_linear(x, w, scale, block=block)) if base_takes_block else (lambda: base.block_fp8_linear(x, w, scale))
        run_new = lambda: new.block_fp8_linear(x, w, scale, block=block)  # noqa: E731
        same = torch.equal(run_base(), run_new())
        tb, tn = _bench(run_base), _bench(run_new)
        print(f"{M:>5} x {N:>5} x {K:>4} {block:>4} {tb:9.2f} {tn:9.2f} {(tn / tb - 1) * 100:+7.1f}%  {same}")


if __name__ == "__main__":
    sys.exit(main())
