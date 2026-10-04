"""Decode selection pipeline of one DSV41 full-width index layer under CUDA-graph replay.

Holds the live history fixed and varies the staged capacity (``max_seq_len / ratio``), so the
numbers show whether the pipeline's cost follows the live history or the capacity. Two pipelines:

* ``torch``  -- the max-width baseline: scores over the whole staged width (dead tiles ``-inf``),
  ``torch.topk`` over it, block maxima + ``topk`` over every block, expansion.
* ``kernels`` -- ``kernel/triton/dsv41``: the worker-grid scorer over the live tiles, the level-wise
  exact top-k and the block-candidate kernels (``dsv41_topk`` / ``dsv41_candidate_blocks``).

Selected score multisets are checked per request; exact ties may select different positions.
V4.1 shape: B=1, 32 index heads, D=128, token
top-k 512, 2048 candidate blocks of 8, ratio 1 (the decoder's kv source) unless overridden.

    python benchmarks/bench_dsv41_selection.py [--live 4096] [--capacities 4096,65536,1048576] [--bs 1]
"""

from __future__ import annotations

import argparse
import time

import torch

from freetoken.kernel.triton.dsv41.indexer import indexer_logits_packed
from freetoken.kernel.triton.dsv41.pack import pack_rows
from freetoken.kernel.triton.dsv41.topk import dsv41_candidate_blocks, dsv41_topk
from freetoken.kvcache.dsv4.v41_row_format import FP4_E8M0_B32

H, D, TOPK, KB, BS = 32, 128, 512, 2048, 8


def torch_pipeline(q, w, pool, locs, ratio, live, T, candidate_source: bool):
    """The previous max-width path (scores fully materialized, torch.topk over the staged width)."""
    B = q.shape[0]
    scores = torch.full((B, 1, T), float("-inf"), device=q.device)
    indexer_logits_packed(q, w, pool, FP4_E8M0_B32, locs, ratio, live, T=T, out=scores)
    picks = scores.topk(min(TOPK, T), dim=-1, sorted=False).indices
    picks = torch.where(picks < live.unsqueeze(-1), picks, torch.full_like(picks, -1))
    big = torch.iinfo(picks.dtype).max
    srt = torch.where(picks < 0, torch.full_like(picks, big), picks).sort(dim=-1).values
    picks = torch.where(srt == big, torch.full_like(srt, -1), srt)
    cand = None
    if candidate_source:
        pad = -T % BS
        blk = torch.nn.functional.pad(scores, (0, pad), value=float("-inf")).view(B, 1, -1, BS).amax(dim=-1)
        nb = blk.shape[-1]
        last = torch.div(live - 1, BS, rounding_mode="floor").unsqueeze(-1)
        blk = blk.masked_fill(torch.arange(nb, device=q.device) == last, float("inf"))
        top = blk.topk(min(KB, nb), dim=-1)
        keep = top.indices.masked_fill(torch.isneginf(top.values), -1).sort(dim=-1).values
        pos = keep.unsqueeze(-1) * BS + torch.arange(BS, device=q.device)
        pos = torch.where((keep < 0).unsqueeze(-1) | (pos >= live.unsqueeze(-1).unsqueeze(-1)), torch.full_like(pos, -1), pos)
        cand = pos.flatten(-2)
    return picks, cand


def kernel_pipeline(q, w, pool, locs, ratio, live, T, candidate_source: bool, scores):
    B = q.shape[0]
    indexer_logits_packed(q, w, pool, FP4_E8M0_B32, locs, ratio, live, T=T, out=scores)
    flat = scores.view(B, T)
    picks = dsv41_topk(flat, live.view(B), TOPK).view(B, 1, TOPK)
    cand = dsv41_candidate_blocks(flat, live.view(B), KB, BS).view(B, 1, -1) if candidate_source else None
    return picks, cand


def _assert_selected_scores(scores, left, right):
    width = max(left.shape[-1], right.shape[-1])

    def selected(indices):
        values = scores.gather(-1, indices.clamp_min(0).long()).masked_fill(indices < 0, -float("inf"))
        values = torch.nn.functional.pad(values, (0, width - values.shape[-1]), value=-float("inf"))
        return values.sort(dim=-1).values

    torch.testing.assert_close(selected(left), selected(right), atol=0, rtol=0)


def _graph_time(fn, iters=50) -> float:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fn()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        graph.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", type=int, default=4096)
    ap.add_argument("--capacities", default="4096,65536,1048576")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--ratio", type=int, default=1)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    B = args.bs
    caps = [int(c) for c in args.capacities.split(",")]
    rows = max(caps) + 8
    pool = pack_rows(torch.randn(rows, D, device=dev, dtype=torch.bfloat16), FP4_E8M0_B32)
    q = torch.randn(B, 1, H, D, device=dev, dtype=torch.bfloat16)
    w = torch.rand(B, 1, H, device=dev)
    live = torch.full((B, 1), args.live, dtype=torch.int32, device=dev)
    print(f"live={args.live} B={B} ratio={args.ratio}  (us per full-width layer: scoring + top-k [+ candidate blocks])")
    print(f"{'capacity':>10} {'torch score+topk':>17} {'kernels score+topk':>19} {'torch +cand':>12} {'kernels +cand':>14}")
    for T in caps:
        locs = torch.arange(T * args.ratio, device=dev, dtype=torch.int64).view(1, -1).expand(B, -1).contiguous()
        scores = torch.empty((B, 1, T), device=dev)
        res = []
        for candidate_source in (False, True):
            p_t, c_t = torch_pipeline(q, w, pool, locs, args.ratio, live, T, candidate_source)
            p_k, c_k = kernel_pipeline(q, w, pool, locs, args.ratio, live, T, candidate_source, scores)
            _assert_selected_scores(scores, p_t, p_k)
            if candidate_source:
                valid_scores = scores.masked_fill(torch.arange(T, device=dev) >= live.unsqueeze(-1), -float("inf"))
                blocks = torch.nn.functional.pad(valid_scores, (0, -T % BS), value=-float("inf")).view(B, 1, -1, BS).amax(-1)
                last = torch.div(live - 1, BS, rounding_mode="floor").unsqueeze(-1)
                blocks = blocks.masked_fill(torch.arange(blocks.shape[-1], device=dev) == last, float("inf"))
                _assert_selected_scores(blocks, c_t[..., ::BS] // BS, c_k[..., ::BS] // BS)
            res.append(_graph_time(lambda: torch_pipeline(q, w, pool, locs, args.ratio, live, T, candidate_source)))
            res.append(_graph_time(lambda: kernel_pipeline(q, w, pool, locs, args.ratio, live, T, candidate_source, scores)))
        print(f"{T:>10} {res[0]:>17.1f} {res[1]:>19.1f} {res[2]:>12.1f} {res[3]:>14.1f}")


if __name__ == "__main__":
    main()
