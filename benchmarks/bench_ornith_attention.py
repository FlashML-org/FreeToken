"""Ornith Q4_0 attention benchmark: decode/prefill/extend at the exact 16Q/2KV/D256 geometry.

Ornith (Qwen3.5-MoE-family GQA: 16 query heads, 2 KV heads, head_dim 256) is the shape
``decode_launch_config`` in ``kernel/triton/attention.py`` special-cases for the packed
int4 (GGML Q4_0 / ``--kv-cache-dtype q4_0``) KV pool: BLOCK_N=32 / 32 splits / 4 warps,
selected because the generic BLOCK_N=16 tuning silently corrupted this geometry's packed
loads (see ``tests/kernels/test_kv_quant.py::test_ornith_q4_tuned_decode_matches_dequantized_oracle``,
which this bench exercises at benchmark scale). This script calls the SAME production
Triton entry points (``decode_paged_attention`` / ``paged_attention`` /
``extend_paged_attention``) no server, no model weights.

Three op families, matching how a long-context Ornith session actually drives attention:

  decode  -- one new token against a cached context of length L (the tuned split-K
             kernel above). Swept over context length, batch size, and the max_kv_splits
             scratch ceiling (``launch_splits = min(preferred, ceiling)``, so a ceiling
             below the tuned value reproduces the degraded-scratch/capture-buffer case
             the kernel's docstring calls out).
  prefill -- a fresh chunk attending only to itself (first chunk of a prompt, or a
             --max-prefill-length-sized chunk), via the fused causal kernel.
  extend  -- a fresh chunk attending to a long CACHED (quantized) prefix plus itself,
             via the split extend kernel -- the realistic shape of chunk N>1 of a long
             prompt, or of extending a session's context.

Correctness gate (always on unless --skip-verify): before ANY case is timed, its
production (quantized) call is checked against the SAME kernel fed the pool's
dequantized values -- the dequantized values are the oracle, not a separate reference
implementation, so this isolates "does the packed-Q4 dequant-and-dot path compute the
same attention" from "how fast is it". Tolerance matches the kernel's own regression
tests (rtol=atol=2e-2, bf16 accumulation).

Run:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python benchmarks/bench_ornith_attention.py
    ... --decode-lengths 8192 32768 131072 200000 --kv-quant int4 q8_0 --json out.jsonl
    ... --ops decode --batch-sizes 1 4 16 --max-kv-splits 8 16 32
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field

# Geometry this bench targets by default: Ornith-1.5-35B-A3B's attention (16 query
# heads, 2 KV heads, head_dim 256) -- the exact shape decode_launch_config tunes for.
Q_HEADS = 16
KV_HEADS = 2
HEAD_DIM = 256

QUANT_CHOICES = ("int4", "q4_0", "q8_0", "fp8_e4m3", "bf16")
# CLI spelling -> freetoken.kvcache.quant spec name. "q4_0"/"int4" are the same packed
# scheme (llama.cpp naming vs the internal one); "bf16" means the unquantized pool.
_QUANT_ALIAS = {"int4": "int4", "q4_0": "int4", "q8_0": "q8_0", "fp8_e4m3": "fp8_e4m3", "bf16": "auto"}

OPS = ("decode", "prefill", "extend")


# --------------------------------------------------------------------------------------
# Pure case-building (no torch/CUDA import) -- kept separate so it is unit-testable on a
# machine with no GPU.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodeCase:
    op: str = field(default="decode", init=False)
    quant: str = ""
    ctx_len: int = 0
    batch: int = 0
    max_kv_splits: int | None = None  # None = use the kernel's own tuned preference


@dataclass(frozen=True)
class PrefillCase:
    op: str = field(default="prefill", init=False)
    quant: str = ""
    chunk_len: int = 0


@dataclass(frozen=True)
class ExtendCase:
    op: str = field(default="extend", init=False)
    quant: str = ""
    prefix_len: int = 0
    chunk_len: int = 0


def build_cases(args: argparse.Namespace) -> list:
    """Expand the CLI sweeps into one case object per (op, quant, config) point.

    Deterministic order (op, then quant, then the op's own axes) so --json output and
    the printed table always list cases the same way for a given set of flags.
    """
    cases: list = []
    quants = args.kv_quant  # kept as the CLI spelling for display; runners resolve via _QUANT_ALIAS
    max_kv_splits = args.max_kv_splits or [None]

    if "decode" in args.ops:
        for quant in quants:
            for ctx_len in args.decode_lengths:
                for batch in args.batch_sizes:
                    for splits in max_kv_splits:
                        cases.append(DecodeCase(quant=quant, ctx_len=ctx_len, batch=batch, max_kv_splits=splits))
    if "prefill" in args.ops:
        for quant in quants:
            for chunk_len in args.prefill_chunk_sizes:
                cases.append(PrefillCase(quant=quant, chunk_len=chunk_len))
    if "extend" in args.ops:
        for quant in quants:
            for prefix_len in args.decode_lengths:
                for chunk_len in args.extend_chunk_sizes:
                    cases.append(ExtendCase(quant=quant, prefix_len=prefix_len, chunk_len=chunk_len))
    return cases


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--ops", nargs="+", default=list(OPS), choices=OPS)
    p.add_argument(
        "--kv-quant", nargs="+", default=["int4"], choices=QUANT_CHOICES,
        help="pool format(s) to bench; 'bf16' is the unquantized baseline (no oracle check)",
    )
    p.add_argument(
        "--decode-lengths", type=int, nargs="+", default=[2048, 16384, 65536, 131072],
        help="decode: cached context length before the new token. extend: cached prefix length.",
    )
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1], help="decode: concurrent requests")
    p.add_argument(
        "--max-kv-splits", type=int, nargs="+", default=None,
        help="decode: scratch ceiling for the split-K reduction (launch uses min(tuned, ceiling)); "
        "default lets each case use its quant's own tuned preference",
    )
    p.add_argument(
        "--prefill-chunk-sizes", type=int, nargs="+", default=[512, 2048, 8192],
        help="prefill: fresh (no-prefix) chunk sizes; 8192 matches the server's default --max-prefill-length",
    )
    p.add_argument(
        "--extend-chunk-sizes", type=int, nargs="+", default=[2048],
        help="extend: freshly-computed chunk size attending to the cached --decode-lengths prefix",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-verify", action="store_true", help="skip the dequantized-oracle correctness gate")
    p.add_argument("--json", dest="json_out", default=None, help="append one JSON line per case here")
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------
# CUDA execution -- imports torch/triton lazily so --help and case-building stay usable
# without a GPU.
# --------------------------------------------------------------------------------------


def _make_kv(tokens: int, heads: int, dim: int, device, seed: int):
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(tokens, heads, dim, generator=g, device=device, dtype=torch.bfloat16)


def _quantize_pool(spec, k, v):
    """Store bf16 k/v into a pool of the given quant spec.

    Returns ``(k_cache, k_scale, v_cache, v_scale, k_oracle, v_oracle)``. For the
    unquantized spec the cache tensors ARE k/v and there is no oracle (nothing to check
    the production path against) -- callers must skip the correctness gate in that case.
    """
    import torch

    from freetoken.kernel.triton.kv_quant import store_kv_quant
    from freetoken.kvcache.quant import BLOCK

    if not spec.enabled:
        return k, None, v, None, None, None

    slots, heads, dim = k.shape
    epb = spec.elements_per_byte
    kq = torch.zeros(slots, heads, dim // epb, device=k.device, dtype=spec.storage_dtype)
    vq = torch.zeros_like(kq)
    ks = torch.zeros(slots, heads, dim // BLOCK, device=k.device, dtype=torch.float16)
    vs = torch.zeros_like(ks)
    indices = torch.arange(slots, device=k.device, dtype=torch.int32)
    store_kv_quant(kq, ks, vq, vs, indices, k, v, spec)
    k_oracle = spec.dequantize(kq.float(), ks).to(torch.bfloat16).reshape(slots, heads, dim)
    v_oracle = spec.dequantize(vq.float(), vs).to(torch.bfloat16).reshape(slots, heads, dim)
    return kq, ks, vq, vs, k_oracle, v_oracle


def _check_oracle(got, want, label: str) -> float:
    import torch

    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2, msg=lambda m: f"{label}: {m}")
    return (got.float() - want.float()).abs().max().item()


def _time_ms(fn, warmup: int, iters: int) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(10**7)  # settle clocks before the timed call, like bench_offload_cache_copy
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def run_decode(case: DecodeCase, quant_specs, device, args) -> dict:
    import torch

    from freetoken.kernel.triton.attention import decode_launch_config, decode_paged_attention

    spec = quant_specs[_QUANT_ALIAS[case.quant]]
    slots = case.ctx_len * case.batch
    k = _make_kv(slots, KV_HEADS, HEAD_DIM, device, args.seed)
    v = _make_kv(slots, KV_HEADS, HEAD_DIM, device, args.seed + 1)
    q = _make_kv(case.batch, Q_HEADS, HEAD_DIM, device, args.seed + 2)
    k_cache, k_scale, v_cache, v_scale, k_oracle, v_oracle = _quantize_pool(spec, k, v)

    indptr = torch.arange(0, slots + 1, case.ctx_len, device=device, dtype=torch.int32)
    indices = torch.arange(slots, device=device, dtype=torch.int32)
    q_pos = torch.full((case.batch,), case.ctx_len - 1, device=device, dtype=torch.int32)
    sm_scale = HEAD_DIM**-0.5

    quant_name = "int4" if spec.enabled and spec.elements_per_byte == 2 else ("quant8" if spec.enabled else None)
    preferred, block_n, num_warps = decode_launch_config(
        quant_name=quant_name, head_dim=HEAD_DIM, num_q_heads=Q_HEADS, num_kv_heads=KV_HEADS
    )
    ceiling = case.max_kv_splits if case.max_kv_splits is not None else preferred

    def call(k_c, v_c, k_s, v_s):
        # decode_paged_attention picks its OWN preferred split count internally (it
        # differs for the quantized call vs. the dequantized-oracle bf16 call, since
        # decode_launch_config keys off whether a quant scale was given), so nsplits and
        # the logits/lse scratch must be sized from that SAME per-call preference, not a
        # value borrowed from the other call -- otherwise stage2 reduces over splits
        # stage1 never wrote, reading uninitialized scratch.
        qname = quant_name if k_s is not None else None
        call_preferred, _, _ = decode_launch_config(
            quant_name=qname, head_dim=HEAD_DIM, num_q_heads=Q_HEADS, num_kv_heads=KV_HEADS
        )
        splits = max(min(call_preferred, ceiling), 1)
        logits = torch.empty(case.batch, Q_HEADS, splits, HEAD_DIM, device=device, dtype=torch.float32)
        lse = torch.empty(case.batch, Q_HEADS, splits, device=device, dtype=torch.float32)
        nsplits = torch.full((case.batch,), splits, device=device, dtype=torch.int32)
        return decode_paged_attention(
            q, k_c, v_c, indptr, indices, q_pos, logits, lse, nsplits, splits, sm_scale,
            k_scale=k_s, v_scale=v_s,
        )

    scratch_splits = max(min(preferred, ceiling), 1)  # the quantized (production) path's actual launch
    max_abs_err = None
    if spec.enabled and not args.skip_verify:
        got = call(k_cache, v_cache, k_scale, v_scale)
        want = call(k_oracle, v_oracle, None, None)
        max_abs_err = _check_oracle(got, want, f"decode quant={case.quant} ctx={case.ctx_len}")

    time_ms = _time_ms(lambda: call(k_cache, v_cache, k_scale, v_scale), args.warmup, args.iters)
    bytes_moved = slots * KV_HEADS * HEAD_DIM * 2 * spec.bytes_per_element(torch.bfloat16)
    return {
        "op": "decode", "quant": case.quant, "ctx_len": case.ctx_len, "batch": case.batch,
        "max_kv_splits_ceiling": ceiling, "launch_splits": scratch_splits, "tuned_block_n": block_n,
        "tuned_num_warps": num_warps, "time_ms": time_ms, "gbps": bytes_moved / (time_ms * 1e6),
        "max_abs_err": max_abs_err,
    }


def run_prefill(case: PrefillCase, quant_specs, device, args) -> dict:
    import torch

    from freetoken.kernel.triton.attention import paged_attention

    spec = quant_specs[_QUANT_ALIAS[case.quant]]
    n = case.chunk_len
    k = _make_kv(n, KV_HEADS, HEAD_DIM, device, args.seed)
    v = _make_kv(n, KV_HEADS, HEAD_DIM, device, args.seed + 1)
    q = _make_kv(n, Q_HEADS, HEAD_DIM, device, args.seed + 2)
    k_cache, k_scale, v_cache, v_scale, k_oracle, v_oracle = _quantize_pool(spec, k, v)

    indptr = torch.tensor([0, n], device=device, dtype=torch.int32)
    indices = torch.arange(n, device=device, dtype=torch.int32)
    q_to_req = torch.zeros(n, device=device, dtype=torch.int32)
    q_pos = torch.arange(n, device=device, dtype=torch.int32)
    sm_scale = HEAD_DIM**-0.5
    kw = dict(indptr=indptr, indices=indices, q_to_req=q_to_req, q_positions=q_pos, sm_scale=sm_scale)

    def call(k_c, v_c, k_s, v_s):
        return paged_attention(q=q, k_cache=k_c, v_cache=v_c, k_scale=k_s, v_scale=v_s, **kw)

    max_abs_err = None
    if spec.enabled and not args.skip_verify:
        got = call(k_cache, v_cache, k_scale, v_scale)
        want = call(k_oracle, v_oracle, None, None)
        max_abs_err = _check_oracle(got, want, f"prefill quant={case.quant} chunk={case.chunk_len}")

    time_ms = _time_ms(lambda: call(k_cache, v_cache, k_scale, v_scale), args.warmup, args.iters)
    flops = 2 * 2 * n * n * Q_HEADS * HEAD_DIM  # QK^T + PV, causal factor ignored (upper bound)
    return {
        "op": "prefill", "quant": case.quant, "chunk_len": case.chunk_len,
        "time_ms": time_ms, "tflops": flops / (time_ms * 1e9), "max_abs_err": max_abs_err,
    }


def run_extend(case: ExtendCase, quant_specs, device, args) -> dict:
    import torch

    from freetoken.kernel.triton.attention import extend_paged_attention

    spec = quant_specs[_QUANT_ALIAS[case.quant]]
    prefix, chunk = case.prefix_len, case.chunk_len
    total = prefix + chunk
    k = _make_kv(total, KV_HEADS, HEAD_DIM, device, args.seed)
    v = _make_kv(total, KV_HEADS, HEAD_DIM, device, args.seed + 1)
    q = _make_kv(chunk, Q_HEADS, HEAD_DIM, device, args.seed + 2)
    # The cached prefix lives in the (quantized) pool; the new chunk is passed straight
    # through in bf16 via k_extend/v_extend, matching how a real extend step supplies
    # this step's freshly-projected K/V alongside the persistent KV cache.
    k_extend, v_extend = k[prefix:], v[prefix:]
    k_cache, k_scale, v_cache, v_scale, k_oracle, v_oracle = _quantize_pool(spec, k[:prefix], v[:prefix])

    qo_indptr = torch.tensor([0, chunk], device=device, dtype=torch.int32)
    kv_indptr = torch.tensor([0, prefix], device=device, dtype=torch.int32)
    kv_indices = torch.arange(prefix, device=device, dtype=torch.int32)
    prefix_lens = torch.tensor([prefix], device=device, dtype=torch.int32)
    sm_scale = HEAD_DIM**-0.5
    kw = dict(
        qo_indptr=qo_indptr, kv_indptr=kv_indptr, kv_indices=kv_indices, prefix_lens=prefix_lens,
        max_q_len=chunk, sm_scale=sm_scale, k_extend=k_extend, v_extend=v_extend,
    )

    def call(k_c, v_c, k_s, v_s):
        return extend_paged_attention(q=q, k_cache=k_c, v_cache=v_c, k_scale=k_s, v_scale=v_s, **kw)

    max_abs_err = None
    if spec.enabled and not args.skip_verify:
        got = call(k_cache, v_cache, k_scale, v_scale)
        want = call(k_oracle, v_oracle, None, None)
        max_abs_err = _check_oracle(got, want, f"extend quant={case.quant} prefix={prefix} chunk={chunk}")

    time_ms = _time_ms(lambda: call(k_cache, v_cache, k_scale, v_scale), args.warmup, args.iters)
    # QK^T + PV against (cached prefix + causal self) KV, upper-bounded by the full window.
    flops = 2 * 2 * chunk * total * Q_HEADS * HEAD_DIM
    return {
        "op": "extend", "quant": case.quant, "prefix_len": prefix, "chunk_len": chunk,
        "time_ms": time_ms, "tflops": flops / (time_ms * 1e9), "max_abs_err": max_abs_err,
    }


_RUNNERS = {"decode": run_decode, "prefill": run_prefill, "extend": run_extend}


def _print_row(row: dict) -> None:
    op = row["op"]
    err = f"{row['max_abs_err']:.4f}" if row["max_abs_err"] is not None else "n/a"
    if op == "decode":
        print(
            f"decode  quant={row['quant']:<9} ctx={row['ctx_len']:>7} batch={row['batch']:<3} "
            f"splits={row['launch_splits']:>2}/{row['max_kv_splits_ceiling']:<3} "
            f"{row['time_ms']:8.4f} ms  {row['gbps']:7.1f} GB/s  oracle_err={err}",
            flush=True,
        )
    elif op == "prefill":
        print(
            f"prefill quant={row['quant']:<9} chunk={row['chunk_len']:>7} "
            f"{row['time_ms']:8.3f} ms  {row['tflops']:7.2f} TFLOP/s  oracle_err={err}",
            flush=True,
        )
    else:
        print(
            f"extend  quant={row['quant']:<9} prefix={row['prefix_len']:>7} chunk={row['chunk_len']:>6} "
            f"{row['time_ms']:8.3f} ms  {row['tflops']:7.2f} TFLOP/s  oracle_err={err}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import torch

    from freetoken.kvcache.quant import FP8_E4M3, INT4, NONE, Q8_0

    assert torch.cuda.is_available(), "CUDA is required"
    torch.cuda.set_device(args.device)
    device = torch.device("cuda")
    quant_specs = {"int4": INT4, "q8_0": Q8_0, "fp8_e4m3": FP8_E4M3, "auto": NONE}

    print(f"gpu {torch.cuda.get_device_name(device)}  geometry q_heads={Q_HEADS} kv_heads={KV_HEADS} "
          f"head_dim={HEAD_DIM} (Ornith)", flush=True)

    cases = build_cases(args)
    rows = []
    for case in cases:
        row = _RUNNERS[case.op](case, quant_specs, device, args)
        _print_row(row)
        rows.append(row)
        if args.json_out:
            with open(args.json_out, "a") as f:
                f.write(json.dumps(row) + "\n")

    print(f"\n{len(rows)} cases run, {sum(1 for r in rows if r.get('max_abs_err') is not None)} "
          f"oracle-verified before timing", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
