"""Paged KV storage/scatter/decode microbenchmark, independent of model weights.

Run with PYTHONPATH=python:. uv run python benchmarks/bench_kv_quant.py.
Compare identical arguments on the baseline and candidate; this does not measure
end-to-end model quality, TTFT, or serving throughput.
"""

import argparse
import json

import torch
import triton.testing

from freetoken.distributed import set_tp_info
from freetoken.kernel.triton.attention import decode_paged_attention
from freetoken.kvcache.mha_pool import MHAKVCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formats", default="none,fp8,nvfp4")
    parser.add_argument("--lengths", default="1024,8192,32768")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()
    set_tp_info(rank=0, size=1)
    torch.manual_seed(42)
    batch, heads, dim = args.batch, args.heads, args.dim
    qheads = heads * args.group
    device = torch.device("cuda")
    results = []
    for length in map(int, args.lengths.split(",")):
        slots = batch * length
        k, v = [torch.randn(slots, heads * dim, device=device, dtype=torch.bfloat16)
                for _ in range(2)]
        loc = torch.arange(slots, device=device, dtype=torch.int32)
        q = torch.randn(batch, qheads, dim, device=device, dtype=torch.bfloat16)
        indptr = torch.arange(batch + 1, device=device, dtype=torch.int32) * length
        pos = torch.full((batch,), length - 1, device=device, dtype=torch.int32)
        scratch = torch.empty(batch, qheads, 8, dim, device=device)
        lse = torch.empty(batch, qheads, 8, device=device)
        splits = torch.full((batch,), 8, device=device, dtype=torch.int32)
        out = torch.empty_like(q)
        for quant in args.formats.split(","):
            pool = MHAKVCache(heads, 1, dim, slots, 1, q.dtype, device, kv_quant=quant)
            pool.store_kv(k, v, loc, 0)
            extra = {}
            if quant == "nvfp4":
                extra = dict(kv_quant=quant, k_block_scale=pool.k_block_scale(0),
                             v_block_scale=pool.v_block_scale(0))
            kc, vc = [getattr(pool, name)(0).flatten(0, 1) for name in ("k_cache", "v_cache")]

            def decode():
                return decode_paged_attention(q, kc, vc, indptr, loc, pos,
                    scratch, lse, splits, 8, dim ** -.5, out=out,
                    k_scale=pool.k_scale(0), v_scale=pool.v_scale(0), **extra)

            decode()
            decode_ms = triton.testing.do_bench(decode, warmup=100, rep=300)
            store_ms = triton.testing.do_bench(
                lambda: pool.store_kv(k[-batch:], v[-batch:], loc[-batch:], 0),
                warmup=100, rep=300)
            record = dict(format=quant, length=length, batch=batch,
                          kv_bytes=pool.unit_bytes()[0] * slots,
                          decode_ms=decode_ms, store_ms=store_ms)
            results.append(record)
            print(json.dumps(record), flush=True)
            del pool, kc, vc
        del k, v
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                         heads=heads, group=args.group, dim=dim, results=results)))


if __name__ == "__main__":
    main()
