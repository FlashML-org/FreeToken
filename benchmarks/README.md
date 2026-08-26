# benchmarks

Run from the repo root with `PYTHONPATH=python:.`, pinned to one GPU
(`CUDA_VISIBLE_DEVICES=0`). Each script's `--help` / docstring has the details.

**`bench_decode_moe.py`** — bs=1 decode tok/s of a served MoE model. Spawns `ft serve`
per backend and times token arrivals over streamed `/v1/chat/completions`, so numbers
include the full serving path. AIME-25 prompt, checkpoint-recommended sampling.

```bash
python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload,cpu,hybrid
```

**`bench_load_weight_generic.py`** — expert-bank load time: serial vs parallel O_DIRECT
vs pre-repacked FTW, each mode in its own subprocess. Linux-only; stages the FTW under
`/var/tmp` (`--ftw-dir` overrides; roughly checkpoint-sized).

```bash
python benchmarks/bench_load_weight_generic.py --model /path/to/model
```

**`bench_offload_cache_copy.py`** — synthetic (no checkpoint): per-layer decode expert
copy cost (`ensure_experts` + `copy_missing`), swept over bank layout x cache slots x
batch size x miss rate.

```bash
python benchmarks/bench_offload_cache_copy.py
```

**`bench_ornith_attention.py`** — synthetic (no checkpoint): Ornith's exact attention
geometry (16 query heads, 2 KV heads, head_dim 256 — the GQA shape `decode_launch_config`
tunes packed-int4/Q4_0 decode for) through the production Triton kernels directly
(`decode_paged_attention` / `paged_attention` / `extend_paged_attention`, no server).
Sweeps decode context length x batch size x the `max_kv_splits` scratch ceiling, plus
representative prefill (fresh chunk) and extend (cached prefix + new chunk) cases, over
one or more `--kv-quant` pool formats (`int4`/`q4_0`, `q8_0`, `fp8_e4m3`, `bf16`). Every
quantized case is checked against the same kernel fed the pool's dequantized values
before it is timed — the correctness gate `test_ornith_q4_tuned_decode_matches_dequantized_oracle`
pins at unit scale, exercised here at benchmark scale.

```bash
python benchmarks/bench_ornith_attention.py
python benchmarks/bench_ornith_attention.py --decode-lengths 8192 32768 131072 200000 \
    --kv-quant int4 q8_0 --batch-sizes 1 4 16 --json out.jsonl
```

For host RAM vs PCIe bandwidth and the offload/hybrid backend pick, use `ft bench bw`
instead — it writes the JSON profile the engine reads.

`bench_decode_moe.py` also accepts `--max-context` (full-context `--max-seq-len-override`
+ `--num-tokens`), `--kv-cache-dtype`, `--prefill-chunk` (`--max-prefill-length`), and
`--prefill-hit-d2d`, for reproducing the long-context configurations in `docs/models.md`
(e.g. Ornith Q4_0 at 200K) through the real serving path — all optional, defaults
unchanged when omitted.
