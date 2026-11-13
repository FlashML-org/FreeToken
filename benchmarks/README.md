# benchmarks

Run from the repo root with `PYTHONPATH=python:.`, pinned to one GPU
(`CUDA_VISIBLE_DEVICES=0`). Each script's `--help` / docstring has the details.

**`bench_serving.py`** — a client-wall performance harness for an already-running FreeToken
server. It calibrates prompts with `/v1/messages/count_tokens`, measures streamed
`/v1/chat/completions` TTFT/decode/total latency, and records live cache geometry from
`/v1/cache/status`.

The first version measures one configuration at a time. It does not start the server, sweep
backends, or run concurrent requests; those concerns can consume the same JSON schema later.

```bash
PYTHONPATH=python:. python benchmarks/bench_serving.py \
    --base-url http://127.0.0.1:1919 \
    --prefill-sizes 512,1024,2048,4096 \
    --decode-tokens 256 \
    --cache-prefix-tokens 4096 \
    --repetitions 3 \
    --json serving-bench.json
```

`client_wall_*` fields are measured at the HTTP client boundary. They include queueing,
prefill, scheduling, detokenization, and SSE delivery; they are not engine-internal
throughput counters. Prefix-cache samples are a sequence (`fresh`, `identical`, and
`small_suffix`) so cache reuse is measured explicitly. Run the CPU-only tests with:

```bash
pytest tests/benchmarks/test_bench_serving.py
```

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

For host RAM vs PCIe bandwidth and the offload/hybrid backend pick, use `ft bench bw`
instead — it writes the JSON profile the engine reads.
