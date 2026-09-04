# benchmarks

Run from the repo root with `PYTHONPATH=python:.`, pinned to one GPU
(`CUDA_VISIBLE_DEVICES=0`). Each script's `--help` / docstring has the details.

**`bench_decode_moe.py`** — bs=1 decode tok/s of a served MoE model. Spawns `ft serve`
per backend and times token arrivals over streamed `/v1/chat/completions`, so numbers
include the full serving path. AIME-25 prompt, checkpoint-recommended sampling.

```bash
python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload,cpu,hybrid
```

For promotion evidence, use one immutable local AIME JSONL (or Hub revision plus SHA-256),
the exact same GGUF, `--decode 512 --context 9216 --batch 512 --ubatch 512 --kv-type q8_0`,
and ten repeats. Validate rows with `benchmarks/check_decode_gate.py`; missing fixture,
runtime KV, graph, thermal, model, or reference identity rejects promotion. Set
`FREETOKEN_GGUF_MOE_IMPL=legacy` for rollback. `FREETOKEN_GRAPH_SAMPLER=1` enables
capture-safe greedy sampler/device-token-chain evidence; dynamic sampling keeps fallback.

**`bench_decode_replay.py`** — pins one legacy FreeToken greedy continuation and runs a
separate teacher-forced replay lane. Route capture is untimed and never contributes to
replay timing. The golden command must run on an idle GPU:

```bash
python benchmarks/bench_decode_replay.py golden \
  --model /path/to/model --aime /path/to/test.jsonl \
  --aime-sha256 SHA256 --out .plans/rocm-ollama-gap/replay-manifest.json
python benchmarks/bench_decode_replay.py replay-freetoken \
  --model /path/to/model --manifest .plans/rocm-ollama-gap/replay-manifest.json \
  --aime /path/to/test.jsonl --aime-sha256 SHA256 --routes \
  --json .plans/rocm-ollama-gap/replay-freetoken.jsonl
```

`bench_llama_cpp_hip.py --replay-manifest` delegates fixed-token replay to a supplied
`llama-server`; route hashes remain unavailable unless reference binary is instrumented.

`bench_gguf_moe_kernels.py --paired --impl gfx1100` interleaves legacy/candidate
exact-shape calls and emits raw samples plus paired median recovery and bootstrap CI.
It requires a GPU when run; no result is considered promotion evidence until output
bytes, quant shape, baseline drift, and fallback count pass their gates.

`profile_decode_rocm.py` parses an existing rocprof artifact or launches only an
explicit command after `--`. It fails closed without clock-correlation records and
reports disjoint token ledgers plus warm-offload event counts/bytes.

Promotion gate output contains independent `gate_a` (sampled absolute) and `gate_b`
(q8/q8 teacher-forced replay) objects. Gate B requires matched IDs, route hashes,
identities, and paired performance; absence of replay evidence never promotes q8.

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
