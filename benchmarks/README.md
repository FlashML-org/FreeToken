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

For host RAM vs PCIe bandwidth and the offload/hybrid backend pick, use `ft bench bw`
instead — it writes the JSON profile the engine reads.

## ROCm evidence

`bench_rocm_matrix.py` creates content-hashed workload/runtime manifests. Keep
sampled, greedy, and teacher-forced replay lanes separate. Validate candidate
versus incumbent evidence with `check_decode_gate.py`; speed promotion accepts only
`teacher_forced_replay` rows carrying replay schema v2, prompt/continuation hashes,
an independent oracle ID, matching route digest, zero fallbacks, finite logits, and
exact completion count. Sampled/greedy rows remain correctness evidence only.

Use unique `TORCH_EXTENSIONS_DIR` per run. Fresh cache proves JIT hygiene, not
correctness or speed. Promotion requires same-model A/B served results and at
least three fresh-process repeats; direct-kernel timings remain diagnostics.

`profile_decode_rocm.py` wraps `rocprofv3` only for an explicit command after `--`,
or summarizes `--trace` JSON. Pass observed `--route`, `--graph-mode`, and `--lane`
when artifact does not contain them. It returns exit code 2 and writes `incomplete`
when clock, route, token, or lane evidence is missing; missing profiler data never
becomes zero overhead.
