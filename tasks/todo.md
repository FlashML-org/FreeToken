# Ornith RTX 5080 full-context optimization

- [x] Add reproducible Ornith Q4_0 attention and serving benchmark controls.
- [x] Record unchanged-main RTX 5080 synthetic baselines.
- [x] Sweep and implement numerically safe sm_120 attention launch geometry.
- [x] Sweep Ornith Q4_K/Q6_K dense and routed-expert GGUF dispatch on sm_120.
- [x] Implement only repeatable sm_120 wins with safe non-sm_120 fallbacks.
- [x] Verify focused tests, full non-slow tests, and lint.
- [x] Validate a live near-262K request and long-context retrieval.
- [x] Document the final RTX 5080 launch command and measured before/after results.

## Review (2026-08-25, RTX 5080 sm_120, Torch 2.11/CUDA 13/Triton 3.6, WSL)

### Implemented
- `decode_launch_config` is architecture-aware (`compute_capability` param). Ornith
  Q4_0 decode on sm_120 uses (kv_splits=64, block_n=64, warps=8): 0.356 ms vs
  0.822 ms per 262K full-attention layer with the sm_89 tuple (2.31x). sm_89 and
  the conservative fallback are unchanged; scratch/CUDA-graph capacity follows.
- `extend_paged_attention` uses num_warps=4 on sm_120 (8 elsewhere): 1.12x on the
  production long-Q4-prefix split kernel, 2.01x on the fused cold-prefill kernel.
  BLOCK_N=16 was reconfirmed to silently corrupt the packed loader on sm_120 in
  BOTH extend kernels and remains excluded.
- `benchmarks/bench_ornith_attention.py` (oracle-gated decode/prefill/extend
  sweeps at the exact Ornith geometry) + full-context flags on
  `bench_decode_moe.py` (`--max-context`, `--kv-cache-dtype`, `--prefill-chunk`,
  `--prefill-hit-d2d`); GPU-free unit tests under `tests/benchmarks/`.
- `docs/models.md`: sm_120 full-262K launch command (explicit
  `--attention-backend triton`; sm_120 auto-resolves to FlashInfer, which cannot
  read the quantized KV pool).

### Measured but intentionally not changed
- The vendored GGUF DP4A kernels predate llama.cpp's int8-tensor-core MMQ
  rewrite (as do vLLM/SGLang's copies); porting that is phase 2b (below).

## Phase 2a (2026-08-25): arch-aware GGUF dispatch thresholds

- [x] `layers/gguf.py`: `dequant_gemm_min_rows(cc)` — 24 on sm_120, 32
  elsewhere (Q4_K attn shapes cross at 24: dequant 0.0645 vs MMQ 0.0778 ms;
  16 would regress the Q6_K lm_head where MMQ still wins at 16).
- [x] `moe/fused_gguf.py`: `mmq_min_tokens(cc)` — 16 on sm_120, 32 elsewhere
  (grouped MMQ 0.314 vs vec 0.324 ms @16; 0.382 vs 0.475 @24).
- [x] `_MMVQ_SAFE` left at 6 (per-shape ambiguous on sm_120).
- [x] Tests: `tests/kernels/test_gguf_dispatch.py` — pure threshold tests for
  both archs + CUDA dispatch-branch tests with faked capability (18 passed).
- [x] Live re-verify on real Ornith blk.3.attn_q (Q4_K): 24 rows now 0.0690 ms
  via dequant vs 0.0806 ms with the old threshold; 16/32 rows unchanged.
- [x] Kernel+benchmark suites 123 passed; full non-slow suite failures A/B
  identical to clean main (7 failures; laguna errors are suite-order artifacts
  present in the clean-main baseline too); ruff clean.

## Phase 2b (2026-08-25): upstream int8-MMA MMQ port (Q4_K/Q6_K)

- [x] Vendored llama.cpp master `eab8ee41` CUDA MMQ verbatim into
  `python/freetoken/kernel/csrc/gguf_mmq/` (mmq/mma/load-tiles/vec-dot/configs/
  quantize/mmid + the ggml headers). `mmq_ext.cu` is the only hand-written
  file: backend shims (device info, torch-allocator pool, abort/error) + torch
  bindings; only Q4_K/Q6_K `mul_mat_q` cases are instantiated.
- [x] Dense: `ggml_mul_mat_a8_mma` wired into `fused_mul_mat_gguf` on sm_120
  for rows > `_MMVQ_SAFE` (replaces BOTH the DP4A-MMQ band and dequant+cuBLAS).
  Measured (real Ornith tensors, embedder stopped): Q4_K attn_q 8192 rows
  1.79 ms vs dequant 2.40 vs DP4A 22.9; Q6_K lm_head @2048 tokens 17.5 vs 19.2
  vs 262; wins at every rows >= 4. Build failure falls back to the old path.
- [x] MoE: `ggml_moe_a8_mma` (upstream ids path: mm_ids_helper +
  scatter/gather q8_1_mmq quantize + expert_bounds mul_mat_q) wired into
  `_moe_matmul` for 320 <= tokens <= 16384 on sm_120; broadcast (gate/up) and
  per-slot (down) forms; padded flat slots addressed in whole blocks (stride
  must divide by block size -- true for Ornith banks). Ornith geometry
  (E=256 top-8): 8192 tokens gate_up 4.16 ms vs DP4A 23.2, down 4.90 vs 15.3;
  DP4A keeps 16..319 (crossover ~288), MMVQ keeps decode.
- [x] Numerics: MMA matches dequant reference within activation-quant noise on
  real Ornith tensors (rel <= 0.013, on par with DP4A) and on random-safe
  packed bytes vs gguf-py (`tests/kernels/test_gguf_mma.py`); MoE broadcast +
  gather verified vs per-expert dense reference and vs the vec kernel
  (end-to-end `fused_experts_gguf` rel ~1e-3).
- [x] Tests: dispatch-branch tests extended (MMA seams, `_use_mma_moe` gate);
  `test_dense_gguf_prefill_uses_dequantized_cublas_result` pinned to the
  dequant branch it validates. Kernels+benchmarks: 336 passed, 1 skipped.
  Full non-slow suite: same 7 pre-existing failures as clean main.
- [x] Live A/B at the production 262K config (2026-08-26, hostile prompt:
  28K words seeded non-repetitive text, ~50K tokens / 6+ chunks, three distinct
  needles at 10/50/90% depth, greedy, 320-token decode):
  - MMA: 13.88 s wall, 3/3 needles exact, 0 tracebacks (prefill ~4,700 tok/s).
  - Fallback (FREETOKEN_GGUF_DISABLE_MMA=1, same box/prompt minutes apart):
    21.94 s wall, 3/3 needles exact, 0 tracebacks (prefill ~2,700 tok/s).
  - Net: ~1.75x prefill, 1.58x end-to-end TTFT+decode; identical answers.
  - Path evidence: the MMA leg's worker warmup demonstrably blocked polling the
    freetoken_gguf_mmq JIT lock (only `_mma_module()` touches it) and ran at
    the faster wall time; the fallback leg never touched that cache. The
    in-log `int8-MMA MMQ ACTIVE` INFO line is swallowed by the server's log
    handler -- known instrumentation gap, not a dispatch gap.
- OPS HAZARD found: `torch.utils.cpp_extension.load` leaves a stale `lock` in
  ~/.cache/torch_extensions/ when a building/loading process is SIGKILLed; the
  next server then hangs in warmup FOREVER, sleep-polling it (looks like a
  startup hang). Fix on sight: `rm ~/.cache/torch_extensions/py312_cu130/*/lock`.
  Also: 3 host OOMs during testing were the ~20 GiB shmem expert banks + agent
  processes on the 29 GiB WSL box; mitigated with a 12 GiB swapfile
  (/swapfile-claude, left enabled) + oom_score_adj (serve 800, terminal -600).
  A killed `ft serve` leaves `multiprocessing.spawn` workers holding the banks:
  `pkill -9 -f "FreeToken/.venv/bin/python3"` and check `free -g`.
- Flaky pre-existing `test_reference_roundtrip_error_is_within_the_scheme_envelope[int4]`
  (~5% failure, unseeded randn) now seeded.

### Live 262K validation (Ornith-1.5-35B-Q4_K_M, one RTX 5080 16 GB)
Command: `ft serve --model ~/ai/models/Ornith-1.5-35B-Q4_K_M.gguf
--attention-backend triton --kv-cache-dtype q4_0 --num-tokens 262144
--kv-reserve-tokens 262144 --max-seq-len-override 262144
--max-running-requests 1 --moe-backend offload --moe-cache-auto
--max-prefill-length 8192`
- Auto-sizing: 4,835 expert slots + 262,263 KV pages, prefill overlap on.
- Cold ~259,400-token prefill: 210–220 s (~1,230 tok/s sustained).
- Decode at ~259K context: 99–104 tok/s (Ada baseline: 33.67 tok/s at 170K).
- NIAH 3/3 exact: passcode `7391-ALPHA` recovered at 10%/50%/90% depth
  (greedy, through the model's `<think>` block).
- Radix-cached TTFT for a repeated full-context prompt: 4.9 s.
- No crash, OOM, or worker restart across the whole run.

### Test/lint status
- `tests/kernels/test_triton_attention.py` + `test_kv_quant.py`: 75 passed.
- `tests/benchmarks`: 8 passed; ruff clean on all changed files.
- Full non-slow suite: 9 failures + 6 errors pre-exist on clean main
  (`moe_pageable_gpu` config-test drift), byte-identical A/B vs baseline.
