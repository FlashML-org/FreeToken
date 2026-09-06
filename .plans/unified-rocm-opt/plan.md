# Unified ROCm Path Optimization

## Context

FreeToken now contains the PR-217 ROCm/GGUF implementation consolidated into
PR-132, including generic ROCm execution, GGUF dispatch, HIP JIT/AOT seams,
graph-capture gating, offload-cache movement, and exact gfx1100 candidate
guards. The next problem is performance and maintainability: these paths need
one shared ROCm foundation while retaining generic fallbacks for every target
and keeping CUDA behavior unchanged.

The primary workload is non-MTP, single-request, batch-1 GGUF decode on the
exact `Qwen_Qwen3.5-35B-A3B-Q4_K_S.gguf` checkpoint. Inc 1B records its full
SHA-256 before any comparison; historical Qwen3.6 results are directional
context only. Shape: hidden size 2048, intermediate size 512, top-k 8, Q4_K
gate/up, and Q8_0 down banks. The primary metric is client-arrival decode
tokens/s with TTFT excluded. Exact model, checkpoint hash, prompt,
continuation, KV mode, graph mode, runtime, and route identity must match
before comparing numbers.

Historical results guide hypotheses only. Prior plans recorded FreeToken
medians of 34.69, 61.543, and 62.295 tok/s under different revisions, blobs,
graph states, and evidence contracts; they are not the current baseline. A
current 32-token run on this branch loaded ROCm Torch successfully but exceeded
the 300-second startup gate during serial expert-bank construction, so no
current end-to-end tok/s baseline exists yet. Current synthetic copy evidence
is diagnostic only and cannot promote decode performance.

This plan optimizes the unified path in dependency order. Shared foundational
code may add portable contracts, capability detection, instrumentation, and
generic fallbacks. Hardware-specific code is allowed only behind exact target,
operation, quantization, shape, ABI, self-test, and promotion gates.

## Architectural decisions

- **One shared ROCm foundation.** Extend the existing architecture/toolchain,
  dispatch, graph-gate, and benchmark seams instead of adding a second ROCm
  resolver or a parallel backend. Shared APIs must remain usable by all ROCm
  targets and by CUDA builds.
- **Generic ROCm remains default.** Existing `legacy` remains the compatible
  generic selector and `auto` is its explicit policy alias. Both select the
  generic path whenever a target-specific candidate is absent, unproven,
  unsupported, or fails a runtime probe. Existing `rdna3_mmid`, `rdna3_mmvdq`,
  and `grouped_mmq` names remain valid. Forced candidate modes fail loudly and
  never silently fall back.
- **Hardware-specific code is isolated.** gfx1100 code remains separate from
  shared HIP/CUDA sources and is compiled/loaded only for exact `gfx1100`.
  Future gfx12 or CDNA variants get separate capability records and source
  units; family labels never imply candidate support.
- **Dispatch is contract-driven.** A route is selected from backend, target,
  operation, quantization, shape, phase, ABI, and capability status. Compile
  success alone never enables a candidate.
- **Optimization starts with measurement.** Every candidate needs a causal
  stage profile, exact output/logit/route evidence, and fresh-process server
  medians. Microbenchmarks prioritize work; they cannot replace serving proof.
- **CUDA is a protected control.** CUDA-only packages, flags, kernels, graph
  behavior, and NVFP4/Marlin paths remain unchanged except for additive shared
  interfaces with explicit non-ROCm branches.
- **Graph capture remains fail-closed.** ROCm graph use requires a target-valid
  `pass`; unknown, stale, unavailable, TP, or explicit-GPU parent probes use
  eager kernel-launch decode. A graph variant may be enabled only after a
  worker-start probe and replay correctness evidence.
- **Base decode only.** MTP, speculative decoding, batch-throughput claims,
  Q4_K-to-NVFP4 substitutions, and PLE/SSD redesign are outside this plan.

## Reused decisions from prior optimization plans

- Keep median-of-three or stronger fresh-process comparisons, with raw rows,
  spread, machine identity, model/checkpoint identity, route, graph, cache, KV,
  compiler, and kernel identity recorded.
- Use stage markers and a causal timeline before changing cache size, MMVQ
  launch geometry, attention, sampler, or host handoff.
- Reuse the in-repo Triton router on ROCm when available, with pure-Torch
  routing as a generic fallback and explicit output/route parity checks.
- Reuse graph-gate variants and worker-start environment propagation; do not
  apply BLAS policy late during graph capture.
- Reuse exact-shape GGUF MMVQ tests for Q4_K and Q8_0, and keep target-specific
  tuning behind the dispatch contract.
- Close known-negative branches unless hypothesis changes: repeated
  `MMV_Y` sweeps, rotated `sdot4` candidate promotion, cache enlargement
  without miss/fetch evidence, and tinygrad fused output as an independent
  correctness oracle.
- Do not compare FreeToken GGUF against a different Ollama blob as an
  identity-complete speed claim. If same-file proof is unavailable, label the
  comparison directional.

## Assumptions and answers from code

- Decision: current integration target is `help/pr-132` at local commit
  `ab5f46c`. Source: `git branch --show-current`, `git log -1`.
- Decision: architecture and candidate records already exist separately in
  `python/freetoken/utils/arch.py:20-250`; extend this contract rather than
  infer candidate support from family membership. Source: current branch code.
- Decision: GGUF dispatch already separates generic and candidate routes and
  rejects cross-backend architectures. Source:
  `python/freetoken/kernel/gguf.py:163-338`.
- Decision: native GGUF dense dispatch is operation/quant/shape based, while
  MoE execution passes per-operation metadata. Source:
  `python/freetoken/layers/gguf.py:47-75` and
  `python/freetoken/moe/fused_gguf.py:283-456`.
- Decision: current graph policy is split between engine configuration,
  `GraphRunner`, `graph_gate`, and launch-time worker environment. Source:
  `python/freetoken/engine/engine.py:44-84`,
  `python/freetoken/engine/graph.py:128-222`,
  `python/freetoken/utils/graph_gate.py:249-410`, and
  `python/freetoken/server/launch.py:165-181`.
- Decision: offload movement already has batch-copy, hit/miss, prefetch, and
  stream/event seams. Source: `python/freetoken/moe/offload_cache.py:700-878`
  and `:950-1080`.
- Decision: benchmark gate requires replay schema, independent oracle, route
  digest, finite logits, exact completion count, same identity, and at least
  three runs. Source: `benchmarks/bench_decode_replay.py` and
  `benchmarks/check_decode_gate.py:12-150`.
- Decision: existing workflows provide separate NVIDIA and ROCm gates. Source:
  `.github/workflows/unit-nvidia.yml` and `.github/workflows/unit-rocm.yml`.
- Decision: no repository `.agents/standards/` or
  `.agents/common-mistakes/` files exist. Source: filesystem inspection.
- Assumption: primary runtime is ROCm Torch `2.11.0+rocm7.2` on RX 7900 XTX
  `gfx1100`; this must be refreshed before every performance claim. Source:
  current runtime probe and prior plan artifacts.
- Assumption: a second ROCm target may be unavailable locally. Compile/import/
  contract evidence must be used for unavailable targets, never represented as
  served or performance proof. Source: current hardware inventory and target matrix.
- Assumption: current dirty state is clean after `ab5f46c`; future increments
  preserve unrelated user changes and never edit sibling Ollama/tinygrad trees.
  Source: `git status --short` and repository scope rules.

## Risks accepted

- **Model-load time hides decode regressions.** Mitigation: add a resident
  worker or reusable model-load artifact to the benchmark while retaining a
  fresh-process validation lane; never call startup time decode throughput.
- **ROCm fatal HIP errors kill workers.** Mitigation: probe graph and candidate
  kernels in child processes, persist identity-complete results, and retain
  eager/generic fallback.
- **Shared HIP edits regress CUDA.** Mitigation: compile current CUDA sources
  in NVIDIA CI, run CUDA-safe unit tests, and require explicit `#if`/runtime
  branch review for every shared change.
- **A target candidate helps one SKU but harms another.** Mitigation: exact
  target records, no family-wide enablement, cross-target generic tests, and
  candidate status never higher than measured evidence.
- **Profiler overlap misattributes work.** Mitigation: correlate API wall,
  device events, stream markers, and route counters; do not sum overlapping
  stage times.
- **Triton/library behavior changes across ROCm versions.** Mitigation: record
  Torch, HIP, ROCm, Triton, compiler, BLAS, and cache identity in every gate;
  use generic fallback on unverified combinations.

## Increment DAG

- Inc 1 — Baseline instrumentation and causal timeline (M) — depends on: none —
  unblocks: 1B
- Inc 1B — Startup recovery and identity-complete baseline (M) — depends on: 1 —
  unblocks: 2, 4, 5
- Inc 2 — Unified ROCm capability and dispatch contract (M) — depends on: 1B —
  unblocks: 3, 4, 5, 6
- Inc 3 — Generic ROCm execution and CUDA integrity (M) — depends on: 2 —
  unblocks: 6, 7
- Inc 4 — Portable operation-path optimization (L) — depends on: 1B, 2 —
  unblocks: 6
- Inc 5 — Exact-target kernel candidates (L) — depends on: 1B, 2, 4 —
  unblocks: 6
- Inc 6 — Memory, graph, and host-cadence integration (L) — depends on: 3, 4,
  5 — unblocks: 7
- Inc 7 — Cross-ROCm/CUDA CI and promotion gate (M) — depends on: 2, 3, 6 —
  unblocks: 8
- Inc 8 — Final identity-complete benchmark and handoff (S) — depends on: 7 —
  unblocks: none

## Increments

### Inc 1 — Baseline instrumentation and causal timeline (M)

**Status:** done

**Depends on:** none
**Unblocks:** 1B
**Done criteria:** Benchmark/replay schema and one causal trace are implemented
and validated on hardware-free fixtures. Trace separates model load, prefill,
decode GPU work, offload movement, sampler, and host/API cadence. A startup
blocker is an incomplete increment, not permission to continue performance work.

#### Files to touch

##### `benchmarks/bench_decode_moe.py`
- What changes: add or complete repeat/resident-worker controls and JSONL
  identity fields without changing timing semantics.
- Functions: `parse_args`, `serve_cmd`, `run_one`, `main`.
- Data shapes: bs=1, exact prompt and continuation IDs, fixed decode window,
  explicit sampling, `mtp=false`, graph/cache/KV/backend fields.
- Integration points: `ft serve`, `/health`, `/v1/models`, streamed
  `/v1/chat/completions`, `/v1/stats`.
- Error paths: incomplete usage, missing token events, server startup failure,
  model/hash mismatch, or output/route evidence mismatch produces incomplete
  artifact and nonzero status.

##### `benchmarks/bench_decode_replay.py`, `benchmarks/check_decode_gate.py`, `benchmarks/bench_rocm_matrix.py`
- What changes: ensure one schema carries prompt/continuation identity, route
  digest, oracle ID, finite-logit result, fallback count, and runtime identity.
- Functions: `build_replay_record`, `validate_replay_record`,
  `evaluate_gate`, `validate_manifest`.
- Error paths: sampled/greedy rows remain correctness-only; absent oracle,
  route mismatch, incomplete rows, or fewer than three runs rejects promotion.

##### `benchmarks/profile_decode_rocm.py`, `python/freetoken/utils/step_profiler.py`, `python/freetoken/engine/engine.py`, `python/freetoken/scheduler/scheduler.py`
- What changes: correlate low-overhead phase markers with device timing and
  request lifecycle; preserve plain serving when profiling is disabled.
- Functions/data: named phases for router, attention, gate/up, down, cache
  gather/fetch, GDN/norms, sampler, token D2H, scheduler, and HTTP.
- Error paths: missing trace, missing route, or overlapping/incomplete timing
  marks report incomplete evidence rather than zero cost.

#### Edge cases

- Startup JIT/cache lock or serial expert-bank construction exceeds timeout.
- Graph gate changes active path between baseline runs.
- Dynamic sampling changes route/output identity.
- Ollama or another process owns GPU; do not launch competing work.

#### Verification

- Run hardware-free schema tests first. This increment does not claim tok/s.
- Do not unblock performance increments until Inc 1B produces a valid baseline;
  blocker artifacts must use nonzero status and `status=incomplete`.
- Run profiler once with full trace in `/tmp`; summarize markers and top kernels
  into plan artifacts only after identity validation.
- Done only when instrumentation and incomplete-artifact behavior are validated.

### Inc 1B — Startup recovery and identity-complete baseline (M)

**Status:** blocked — exact checkpoint and ROCm Torch runtime unavailable in
this checkout; no throughput claim emitted.

**Depends on:** 1
**Unblocks:** 2, 4, 5
**Done criteria:** A valid generic ROCm serving baseline exists for the exact
primary checkpoint, or this increment remains blocked. Startup evidence alone
never unblocks optimization or creates a throughput number.

#### Files to touch

##### `benchmarks/bench_decode_moe.py`, `benchmarks/bench_decode_replay.py`, `benchmarks/bench_rocm_matrix.py`
- What changes: add an explicit resident-worker protocol while retaining the
  fresh-process lane. Resident mode starts one server, records startup once,
  performs one discarded warm request, then records exactly three or more
  measured requests with identical prompt, decode count, cache, graph, KV,
  sampling, and route policy. Fresh-process mode remains the final validation
  lane.
- Error paths: startup/JIT/model-bank failure writes an incomplete artifact and
  exits nonzero; it never emits `median_tok_s`.

##### `.plans/unified-rocm-opt/artifacts/` (local raw artifacts only)
- What changes: record checkpoint SHA-256, runtime identity, server mode,
  startup duration, warmup count, measured rows, cache reset/reuse policy, and
  route/fallback counters.

#### Edge cases

- Resident mode accidentally reuses prompt/KV state or changes cache residency.
- Expert-bank reuse hides a required initialization step.
- Startup exceeds timeout despite valid model and runtime identity.

#### Verification

- Run exact checkpoint/hash, greedy correctness, and teacher-forced replay lanes.
- Require at least three positive measured rows after warmup and exact
  completion count. Compare only rows sharing all gate identity fields.
- Record historical numbers as context only; current baseline must be local to
  this checkout, runtime, checkpoint, and commit.

### Inc 2 — Unified ROCm capability and dispatch contract (M)

**Depends on:** 1B
**Unblocks:** 3, 4, 5, 6
**Done criteria:** All ROCm selection points consume one shared runtime/target
  identity and route report; generic fallback is default; unsupported target,
  ABI, quant, shape, and phase combinations fail or fall back according to
  mode, with CUDA path tests unchanged.

#### Files to touch

##### `python/freetoken/utils/arch.py`, `python/freetoken/utils/graph_gate.py`, `python/freetoken/kernel/backend.py`, `python/freetoken/kernel/utils.py`, `python/freetoken/kernel/gguf.py`, `python/freetoken/kernel/_toolchain.py`, `python/freetoken/engine/engine.py`, `python/freetoken/server/args.py`, `python/freetoken/server/launch.py`, `python/freetoken/kernel/batch_memcpy.py`, `python/freetoken/moe/offload_kernels.py`, `python/freetoken/kvcache/cache_status.py`
- What changes: expose one immutable runtime identity containing backend,
  normalized target, device, Torch/HIP/ROCm/toolchain versions, and relevant
  policy environment. All runtime ROCm selectors listed here consume it.
  Standalone `setup.py` and `freetoken-kernel-cache/build_backend.py` use a
  separate build identity with the same backend/target normalization; build
  identity must not be treated as served hardware evidence. Reuse runtime
  identity for graph-cache keys and dispatch diagnostics.
- Functions: `device_kind`, `get_rocm_gfx_arch`, `rocm_arch_capability`,
  `rocm_candidate_capability`, `_cache_identity`.
- Data shape: `RocmRuntimeIdentity` fields are strings/booleans only and JSON
  serializable; unknown fields remain explicit `None`/`unknown`.
- Error paths: missing runtime/device returns generic unavailable status; never
  infer an exact target from family name or compile flags.

##### `python/freetoken/kernel/gguf.py`, `python/freetoken/layers/gguf.py`, `python/freetoken/layers/moe.py`, `python/freetoken/moe/fused_gguf.py`
- What changes: centralize route request/decision metadata and ensure dense,
  gate/up, and down dispatch use the same contract.
- Functions: `gguf_dispatch`, `_candidate_dispatch_contract`,
  `fused_mul_mat_gguf`, `fused_experts_gguf`.
- Data shape: report backend, target, phase, operation, quant type, rows, cols,
  token count, route, fallback route, ABI, and rejection reason. Preserve
  existing CLI/environment names: `legacy` and `auto` map to generic;
  `rdna3_mmid`, `rdna3_mmvdq`, and `grouped_mmq` are named requests.
- Error paths: forced candidate raises before extension call; `auto` selects
  generic route with a reason counter; CUDA never imports/probes ROCm candidate.

##### `tests/utils/test_rocm_arch.py`, `tests/utils/test_graph_gate.py`, `tests/kernels/test_gguf_rocm.py`
- What changes: cover identity invalidation, generic fallback, exact candidate
  rejection, unknown graph status, explicit graph disable, and CUDA isolation.
- Done: each route decision has a hardware-free negative test and target matrix
  tests do not claim served support from compile-only records.

#### Edge cases

- Multi-GPU/TP parent cannot probe each worker target.
- `FREETOKEN_ROCM_ARCH` lists multiple targets while runtime sees one device.
- Stale graph cache after Torch, HIP, driver, BLAS, or target change.
- CUDA build has ROCm environment variables installed.

#### Verification

- Run focused dispatch/graph/arch tests on CPU-only import path and ROCm venv.
- Assert CUDA branch does not call graph gate, ROCm candidate loader, or HIP
  toolchain resolver, even when ROCm environment variables are present.
- Review all route reports for stable schema and explicit fallback reasons.

### Inc 3 — Generic ROCm execution and CUDA integrity (M)

**Depends on:** 2
**Unblocks:** 6, 7
**Done criteria:** Generic ROCm path has hardware-free contract coverage and
compile/import coverage for every declared quant/operation pair. Served runtime
status is claimed only for physically available targets; compile-only targets
remain compile-only. HIP toolchain/build/runtime errors stay isolated; CUDA
compile/import tests show no semantic or dependency regression.

#### Files to touch

##### `python/freetoken/kernel/_toolchain.py`, `freetoken-kernel-cache/build_backend.py`, `setup.py`, `python/freetoken/kernel/utils.py`
- What changes: consolidate HIP compiler discovery, Torch/ROCm compatibility
  checks, cache tags, and flags; keep CUDA `nvcc`/cudart behavior separate.
- Functions: toolchain checks, runtime library path resolution, JIT/AOT flag
  builders.
- Data shape: backend-specific flag lists and cache namespace include backend,
  target, Torch/ROCm identity, and source ABI.
- Error paths: missing `hipcc`/SDK fails with actionable ROCm error; CUDA build
  never requires ROCm; ROCm AOT does not bypass Torch/toolchain validation.

##### `python/freetoken/kernel/csrc/include/freetoken/utils.cuh`, `python/freetoken/kernel/csrc/jit/batch_memcpy.cuh`, `python/freetoken/kernel/batch_memcpy.py`
- What changes: retain shared wrapper contracts while using HIP stream-ordered
  copies/launches only under HIP; keep CUDA batch-copy ABI unchanged.
- Functions/data: explicit non-legacy stream, pointer arrays, sizes, async
  metadata copies, kernel launch, async frees.
- Error paths: reject null/legacy stream, invalid device kind, unsupported
  runtime API, or compile mode with clear errors; generic copy loop remains
  available when fused copy is unavailable.

##### `.github/workflows/unit-nvidia.yml`, `.github/workflows/unit-rocm.yml`, `tests/`
- What changes: keep separate build/test jobs and add compile/import checks for
  shared sources plus generic ROCm contracts.
- Done: NVIDIA job does not install or execute ROCm-only candidates; ROCm job
  runs target-matrix contracts and exact candidate tests only on matching labels.

#### Edge cases

- ROCm Torch has CUDA-named PyTorch APIs but no CUDA libraries.
- HIP async allocation/copy ordering differs across SDK versions.
- CUDA 13 eight-argument batch-copy ABI must remain byte-compatible.
- Host has ROCm SDK but no CUDA toolkit, or vice versa.

#### Verification

- Run ROCm compile/import/unit checks in pinned ROCm venv.
- Run NVIDIA workflow-equivalent static/import/build checks where local nvcc is
  absent; rely on CI compile gate for actual CUDA object generation.
- Run `git diff --check`, Python compilation, and targeted C++ preprocessor/
  compile checks for each available toolchain.

### Inc 4 — Portable operation-path optimization (L)

**Depends on:** 1B, 2
**Unblocks:** 6
**Done criteria:** At least one profiler-confirmed generic ROCm bottleneck has a
  measured improvement or is closed as below materiality, with finite-logit,
  route/token, and CUDA integrity evidence.

**Scope gate:** Select exactly one branch from the Inc-1/1B trace: router,
GGUF/MMVQ, attention, or launch policy. Do not edit unrelated branches in the
same increment. Each branch gets its own before/after artifact and remains
generic across targets.

#### Files to touch

##### `python/freetoken/moe/fused.py`, `python/freetoken/kernel/triton/moe_router.py`
- What changes: select the portable in-repo Triton router on ROCm when its
  capability probe passes; retain pure-Torch fallback on import/compile/runtime
  failure.
- Functions/data: `fused_topk(hidden_states, gating_output, topk, renormalize,
  num_token_non_padded)` returns fp32 weights `[M, K]` and int32 IDs `[M, K]`.
- Error paths: Triton failure downgrades once with reason; CUDA existing router
  priority remains unchanged; tie behavior is recorded.

##### `python/freetoken/kernel/csrc/gguf/gguf_kernel.cu`, `python/freetoken/kernel/csrc/gguf/mmvq*.cuh`, `python/freetoken/attention/triton.py`, `python/freetoken/layers/gguf.py`
- What changes: test generic ROCm library/launch policies and operation
  dispatch selected by the Inc-1/1B trace; change only measured knobs.
- Data shapes: Q4_K/Q5_K/Q6_K/Q8_0 block geometry; Qwen hidden 2048,
  intermediate 512, top-k 8; batch-1 decode and matching prefill cases.
- Error paths: unsupported quant/shape uses generic reference or explicit
  unsupported result; no candidate assumption enters generic route.

##### `tests/moe/test_fused_moe.py`, `tests/kernels/test_gguf_rocm.py`, `tests/attention/test_torch_backend.py`
- What changes: compare router, dequant, MMVQ, and attention against independent
  references; cover non-power-of-two top-k, masked rows, zero rows, and finite
  outputs.

#### Edge cases

- Router ties alter expert ordering without changing mathematically equivalent
  output; flag token/route hash changes.
- Triton compile happens during graph capture or selects unsupported warp size.
- K-quant row padding and mixed gate/up/down quant types disagree.
- Dense GEMM backend reports a policy different from requested environment.

#### Verification

- Profile before/after in fresh processes; require materiality threshold set by
  Inc-1/1B trace and no regression outside exact target/shape.
- Run generic ROCm tests on every available target; candidate code must not be
  imported by generic tests.
- Re-run CUDA-safe tests and inspect shared-source diff for unchanged CUDA path.

### Inc 5 — Exact-target kernel candidates (L)

**Depends on:** 1B, 2, 4
**Unblocks:** 6
**Done criteria:** Each candidate either improves the exact target in an
  identity-complete server comparison or is closed with evidence; unsupported
  ROCm targets remain on generic path and forced candidate mode fails loudly.

#### Files to touch

##### `python/freetoken/kernel/csrc/gguf/gguf_moe_gfx1100.cu`, `python/freetoken/kernel/csrc/gguf/moe_vec_gfx1100.cuh`, `python/freetoken/kernel/csrc/gguf/moe_vec_gfx1100_hip.cuh`
- What changes: tune only measured gfx1100 MMVQ/MoE launch/layout candidates;
  keep source units separate from portable GGUF kernels.
- Data shapes: exact Qwen GGUF gate/up and down bank layouts, hidden 2048,
  intermediate 512, top-k 8, decode-only candidate contract.
- Integration points: `gguf_dispatch` candidate ABI and self-test loader.
- Error paths: compiler/ABI/self-test/shape mismatch leaves candidate
  unavailable; forced mode raises; `auto` routes generic.

##### `python/freetoken/kernel/gguf.py`, `python/freetoken/utils/arch.py`, `tests/kernels/test_gguf_rocm.py`
- What changes: register source version, binary ABI, supported quant/shape,
  target, and self-test result; never mark performance from configuration.
- Done: `gfx1100` exact positive tests, every other declared target
  (`gfx1101`, `gfx1102`, `gfx1103`, `gfx1150`, `gfx1151`, `gfx1200`,
  `gfx1201`) negative candidate tests, generic fallback tests, and CUDA
  no-probe tests. Candidate predicate rejects phase, canonical shape, mixed
  quant, stride, and ID-space mismatches before extension load.

##### `benchmarks/bench_gguf_moe_kernels.py`
- What changes: measure direct kernel stages only as diagnostic, with packed
  byte identity, output hash, route identity, compiler/cache identity, and
  fresh process.
- Error paths: source/build hash missing or reference mismatch invalidates row.

#### Edge cases

- Candidate source accidentally compiles into generic module.
- `sdot4` intrinsic exists but generated ISA/occupancy loses to scalar path.
- Q4_K gate/up and Q8_0 down use different row strides.
- Candidate works for decode but not prefill; keep phase gate explicit.

#### Verification

- Compile/import candidate on exact target in isolated cache namespace.
- Run numerical/reference self-tests before any server test.
- Run three or more candidate/control server runs only if self-tests pass;
  preserve generic fallback and output hash evidence.

### Inc 6 — Memory, graph, and host-cadence integration (L)

**Depends on:** 3, 4, 5
**Unblocks:** 7
**Done criteria:** Trace-selected memory, graph, or host-cadence change reduces
  measured base decode wall time without changing identity or forcing unsafe
  target behavior; otherwise branch closes with measured no-op evidence.

**Scope gate:** Select exactly one branch from the Inc-1/1B trace: cache movement,
graph policy, or host cadence. Do not combine branches in one comparison or
attribute a mixed change to one stage.

#### Files to touch

##### `python/freetoken/moe/offload_cache.py`, `python/freetoken/moe/offload_kernels.py`, `python/freetoken/kernel/csrc/jit/batch_memcpy.cuh`
- What changes: optimize only confirmed miss/fetch/slot-remap cost; preserve
  event ordering, hit/miss counters, and generic stream-safe copy fallback.
- Data shapes: cache slot IDs, `[tokens, topk]` route IDs, per-bank packed rows,
  batch-1 and observed miss rates.
- Error paths: cache boundary/thrash, unavailable copy extension, stale events,
  or unsupported stream uses safe existing path with explicit reason.

##### `python/freetoken/utils/graph_gate.py`, `python/freetoken/server/launch.py`, `python/freetoken/engine/graph.py`, `python/freetoken/engine/engine.py`
- What changes: test graph variants only when target-specific worker probe,
  BLAS identity, capture/replay correctness, and cache identity all pass.
- Error paths: unknown/stale/fatal probe, TP/explicit-GPU ambiguity, or dynamic
  shape returns eager decode; CUDA graph behavior remains unchanged.

##### `python/freetoken/engine/engine.py`, `python/freetoken/scheduler/scheduler.py`, `python/freetoken/benchmark/perf.py`
- What changes: reduce avoidable host/token handoff/allocation cadence only
  after marker evidence; preserve two-slot pinned D2H and scheduler semantics.
- Data shape: one decode token per iteration, finite logits, exact completion
  count, event/fence ownership recorded.

#### Edge cases

- Graph replay captures a late JIT compile or allocator operation.
- Offload copy and graph stream reuse races after cache rebuild.
- Reducing synchronization changes token order or route digest.
- A faster microbench path increases API wall time.

#### Verification

- Run graph-on only after probe pass; run eager control in every comparison.
- Validate finite logits, token IDs, route digest, completion count, fallback
  count, and no hidden candidate fallback.
- Require end-to-end median improvement beyond measured noise; otherwise record
  no-op and keep generic path.

### Inc 7 — Cross-ROCm/CUDA CI and promotion gate (M)

**Depends on:** 2, 3, 6
**Unblocks:** 8
**Done criteria:** CI and local gates exercise generic ROCm contracts on every
  labeled target, exact candidates only on matching hardware, and CUDA shared
  sources compile/import without ROCm dependency; promotion tool rejects
  incomplete or incomparable evidence.

#### Files to touch

##### `.github/workflows/unit-rocm.yml`, `.github/workflows/unit-nvidia.yml`, `pyproject.toml`, `freetoken-kernel-cache/pyproject.toml`
- What changes: pin supported runtime/toolchain matrix, separate generic and
  candidate labels, and compile both shared and target-specific sources under
  correct guards. ROCm runtime jobs must compare requested target to actual
  `torch.cuda.get_device_properties(...).gcnArchName`; environment override
  alone is never hardware identity. Add separate compile-only jobs for targets
  without matching runners.
- Error paths: target mismatch fails before candidate tests; missing hardware
  labels produce explicit compile-only/skipped status and never pass as served
  or performance evidence.

##### `tests/utils/`, `tests/kernels/`, `tests/moe/`, `tests/attention/`, `tests/engine/`
- What changes: add contract tests for all shared fallback/error paths and
  target-negative cases; preserve existing CUDA tests.

##### `docs/reproducibility.md`, `benchmarks/README.md`
- What changes: document one benchmark command, artifact schema, runtime
  identity, target matrix, cache policy, and `NO-PROMOTION` requirements.

#### Edge cases

- CI ROCm runtime version differs from local plan identity.
- NVIDIA runner sees ROCm variables or imports HIP-only module.
- Candidate artifact is reused after source/compiler/driver change.
- Benchmark has sampled output but no independent replay oracle.

#### Verification

- Run local hardware-free suite and workflow YAML/schema checks.
- Run target-labeled ROCm compile/import/contract checks where available.
- Assert served target job has matching runner label, runtime `gcnArchName`, and
  requested target; cross-compile jobs cannot promote.
- Run CUDA workflow-equivalent checks and inspect dependency closure.
- Run `check_decode_gate.py` against intentionally invalid manifests and confirm
  explicit rejection reasons.

### Inc 8 — Final identity-complete benchmark and handoff (S)

**Depends on:** 7
**Unblocks:** none
**Done criteria:** Fresh baseline/control/candidate artifacts contain at least
  three valid runs per side, exact shared identity, independent replay oracle,
  route/token parity, finite logits, exact completion count, and raw/median/
  spread numbers; result is `PROMOTE` only if all gates pass, else `NO-PROMOTION`.

#### Files to touch

##### `.plans/unified-rocm-opt/artifacts/` (local raw artifacts only)
- What changes: store JSONL manifests, logs, environment snapshot, source and
  binary hashes, profiler summaries, and final gate output.
- Error paths: incomplete artifacts remain marked incomplete; no synthetic or
  self-generated number is promoted.

##### `.plans/unified-rocm-opt/plan.md`, `docs/reproducibility.md`
- What changes: record final accepted route, target matrix, benchmark numbers,
  rejected candidates, rollback path, and known blockers.

#### Edge cases

- Model file changes between runs.
- Graph/cache/KV/runtime differs between control and candidate.
- Candidate speed win falls within noise or changes greedy hash.
- Only one ROCm target is physically available.

#### Verification

- Run same-model, same-file, same-prompt, same-token-count, non-MTP A/B in
  fresh processes or validated resident-worker protocol.
- Run teacher-forced replay gate and separate greedy serving correctness.
- Report median, raw runs, spread, TTFT, VRAM, route, graph, cache, runtime,
  compiler, kernel/ABI, and commit; do not report copy-only metrics as tok/s.

## Cross-cutting verification

- Every increment runs focused tests before wider tests; no increment relies on
  an unrecorded local state.
- Shared changes require both `is_rocm()==True` and `is_rocm()==False` tests.
- Generic ROCm path is the default in all unsupported, unknown, or failed
  candidate cases. Forced candidate mode has no fallback.
- Candidate code is target/operation/quant/shape/ABI/self-test gated and never
  imported by CUDA paths.
- Benchmark artifacts preserve model hash, tokenizer/prompt/continuation
  identity, route digest, oracle ID, quant/KV/graph/cache/runtime identity,
  completion count, finite-logit status, fallback count, and source/binary hash.
- Inc 1B startup blocker must be solved before Inc 2/4/5 performance work or
  any end-to-end tok/s claim. Use resident worker or complete expert-bank build,
  then rerun baseline and all candidate comparisons.
- Every optimization branch has a reversible selector/flag, explicit default
  preservation, failure reason, and rollback artifact. Candidate rollback is
  automatic on target/shape/ABI/self-test failure; generic remains default.
- Final gate vocabulary: `PROMOTE` only after correctness, serving, CUDA
  isolation, cross-ROCm contracts, and teacher-forced performance pass;
  otherwise `NO-PROMOTION` with generic fallback retained.

## Standards / common-mistakes referenced

- `CONTRIBUTING.md` — use `uv`, add tests for fixes, include reproducible
  hardware/checkpoint/command evidence, preserve Conventional Commit style.
- `../FreeToken/.plans/rocm-parity-next/plan.md` (historical sibling checkout;
  context only) — reuse causal timeline, worker-start graph
  policy, generic default, and explicit candidate rollback.
- `../FreeToken/.plans/rocm-perf-parity/plan.md` (historical sibling checkout;
  context only) — reuse stage attribution, graph variants,
  median-based comparison, and CUDA integrity posture.
- `../FreeToken/.plans/qwen-moe-speed/plan.md` (historical sibling checkout;
  context only) — reuse base-speed scope, exact workload
  identity, known-negative branch closure, and no MTP policy.
- `../FreeToken/.plans/rocm-ollama-gap/plan.md` (historical sibling checkout;
  context only) — reuse replay/oracle/schema requirements,
  resident-phase accounting, and incomplete-evidence handling.
- `.agents/learnings/` — no repository learning files were present in this
  checkout; memory-derived historical results remain time-sensitive and must be
  refreshed before use.

## Open questions (CONSIDER from review)

- Which second ROCm target is physically available for cross-target compile and
  runtime evidence?
- Can current expert-bank serial build be made resident/reusable without
  changing model semantics or benchmark identity?
- Is a same-file Ollama/llama.cpp reference available for directional gap
  analysis, or should this plan remain FreeToken-only?
- Which profiler is accepted in CI: `rocprofv3`, Torch profiler, or both?
- Should a successful ROCm graph variant be enabled only per exact Torch/HIP
  identity, or also require a driver/toolchain build fingerprint?

## Out of scope

- MTP, speculative decoding, verify batches, or MTP-derived speed claims.
- Editing sibling Ollama, llama.cpp, or tinygrad repositories.
- Replacing GGUF Q4_K/Q8_0 with NVFP4/Marlin or changing model quality.
- Making gfx1100 candidate behavior generic to gfx11 or gfx12 by family label.
- Treating compile success, IPC rate, synthetic copy bandwidth, HTTP 200, or
  self-generated output as decode proof.
- Repeating known-negative MMV_Y, cache-size, or rotated-sdot4 experiments
  without a new compiler, runtime, shape, or measured-causal hypothesis.
- PLE/SSD residency redesign, Windows support, TP>1 performance, or unrelated
  model architectures.
- Remote push, PR updates, merge, or release publication.
