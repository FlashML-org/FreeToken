# ROCm PR-217 Consolidation into PR-132

## Objective

Consolidate relevant work from original PR #217 into help PR #132. Preserve generic ROCm behavior, gate hardware-specific paths by exact capability, retain safe fallbacks, prove CUDA isolation, cross-check other ROCm targets, and report reproducible performance evidence.

This plan covers source reconciliation, implementation, tests, benchmark tooling, and validation. It does not authorize push, PR updates, or remote merge.

## Current context

- Working branch: `help/pr-132`.
- Current local tip contains one local ROCm hardening commit after PR-132 work.
- PR #217 contains a larger AMD/ROCm stack with overlapping but independently evolved runtime, kernel, benchmark, and documentation changes.
- Target branch already has ROCm foundation contracts and CUDA isolation work. Consolidation must preserve those contracts while importing behavior, not blindly replaying commits.
- Current generic ROCm capability records report compile-only status. Exact candidate support is narrower and must not become default dispatch.
- Existing benchmark helpers contain `freetoken-rocm-manifest-v1`; PR #217 also contains `freetoken-replay-manifest-v1`. These schemas cannot be silently conflated.

## Success criteria

1. Relevant PR-217 runtime behavior exists in one coherent PR-132 implementation.
2. Generic ROCm dispatch remains usable on supported ROCm targets without candidate-only assumptions.
3. Hardware-specific kernels require exact target, operation, quantization, shape, ABI, and self-test gates.
4. Forced candidate mode fails loudly when capability is absent or validation fails; auto mode falls back to generic ROCm.
5. CUDA imports, dispatch, build, tests, and runtime behavior remain unaffected.
6. Other ROCm targets receive compile/import/contract coverage and no accidental exact-target routing.
7. Performance promotion uses identity-complete teacher-forced replay with independent route/token evidence. Sampled or greedy lanes cannot pass speed promotion.
8. Final result is `PROMOTE` only with passing correctness, serving, and performance gates; otherwise report `NO-PROMOTION` with usable generic fallback.

## Architectural decisions

### Generic versus candidate capability

Keep generic ROCm support and candidate support as separate records. Do not infer candidate support from architecture family or compile success.

```text
RocmArchCapability:
  target
  family
  wave_size
  generic_gguf_types
  graph_features
  status: compile-only | served | correctness | performance

CandidateCapability:
  target
  operations
  quant_types
  supported_shapes
  abi
  source_version
  self_test
  status: unavailable | compile-only | correctness | performance
```

`auto` may select generic ROCm only when generic capability is `served` or better. Candidate route requires exact target match, complete operation/quant/shape/ABI match, passing self-test, and explicit candidate opt-in. Candidate `performance` status is evidence-driven, not configured by default.

### Dispatch policy

- `auto`: generic ROCm first; candidate only when exact capability record and promotion gate permit it.
- `rocm_generic`: force generic ROCm fallback; candidate code is not probed.
- `rocm_candidate`: exact-target candidate only; fail before execution on unsupported target, shape, quant, ABI, or failed self-test.
- `cuda`: existing CUDA route; no ROCm probe or ROCm candidate import may alter it.
- Unsupported or unavailable routes fail with target, requested route, reason, and fallback guidance.

### Validation lanes

Keep correctness lanes separate from speed lanes:

- `teacher_forced_replay`: required for performance promotion; replay continuation token IDs from an independent oracle and compare logits/route evidence.
- `greedy_decode`: served-path correctness and completion checks only.
- `sampled_decode`: parser/serving smoke only; never a speed-promotion input.

Performance comparison must use candidate and baseline records with the same model hash, prompt/continuation identity, token count, quantization, KV settings, graph mode, runtime identity, and route digest. A self-generated candidate stream is not an independent oracle.

### Merge strategy

Use semantic porting into `help/pr-132`, grouped by file contract and dependency order. Preserve one coherent final PR-132 history. Do not merge remote branches or push from agent session. Keep a disposable worktree and a recorded rollback ref so conflicts or failed validation can be discarded without touching unrelated user work.

## Assumptions and decisions

- PR #217 means `samuelishida:feat/amd-rocm-gfx1100-support` and PR #132 means `zihaomu:feat/rocm-rdna3-rdna4-foundation`. Source: repository remote refs and GitHub PR pages inspected during planning.
- Current working branch is intended integration target. Source: `git branch --show-current` and repository status.
- Non-MTP base decode is scope. Source: project performance guidance and existing reproducibility policy.
- Matching GGUF Q4_K/Q8_0 lanes are preferred; NVIDIA-only NVFP4/Marlin comparisons are excluded. Source: project performance guidance.
- Hardware-specific candidate behavior remains opt-in until independent correctness and performance gates pass. Source: current `gguf.py` candidate dispatch and target capability records.
- Local benchmark execution may be unavailable or unsafe if another process owns GPU. Source: workspace operating constraints. Record unavailable hardware evidence instead of claiming it.

## Execution preflight (not an increment)

1. Read `CONTRIBUTING.md`, verify branch, worktree status, remotes, and active GPU processes.
2. Create disposable integration worktree from current `help/pr-132` tip. Record `HEAD`, `upstream/main`, `upstream/pr-132`, `upstream/pr-217`, merge-bases, Python/uv versions, ROCm/CUDA versions, and GPU visibility.
3. Build a local source map in `/tmp/freetoken-pr132-source-map.tsv` or another explicitly temporary location. Do not add source-map artifacts under `.plans`; internal planning artifacts are out of scope for product changes.
4. Classify every PR-217 path as `port`, `adapt`, `replace`, or `exclude` before editing. Every excluded production path needs a reason in the plan execution log.
5. Port by contract order below. For each conflict, record both sides, selected behavior, test proving it, and rollback point. Do not resolve conflicts by favoring one branch globally.
6. Run focused checks after every increment. Stop implementation if a prerequisite contract is not proven.
7. Keep final branch changes reviewable and squash-ready. Human owner performs any final commit, PR update, merge, or push.

## Dependency graph

```text
Inc 1 ROCm capability contract (M)
  -> Inc 2 GGUF dispatch and fallback policy (M)
  -> Inc 3 HIP/JIT and extension isolation (M)
  -> Inc 4 Portable runtime fallback seams (M)
       -> Inc 5 Generic GGUF/Qwen runtime (L)
       -> Inc 6 Guarded native GGUF candidates (L)
Inc 2 -> Inc 3, Inc 4, Inc 5, Inc 6
Inc 3 -> Inc 4, Inc 5, Inc 6, Inc 7
Inc 4 -> Inc 5, Inc 6, Inc 7
Inc 5 -> Inc 7
Inc 6 -> Inc 7
Inc 7 Evidence tooling and platform CI (M)
```

Each increment below is independently shippable code plus tests. Final A/B validation is cross-cutting work after Inc 7, not an artifact-only increment.

## Inc 1 - ROCm capability contract (M)

**Status:** done

**Depends on:** none

**Unblocks:** Inc 2, Inc 3, Inc 4

### Scope

Define target detection and capability data without claiming runtime support from compile success. Keep generic ROCm support independent from exact native candidate support.

### Files to touch

- `python/freetoken/utils/arch.py`
- `tests/utils/test_rocm_arch.py`
- Existing capability/config import seams only when required by tests; do not move dispatch here.

### Implementation

- Normalize target strings and preserve raw device identity for diagnostics.
- Add explicit generic fields and candidate fields described above.
- Keep status transitions monotonic: compile-only cannot satisfy served/correctness/performance gates.
- Represent unknown targets as generic-eligible only when generic runtime contract says so; never map unknown target to candidate target.
- Expose deterministic capability query usable without allocating model tensors.

### Edge cases

- Missing ROCm runtime or unavailable device query.
- `gfx1100`, `gfx1101`, `gfx1102`, `gfx1103`, `gfx1150`, `gfx1151`, `gfx1200`, and `gfx1201` spelling/normalization.
- Multi-device visibility where selected device differs from first visible device.
- CUDA process importing module on host without ROCm.
- Compile-only target accidentally reaching `auto` served route.

### Rollback and observability

Rollback is one commit reverting capability schema and tests. Emit selected target, generic status, candidate status, and reason for every route query. Do not log device memory contents or credentials.

### Verification

- Unit-test normalization, unknown target, missing runtime, and exact candidate match.
- Run focused `uv run pytest tests/utils/test_rocm_arch.py`.
- Run import/compile checks for CUDA-safe import path.
- Done only when capability status cannot be mistaken for runtime validation.

## Inc 2 - GGUF dispatch and fallback policy (M)

**Status:** done

**Depends on:** Inc 1

**Unblocks:** Inc 3, Inc 4, Inc 5, Inc 6

### Scope

Centralize generic ROCm fallback and exact candidate gating across GGUF layers, fused MoE, and kernel dispatch. Candidate implementation remains in Inc 6.

### Files to touch

- `python/freetoken/kernel/gguf.py`
- `python/freetoken/layers/gguf.py`
- `python/freetoken/moe/fused_gguf.py`
- `tests/kernels/test_gguf_rocm.py`
- `tests/kernels/test_gguf_linear.py`
- `tests/kernels/test_gguf_moe.py`

### Implementation

- Define route selection result with route, target, reason, capability status, and fallback route.
- Make generic ROCm route usable for all supported generic quant types.
- Require exact candidate capability for candidate route; no substring or family-only match.
- Ensure `auto` falls back to generic route on candidate absence or self-test failure.
- Ensure forced candidate mode fails loudly and never silently falls back.
- Keep CUDA route selection unchanged and avoid importing candidate modules on CUDA.

### Edge cases

- Candidate module installed but wrong target.
- Candidate supports one quant type while request uses another.
- Candidate self-test unavailable, stale, or returns non-finite output.
- Mixed batch shapes where only part of request matches candidate.
- Generic fallback unavailable: return actionable error before partial execution.

### Rollback and observability

Rollback route policy independently from capability schema. Route result must record requested route, selected route, target, quant, shape class, candidate version/ABI, fallback reason, and graph mode. Add a disable switch for candidate probing.

### Verification

- Unit-test auto fallback, forced fail-loud, exact target/quant/shape rejection, and CUDA non-probe behavior.
- Run focused kernel tests listed above.
- Static-check no unconditional candidate import or CUDA path change.
- Done when every route has deterministic behavior and test evidence.

## Inc 3 - HIP/JIT and extension isolation (M)

**Status:** done

**Depends on:** Inc 1, Inc 2

**Unblocks:** Inc 4, Inc 5, Inc 6, Inc 7

### Scope

Port PR-217 HIP/JIT/build seams while isolating ROCm-only compile flags, extension loading, launch kwargs, memcpy, pinned memory, and attention backend behavior from CUDA.

### Files to touch

- `setup.py`
- `pyproject.toml`
- `freetoken-kernel-cache/build_backend.py`
- `python/freetoken/kernel/_toolchain.py`
- `python/freetoken/kernel/utils.py`
- `python/freetoken/kernel/batch_memcpy.py`
- `python/freetoken/kernel/pinned_tensor.cpp`
- `python/freetoken/kernel/csrc/include/freetoken/hip_compat.h`
- `python/freetoken/kernel/csrc/include/freetoken/device_api.h`
- `python/freetoken/kernel/csrc/include/freetoken/utils.cuh`
- `python/freetoken/kernel/csrc/jit/index.cu`
- `python/freetoken/kernel/csrc/jit/store.cu`
- `python/freetoken/kernel/csrc/jit/batch_memcpy.cuh`
- `python/freetoken/kernel/csrc/jit/fast_index_copy.cuh`
- `python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp`
- `python/freetoken/attention/__init__.py`
- `python/freetoken/attention/torch.py`
- `python/freetoken/attention/triton.py`
- `tests/kernels/test_rocm_launch_kwargs.py`
- `tests/kernels/test_jit_index_store.py`
- `tests/kernels/test_pinned_tensor.py`
- `tests/attention/test_torch_backend.py`
- `tests/kernels/test_attention_q8.py`

### Implementation

- Select HIP compiler/toolchain only under ROCm conditions; retain CUDA compiler and flags for CUDA.
- Normalize launch keyword translation at one boundary.
- Guard HIP-only headers, intrinsics, and compilation units from CUDA compilation.
- Make JIT cache identity include backend, target, compiler, flags, ABI, and source version.
- Keep memcpy and pinned-memory ownership semantics backend-neutral where possible.
- Preserve attention backend ordering and fail-loud behavior for unsupported forced backend.

### Edge cases

- ROCm installed without `hipcc` in PATH.
- CUDA build on host with HIP headers present.
- Stale JIT cache produced by different backend or GPU target.
- Zero-length or non-contiguous copy.
- Stream/event kwargs accepted by CUDA but not HIP.
- CPU-only import and extension discovery.

### Rollback and observability

Cache invalidation can be disabled through existing cache controls. Record compiler, backend, target, ABI, cache key, selected attention backend, and fallback reason. Roll back build/JIT changes separately from runtime route changes.

### Verification

- Run no-GPU import and compile checks for CUDA and ROCm conditionals.
- Run focused tests listed above with mocked backend probes.
- If hardware is available, compile one generic ROCm extension and one CUDA extension separately; capture exact commands and logs.
- Done when CUDA compile/import path is unchanged by ROCm-only code.

## Inc 4 - Portable runtime fallback seams (M)

**Status:** done

**Depends on:** Inc 1, Inc 2, Inc 3

**Unblocks:** Inc 5, Inc 6, Inc 7

### Scope

Port PR-217 runtime seams for configuration, graph policy, server launch, KV cache, communication, profiling, and MoE fallback. This increment makes generic serving path complete before native candidate work.

### Files to touch

- `python/freetoken/engine/config.py`
- `python/freetoken/engine/graph.py`
- `python/freetoken/engine/engine.py`
- `python/freetoken/engine/resident_budget.py`
- `python/freetoken/utils/graph_gate.py`
- `python/freetoken/utils/step_profiler.py`
- `python/freetoken/utils/torch_utils.py`
- `python/freetoken/server/args.py`
- `python/freetoken/server/launch.py`
- `python/freetoken/server/api_server.py`
- `python/freetoken/server/stats.py`
- `python/freetoken/kvcache/__init__.py`
- `python/freetoken/kvcache/base.py`
- `python/freetoken/kvcache/cache_status.py`
- `python/freetoken/kvcache/mha_pool.py`
- `python/freetoken/core.py`
- `python/freetoken/kernel/pynccl.py`
- `python/freetoken/moe/cpu_executor.py`
- `python/freetoken/moe/offload_kernels.py`
- `python/freetoken/moe/nvfp4_backends.py`
- `tests/engine/test_rocm_communication.py`
- `tests/engine/test_cpu_moe_graph_safety.py`
- Relevant server, KV-cache, graph, and MoE tests mirrored under `tests/`.

### Implementation

- Route ROCm graph behavior from capability/config, with safe eager fallback for unsupported graph features.
- Keep CUDA graph defaults and communication setup unchanged.
- Select RCCL/PyNCCL only on ROCm and preserve CUDA NCCL path.
- Make server arguments expose backend/candidate mode without changing existing defaults.
- Ensure KV cache, resident budget, CPU MoE, and offload paths do not assume CUDA-only tensor/device APIs.
- Keep NVFP4 CUDA path intact; ROCm routes to supported generic implementation or explicit unsupported error.
- Add structured runtime identity and route status to serving diagnostics.

### Edge cases

- ROCm graph capture unavailable or invalidated by dynamic shapes.
- RCCL unavailable in single-device run.
- Multi-device communication initialization failure.
- KV cache on CPU or mixed device.
- Server started with candidate flag on unsupported target.
- CUDA environment with ROCm packages installed.

### Rollback and observability

Provide config switches for graph disable, candidate disable, and communication backend selection. Record graph requested/selected, communication backend, KV placement, MoE route, fallback reason, and runtime ABI. Roll back server/config seams independently from cache internals.

### Verification

- Run focused engine, graph, communication, server, KV-cache, and MoE unit tests.
- Run no-GPU CUDA config tests and assert CUDA graph/NCCL/NVFP4 selection is unchanged.
- Run generic ROCm mocked-device tests for eager fallback and RCCL selection.
- Done when generic ROCm serving can proceed without native candidate code.

## Inc 5 - Generic GGUF/Qwen runtime (L)

**Status:** done

**Depends on:** Inc 2, Inc 3, Inc 4

**Unblocks:** Inc 7

### Scope

Port PR-217 generic GGUF reader, dequantization, Qwen model, GDN, MoE, expert-bank, tokenizer, generation, and API behavior. Keep generic route usable independent of exact native candidate kernels.

### Files to touch

- `python/freetoken/checkpoint/gguf_config.py`
- `python/freetoken/checkpoint/gguf_dequant.py`
- `python/freetoken/checkpoint/gguf_reader.py`
- `python/freetoken/checkpoint/gguf_tokenizer.py`
- `python/freetoken/models/qwen35_gguf.py`
- `python/freetoken/models/qwen35_moe_gguf.py`
- `python/freetoken/models/qwen35_gdn.py`
- `python/freetoken/models/registry.py`
- `python/freetoken/models/weight.py`
- `python/freetoken/layers/gguf.py`
- `python/freetoken/layers/moe.py`
- `python/freetoken/moe/expert_banks.py`
- `python/freetoken/moe/fused_gguf.py`
- `python/freetoken/moe/offload_cache.py`
- `python/freetoken/server/generation.py`
- `python/freetoken/server/openai_api.py`
- `python/freetoken/server/api_models.py`
- Matching `tests/checkpoint/`, `tests/models/`, `tests/layers/`, `tests/moe/`, and `tests/server/` files.

### Implementation

- Port only behavior compatible with Inc 1-4 contracts.
- Preserve GGUF metadata validation, tensor layout, quantization identity, tokenizer identity, and model registry behavior.
- Keep Qwen GDN/attention/MoE paths generic on ROCm; route native candidate only through Inc 2 policy.
- Preserve generation parser, penalties, stop handling, and API compatibility.
- Add deterministic route and quantization metadata to replay records.

### Edge cases

- Missing GGUF metadata or unsupported quant type.
- Qwen variant with absent GDN or alternate expert layout.
- Empty expert selection, duplicate experts, and CPU/GPU mixed expert placement.
- Long context and KV cache capacity boundary.
- Tokenizer mismatch between baseline and candidate.

### Rollback and observability

Keep generic route selectable independently of candidate route. Emit model/checkpoint hash, tokenizer hash, quant types, layer route, expert route, and fallback counts. Roll back model/runtime ports without removing capability or gate contracts.

### Verification

- Run focused checkpoint/model/layer/MoE/server unit tests.
- Run finite-logit and short completion checks on generic ROCm when hardware is available.
- Verify model/tokenizer metadata identity in replay fixture.
- Code complete and hardware evidence are separate states: fixture-pending is not serving validation and cannot support final serving claim.

## Inc 6 - Guarded native GGUF candidates (L)

**Status:** done

**Depends on:** Inc 2, Inc 3, Inc 4, Inc 5

**Unblocks:** Inc 7

### Scope

Port PR-217 native HIP/GGUF kernels and candidate routing as opt-in exact-target acceleration. Keep generic fallback and CUDA routes intact.

### Files to touch

- `python/freetoken/kernel/csrc/gguf/gguf_kernel.cu`
- `python/freetoken/kernel/csrc/gguf/dispatch.h`
- `python/freetoken/kernel/csrc/gguf/ggml-common.h`
- `python/freetoken/kernel/csrc/gguf/ggml-common_hip.h`
- `python/freetoken/kernel/csrc/gguf/dequantize_hip.cuh`
- `python/freetoken/kernel/csrc/gguf/mmq_hip.cuh`
- `python/freetoken/kernel/csrc/gguf/mmvq_hip.cuh`
- `python/freetoken/kernel/csrc/gguf/moe_hip.cuh`
- `python/freetoken/kernel/csrc/gguf/moe_vec_hip.cuh`
- `python/freetoken/kernel/csrc/gguf/vecdotq_hip.cuh`
- Any PR-217 native GGUF translation units required by build graph, with each path classified in preflight.
- `python/freetoken/kernel/gguf.py`
- `python/freetoken/layers/moe.py`
- `python/freetoken/moe/fused_gguf.py`
- `tests/kernels/test_gguf_moe.py`
- `tests/kernels/test_gguf_rocm.py`
- Native candidate self-test fixtures.

### Implementation

- Port kernels behind HIP/ROCm compile guards and explicit target feature checks.
- Register exact operation, quant, shape, ABI, and source-version capabilities.
- Keep candidate module lazy-loaded and absent from CUDA path.
- Run candidate self-test against independent reference before candidate execution.
- Make `rocm_candidate` fail loudly on any mismatch; make `auto` select generic fallback.
- Do not mark candidate correctness/performance status in source defaults; status comes from validation artifacts.

### Edge cases

- RDNA3 versus RDNA4 wave/packing differences.
- Unsupported Q4_K/Q8_0 shape or alignment.
- Kernel compile success with wrong runtime ABI.
- Non-finite candidate output, route mismatch, or silent fallback.
- Candidate module present on another ROCm target.
- CUDA compiler seeing HIP-only translation unit.

### Rollback and observability

Candidate can be disabled globally and per target. Record kernel name, target, quant, shape, ABI, source version, self-test result, and fallback reason. Preserve generic route as immediate rollback. Revert native files without reverting Inc 1-5.

### Verification

- Compile candidate for exact target and run self-test.
- Run candidate-versus-generic finite-logit and teacher-forced token parity checks.
- Run cross-target negative tests proving candidate is not selected on other ROCm targets.
- Run CUDA compile/import tests proving no candidate probe or symbol collision.
- No performance promotion here; benchmark gate is Inc 7.

## Inc 7 - Evidence tooling and platform CI (M)

**Status:** done

**Depends on:** Inc 3, Inc 4, Inc 5, Inc 6

**Unblocks:** none

### Scope

Port and harden benchmark/replay/gate tooling, manifests, README instructions, and CI coverage. This increment supplies evidence required by final cross-cutting validation.

### Files to touch

- `benchmarks/bench_decode_replay.py`
- `benchmarks/bench_rocm_matrix.py`
- `benchmarks/check_decode_gate.py`
- `benchmarks/profile_decode_rocm.py`
- `benchmarks/README.md`
- `docs/reproducibility.md`
- `.github/workflows/unit-nvidia.yml`
- `.github/workflows/unit-rocm.yml`
- Benchmark and gate tests under `tests/benchmarks/` or repository-equivalent test location.

### Implementation

- Port replay capture and validation behavior while preserving independent-oracle provenance.
- Make benchmark records identity-complete before timing is accepted.
- Add explicit schema readers/migration fixtures and reject ambiguous records.
- Make promotion gate reject non-teacher-forced speed rows and route mismatches.
- Keep NVIDIA CI no-GPU-safe and add ROCm matrix jobs only behind available runner labels and explicit hardware evidence.

### Manifest migration

- Do not reinterpret `freetoken-rocm-manifest-v1` or `freetoken-replay-manifest-v1` in place.
- If route digest, oracle identity, or capability fields change, write `freetoken-replay-manifest-v2` or `freetoken-rocm-manifest-v2` explicitly.
- Keep readers for v1 as historical read-only inputs, or provide a named migration function with fixtures proving lossless/explicit field mapping.
- Reject unknown schema versions; never silently drop identity fields.
- Add fixtures for both existing v1 forms and new v2 form.

### Performance gate rules

- Candidate and baseline rows must both use `teacher_forced_replay`.
- Replay record must contain prompt hash, continuation token IDs/hash, model/checkpoint/tokenizer identity, quant/KV/graph identity, independent oracle ID, route digest, and completion/logit finiteness evidence.
- Candidate and baseline route digests must match expected semantic route or be explicitly classified as a different lane; route mismatch cannot pass promotion.
- Candidate route must show no unplanned generic fallback during measured region.
- `sampled_decode` and `greedy_decode` rows may support correctness/serving gates but must be rejected for speed promotion.
- Add tests rejecting sampled speed rows, greedy speed rows, missing oracle ID, route mismatch, malformed continuation hash, and schema ambiguity.

### CI and platform coverage

- Keep NVIDIA workflow no-GPU-safe and add assertions that ROCm probes do not alter CUDA configuration.
- Run ROCm workflow across available self-hosted targets, at minimum generic compile/import contracts and exact candidate target when hardware exists.
- Separate compile-only, correctness, serving, and performance artifacts/statuses.
- Store commands, environment identity, raw timings, median/spread, route logs, and gate result.

### Edge cases

- Benchmark process starts with stale JIT cache.
- Worker/model identity changes between baseline and candidate.
- Replay record from old schema.
- Completion shorter than requested or route changes mid-run.
- GPU unavailable, busy, or different target than manifest.
- CI has ROCm runtime but no model/checkpoint.

### Rollback and observability

Gate changes are independently revertible. Gate output must state schema, lane, model identity, target, backend, route digest, fallback count, oracle ID, and exact rejection reason. Preserve raw artifacts even on `NO-PROMOTION`.

### Verification

- Run benchmark/gate unit tests, including all negative promotion cases.
- Run no-GPU NVIDIA workflow-equivalent tests locally where possible.
- Run ROCm matrix compile/import/contract checks on every available target.
- Run one end-to-end replay with independent oracle when hardware is available.
- Done when tooling can produce either valid promotion evidence or explicit non-promotion evidence.

## Cross-cutting final validation

Run only after Inc 7. No final serving or speed claim is valid from unit tests, compile success, HTTP 200, IPC rate, or self-generated output alone.

### Correctness and serving gate

- Start resident worker with explicit device/backend, model path/hash, context, KV type, graph mode, and MTP disabled.
- Run teacher-forced replay against independent oracle; verify finite logits, token ID parity, route digest, completion count, and no hidden fallback.
- Run greedy served completion to verify API/streaming behavior, but keep it separate from speed promotion.
- Validate candidate forced mode fail-loud and auto generic fallback on unsupported target/shape/quant.

### A/B performance gate

- Compare candidate and generic baseline in fresh processes with same model/checkpoint/tokenizer, prompt and continuation IDs, token count, quantization, KV settings, graph mode, batch/concurrency, and runtime flags.
- Use at least three valid runs per side after warmup; report raw runs, median, spread, device, driver, ROCm, compiler, SPIR-V/ABI or kernel identity, cache state, and commit.
- Measure non-MTP base decode. Prefer matching Q4_K and Q8_0 lanes. Do not compare candidate lane with sampled/greedy lane.
- Historical numbers such as prior 34.69 tok/s are context only, not current evidence.
- Require independent route/token evidence and teacher-forced rows before calculating promotion gain.

### Cross-ROCm and CUDA checks

- On each available ROCm target, run generic compile/import/contract checks; run candidate only on exact target.
- On CUDA, run unit/import/build checks and, if hardware is available, unchanged baseline smoke. Confirm no HIP-only import, candidate probe, flag, or cache collision.
- Record unavailable hardware explicitly; do not convert missing evidence into pass.

### Final result

Use `PROMOTE` only when generic correctness, candidate correctness, serving, CUDA isolation, cross-ROCm contracts, and teacher-forced performance gates pass. Otherwise use `NO-PROMOTION`; keep generic ROCm default and attach failure evidence plus next gate.

## Execution result

Implemented all seven increments in this checkout. Existing PR-217-derived generic GGUF/Qwen,
MoE, benchmark, and CI work was retained; missing gfx1100 candidate translation units were
restored and connected to the exact-target ABI gate. Added final hardening for CUDA isolation,
ROCm graph/BLAS fallback, runtime route diagnostics, and replay promotion evidence.

Validation evidence:

- `python3 -m pytest -q tests/utils/test_graph_gate.py tests/utils/test_rocm_arch.py tests/utils/test_decode_replay.py tests/utils/test_decode_gate.py tests/utils/test_decode_benchmark_metadata.py`: `29 passed, 8 skipped`.
- `python3 -m compileall -q benchmarks python/freetoken freetoken-kernel-cache`: passed.
- `python3 -m py_compile` on changed Python modules: passed.
- `git diff --check`: passed.
- ROCm identity: Radeon RX 7900 XTX, `gfx1100`, driver `6.17.0-35-generic`.
- Full `uv` test/build path: blocked before collection by `RuntimeError: CUDA_HOME is required to build`.
- Engine/communication runtime tests: unavailable because host Torch cannot load `libcublasLt.so.*`.
- ROCm GGUF and Torch-attention collection: blocked by missing `libcudart.so.13` and `libcublasLt.so.*`.
- Current benchmark: unavailable; no valid tok/s number collected. Historical `34.69 tok/s` remains context only.

Review remediation: graph capture now requires an explicit `pass`, parent-side ROCm probing is
disabled for TP/explicit GPU targets, graph-gate cache identity includes runtime/toolchain/policy
inputs, ROCm cache builds validate Torch toolchain compatibility, HIP metadata copies are stream
ordered, and duplicate ROCm packaging metadata was removed.

Final gate: `NO-PROMOTION`. Generic ROCm default remains protected; correctness/serving and
teacher-forced A/B performance evidence remain pending a working ROCm Torch environment and
model fixture. No commit, push, or remote merge performed.

## Commit and handoff

Suggested final title, subject to actual diff:

```text
feat(rocm): consolidate portable GGUF execution and harden target gates
```

Human-readable body, filled only from collected artifacts:

```text
Consolidate PR-217 ROCm runtime and GGUF work into PR-132.

- keep generic ROCm fallback as default (`python/freetoken/kernel/gguf.py`, `python/freetoken/utils/arch.py`)
- gate native kernels by exact target, quant, shape, ABI, and self-test (`python/freetoken/kernel/gguf.py`)
- fail closed on ROCm graph uncertainty and preserve CUDA isolation (`python/freetoken/engine/graph.py`, `python/freetoken/utils/graph_gate.py`, `python/freetoken/server/launch.py`)
- validate ROCm cache toolchains and stream-order HIP metadata copies (`freetoken-kernel-cache/build_backend.py`, `python/freetoken/kernel/csrc/jit/batch_memcpy.cuh`)
- add replay schema migration and teacher-forced promotion gates (`benchmarks/bench_decode_replay.py`, `benchmarks/check_decode_gate.py`)

Validation:
- correctness: 29 passed, 8 skipped in focused unit suite; runtime blocked by missing Torch ROCm/CUDA libraries
- serving: not run; same Torch loader blocker
- performance: no valid current tok/s; historical 34.69 tok/s is context only
- final gate: NO-PROMOTION; teacher-forced A/B evidence unavailable
```

Use Conventional Commits, lowercase imperative subject, no trailing period. Do not commit until human owner asks. Do not push or update PR remotely from agent session.

## Open questions to resolve during preflight

- Confirm whether local PR-132 helper commit or upstream PR-132 head is authoritative when behavior differs; resolve by contract and test, not commit order.
- Confirm which second ROCm target is physically available for cross-target evidence.
- Confirm artifact storage location accepted by project CI; keep raw artifacts local until owner chooses publication.

## Out of scope

- MTP/speculative decode performance.
- Automatic fallback in forced candidate mode.
- Treating compile-only status as served/correctness/performance proof.
- NVIDIA-only NVFP4/Marlin comparison against ROCm GGUF.
- PLE/q-star streaming, Windows, TP>1, or unrelated model architectures.
- Personal `.vscode` settings, local environment files, generated caches, and temporary source maps.
- Remote push, PR creation/comments, or final remote merge by agent.
