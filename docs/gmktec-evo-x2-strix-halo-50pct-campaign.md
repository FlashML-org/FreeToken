# GMKtec EVO-X2 Strix Halo 50 percent performance campaign

## Objective

Increase the client-visible steady-state decode speed of the native ROCm/HIP
FreeToken Qwen3.6-35B-A3B Q4 service on GMKtec EVO-X2 by up to 50 percent over the
currently accepted exact-Q4 baseline, while retaining equivalent output quality
and operational reliability.

The target is an engineering hypothesis, not a promised result.  Every claimed
gain must be measured against the same model, prompt, decoding contract, and
quality suite.  The campaign stops at the measured limit of the approved
software and host scope if the target cannot be achieved without a regression.

## Baseline and numeric target

The accepted comparison baseline is the exact Qwen3.6-35B-A3B Q4_K_M FreeToken
profile with `memory_ratio=0.25`, measured through the local OpenAI-compatible
API after warmup.  Its recorded mean decode speed is 47.960 tokens per second.

| Measure | Value |
| --- | ---: |
| Accepted FreeToken baseline | 47.960 decode tokens per second |
| 50 percent campaign target | 71.940 decode tokens per second |
| Matched llama.cpp ROCm control | 48.831 decode tokens per second |

This document does not treat an isolated kernel time, server-internal counter,
batch-only aggregate, or a different quantization as a substitute for the
baseline metric.  Those are diagnostic measurements and must be labelled as
such.

## Scope boundaries

- Target host: GMKtec EVO-X2 only, Radeon 8060S `gfx1151`.
- Target runtime: native FreeToken ROCm/HIP path only.
- Target model: the exact qualified Qwen3.6-35B-A3B Q4_K_M artifact.
- Candidate servers bind only to loopback test ports in isolated clean
  worktrees.
- The protected normal Qwen service is stopped only inside an explicit
  time-share window and must be verified healthy after every window.
- Do not alter llama-swap, LAN routes, production model files, BIOS settings,
  kernel, system ROCm packages, or host power limits under this campaign.
- Preserve every rejected result with its failure or rejection reason.

## Non-negotiable acceptance gate

A candidate may replace the accepted baseline only when all conditions hold:

1. It improves the median client-visible decode rate by at least one percent
   over the accepted baseline in two independently launched API matrices.
2. The exact deterministic canaries remain byte-identical for same-weight,
   same-template, greedy decoding comparisons.
3. The versioned functional suite passes, including arithmetic, structured
   JSON, retrieval, multi-turn, and long-context cases.
4. Mean TTFT does not regress by more than five percent and p99 token-gap
   latency does not materially worsen.
5. It introduces no NaN, malformed SSE sequence, crash, stale process group,
   SVM memory failure, unbounded growth, or failed protected-service recovery.
6. It records the full commit, patch, runtime versions, exact commands, model
   identity, raw outputs, telemetry, and accept or reject decision.

A fast candidate that fails any quality or reliability gate is rejected even if
it exceeds 71.940 tokens per second.

## Campaign ladder

### Stage 0: lock and stress the control

1. Complete the running 24-hour minute-cadence Q4 endurance battery.
2. Confirm all request sessions pass after excluding the documented initial
   warmup effect.
3. Confirm the normal NVFP4 Qwen endpoint is restored by the controller and
   answers its health check with the intended model identity.
4. Archive a signed baseline manifest, API matrix, quality outputs, and
   telemetry summary before any new candidate starts.

The first long battery may be deliberately concluded after a successful
six-hour checkpoint when an active optimization window is more valuable than
additional identical idle-duration coverage.  Such a run is always labelled
`incomplete_checkpoint`, never reported as a completed 24-hour endurance pass.
Before the next candidate starts, the controller must stop the Q4 service,
restore the protected NVFP4 server, and reach a real `serving` health state.

### Stage 1: make quality difficult to accidentally regress

The existing small exact suite is necessary but insufficient for aggressive
kernel and scheduling changes.  Extend it in a versioned corpus with:

- Greedy exact-response canaries for routing and numerical-order changes.
- Machine-scored arithmetic and constrained reasoning answers.
- JSON and tool-call-shaped schema validation.
- Code snippets with a local execution test harness.
- 2K, 8K, 16K, and maximum-qualified-context retrieval cases.
- Multi-turn correction and cache-reuse cases.
- A small fixed Gemma 4 text and image control set, run separately so Qwen
  improvements never hide a Gemma regression.

The suite stores prompts, generated text, token counts, finish reasons, scorer
results, and output hashes.  It is a gate, not a performance workload.

### Stage 2: profile the actual server

Collect the following in a dedicated Q4 candidate window:

1. Low-overhead application counters for a warm 256-token decode, a long
   context request, and concurrency levels 1, 2, 4, and 8.
2. HIP event timing around router, cache operations, dense projections, MoE
   projections, attention, sampling, and synchronizations.
3. A representative ROCm trace using the wheel-compatible profiler wrapper.

Rank candidates by end-to-end decode contribution.  Do not optimize a
microbenchmark only because it looks slow outside the actual server trace.

The trace protocol launches the disposable Q4 process through the
wheel-compatible ROCm profiler wrapper.  The host profiler cannot safely attach
to the running PyTorch ROCm wheel on this machine, and raw profiler throughput
is intentionally excluded from every TPS comparison because trace collection
is intrusive.  Capture kernel dispatch, HIP runtime, memory-copy, and KFD
events for one warmed fixed-length decode, then use a read-only database
aggregate to rank the final active window.

### Stage 3: run three independent optimization lanes

#### Lane A: RDNA3.5 dense FP8 decode

The earlier trace identifies the dense FP8 `_gemv_splitk_kernel` as the largest
measured GPU-time consumer.  This lane investigates exact Qwen matrix shapes
only, preserving split-K reduction ordering and accumulation precision.

Screen workgroup geometry, vectorized load alignment, wave occupancy, register
pressure, LDS use, and shape-specialized dispatch.  Inspect generated ISA
before claiming an intrinsic or coalescing improvement.  Reject a candidate
that changes deterministic canaries.

#### Lane B: UMA-aware MoE cache and expert movement

Static larger cache residency previously reduced cache misses without producing
an API throughput gain and the high-memory profile showed SVM instability.
This lane therefore measures a contention curve rather than assuming more cache
is better.

Test cache target, KV allocation, active request count, route locality, and
safe-point resizing.  A policy may increase cache only while verified memory
headroom remains above a configured guard threshold.  It must back off with
hysteresis before paging or a driver fault, and it must never silently change
the model or precision.

#### Lane C: scheduler and launch overhead

Measure whether decode is limited by CPU launch chains, small kernel dispatches,
or poor request coalescing.  Test scheduler policies at controlled concurrency
while separately reporting per-user TPS, aggregate TPS, TTFT, queue time, and
tail token gap.

Only investigate graph capture, persistent execution, or layer-local route
batching when the trace establishes that launch or synchronization cost is
large enough to justify the complexity.  A concurrency gain is reported as an
aggregate-throughput result and never presented as a single-user TPS gain.

### Stage 4: compose accepted improvements

Accepted changes are combined one at a time.  After each composition, rerun the
full single-user and concurrent API matrix, full quality suite, long-context
test, multi-turn battery, controlled cancellation, and service-recovery test.
This prevents individually safe changes from hiding an interaction regression.

### Stage 5: controlled platform qualification

Only after exhausting code and policy work, consider a separate ROCm, kernel,
or firmware qualification project.  It requires a specific approval because it
changes host-level software outside this campaign.  The justification must
include a current SVM or compiler limitation, a rollback plan, and the exact
same before-and-after test matrix.

## Iteration protocol

For every candidate:

1. Create a clean worktree and give the candidate a short, immutable ID.
2. Write a design note naming the bottleneck, hypothesis, expected upside,
   quality risk, and rollback method.
3. Run the relevant microbenchmark only as an initial screen.
4. Start on a loopback candidate port and verify health, model identity, and
   native extension identity.
5. Run the complete quality gate before throughput work.
6. Run five warm API samples plus the concurrent matrix with aligned telemetry.
7. Run long-context, multi-turn, cancellation, stop, and recovery validation.
8. Compare against a fresh baseline from the same host state whenever possible.
9. Mark the candidate accepted, rejected, or inconclusive with raw evidence.
10. Restore and verify the protected normal service before leaving the window.

## Reporting

Each accepted or rejected candidate receives a row with:

| Candidate | Baseline TPS | Candidate TPS | Delta | TTFT | p99 gap | Quality | Reliability | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |

The final report will separate:

- Single-user client-visible decode TPS.
- Concurrent aggregate throughput and per-user latency.
- Kernel-only diagnostic changes.
- Stable production-eligible configurations.
- Experimental configurations that are faster but not yet reliable.
- The remaining measured bottleneck if the 50 percent target is not reached.

## Experiment log

### C01: two-row HIP GGUF matrix-vector blocks

The first post-checkpoint ROCm trace used the exact Qwen3.6-35B-A3B Q4_K_M
workload and isolated Q4 server.  Its final 30-second active window ranked the
GGUF vector kernels, rather than the NVFP4 dense FP8 path, as the primary work:

| Kernel family | Calls | GPU time in traced window |
| --- | ---: | ---: |
| Q8_0 vector matrix multiply | 81,600 | 3,660.029 ms |
| Q4_K routed MoE vector multiply | 20,480 | 3,103.535 ms |
| Q5_K routed MoE vector multiply | 18,944 | 1,971.720 ms |
| Q6_K vector matrix multiply | 512 | 920.072 ms |
| Routed cache gather | 22,016 | 684.485 ms |

The upstream matrix-vector launch used one 32-thread output row per block.
The candidate grouped two independent rows into a 64-thread HIP block, which
fills one RDNA wavefront while retaining the same per-row quantization and
reduction.  It built successfully in clean worktree `218104c`, passed the
three deterministic Qwen API controls, and completed three fixed-workload API
samples.

| Measure | Stable baseline | C01 candidate | Change |
| --- | ---: | ---: | ---: |
| Mean decode TPS | 47.960 | 48.081 | +0.25% |
| Median decode TPS | 48.075 | 48.083 | +0.02% |
| Mean TTFT | 0.453 s | 0.439 s | diagnostic only |
| Quality controls | 3/3 pass | 3/3 pass | unchanged |

**Decision: rejected.** The candidate is numerically safe in the screened
controls, but its 0.25 percent gain is below the one percent acceptance floor
and is within normal run-to-run variation.  The change was reverted in
`0a1b709`; its complete candidate artifact remains on GMKtec EVO-X2 for comparison.

### C02: opt-in HIP unsafe-math optimizations

Current llama.cpp HIP build guidance uses `-funsafe-math-optimizations`, which
is narrower than `-ffast-math`.  The candidate made that flag opt-in through
`FREETOKEN_HIP_GGUF_FAST_MATH=1`, so its generated HIP extension has a distinct
build configuration and cannot alter the conservative default path.  It was
built in clean worktree `dd8bc3b` and ran the exact qualified Q4 model, the
three deterministic API controls, and the fixed 256-token throughput workload.

| Measure | Stable baseline | C02 candidate | Change |
| --- | ---: | ---: | ---: |
| Mean decode TPS | 47.960 | 41.391 | -13.70% |
| Median decode TPS | 48.075 | 48.023 | -0.11% |
| Best sample TPS | 48.075 | 48.558 | +1.00% |
| p99 token gap | 0.02490 s | 0.02481 s | diagnostic only |
| Quality controls | 3/3 pass | 3/3 pass | unchanged |

One of the three candidate samples contained a 3.943-second token stall.  The
other two samples were approximately 48 TPS, which is indistinguishable from
the stable baseline and far below the campaign acceptance threshold.  The
candidate therefore has no demonstrated decode gain, while its mean result is
materially worse because of the stall.

**Decision: rejected.** Preserve the raw quality and benchmark artifacts at
`qwen35moe-q4-hipmath-20260901T081500Z` on GMKtec EVO-X2, but remove the experimental
compiler flag from the branch.  Further work should target the measured Q4_K
and Q5_K routed-MoE vector kernels, not generic compiler flags.

### C03: four-chunk HIP Q4_K and Q5_K vector work

The model-shape inventory confirmed that every routed gate and up projection is
Q4_K with 512 input values and 2,048 output values, while the routed down
projection is Q5_K with 2,048 inputs and 512 outputs.  The candidate doubled
the per-lane vector-dot ratio from two to four only under HIP, preserving the
packed weight layout and reduction expression while reducing the number of
chunks each lane processes.

The exact-Q4 candidate built in clean worktree `921ec3f` and reached its
loopback endpoint.  It failed all three deterministic visible-output controls:
the exact canary emitted a control token, the arithmetic control emitted an
incorrect sentence, and the JSON control was not valid JSON.  No throughput
claim was measured or retained because the mandatory quality precondition
failed.

**Decision: rejected for correctness.** The wider vector ratio changes the
kernel's coverage or reduction mapping on this HIP path.  Preserve the failed
quality artifact at `qwen35moe-q4-vdr4-20260901T084500Z` on GMKtec EVO-X2, revert the
source candidate, and restore the protected normal Qwen service before the
next investigation.

### C04: shared-activation two-row Q4_K and Q5_K routed-MoE vectors

The next candidate retained the established two-chunk vector-dot mapping and
separate XOR reduction for each output row.  Instead of changing quantization
coverage, one HIP logical wave computed two adjacent rows for a route and
shared the selected expert and Q8_1 activation address.  It built in clean
worktree `3ffb1c6`, passed all three deterministic Qwen API controls, and ran
the fixed warmup plus three scored 256-token API samples.

| Measure | Stable baseline | C04 candidate | Change |
| --- | ---: | ---: | ---: |
| Mean decode TPS | 47.960 | 47.745 | -0.45% |
| Median decode TPS | 48.075 | 47.833 | -0.50% |
| p99 token gap | 0.02490 s | 0.02489 s | diagnostic only |
| Quality controls | 3/3 pass | 3/3 pass | unchanged |

**Decision: rejected.** Correctness was preserved, but sharing the activation
address did not offset the extra live accumulator and register pressure.  The
result is below baseline and below the one-percent acceptance floor.  Preserve
the artifact at `qwen35moe-q4-k2row-20260901T093500Z` on GMKtec EVO-X2 and revert the
candidate source.

### C05: wider HIP Q8_0 vector-dot ratio

The representative trace ranked Q8_0 vector matrix multiply as the largest
single kernel family.  This HIP-only candidate changed its vector-dot ratio
from two to four, which the Q8 dot helper supports directly, while retaining
the CUDA two-group behavior.  Clean worktree `dea5d6f` built successfully and
passed all three deterministic Qwen API controls.

| Measure | Stable baseline | C05 candidate | Change |
| --- | ---: | ---: | ---: |
| Mean decode TPS | 47.960 | 47.546 | -0.86% |
| Median decode TPS | 48.075 | 47.585 | -1.02% |
| p99 token gap | 0.02490 s | 0.02606 s | diagnostic only |
| Quality controls | 3/3 pass | 3/3 pass | unchanged |

**Decision: rejected.** The wider Q8 work ratio is numerically safe but slows
the end-to-end Q4 workload.  The extra per-lane work does not repay its
occupancy and register cost on gfx1151.  Preserve the artifact at
`qwen35moe-q4-q8vdr4-20260901T104200Z` on GMKtec EVO-X2 and revert the candidate.

### C06: modern MMVQ component replacement investigation

The prior candidates establish that changing local launch dimensions or
per-lane work ratios in the older vendored GGUF kernels does not produce a
safe gain on gfx1151.  GMKtec EVO-X2 reports a 32-lane HIP warp, so the existing
32-thread logical reduction is not accidentally running at half its physical
wave width.

Current llama.cpp has evolved from the older MMVQ donor used here into an
architecture-aware implementation.  It selects launch geometry by GPU family,
uses a newer parameter table, and has a dedicated routed-expert vector path.
The relevant upstream components are `ggml-cuda/mmvq.cu` and `vecdotq.cuh` in
the current llama.cpp tree.  FreeToken's GGUF extension has a narrower PyTorch
binding and different packed-bank interface, so copying the file wholesale
would be unsafe.

The next component lane is therefore a selective port with these gates:

1. Extract only the Q4_K, Q5_K, Q6_K, and Q8_0 vector-dot helpers plus the
   routed-expert launch geometry needed by the exact Qwen model.
2. Preserve FreeToken's existing packed `[expert, row, row_bytes]` bank and
   `topk_ids` interface.  Do not change quantization, routing, model files, or
   sampling behavior.
3. Add a model-shape microbenchmark using the actual 512-to-2,048 Q4_K gate/up
   projections, 2,048-to-512 Q5_K down projection, and Q8_0 dense shapes.
4. Compare candidate tensors to the accepted kernel before API startup, then
   run the deterministic API suite.  Any mismatch is an immediate rejection.
5. Use the full fixed API workload, tail-latency telemetry, long-context,
   multi-turn, and recovery gates before accepting a candidate.

This is the remaining software-only path with credible headroom.  The evidence
does not support promising a 50-percent single-user decode gain from it: the
current accepted FreeToken exact-Q4 result is already within 1.78 percent of
the matched llama.cpp ROCm control.  Any larger claim requires measured proof,
not extrapolation from CUDA-oriented paper results.

#### C06 baseline: exact packed-expert microbenchmark

The new screening harness completed its initial GMKtec EVO-X2 baseline with real
packed bytes from layer 0 of the qualified Qwen GGUF.  It copied the eight
routed expert slices only, used the production `ggml_moe_a8_vec` binding, and
excluded model load, HTTP, router, scheduler, and JIT time from GPU-event
measurements.

| Projection | Quantization | Exact shape | Mean device time |
| --- | --- | --- | ---: |
| Gate | Q4_K | 512 to 2,048, eight selected experts | 21.970 microseconds |
| Up | Q4_K | 512 to 2,048, eight selected experts | 21.864 microseconds |
| Down | Q5_K | 2,048 to 512, eight selected experts | 20.624 microseconds |
| Three projections | mixed | one routed token's screen workload | 64.459 microseconds |

This is a selection baseline, not server TPS.  It makes later component work
auditable: a candidate must improve this real-shape screen and still pass all
end-to-end quality, latency, and recovery gates.  The artifact is
`qwen35moe-q4kq5k-microbaseline-20260901T141100Z` on GMKtec EVO-X2.

### C06 execution contract: selective modern MMVQ port

The next iteration is deliberately limited to the modern llama.cpp component
surface that is relevant to this Qwen artifact: the Q4_K and Q5_K routed
expert helpers, Q6_K and Q8_0 dense helpers, and the architecture-aware MMVQ
launch selection.  It does not alter the GGUF packing, router semantics,
quantization, sampling, cache capacity, model files, ROCm installation, or
normal service configuration.

The candidate may advance only in this order:

1. Compile in an isolated source tree with a separate ROCm kernel cache.
2. Match the accepted implementation on real packed Qwen expert tensors before
   starting an HTTP server.  Any element mismatch rejects the candidate.
3. Improve the real-shape device microbenchmark by at least 1 percent for a
   traced hot projection without regressing another traced hot projection by
   more than 1 percent.
4. Pass the three deterministic API controls, the functional quality suite,
   long-context retrieval, and multi-turn state-retention suite.
5. Beat 47.960 mean decode TPS by at least 1 percent across repeated
   fixed-workload API samples, with no worse p99 token gap or recovery result.
6. Run the matched llama.cpp control again only after FreeToken clears its own
   acceptance gate.  A useful cross-runtime win is at least 51.3 TPS, roughly
   five percent above the current 48.831 TPS control, rather than an outcome
   inside ordinary run-to-run variation.

The NVIDIA NVFP4 lane remains separate.  It cannot claim comparison with the
published 39.3 TPS RTX 4060 result until the authors' workload, cache, warmup,
generation, stop, source-revision, and policy fields are recovered and frozen.

#### C06.1 source-architecture audit

The source audit separated FreeToken's prefill and decode dispatches.  The
prefill-oriented `moe.cuh` kernels already select eight 32-lane waves with 8
by 128 tiles for the relevant ROCm formats.  The qualified single-token Qwen
decode path, however, calls `ggml_moe_a8_vec`, whose `moe_vec.cuh` Q4_K,
Q5_K, Q6_K, and Q8_0 wrappers still launch one wave per block.  Current
llama.cpp selects eight waves for those simple RDNA4 single-vector helpers.

The resulting C06 candidate changes only that decode dispatcher.  It retains
FreeToken's expert-bank layout, activation packing, vector-dot helpers, row
mapping, and CUDA behavior.  Tensor equality remains mandatory before the
candidate can consume server benchmark time.  This distinction prevents an
already-tuned prefill geometry from being confused with the still-unported
decode geometry.

#### C06.2 decode-wave candidate screen

The HIP-only C06 candidate was built in an isolated source tree and executed
against the real layer-0 packed expert slices from the exact Qwen Q4_K model.
It changed only the four decode wrapper launches from one 32-lane wave to
eight 32-lane waves, matching the family-specific direction seen in current
llama.cpp.  CUDA behavior and all model-visible semantics remained untouched.

| Projection | Baseline device time | C06 device time | Change |
| --- | ---: | ---: | ---: |
| Gate Q4_K | 21.970 microseconds | 23.107 microseconds | +5.18% slower |
| Up Q4_K | 21.864 microseconds | 23.531 microseconds | +7.62% slower |
| Down Q5_K | 20.624 microseconds | 21.672 microseconds | +5.08% slower |
| Three projections | 64.459 microseconds | 68.311 microseconds | +5.98% slower |

The candidate compiled natively with the ROCm 10 runtime and used 30 warmup
iterations plus 300 measured repetitions.  It failed the microbenchmark gate
before tensor-equivalence or HTTP testing was warranted: every traced
projection regressed by more than the allowed one-percent ceiling.  The
likely mechanism is that these relatively small routed-expert matrices do not
provide enough parallel work to repay the added wave coordination.

**Decision: rejected.** C06 is retained only on the isolated experimental
branch and is not merged into the AMD port.  The protected GMKtec EVO-X2 Qwen
service was restarted immediately after the screen and its health endpoint
returned `status: ok` before the iteration was closed.  The immutable screen
artifact is `qwen35moe-q4-c06-micro-20260901T174907Z` on GMKtec EVO-X2.
