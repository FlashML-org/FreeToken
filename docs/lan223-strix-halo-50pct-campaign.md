# LAN-223 Strix Halo 50 percent performance campaign

## Objective

Increase the client-visible steady-state decode speed of the native ROCm/HIP
FreeToken Qwen3.6-35B-A3B Q4 service on LAN-223 by up to 50 percent over the
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

- Target host: LAN-223 only, Radeon 8060S `gfx1151`.
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
`0a1b709`; its complete candidate artifact remains on LAN-223 for comparison.

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
`qwen35moe-q4-hipmath-20260901T081500Z` on LAN-223, but remove the experimental
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
quality artifact at `qwen35moe-q4-vdr4-20260901T084500Z` on LAN-223, revert the
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
the artifact at `qwen35moe-q4-k2row-20260901T093500Z` on LAN-223 and revert the
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
`qwen35moe-q4-q8vdr4-20260901T104200Z` on LAN-223 and revert the candidate.
