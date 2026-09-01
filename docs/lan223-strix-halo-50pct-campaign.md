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
