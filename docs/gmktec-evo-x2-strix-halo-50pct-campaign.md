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

### C07: upstream Triton-router audit

Upstream FreeToken advanced its default router policy in commit `e05cff8`,
which routes `fused_topk` through the in-tree Triton implementation.  This was
investigated as a possible scheduler-overhead improvement because routing runs
once per MoE layer during decode.

The isolated HIP router diagnostic completed without loading a model or
changing the normal service.  On the Qwen shape of 256 experts and top-k 8,
the Triton implementation matched the PyTorch reference IDs and weights in the
diagnostic and reduced synchronized router time as follows:

| Tokens | PyTorch router | Triton router | Microbenchmark speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.02415 ms | 0.01664 ms | 1.45x |
| 4 | 0.02442 ms | 0.01811 ms | 1.35x |

This is not a new production candidate.  The present Q4 production source
intentionally keeps the PyTorch router on ROCm because a prior end-to-end
Qwen canary changed despite isolated router parity.  The upstream change
modifies only the dispatch policy, not the router kernel or its arithmetic, so
it does not address that quality failure.  Repeating a full API test without a
source-level numerical fix would therefore consume a protected-service window
without testing a new hypothesis.

**Decision: rejected as already disproven.** Preserve the router timing
artifact `qwen-router-c07-20260901T180707Z` on GMKtec EVO-X2 as a diagnostic,
but retain the reference route for the exact-Q4 quality baseline.  The normal
GMKtec EVO-X2 service remained on its existing configuration and returned
`status: ok` after the diagnostic.

#### C07 correction and C08-C09 quality requalification

The C07 dispatch conclusion was superseded by a source and deployment audit.
The active service source was older than the branch under test, while the
current branch's router policy selects the in-tree HIP Triton router by
default.  The upstream router-policy change is therefore not an untested new
kernel, but it does expose a branch-versus-deployment qualification gap that
had to be closed before making a performance claim.

First, a quality-only isolated Q4 window using the current branch's router
path passed all three deterministic controls.  A second isolated window used a
clean router-only checkout, excluding the rejected decode-wave modification,
and expanded the test scope:

| Control | Result | Key evidence |
| --- | --- | --- |
| Exact response, arithmetic, structured JSON | 3 of 3 pass | deterministic visible outputs |
| Multi-turn state retention | 3 of 3 pass | maximum TTFT 0.846 s; p99 token gap 24.20 ms |
| Fresh-prefix retrieval | 1 of 1 pass | 1,556 reported prompt tokens; TTFT 5.344 s |
| Higher-context fresh-prefix retrieval | 1 of 1 pass | 5,656 reported prompt tokens; TTFT 22.594 s |

The higher-context control is below the 8,192-token serving limit because its
single request must also reserve output capacity.  A 16K control is outside
this qualified server configuration and is not represented as a passed test.

The C09 window also discovered that an older recovery launcher had created a
healthy but non-dedicated process group.  The controller refused to stop it,
as designed.  After verifying that the group contained only the FreeToken
frontend and its own worker children, it was replaced using the dedicated
`setsid` launcher.  The restored service reported `status: ok`, and its server
PID, process-group ID, and session ID were all identical.  This repair is a
reliability prerequisite for later time-share tests, not a throughput result.

**Decision: performance eligible, not yet accepted.** The current HIP Triton
router configuration has cleared the available deterministic, state, long-
context, and recovery gates.  It must still complete the fixed five-sample API
matrix and concurrent workload with a fresh exact-Q4 reference before it can
replace the 47.960-TPS baseline.  Preserve the quality artifacts
`qwen-router-c08-quality-20260901T181143Z` and
`qwen-router-c09-full-quality-20260901T183253Z` on GMKtec EVO-X2.

### C10: router-only exact-Q4 API and concurrency matrix

The clean router-only checkout then completed the fixed five-sample API matrix
and the three-round concurrency controls.  This is the same checkout that
passed C09 quality, with the rejected decode-wave experiment excluded.  The
benchmark used the exact Q4 model, a fixed 48-line prompt, a 256-token single
decode, and the model's valid tokenizer.  It was served only on the isolated
loopback candidate port while the ordinary NVFP4 API was stopped by the
recovery controller.

| Workload | Mean throughput | p99 TTFT | p99 token gap |
| --- | ---: | ---: | ---: |
| Single request, five samples | 48.282 decode tokens/s | 0.432 s | 24.26 ms |
| Concurrent 1, three rounds | 40.138 aggregate tokens/s | 0.436 s | 40.67 ms |
| Concurrent 2, three rounds | 57.913 aggregate tokens/s | 0.854 s | 53.63 ms |
| Concurrent 4, three rounds | 81.456 aggregate tokens/s | 1.174 s | 76.45 ms |

The single-request result is 0.67 percent above the accepted 47.960-token/s
baseline.  That is within normal run-to-run variation and below the campaign's
minimum promotion gate of a repeatable one percent gain.  The candidate is
therefore quality-qualified and load-stable, but it is not a new performance
baseline and must not be promoted on this evidence alone.

The controller stopped the candidate, restarted the ordinary NVFP4 API, and
verified `status: ok`.  The recovered server PID, process-group ID, and session
ID were identical, and no listener remained on the candidate port.  Preserve
the complete artifact `qwen-router-c10-api-20260901T185412Z` on GMKtec EVO-X2.

**Decision: do not promote.** Retain the current HIP Triton router as a
quality-qualified route, but focus the next iteration on data movement and
expert-cache work, where an end-to-end gain remains plausible.

### C11: upstream expert-cache copy-plan audit

Upstream's newer expert-cache copy-plan changes were merged only into a
disposable source checkout.  The candidate preserves the qualified in-tree
router path and adds the upstream copy-plan and AOT-catalog corrections.  Two
stale test expectations were found and corrected in that disposable checkout:
the router test expected a retired reference dispatch policy, and the AOT test
expected two unsupported legacy copy shapes to be compiled even though the
upstream code intentionally filters them out.

The corrected CPU-side controls passed: three host-residency and locked-layer
copy tests, plus two strict AOT-grid selection tests.  The normal NVFP4 API
reported `status: ok` after the checks.  The saved CPU evidence is
`upstream-cache-c11-retry-20260901T191217Z` on GMKtec EVO-X2.

However, the focused ROCm fused-MoE suite also produced an illegal-memory-
access fault in `fused_moe_kernel` while testing the disposable candidate.
The failed test process was a separate one-process group and was terminated;
GPU utilization returned from 100 percent to 5 percent, and the ordinary API
remained healthy.  This failure cannot be attributed to the copy-plan change
because that fused-expert test path was not changed by the upstream copy-plan
commit.  It is nevertheless a real safety failure on the target ROCm stack.

**Decision: safety-blocked.** Do not merge or benchmark the upstream cache
candidate yet.  The next iteration must stop the normal service, reproduce the
fused-expert fault with serialized kernel dispatch and a minimal shape, compare
the candidate with the accepted source, then repair or replace the failing
kernel before any cache-copy throughput claim is considered.

### C12: accepted-source fused-MoE fault reproduction

The mandatory isolated reproduction was run against the accepted source,
not the upstream cache candidate.  The ordinary NVFP4 API was stopped through
the recovery controller, and the smallest failing test was launched in its own
session with serialized ROCm dispatch.  The test failed in 3.42 seconds with a
HIP illegal-memory-access fault in `fused_moe_kernel` on the float16 grouped
MoE shape: 4 tokens, 37 experts, hidden size 32, intermediate size 24, and
top-k 4.  The exact source revision, test output, exit status, stop log, and
recovery log are saved in `fused-moe-c12-baseline-20260901T191320Z` on GMKtec
EVO-X2.

This establishes that the failure predates the upstream cache-copy candidate.
It also rules out a simple test-runner race because serialized dispatch reports
the same faulting `fused_moe_kernel`.  The normal NVFP4 API was restored by the
controller and again returned `status: ok` after its normal cold load.

**Decision: repair prerequisite confirmed.** The next code change must add a
ROCm-safe grouped-MoE selection or correct the kernel bounds issue for this
shape, backed by the isolated reproducer.  The upstream cache candidate remains
on hold until that repair passes and no longer faults the target ROCm runtime.

### C13-C20: grouped-MoE root cause, repair, and candidate safety checks

The first repair narrowed the second grouped projection to its actual flattened
storage layout.  That removes a real stride and bounds hazard, but the minimal
reproducer still failed at the first projection.  The next diagnostic compared
the two alignment implementations directly.  The compact alignment kernel
returned valid sorted token IDs while assigning every padded block to expert
zero, even when the routed experts were distinct.  Its outputs therefore could
not safely select grouped expert weights on this AMD runtime.

The ROCm path now selects the staged in-tree alignment implementation, which
returned the expected distinct expert IDs for the same route.  The repaired
minimal reproducer passed, followed by the four-shape parity group and a new
regression test that asserts every routed expert is represented in the padded
alignment output.  The saved GPU artifacts are
`fused-moe-c13-stride-20260901T192*Z`,
`fused-moe-c14-align-20260901T193029Z`,
`fused-moe-c15-alt-align-20260901T193812Z`,
`fused-moe-c16-align-fix-20260901T194538Z`,
`fused-moe-c17-full-parity-20260901T195315Z`, and
`fused-moe-c18-regression-suite-20260901T200025Z` on GMKtec EVO-X2.

The candidate containing the upstream cache-copy plan was then merged with the
repair into isolated source revision `340ed31`.  Its focused safety suite
passed 5 tests with 34 intentionally deselected, and its direct fused-copy
comparison matched the legacy per-bank copy for 0, 1, 4, and 8 cache misses.
Those artifacts are `upstream-cache-c19-safety-20260901T200857Z` and
`upstream-cache-c20-fused-copy-20260901T201650Z`.

**Decision: safety gate passed, performance gate not yet passed.** The repair
is eligible for model quality requalification.  No throughput claim follows
from these unit and direct-copy tests alone.

### C21-C22: revision-matched reusable HIP cache

The integrated candidate had no reusable cache for source revision `340ed31`.
The first maintenance wrapper found a helper-file execute-bit defect before it
ran the builder, so it produced no benchmark result and recovery was corrected
immediately by invoking the reviewed helpers through Bash.  The normal service
then completed its measured serial cold recovery in 5 minutes 57 seconds.

The corrected isolated build compiled all 82 explicit C++ and HIP cache modules
for AMD Radeon 8060S Graphics with HIP `7.15.26333`, writing them under
`kernel-cache-rocm-gfx1151-340ed31`.  The subsequent verifier loaded all 82
modules with `FREETOKEN_DISABLE_JIT=1` and reported `status: passed`.  The
artifact `upstream-cache-c22-build-20260901T203402Z` retains the build log,
strict verifier output, and recovery record on GMKtec EVO-X2.

**Decision: reusable-cache gate passed.** Future runs of this exact candidate
must point at this revision-matched cache and retain JIT disabled.  This avoids
per-run native kernel compilation without pretending that a cache from a
different source revision is ABI-safe.  The Q4 model itself is not recompiled
by this process.

### C23-C25: integrated cache and cache-residency measurements

The repaired upstream cache-copy candidate at source revision `340ed31` passed
the deterministic three-case Q4 quality suite, multi-turn state controls, and
fresh-prefix retrieval at 1,556 and 5,656 reported prompt tokens.  Its
revision-matched reusable HIP cache was loaded with `FREETOKEN_DISABLE_JIT=1`.
The three-sample warm API matrix measured 47.848 decode tokens/s at memory
ratio 0.25, 0.23 percent below the accepted 47.960-token/s Q4 baseline.

Decode cache telemetry reported 40,800 layer calls with eight active experts
per layer and a 7.45 percent miss rate.  Raising the cache-residency ratio from
0.25 to 0.30 increased resolved cache slots from 5,470 to 7,051 but reduced
the three-sample mean to 47.032 decode tokens/s, 1.94 percent below baseline.
The 0.30 candidate preserved all three deterministic quality checks; its warm
TTFT mean was 0.454 s and token-gap p99 was 24.01 ms.

**Decision: reject cache-copy and larger-residency as throughput routes.** The
copy candidate is safe and quality-preserving, but neither cache transfer nor
additional resident experts produces a measurable decode gain on this Q4
workload.  The low remaining miss fraction also makes a 50 percent gain from
cache sizing implausible.  Preserve `upstream-cache-c23-q4-quality-20260901T204241Z`,
`upstream-cache-c24-cache-telemetry-20260901T205755Z`, and
`upstream-cache-c25-r030-20260901T210838Z` on GMKtec EVO-X2.  Each candidate
was stopped and the normal NVFP4 API recovery controller was started after its
window.

### C26: single-stream decode graph capture

The existing Q4 launcher deliberately disabled decode graph replay.  A new,
default-off `CUDA_GRAPH_MAX_BS` launcher parameter makes a bounded graph
experiment explicit and reproducible without changing the protected normal
service.  The isolated Q4 candidate captured batch size one successfully,
consuming approximately 0.25 GiB of additional GPU-visible memory and leaving
22.81 GiB free after capture.

The first quality attempt was invalid: the harness was given the API root
instead of the required OpenAI-compatible `/v1` path, so all three requests
received HTTP 404 before model inference.  No TPS result was collected and the
candidate must not be classified as a quality or performance failure.  The
candidate was stopped through its verified dedicated process group, no test
listener remained, and normal-service recovery was started.

**Decision: graph candidate remains pending.** Re-run the deterministic suite
and fixed TPS matrix against the corrected `/v1` endpoint after verified normal
service recovery.  The failed endpoint artifact
`q4-c26-graph-bs1-20260901T212038Z` remains part of the provenance record as a
harness-configuration failure.  The candidate startup also rebuilt its
checkout-local GGUF HIP extension, not the GGUF model.  A later promotion
requires a revision-matched reusable extension cache and a strict no-JIT
verification for that checkout.

### C27: corrected single-stream graph qualification

The graph candidate was repeated after normal-service recovery with the
OpenAI-compatible `/v1` endpoint.  It passed all three deterministic controls,
captured batch size one in 3.26 seconds using the already-built checkout-local
GGUF HIP extension, and kept 22.81 GiB of GPU-visible memory free after graph
capture.

The fixed short canary matrix measured 49.477 decode tokens/s across three
samples, 3.16 percent above the accepted 47.960-token/s baseline.  Its mean
warm TTFT was 0.343 s and token-gap p99 was 24.54 ms.  This result alone was
not sufficient for promotion.  The independent scheduler-shaped three-sample
matrix measured 48.278 decode tokens/s, essentially equal to the previously
qualified non-graph control at 48.282 decode tokens/s.  Its mean warm TTFT was
0.430 s and token-gap p99 was 24.41 ms.

**Decision: reject graph replay as a throughput promotion.** It is functionally
correct on the AMD path, but its apparent short-canary gain did not reproduce
on the scheduler-shaped workload and therefore fails the campaign requirement
for a repeatable gain above normal variation.  Keep the new launcher parameter
defaulted to zero for reproducible future investigation, but do not enable it
for normal service.  Preserve `q4-c27-graph-bs1-v1-20260901T213143Z` on
GMKtec EVO-X2.  The verified candidate process group was stopped, its loopback
ports were clear, and normal NVFP4 service recovery was started.

### C28: two-row GGUF MMV grouping screen

The prior ROCm trace showed that Q4_K and Q5_K routed-expert vector kernels
dominate decode GPU time.  A narrow HIP compile-time experiment therefore
made the MMV output-row grouping explicit.  The default remains one row, while
the isolated candidate used exactly two rows per workgroup through
`FREETOKEN_GGUF_MMV_Y=2`.  Host-side validation tests passed 3 of 3 before GPU
use.  The candidate compiled into a dedicated extension-cache directory, and
the HIP build log records `-DGGML_CUDA_MMV_Y=2` for `gfx1151`.

The two-row candidate passed all three deterministic Q4 quality controls, but
its fixed three-sample throughput mean was 47.954 decode tokens/s.  This is
effectively equal to, and fractionally below, the 47.960-token/s baseline.
Mean warm TTFT was 0.350 s and token-gap p99 was 24.76 ms.

**Decision: reject the two-row MMV grouping.** The configuration preserves
quality but does not create a measurable single-stream decode gain.  Keep the
compile switch defaulted to one row and retain it only as a reproducible
diagnostic control.  Preserve `q4-c28-mmv-y2-20260901T214526Z` on GMKtec
EVO-X2.  The candidate was stopped through its verified process group, its
loopback ports were confirmed clear, and normal NVFP4 service recovery was
started.

### C29: llama.cpp RDNA4 eight-wave MMVQ policy screen

The current llama.cpp ROCm implementation selects eight independent Wave32
rows for simple one-vector MMVQ formats on RDNA4, including Q4_K, Q5_K, Q6_K,
and Q8_0.  FreeToken's vendored Q4_K vector-dot arithmetic is otherwise the
same as the reference implementation, so this candidate changed only the
HIP-only launch geometry for those traced kernel families.  The default
one-row launch remains unchanged.  The candidate required the explicit
`FREETOKEN_GGUF_MMV_Y=8` build setting and compiled into its own revisioned
extension-cache directory with `-DGGML_CUDA_MMV_Y=8` for `gfx1151`.

The candidate source revision `05751ef` passed all three host-side validation
tests, then passed the deterministic Q4 API controls: exact canary, arithmetic,
and JSON schema.  The first quality request included HIP extension compilation
and is retained as startup evidence only.  It was excluded from steady-state
throughput.  The subsequent warm fixed scheduler matrix produced 48.374,
48.103, and 48.357 decode tokens/s, for a 48.278 mean and 48.357 median.
That is a 0.66 percent mean increase over the accepted 47.960-token/s
baseline, below the campaign's one-percent promotion floor.  Warm TTFT averaged
0.425 s and token-gap p99 was 24.91 ms.  The retained per-sample prompt-token
and TTFT fields also yield a client-visible prefill rate of 2,852.315 prompt
tokens/s mean, with a 2,846.519 to 2,856.972 range.  This is an end-to-end
prompt-to-first-text measurement, not the server's narrower internal input
throughput counter.

**Decision: reject the eight-wave MMVQ policy for promotion.** It preserves the
screened output quality and is modestly faster in this one matrix, but the
measured increase is too small to distinguish safely from host variation and
does not meet the repeatable-gain requirement.  Do not merge the candidate
branch or change the default.  Preserve `q4-c29-rdna4-mmvq8-20260901T215745Z`
on GMKtec EVO-X2, including raw quality, per-token timing, HIP build, and
recovery evidence.  The isolated process was stopped; a stale executable bit
on the normal recovery start helper was corrected before normal-service
recovery was launched.

### C30: ROCprof lifecycle qualification

The next optimization decision requires a kernel and memory-copy trace of the
same Q4 control, not an inference-rate estimate from a profiler.  ROCprofv3
was therefore launched through the ROCm SDK bundled with the active PyTorch
wheel.  That avoids loading a second LLVM runtime from the system ROCm tree.
The normal NVFP4 server was stopped only after a healthy API check and was
restarted after every isolated candidate attempt.

The first attempt exposed two setup defects before inference: a source checkout
that did not register the Qwen GGUF architecture, followed by a Qwen-capable
checkout without its required native pinned-memory extension.  The native
extension was then built from that checkout with the installed ROCm 10 HIP
toolchain and successfully imported.  This is build provenance, not a model
conversion or a change to the protected service.

The next run reached the Q4 OpenAI-compatible API and built the checkout-local
GGUF HIP kernel on its first real request.  Its resulting decode output was
intentionally excluded from performance comparison because the profiler uses a
system-memory intercept queue and the request included one-time compilation.
A warm-cache repetition completed one 900-token bounded workload in the
configured collection interval.  The server's diagnostic decode log was about
30 tokens/s under profiling, versus the qualified unprofiled control near 48
tokens/s.  This demonstrates that ROCprof output must not be used as a TPS
measurement.

**Decision: no C30 performance claim.** The candidate and all verified helper
processes were stopped, the disposable loopback listener was confirmed absent,
and the normal NVFP4 API was healthy again.  The forced recovery prevented the
profiler from finalizing a usable `rocpd` database, so the trace cannot yet be
used to rank kernels or justify a code change.  Preserve
`q4-c30d-profile-default-20260901T223610Z`,
`q4-c30e-profile-default-20260901T223951Z`, and
`q4-c30f-profile-warm-cache-20260901T224449Z` as provenance.  The next action
is a documented controller that prewarms the exact isolated cache, runs a
bounded workload during collection, requests graceful profiler finalization,
and verifies the resulting database before normal-service recovery.

### C32: finalized warm-cache ROCprof trace

The revised controller prewarmed the exact Q4 server before profiling, waited
for the scheduler's explicit ready record instead of treating the frontend
models endpoint as inference-ready, and then ran one bounded 900-token request.
It produced a finalized 1.44 GiB `rocpd` SQLite database and the normal NVFP4
API was healthy again after the time-share recovery.  The profiler's own queue
intercept mode changes runtime behavior, so none of these diagnostic values are
used as TPS measurements.

The final 30-second active-dispatch window identifies the decode work that must
be optimized.  The dominant entries were Q8 vector dot at 2,011.326 GPU ms,
Q4_K routed-expert vector dot at 529.036 GPU ms, Q6_K vector dot at 474.200
GPU ms, and Q5_K routed-expert vector dot at 395.507 GPU ms.  The next largest
non-vector components were attention GEMM at 232.153 GPU ms, delta-rule fused
gating at 173.428 GPU ms, grouped decode stage one at 82.297 GPU ms, and cache
index copying at 74.211 GPU ms.  The trace recorded no memory-copy events in
this final window.

**Decision: prioritize the traced GGUF vector-dot path.** The existing cache
residency and graph experiments cannot plausibly deliver the campaign target by
themselves.  Future candidates must preserve the qualified Q4 output checks,
change one vector-kernel dispatch or arithmetic behavior at a time, and pass
two independent throughput matrices before promotion.  Preserve
`q4-c32-rocprof-controller-ready-20260901T225400Z`, including the raw SQLite
database, workload response, controller logs, and normal-service recovery
evidence.
### C33: current-branch MMV-Y4 requalification

The profiler-ranked vector path was rechecked against the current branch rather
than relying on the older candidate artifact. The opt-in
`FREETOKEN_GGUF_MMV_Y=4` build from commit `ff76ede` reached API readiness in
an isolated checkout with 56 GiB free before model loading and 23.07 GiB free
after initialization. Its fixed three-sample scheduler-shaped matrix passed
all requests and measured 45.4603 mean decode TPS, 2,753.3639 mean
client-observed prefill TPS, 0.0927 decode-TPS standard deviation, and 43.781
ms maximum token gap. The canonical current AIME output hash was
`3302eda43396`; the older helper's superseded `0acef4eab6f4` expectation was
classified as a harness-version discrepancy.

**Decision: reject current-branch MMV-Y4 for promotion.** It was approximately
5.8 percent slower than the accepted current Q4 scheduler control near 48.28
decode TPS. The default remains one row, and the Y4 switch remains opt-in for
future architecture-specific investigation. Normal Qwen service recovery was
verified with `status: ok` and `maintenance: serving`.
