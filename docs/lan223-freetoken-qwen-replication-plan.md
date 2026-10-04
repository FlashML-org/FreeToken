# LAN-223 FreeToken Qwen replication and Strix Halo optimization plan

## Decision and success statement

This plan targets only LAN-223, a Ryzen AI Max+ 395 with Radeon 8060S
(`gfx1151`) and shared LPDDR5X memory.  It does not alter LAN-199, LAN-215,
llama-swap, or any production model service.

The first target is the exact model used for FreeToken's documented 8 GB laptop
result: `Qwen/Qwen3.6-35B-A3B`, using the upstream-supported deployment
format that can be reproduced on both systems.  The published claim to
replicate is 39.3 generated tokens per second on an 8 GB RTX 4060 laptop.
This is a model-specific reference, not a general statement that all FreeToken
models fit in 8 GB of VRAM.

The program is successful only when LAN-223 can run the documented Qwen
workload through the native ROCm and HIP FreeToken server with:

1. A fully recorded, exact model and workload contract.
2. Deterministic greedy-output equivalence to a trusted reference for each
   test prompt, plus task-level quality scores where deterministic equality is
   unsuitable.
3. Repeated warm and cold performance measurements with raw artifacts.
4. Explicit measurement of decode TPS, prompt TPS, TTFT, tail token latency,
   memory use, cache behavior, CPU utilization, GPU utilization, clocks,
   temperatures, and throttling.
5. An evidence-backed comparison to the published 39.3 TPS reference on an
   equivalent workload, without claiming equality when prompts, sampling,
   hardware tier, or metric definitions differ.
6. A stable configuration that is safe to expose through FreeToken's
   OpenAI-compatible API after the campaign, but before any llama-swap work.

The longer-term ambition is to exceed the published result on the same model.
That ambition is a hypothesis, not an acceptance assumption.  It must be
supported by a matched benchmark and quality evidence.

## Why this is a different engineering problem on Strix Halo

FreeToken's 8 GB RTX 4060 result uses a discrete GPU, dedicated VRAM, host
DRAM, and a PCIe link.  Its MoE policy can retain hot experts in VRAM while
placing other experts in host memory, fetching misses or computing selected
misses on the CPU.

LAN-223 has UMA.  Its CPU and Radeon 8060S access the same memory pool.  This
can remove PCIe-copy cost and can permit a larger hot-expert cache than an 8 GB
discrete GPU.  It can also be worse if the CPU fallback, GPU compute, KV cache,
and operating system contend for the same LPDDR5X channels.  A direct copy of
the CUDA policy is therefore an invalid optimization target.  The AMD runtime
needs a measured UMA policy.

## Non-negotiable controls

### Scope and safety

- Maintain a LAN-223 host allowlist in every benchmark launcher and refuse any
  other hostname or IP address before contacting a server.
- Use an isolated work directory under `/home/david/freetoken-amd/artifacts/`.
- Bind experiments to loopback or a non-production LAN-223 test port.
- Do not change llama-swap configuration, routes, model aliases, startup
  services, or model files used by production services.
- Store credentials only as environment-variable references.  Do not save,
  print, commit, or upload secrets.
- Do not overwrite previous evidence.  Every experiment receives a UTC
  timestamp, a unique run identifier, and a manifest.

### Environment freeze

For each candidate, save a machine-readable manifest containing:

- Git commit, branch, clean or dirty tree state, and patch hash.
- Linux distribution, kernel, CPU microcode, BIOS version, memory amount,
  configured UMA aperture, and storage mount information.
- ROCm runtime, HIP compiler, AMD GPU driver, PyTorch ROCm build, Triton
  version, Python ABI, compiler flags, and `HSA_OVERRIDE_GFX_VERSION` if set.
- `rocminfo`, `rocm-smi`, CPU topology, NUMA map, memory-frequency data where
  exposed, and the process CPU affinity and priority.
- Exact model revision, all shard checksums, tokenizer revision, configuration
  files, conversion output checksums, and FreeToken weight-format version.
- `TORCH_EXTENSIONS_DIR` location and native-extension binary hash.  A normal
  benchmark must reuse the compiled HIP extension, not compile during timing.

Reject a performance comparison if any material environment item differs and
the difference is not recorded in the comparison table.

## Phase 0: establish the paper replication contract

### 0.1 Extract the authors' actual benchmark protocol

Read the paper, repository history, benchmark scripts, released configs, issue
threads, and desktop defaults.  Produce
`artifacts/<run>/upstream-protocol.md` with citations and exact quotes kept
short.  Resolve, rather than assume:

- Checkpoint name, source revision, quantization, weight format, and total
  downloaded size used for the 39.3 TPS RTX 4060 result.
- RTX 4060 laptop CPU, RAM capacity and speed, operating system, CUDA version,
  driver, FreeToken commit, GPU memory budget, and any desktop-app defaults.
- Prompt text and token count, completion length, warmup procedure, context
  reuse state, concurrency, sampling parameters, stop tokens, and whether the
  first generated token is excluded from decode measurement.
- Whether 39.3 TPS is mean, median, best run, a workload average, or a
  single-run sample; record its confidence interval if provided.
- MoE backend, expert cache budget, CPU thread count, `ft bench bw` result,
  CPU split, and any automatic policy selected by the reference machine.
- TTFT definition and whether server-internal timing or client-observed timing
  is used.

Do not label a LAN-223 result as a reproduction until all fields are known or
explicitly listed as unavailable from the authors.

### 0.2 Define a metric dictionary before testing

All benchmark scripts must emit these definitions unchanged:

| Metric | Definition |
| --- | --- |
| Cold start | Process launch through first successful non-streamed response, including model load and extension compilation only when intentionally requested. |
| Warm TTFT | Client-observed request send to first SSE content token after a completed warmup. |
| Decode TPS | `(completion_tokens - 1) / (last_content_token_time - first_content_token_time)`.  Report zero or one token completions separately. |
| Prompt TPS | Input tokens divided by client-observed TTFT, labelled end-to-end rather than kernel-only. |
| Token-gap p50, p95, p99 | Distribution of streamed content-token intervals, excluding SSE framing-only events. |
| Quality score | Task-specific judged result, with model output, scorer version, and parsing errors retained. |
| Effective memory bandwidth | Actual bytes moved divided by measured wall time for the relevant serving phase.  Never substitute a microbenchmark ceiling. |

Report client and runtime-internal figures in separate columns.  Do not compare
one runtime's internal timing to the other's HTTP timing.

### 0.3 Create the baseline protocol package

Create a versioned benchmark package under `benchmarks/lan223_qwen/` with:

- A static JSON request corpus and expected tokenizer counts.
- A local API client that captures raw SSE timestamps using a monotonic clock.
- A warmup runner, a cold-start runner, a fixed-length decode runner, and a
  multi-turn agentic runner.
- A process guard that checks the host identity and fails closed outside
  LAN-223.
- Telemetry collection with timestamps aligned to each request.
- A manifest writer and checksum verifier.
- A result parser that emits JSON, CSV, and a Markdown table without changing
  raw logs.
- Unit tests for the TPS calculation, token-event parsing, error
  classification, host allowlist, and manifest validation.

## Phase 1: qualify the model and its quality before optimization

### 1.1 Use the exact primary model path

The main candidate is the official `nvidia/Qwen3.6-35B-A3B-NVFP4` checkpoint
already supported upstream and validated functionally on LAN-223.  Preserve
the original model directory as read-only.  Build any FreeToken fast-weight
conversion once, checksum it, and reuse it across every trial.

Run a separate, explicitly labelled compatibility check for the checkpoint and
format used by the authors if it differs from NVFP4.  Do not merge the two
results into one number.

### 1.2 Establish three independent correctness references

Use three types of evidence:

1. **FreeToken NVIDIA reference**: upstream FreeToken on supported NVIDIA
   hardware when available.  Fix greedy decoding and retain raw token IDs.
2. **Independent AMD control**: llama.cpp ROCm on LAN-223 using a compatible
   Qwen quantization and a carefully documented template.  It is a quality
   control, not a performance proxy when the format differs.
3. **Model-level evaluation**: a small, fixed benchmark suite with exact
   prompts and deterministic scoring.

### 1.3 Quality suite

The suite must contain at least:

- Arithmetic and structured reasoning questions with machine-verifiable final
  answers.
- Code generation tasks that execute in a sandboxed test harness.
- Retrieval and instruction-following prompts with explicit required facts.
- Tool-call JSON generation with schema validation.
- A multi-turn editing and correction set that exercises FreeToken's semantic
  cache behavior.
- Long-context prompts at 2K, 8K, 16K, and the largest stable context that
  fits the selected KV allocation.

For every prompt, retain rendered prompt text, token IDs when practical, model
response, finish reason, scorer result, and error class.  Categorize
differences as template or tokenizer, numerical drift, truncation, parsing
failure, serving failure, or genuine task-quality regression.

The gate before performance tuning is:

- No crash, corruption, NaN, malformed streaming sequence, or silent fallback.
- Greedy outputs must be byte-identical where the same tokenizer, template,
  weights, precision, and decode settings are used.
- Where cross-runtime byte identity is impossible, the quality suite must show
  no statistically meaningful regression relative to the selected reference.

## Phase 2: establish unoptimized but comparable LAN-223 baselines

### 2.1 Baseline matrix

Run five clean, independently started server samples per row, after a defined
warmup.  Capture at least:

| Row | Purpose |
| --- | --- |
| Upstream-like automatic policy | Establish how the current port behaves without hand tuning. |
| GPU-resident maximum safe expert cache | Test UMA's likely advantage. |
| Explicit offload | Establish whether FreeToken's discrete-GPU policy transfers at all. |
| Explicit CPU | Measure CPU-only expert-miss cost and shared-memory contention. |
| Explicit hybrid | Measure whether concurrent CPU and GPU execution helps or hurts on UMA. |
| llama.cpp ROCm control | Establish the current AMD alternative using a documented compatible workload. |

For every row record warm and cold measurements, 128-token decode, 512-token
decode, the paper-matched workload, and the agentic workload.  Separate stable
samples from samples contaminated by active disk I/O, CPU contention, thermal
transition, process leaks, or unexpected compilation.

### 2.2 Telemetry and contamination controls

Collect, at one-second cadence and around every request:

- GPU clock, temperature, power, memory activity, GPU busy, and reset events.
- CPU frequency, package power, per-core utilization, migrations, page faults,
  context switches, major faults, and memory pressure.
- RAM and swap use, cache residency, paging activity, I/O throughput, and
  blocked processes.
- Expert-cache hits, misses, evictions, bytes fetched, CPU expert work,
  resident slots, KV bytes, and cache resize events.
- HIP graph-capture status, stream synchronization counts, kernel launch
  counts, and extension cache hits.

Reject and rerun any sample with swapping, thermal throttling, unexpected
compilation, stale server processes, active model-copy jobs, or unexplained
host I/O contention.  Preserve rejected evidence and its rejection reason.

## Phase 3: make the AMD implementation observable

### 3.1 Add low-overhead serving counters

Add a structured, opt-in telemetry mode.  It must never alter numerical output
or become enabled by default.  Counters must include per token and aggregate:

- Router duration, selected expert IDs, unique experts, and route reuse.
- GPU-cache hits, misses, evictions, resident bytes, and cache wait time.
- Expert staging or read time, CPU compute time, HIP copy time if any, and
  queue overlap time.
- Attention, dense projection, MoE projection, normalization, sampling, and
  synchronization time.
- GPU and CPU work submitted versus completed, including backpressure.

Use an event-ring buffer and bulk flush at request completion.  Do not emit
one log line per kernel during a scored run.

### 3.2 Build a profiler ladder

Use three tiers, in order:

1. Application counters for every performance run.
2. HIP events around named execution regions for candidates that pass the
   application gate.
3. ROCm profiler traces only on representative runs because tracing changes
   timing.

For each suspected bottleneck, first prove that its percentage of end-to-end
decode time is large enough to matter.  A microbenchmark improvement is not a
candidate for integration unless it survives a complete API run with identical
quality output.

## Phase 4: derive a Strix Halo UMA execution policy

### 4.1 Measure the actual memory system

Create targeted measurements for:

- CPU-only sequential and realistic expert-shaped reads.
- GPU-only read and quantized GEMV throughput on Qwen dimensions.
- Concurrent CPU expert compute and GPU expert compute.
- Concurrent CPU reads and GPU compute.
- Expert-cache promotion and eviction under the actual route sequence.
- KV-cache growth with 2K, 8K, 16K, and long-context requests.

The critical output is a contention curve, not a peak bandwidth number.  Plot
decode TPS and token latency against CPU contribution, hot-expert cache size,
and KV allocation.

### 4.2 Replace PCIe-centric assumptions

Implement a `uma` policy mode that starts from measured resource contention:

- Prefer GPU execution for hot experts when sufficient shared-memory headroom
  exists.
- Cap CPU fallback when concurrent CPU work reduces GPU progress more than it
  contributes.
- Make the expert-cache target a function of free memory, current KV use,
  observed route locality, and the measured contention curve.
- Allocate pinned host buffers only if profiling proves they help on ROCm UMA.
  Do not assume pinned memory is beneficial just because it helps PCIe.
- Avoid copy paths that duplicate bytes inside the same physical memory pool
  when a direct-access or zero-copy path is correct and measurable.
- Resize the cache only at scheduler safe points, with hysteresis to avoid
  cache thrash during alternating long-context and short decode requests.

The policy must fall back to the existing portable behavior on discrete AMD
hardware and must not modify the CUDA decision path.

### 4.3 Optimizer acceptance rule

For every policy candidate, run the full five-sample API matrix and quality
check.  Accept only when all apply:

- Median decode TPS improves by at least 1 percent over the current accepted
  baseline, or the confidence interval proves a smaller improvement is real.
- Mean TTFT does not regress by more than 5 percent unless the candidate is
  explicitly a decode-only mode.
- p99 token gap does not materially worsen.
- No quality, determinism, memory-safety, crash, or resource-leak regression.
- The improvement persists in a second clean-host matrix.

## Phase 5: HIP and ROCm kernel program

### 5.1 Start with profile-ranked Qwen kernels

Use the Qwen trace, not Gemma's profile, to rank work.  Expected candidates are
the NVFP4 expert GEMV path, quantization deblocking, routed-expert gather and
scatter, router reductions, attention decode, and synchronization between
small expert operations.  Recompute the ranking after every accepted system
policy change.

### 5.2 Kernel development principles

- Maintain separate CUDA and HIP paths behind compile-time guards.
- Use `gfx1151` feature gates that do not accidentally apply to other AMD
  architectures.
- Prefer current ROCm intrinsics and inspect generated ISA before judging a
  kernel by source appearance.
- Match Qwen's real matrix sizes, expert count, top-k, activation dtype, and
  batch shape in every microbenchmark.
- Measure register pressure, occupancy, wave size, LDS use, cache behavior,
  and memory coalescing before changing launch geometry.
- Fuse adjacent operations only when the full decode trace proves that launch
  or global-memory overhead dominates and the fusion does not reduce occupancy.
- Preserve numerically safe accumulation and check output tolerances against a
  high-precision reference.

### 5.3 Likely high-value technical leaps to investigate

These are experiments, not promised outcomes:

1. **Route-aware expert prefetch.**  Predict a small next-token expert set
   from recent routes, stage it into the shared-memory cache during current
   token compute, and measure false-prefetch cost versus cache-miss reduction.
2. **Layer-local route batching.**  Group duplicate expert selections within a
   layer without changing route order or output semantics, reducing tiny launch
   and synchronization overhead.
3. **Persistent decode scheduler.**  Replace host-driven chains of small HIP
   launches with a graph-captured or persistent sequence where ROCm profiling
   proves launch overhead is dominant.
4. **Qwen NVFP4 RDNA3.5 GEMV specialization.**  Specialize the exact expert
   shapes and quantization layout for `gfx1151`, using vectorized loads and
   ROCm-supported dot-product instructions where the generated ISA confirms
   the intended instruction sequence.
5. **Unified KV and expert cache allocator.**  Use a single pressure-aware
   allocator so the cache gives back memory to long context before paging or
   repeated reallocation occurs.
6. **Cooperative CPU-GPU routing budget.**  Choose the CPU share per token or
   per layer from recent observed service time, rather than one static hybrid
   ratio from an isolated bandwidth benchmark.

Each experiment requires a design note, a baseline, a rollback commit or
feature flag, microbenchmark evidence, complete-server evidence, quality
evidence, and a documented accept or reject decision.

## Phase 6: paper-parity and superiority trials

### 6.1 Replication trial

Once protocol fields are resolved, run the exact paper-matched Qwen workload
on LAN-223 with the selected stable configuration:

- At least five independent warm-server samples.
- At least three cold-start samples, reported separately.
- Identical prompt and output length rules.
- Greedy decoding unless the source protocol specifies otherwise.
- Same reported statistic as the paper, plus mean, median, standard deviation,
  min, max, bootstrap confidence interval, and raw samples.
- Full telemetry and quality artifact bundle.

The parity threshold is the paper's 39.3 TPS only if the metric and workload
are matched.  If they are not, report an explicitly labelled comparable result
and enumerate every remaining difference.

### 6.2 Better-than-NVIDIA trial

Only after a successful replication trial, test claimed advantages of UMA:

- Higher expert-cache residency at equivalent KV capacity.
- Lower cache-miss cost without PCIe transfer.
- Stable decode under long multi-turn contexts.
- Better tail token latency under the same quality and concurrency settings.
- Sustained throughput with no thermal or memory-pressure degradation.

Use the NVIDIA reference as a published comparison point, not as a reason to
hide protocol differences.  A claim that LAN-223 exceeds the NVIDIA result
requires a same-model, same-precision, same-workload, same-TPS-definition
comparison, or a clearly bounded claim such as "higher end-to-end warm decode
TPS on this specified request."

## Phase 7: reliability and API qualification

Run a 24-hour endurance test only after performance and quality gates pass.
It must use bounded test traffic, a non-production endpoint, and rotate among
short, paper-matched, long-context, streaming, non-streaming, and cache-edit
workloads.  Record:

- Request success rate, error taxonomy, restarts, memory high-water marks,
  cache resizes, context truncations, GPU resets, and server leaks.
- TPS and TTFT trend over time, including first-hour versus final-hour values.
- Exact output hash for repeated deterministic canary prompts.
- Process cleanup after shutdown and no orphan worker, blocked I/O, or runaway
  compilation processes.

Pass conditions: no data corruption, no silent fallback to CPU-only or Vulkan,
no swap thrash, no unbounded growth, no unacceptable output regression, and a
documented recovery procedure tested once.

## Phase 8: publication and upstream readiness

Publish a reproducibility bundle in the fork containing:

- Executive result table that clearly separates functionality, quality,
  paper-parity, and superiority claims.
- Full build guide, environment manifest, model provenance, and checksums.
- Benchmark harness source, fixed request corpus where licensing permits, raw
  result JSON, sanitised logs, and result-generation script.
- Performance tables with client versus runtime timing clearly separated.
- Rejected-candidate register so future work does not repeat failed paths.
- Architecture explanation of why UMA differs from the CUDA offload design.
- Known limitations, non-goals, and exact commands required to reproduce.

Before updating the existing upstream pull request, split changes into focused
commits: portable HIP correctness, instrumentation and tests, and optionally a
portable AMD optimization.  Keep LAN-223-specific evidence and tuning defaults
in this fork unless upstream maintainers request them.  Do not claim general
AMD support from a single `gfx1151` result.

## Required acceptance table

| Gate | Evidence required | Status at plan creation |
| --- | --- | --- |
| Native ROCm and HIP execution | Compiled HIP extension, ROCm telemetry, no substitute backend | Achieved for existing Qwen and Gemma validation |
| OpenAI-compatible serving | `/v1/models`, streaming and non-streaming responses | Achieved for existing Qwen and Gemma validation |
| Qwen model quality | Exact reference or scored task suite | Not yet complete |
| Authors' 39.3 TPS protocol reconstructed | Cited protocol artifact with every field resolved or marked unavailable | Not yet complete |
| Matched Qwen TPS replication | Five-sample paper-matched result and raw evidence | Not yet complete |
| Exceeds paper result | Same-model, same-workload, same-metric evidence | Not yet complete |
| Beats AMD llama.cpp Qwen control | Matched ROCm quality and performance matrix | Not yet complete |
| UMA policy is beneficial | Complete API matrix, telemetry, second clean-host confirmation | Not yet complete |
| Long-run reliability | 24-hour isolated endurance artifact | Not yet complete |
| Upstream-ready documentation | Reviewed, reproducible, secret-safe bundle | Not yet complete |

## Immediate next actions

1. Resolve the authors' 39.3 TPS protocol and freeze the Qwen benchmark
   contract.
2. Implement the LAN-223-only harness and manifest schema before altering
   another performance kernel.
3. Re-run the current Qwen NVFP4 baseline with five samples, correct telemetry,
   and quality canaries.
4. Measure UMA contention across cache size and CPU contribution.
5. Use the resulting trace to select one systems-policy candidate and one
   profile-ranked HIP-kernel candidate.

No larger MoE model is admitted to the performance campaign until the Qwen
paper-replication gate is complete.  A larger model can receive a separate
capacity feasibility assessment, but it must not consume the evidence or
optimization budget needed to establish this primary result.
