# LAN-223 Qwen router and cache optimization, 2026-08-29

## Scope

This record covers only the isolated FreeToken server on LAN-223's Radeon 8060S
(`gfx1151`). It did not start, stop, unmask, or reconfigure llama-swap or the
production llama.cpp service. All server instances bound only to `127.0.0.1:1919`.

## Rejected router candidate

FreeToken's vendored Triton softmax top-k router was evaluated on ROCm.
Qwen3.6 NVFP4 uses 256 experts and selects eight experts per token. On
LAN-223, the candidate matched the PyTorch reference in the isolated router
test and reduced router-only latency at the production shape.

| Router microbenchmark | PyTorch reference | HIP Triton | Speedup |
| --- | ---: | ---: | ---: |
| 1 token, 256 experts, top-8 | 0.02449 ms | 0.01512 ms | 1.62x |
| 4 tokens, 256 experts, top-8 | 0.02480 ms | 0.01518 ms | 1.63x |

The focused ROCm test set passed 11 tests, including routing parity cases.
That evidence was necessary but not sufficient: an earlier end-to-end greedy
AIME run produced a different output hash with this router. The candidate is
therefore rejected and ROCm retains the exact PyTorch router.

## Transport canary and decode experiments

The Qwen API harness sends greedy sampling and `reasoning_effort=none`. This
is required because otherwise Qwen can stream its reasoning trace until the
output cap before returning a final answer. The exact-answer canary returned
`LAN223` on warmup and scored requests, proving API transport and parser
behavior only. It is not an end-to-end model-quality acceptance test.

The sustained-decode workload is a 1,212-prompt-token, repeated scheduler
paragraph with 251 forced generated tokens and three scored samples. It is a
warm cache workload, so its input TPS is not an uncached-prefill claim.

| Configuration | Quality | Mean output TPS | Sample output TPS | First-text TTFT |
| --- | --- | ---: | --- | ---: |
| Reference PyTorch router, 8,990 slots | Transport passed previously | 26.731 | 26.724, 26.728, 26.740 | not measured correctly by earlier harness |
| HIP Triton router, 8,990 slots, no graph | Transport passed, AIME regression found later | 29.186 | 29.201, 29.183, 29.175 | 400 to 405 ms |
| HIP Triton router, 10,006 slots, no graph | Transport passed, inherits router regression | 29.080 | 29.077, 29.078, 29.086 | 359 ms canary |
| HIP Triton router, 8,990 slots, graph batch 1 | Transport passed, inherits router regression | 28.830 | 28.827, 28.823, 28.839 | 398 to 404 ms |

The accepted quality configuration retains the PyTorch router. The faster
Triton rows are retained as performance evidence, but must not be used as an
accepted model-serving configuration until their AIME output differs only for
an independently justified numerical reason and task-level quality is proven.

### Current quality restoration proof

After restoring the exact PyTorch router, the same AIME-25 problem zero was
warmed once and measured once against the live LAN-223 server. The checkpoint
used greedy sampling, a thinking-enabled template, and a forced 128-token
decode. The 54-token prompt produced the historic output SHA-1
`0acef4eab6f4` exactly. The dedicated script
`scripts/lan223/verify_qwen_aime_quality.py` now makes this a repeatable
quality gate for every future performance candidate.

The gate now records client-visible timing from the same streamed request. Three
additional warm, quality-matched repeats all produced the reference hash:

| Measure | Result |
| --- | ---: |
| Mean decode TPS | 27.880 |
| Decode TPS samples | 26.786, 28.422, 28.431 |
| Mean warm TTFT | 409.0 ms |
| Prompt / completion tokens | 54 / 127 |
| Output hash | `0acef4eab6f4` in every run |

The first run includes a modest cache or scheduler outlier, with a 50.49 ms
p99 event gap, while the two later runs had 37.81 ms and 37.29 ms p99 gaps.
The three-run mean is 3.6 percent below the historical 28.935 TPS
quality-matched reference. It is therefore a bounded regression, not evidence
that the rejected Triton router should be restored.

## Calibration and rejected alternatives

`ft bench bw` measured Qwen NVFP4's real expert kernels on LAN-223. The CPU
expert path reached 4.8 GB/s, while HIP expert gather reached 92.5 GB/s. That
is a 0.05x CPU-to-gather ratio, so the calibration selected `offload`, not
`hybrid`. CPU and GPU hybrid execution is therefore not a sound optimization
candidate for this checkpoint on this host.

Increasing `--memory-ratio` from 0.35 to 0.38 increased automatic cache
residency from 8,990 to 10,006 slots and reduced free memory from 19.44 GiB to
17.78 GiB. It did not improve decode throughput, so the default remains 0.35.

ROCm graph capture was accepted and completed for batch size one, but reduced
sustained decode throughput by about 1.2 percent. It remains disabled in the
accepted isolated launcher.

## Evidence locations on LAN-223

```text
/home/david/freetoken-amd/artifacts/qwen-reboot-recovery-20260829T085317Z/
/home/david/freetoken-amd/artifacts/qwen-reboot-recovery-20260829T090601Z/
/home/david/freetoken-amd/artifacts/qwen-reboot-recovery-20260829T091716Z/
/home/david/freetoken-amd/artifacts/qwen-reboot-recovery-20260829T093921Z/aime-quality-tps-run1.json
/home/david/freetoken-amd/artifacts/qwen-reboot-recovery-20260829T093921Z/aime-quality-tps-run2.json
/home/david/freetoken-amd/artifacts/qwen-reboot-recovery-20260829T093921Z/aime-quality-tps-run3.json
```

The current best configuration is reloading under:

```text
/home/david/freetoken-amd/artifacts/qwen-reboot-recovery-20260829T092725Z/
```

## Remaining gap

The 29.186 client decode TPS is an informative but rejected performance-only
result, not a quality-validated serving claim. The quality-preserving reference
router is now measured at 27.880 mean TPS with the improved hash-gated harness.
The paper's exact prompt sequence, stop policy, warm-cache state, and source
revision remain unrecovered, so this is not a strict paper-parity comparison.
Further work should profile per-layer NVFP4 expert execution and the Qwen
linear-attention path under native HIP, then repeat this task-level quality and
throughput protocol.

## Native HIP trace and FP8 dense-path investigation

### Trace method and limitations

`rocprofv3 --attach` cannot instrument an already running server with this
PyTorch ROCm wheel because the wheel does not provide the ROCProfiler SDK
attachment registration thread. The evidence was instead captured by launching
the same isolated Qwen command directly through the wheel-compatible
`scripts/lan223-rocprof-wheel-sdk.sh` wrapper. That run created the native
ROCm SQLite trace below and passed the deterministic AIME output gate.

```text
/home/david/freetoken-amd/artifacts/qwen-reboot-recovery-20260829T100601Z/
  rocprof-full-qwen/david-Gmktec-x2-2/54976_results.db
```

The profiler recorded 353,457 dispatches. Its 15.61 decode TPS is intrusive
trace overhead, not serving performance and must never be compared with the
unprofiled client TPS rows in this report.

The added `scripts/lan223/inspect_rocprof_db.py` is a standard-library,
read-only companion for that evidence. It opens the SQLite artifact with
`mode=ro&immutable=1`, inventories ROCm's version-specific table names, and
aggregates a requested final kernel window without altering the database or
requiring a host `sqlite3` package.

### Dominant kernel

The final 120-second active window showed that the largest GPU-time consumer is
not the routed NVFP4 expert kernel. It is the dense mixed-FP8 decode kernel
`_gemv_splitk_kernel` from `fp8_pertensor_linear.py`.

| Kernel | Calls | GPU time in final window |
| --- | ---: | ---: |
| `_gemv_splitk_kernel` | 20,320 | 5,631.844 ms |
| `_gemm_kernel` | 160 | 1,676.018 ms |
| `_decode_nvfp4_marlin_kernel` | 20,320 | 1,566.004 ms |
| `fast_index_copy` | 10,240 | 593.192 ms |
| `_nvfp4_gemv_kernel` | 256 | 322.227 ms |

Qwen's relevant dense projection shapes include `[8192, 2048]`, `[4096,
2048]`, `[2048, 4096]`, and `[512, 2048]`. The initial split-K policy was
written for NVIDIA's much larger GPU target and partitions the K dimension.
Changing that policy changed the numerical reduction grouping, so it cannot be
treated as a quality-neutral performance switch.

### Rejected split-K candidate

An isolated target-512 split-K experiment completed at 22.504 decode TPS and
produced output SHA-1 `1cae5bae914f`, instead of the required
`0acef4eab6f4`. It was both slower and incorrect under the deterministic gate.
The source override was removed and the normal split-K policy restored.

The next candidate is constrained to output-row tiling only: it preserves the
K chunks, each row's FP32 accumulation, partial-buffer layout, and final
split-K reduction order. It is still a candidate, not an accepted optimization,
until it has a saved exact-hash response and an unprofiled TPS result.

### Quality-preserving but inconclusive output-row candidate

The first gfx1151 candidate changed the dense FP8 GEMV output-row tile from 16
to 32. Its three AIME responses matched the required SHA-1 exactly, but a
post-run source audit found that the old automatic split-K calculation was
derived from the output-row tile. The candidate therefore also changed the K
partition, despite being intended as a row-only experiment.

| Tile | Output SHA-1 | Output TPS samples | Mean output TPS | Decision |
| --- | --- | --- | ---: | --- |
| 16 baseline | `0acef4eab6f4` | 26.786, 28.422, 28.431 | 27.880 | Validated baseline |
| 32 first candidate | `0acef4eab6f4` | 28.677, 26.867, 28.683 | 28.075 | Prompt gate passed, arithmetic scope corrected afterward |

The candidate mean is 0.7 percent higher than the earlier baseline mean, but
the 26.867 TPS sample had a 114.51 ms p99 stream-event gap and the immediate
post-reboot tile-16 validation measured 28.596 TPS. This is within normal
measurement variation, not evidence of a repeatable throughput improvement.
More importantly, the inadvertent split-K coupling means this candidate cannot
establish a row-only quality claim. It is not the default. The implementation
now derives split-K from the validated 16-row reference tile even when a
different output-row tile is requested, then the corrected candidate must be
retested from scratch.

### Corrected HIP kernel screen

After decoupling split-K from the output-row tile, the isolated microbenchmark
used deterministic synthetic tensors at Qwen's real `[N, 2048]` shapes. It
warms each compiled kernel, records 100 native HIP event timings, and hashes
the raw BF16 result buffer. Every compared row below has the same output hash
as its tile-16 baseline for that shape. This is a kernel-level numerical check,
not a replacement for the end-to-end AIME gate.

| Shape | Candidate | Baseline median | Candidate median | Result |
| --- | --- | ---: | ---: | --- |
| `[8192, 2048]` | 32 output rows, fixed split-K | 0.0764 ms | 0.0779 ms | Slower |
| `[4096, 2048]` | 32 output rows, fixed split-K | 0.0392 ms | 0.0396 ms | Slower |
| `[2048, 2048]` | 32 output rows, fixed split-K | 0.0325 ms | 0.0330 ms | Slower |
| `[8192, 2048]` | two waves, 16 output rows | 0.0764 ms | 0.0765 ms | No material gain |
| `[8192, 2048]` | activation-side exact FP8 scale | 0.0764 ms | 0.0766 ms | No material gain |

The activation-scale candidate decodes each FP8 byte as an exact fp16 value
divided by 256, then applies the compensating exact power-of-two scale once to
the FP32 activation. It was bit-identical in the screen but did not lower
latency. It remains disabled. All of these variants are rejected from default
serving because the target is a repeatable end-to-end TPS gain with unchanged
quality, not merely a different kernel that happens to pass one output check.

The current source passed the native focused regression suite after these
experiments: `22 passed, 11 skipped` in
`tests/kernels/test_fp8_pertensor_linear.py` on LAN-223.

### Hardware counters and NVFP4 follow-up

The wheel-compatible ROCm profiler wrapper also supports isolated performance
counter collection. A direct host `rocprofv3` launch aborts before Python starts
because it injects a second LLVM and registers `spirv-expand-step` twice. The
existing wheel-SDK wrapper avoids that conflict and captured the dense FP8
`[8192, 2048]` GEMV successfully. `FetchSize` and `VALUUtilization` are not
available for this `gfx1151` agent through the installed SDK, but the available
counters are sufficient to classify the bottleneck:

| Counter | Observed range on steady GEMV dispatches | Interpretation |
| --- | ---: | --- |
| `MemUnitBusy` | about 89 to 93% | The memory unit is near saturation |
| `L2CacheHit` | about 39 to 51% | Large streamed FP8 weights do not persist fully in L2 |

That evidence explains why row tiles, additional waves, and relocation of an
exact power-of-two FP8 scale did not create a repeatable gain. The next profile
consumer was the Marlin-style inline-NVFP4 MoE decode kernel, so it received an
equally strict screen at Qwen's actual eight-route shapes: gate/up `[1024,
2048]` and down `[2048, 512]`.

| Projection | Candidate | Raw BF16 hash | Timing result | Decision |
| --- | --- | --- | --- | --- |
| Gate/up | 8 output rows | Changed | Faster in isolation | Rejected: exact output changed |
| Gate/up | 32 output rows | Matched | 0.0946 ms vs 0.0615 ms baseline screen | Rejected: slower |
| Gate/up | 2 waves | Matched | 0.0941 ms | Rejected: slower |
| Gate/up | 8 waves | Changed | 0.0557 ms | Rejected: exact output changed |
| Down | 8, 16, or 32 output rows; 2, 4, or 8 waves | Matched | No faster result | Rejected: no repeatable gain |

The helper `scripts/lan223/bench_nvfp4_marlin_decode.py` creates layout-correct
NVFP4 banks and evaluates the production decode kernel directly. It deliberately
uses raw output SHA-1 as the first gate, so numerically faster variants cannot
leak into a full-model reload merely because they are faster.

### Current-main integration validation

The AMD branch merged FreeToken upstream commit `58f4b9e`, which fixes an
NVIDIA Ada row-wise W8A8 prefill issue. The merge is current-main compatible
and does not alter LAN-223's ROCm W8A16 dense decode route, but it was still
validated from a fresh isolated server launch rather than inferred from source
inspection. The combined focused native test suite completed with `28 passed,
22 skipped`.

The reloaded current-main server passed the deterministic AIME gate with the
required output SHA-1 `0acef4eab6f4`, 28.775 output TPS, 403.57 ms TTFT, and
36.77 ms p99 stream-event gap. The newer source resolved 8,974 cache slots and
2,068 KV pages with 19.45 GiB free memory, versus 8,990 slots and 2,081 pages
in the prior baseline. This allocation difference is recorded as a source
revision effect, not an optimization result.

```text
/home/david/freetoken-amd/artifacts/qwen-reboot-recovery-20260829T113506Z/
  aime-quality-current-main.json
```

The final current-main service health check returned `status: ok` on
`127.0.0.1:1919`. A failed direct host-profiler run left one non-serving Python
child behind; it was identified by its profiler benchmark command, terminated,
and then force-cleared when it ignored SIGTERM. The serving parent and current
worker were verified separately before and after that cleanup.

### Unified-memory expert-cache copy screen

The full ROCm trace showed `fast_index_copy` at 593.192 ms across 10,240
dispatches. That total makes the helper worth measuring, but it does not prove
that cache fills limit observed single-stream decode TPS. The existing cache
copy benchmark did not previously encode Qwen3.6-35B-A3B-NVFP4's real model
geometry, so the AMD branch adds a documented profile: 40 MoE layers, 256
experts per layer, top-8 routing, hidden size 2048, intermediate size 512, and
the production six-bank NVFP4 layout.

On LAN-223, with a 513-slot cache, one active token and all eight routed experts
missing, the benchmark copied 13.5 MiB in 0.097 ms, or 146.8 GB/s. Across all
40 MoE layers, its documented extrapolation is 3.87 ms per decode token. The
all-hit case took 0.023 ms. This is a native HIP measurement using the actual
allocation and copy path, not a theoretical memory-bandwidth figure.

| Batch | Active experts | Miss rate | Copy amount | Median time | Bandwidth |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 0% | 0.0 MiB | 0.023 ms | not applicable |
| 1 | 8 | 100% | 13.5 MiB | 0.097 ms | 146.8 GB/s |

The measured worst-case fill is materially smaller than the approximate 35 ms
per-token service interval at 28.775 output TPS. It does not support changing
`fast_index_copy` parameters or bypassing cache maintenance: cache behavior is
correctness-sensitive, and the directly profiled dense FP8 and NVFP4 decode
kernels remain the dominant performance targets. The full reproducibility log
and exit code are retained at:

```text
/home/david/freetoken-amd/artifacts/qwen-copy-bench-20260829T114800Z/
```

### Live clock and power-state verification

The service process is configured for the native ROCm 10 HIP runtime and its
CPU host was already in the Linux `performance` governor. Idle sensor readings
reported the expected 600 MHz shader clock, which is not suitable evidence for
a decode-performance diagnosis. A fixed 256-token, three-sample API workload
therefore ran on the unchanged loopback service while ROCm SMI collected one
sample per second.

The workload completed all three samples at 28.270 mean output TPS with a
0.014 TPS standard deviation. During its steady portion, GPU utilization was
100 percent in 24 samples, shader clocks reached and held the 2.9 GHz state,
memory clock remained at 1.0 GHz, and package graphics power was typically
about 70 to 90 W, with a 114 W peak sample. The service continued to report
`status: ok` after the workload.

This excludes an inactive CPU governor, idle shader state, or obvious
power-state failure as the explanation for the present decode ceiling. It is
consistent with the isolated kernel counters: decode is actively executing at
the device's performance state and the dense FP8 memory unit is already near
saturation. Hardware clock forcing is therefore not a justified safe
optimization. Reproducible workload and sensor artifacts are retained at:

```text
/home/david/freetoken-amd/artifacts/qwen-live-telemetry-20260829T050800Z/
```

### Reusable gfx1151 C++ and HIP cache

LAN-223 initially had no `freetoken_kernel_cache` package and therefore no
formal prebuilt helper-kernel inventory. The AMD branch now includes
`scripts/lan223/build_rocm_kernel_cache.sh`. It validates the native HIP
runtime and gfx1151 device, derives a source-revision-scoped cache path, and
compiles the complete explicit model catalog with four bounded compiler jobs.
The startup script resolves that cache and sets `FREETOKEN_DISABLE_JIT=1`, so a
missing FreeToken C++ or HIP helper fails explicitly instead of compiling during
an inference request.

The first full build found and corrected a portability defect in the shared AOT
catalog: 240-byte and 400-byte scale-bank rows were incorrectly emitted for the
legacy per-bank kernel even though its vector loop requires whole 128-byte
worker rows. Those small rows are supported by the production fused multi-bank
path, which has tail handling. The branch now excludes only the impossible
legacy templates and has a regression test that pins the rule. The repaired
ROCm 10 build produced all 80 valid catalog modules for gfx1151:

```text
/home/david/freetoken-amd/cache/kernel-cache-rocm-gfx1151-d6ee8cef479c/
```

`scripts/lan223/verify_rocm_kernel_cache.py` then loaded every one of those 80
modules with `FREETOKEN_DISABLE_JIT=1`. This verifies ABI-compatible loading
through the installed Python, TVM FFI, ROCm 10 and HIP runtime, which a shared
object file count alone cannot prove. The verifier neither starts a model nor
modifies the cache.

The strict cache launch completed the normal serial NVFP4 expert-bank load,
then passed the AIME output gate with the required SHA-1 `0acef4eab6f4` at
28.504 output TPS, 399.99 ms TTFT, and 36.62 ms p99 stream-event gap. It
resolved the same 8,974 MoE cache slots and 2,068 KV pages as the earlier
current-main validation. The startup artifact is retained at:

```text
/home/david/freetoken-amd/artifacts/qwen-reboot-recovery-20260829T120405Z/
```

Its fixed 256-token scheduler workload also completed three of three scored
samples at 28.018 mean output TPS, with a 28.017 median and 0.007 TPS standard
deviation. This is consistent with the preceding quality-gated runs and shows
that enforcing the reusable C++ and HIP cache changes startup compilation
behavior, not steady-state decode arithmetic or output quality.

This cache eliminates FreeToken's C++ and HIP helper JIT for the catalog it
contains. It does not claim to precompile every Triton specialization or a
GGUF kernel: Qwen3.6-35B-A3B-NVFP4 is a safetensors NVFP4 checkpoint, not a
GGUF model, and Triton maintains its own architecture- and source-keyed
persistent cache.

### Rejected eight-wave FP8 GEMV candidate

An eight-wave gfx1151 FP8 GEMV launch was screened because it produced the
same isolated raw BF16 result hash as the one-wave baseline and reduced the
microbenchmark median from 0.07703 ms to 0.07574 ms for the 8192 by 2048
matrix. It passed the deterministic API quality gate with the required AIME
SHA-1 `0acef4eab6f4`. That isolated result did not carry over to the actual
Qwen decode workload: the fixed three-sample, 256-output-token scheduler test
averaged 26.260 TPS with 0.0008 TPS standard deviation, compared with the
strict-cache one-wave baseline of 28.018 TPS. This is a 6.3 percent regression.

The eight-wave option was therefore removed from the accepted launcher and
benchmark allowlists. The candidate artifacts are retained for reproducibility
at:

```text
/home/david/freetoken-amd/artifacts/qwen-reboot-recovery-20260829T121916Z/
```

This result demonstrates why raw tensor equality and a favorable isolated
kernel timing are necessary but not sufficient acceptance conditions. The
multi-layer MoE decode schedule has materially different occupancy and cache
interaction from a single dense GEMV invocation.

### Rejected FP8 GEMV pipeline-depth screen

The next quality-safe dense FP8 candidate was an explicit Triton pipeline depth
for the existing 16-row, one-wave, fixed split-K GEMV. Pipeline depth changes
software scheduling but not the arithmetic, output-row tile, K chunks, or
split-K reduction tree. All five tested depths produced the same raw BF16 SHA-1
`0aca8b9e38ebfaa91893366a175970f1c45599b9` on the Qwen-shaped 8192 by 2048
screen.

The initial 200-iteration pass made stage 2 look marginally favorable, with a
0.07698 ms median versus 0.07734 ms for stage 3. A new, longer 500-iteration
paired measurement reversed that apparent advantage: stage 3 measured 0.07689
ms median versus 0.07704 ms for stage 2, while their means were effectively
identical at 0.07702 ms. Stages 1, 4, and 5 were slower in the initial screen.
The small difference is ordinary device-timing variation, not a defensible
end-to-end improvement. The staging override was removed without a model
reload, preserving the established default Triton pipeline policy.

### Measured expert-cache residency and rejected capacity increase

The AMD branch now exposes a read-only `/v1/cache/stats` endpoint through the
existing API, tokenizer, and scheduler control-message path. It transfers an
already accumulated device-counter snapshot only when explicitly requested;
it does not alter cache contents, scheduling, routing, model weights, or the
normal no-statistics serving path. The associated `--moe-collect-stats` launch
option is disabled by default because its counter updates are diagnostic work.

At the validated 0.35 memory ratio, Qwen allocated 8,974 cache slots, 2,068
KV pages, and 24 GDN state slots. The fixed workload accumulated 33,920
MoE-layer decode calls: eight active experts per layer and 0.671 misses per
layer, an 8.39 percent miss rate. A 0.38 memory-ratio candidate raised
residency to 9,990 slots while retaining 2,055 KV pages and 24 GDN slots. It
reduced the realized miss rate to 7.33 percent, but its three-sample scheduler
throughput fell from 28.038 TPS to 27.908 TPS. The capacity increase is
therefore rejected. It consumes roughly 1.7 GiB of additional headroom without
producing a serving improvement.

The counters establish that a small number of expert fetches remains, but not
that larger static residency is a profitable AMD optimization. Direct cache
copy measurements and the sustained TPS result agree: the dense FP8 decode
path remains the more valuable target. The two diagnostic artifact roots are:

```text
/home/david/freetoken-amd/artifacts/qwen-reboot-recovery-20260829T124643Z/
/home/david/freetoken-amd/artifacts/qwen-reboot-recovery-20260829T125511Z/
```

### Rejected 64-block fused expert-copy candidate

The production fused expert-cache copy helper was also screened with the real
Qwen3.6-35B-A3B-NVFP4 six-bank layout. Both candidate grids use precompiled
gfx1151 HIP modules, run with `FREETOKEN_DISABLE_JIT=1`, and were required to
copy every selected source row byte-for-byte into its requested cache slot
before a timing sample was recorded. With one missing expert, widening from
eight to 64 blocks per bank reduced median copy time from 0.01872 ms to
0.01812 ms. With eight missing experts, it reduced the median from 0.07984 ms
to 0.05982 ms, increasing the measured copy rate from 177.9 GB/s to 237.5
GB/s.

That isolated improvement preserved deterministic model output: the 64-block
server returned the required AIME SHA-1 `0acef4eab6f4`, with 28.582 output TPS
on that quality request. It did not improve the fixed three-sample scheduler
workload. The end-to-end result was 28.070 TPS mean, 28.078 TPS median, and
0.016 TPS standard deviation, below the fresh default eight-block result of
28.153 TPS mean. The larger launch grid is therefore rejected as the serving
default. The code retains an explicitly bounded `8|64` diagnostic selection so
future cache changes can be remeasured without a new JIT specialization, but
the launcher defaults to the accepted eight-block grid.

The full candidate artifacts, including exact-copy microbenchmark data, AIME
quality result, and scheduler samples, are retained at:

```text
/home/david/freetoken-amd/artifacts/qwen-reboot-recovery-20260829T131629Z/
```

### Native-library replacement screen

The Qwen checkpoint carries calibrated `input_scale` tensors, so a W8A8
hipBLASLt replacement was investigated as a possible way to replace the
memory-bound W8A16 dense decode kernel. LAN-223 is running ROCm 10.0 with
hipBLASLt 1.4 and PyTorch `2.13.0+rocm10.0.0`, but the route is not available
for this model and GPU. PyTorch's native `_scaled_mm` call on gfx1151 rejects
the operation before dispatch, reporting that it is supported only on CUDA
compute capability 8.9 or 9.0 devices, or ROCm MI300-class devices. This is a
runtime capability gate, not a FreeToken configuration error.

The installed hipBLASLt 1.4 headers also expose `HIP_R_8F_E5M3_EXT` but not an
OCP E4M3 matrix data type. Qwen's dense weights are OCP E4M3, so directly
calling that ABI would require a representation conversion and would no longer
be a like-for-like W8A8 replacement. A BF16 conversion would double weight
traffic in the already memory-bound decode path. Neither alternative is an
acceptable serving optimization or a valid quality-preserving AMD port.

The conclusion is deliberately limited to this ROCm 10, PyTorch 2.13, and
gfx1151 environment. The result keeps the verified native Triton W8A16 route
as the active path and directs follow-up work toward custom kernels that retain
the checkpoint's OCP E4M3 bytes and its exact output contract.

### Rejected direct HIP OCP-E4M3 GEMV prototype

To validate that a custom HIP component remains possible despite the library
gate, an isolated one-Wave32-per-output-row W8A16 GEMV was compiled with
ROCm 10 hipcc for gfx1151. It reads the checkpoint's raw OCP E4M3 bytes,
streams one shared BF16 activation vector, accumulates in FP32, applies the
existing per-row FP32 scale, and avoids the production split-K partial buffer.
The prototype is intentionally outside the API serving path.

The build succeeded after supplying rocThrust as a compiler-only system include
to PyTorch's ROCm extension machinery. At Qwen's `[8192, 2048]` dense shape,
however, the candidate measured 0.10674 ms median versus 0.07687 ms for the
production Triton kernel, a 38.9 percent regression. Its raw BF16 SHA-1 also
differed (`e08b284faedd608850119511655e1e94cab87b05` versus
`0aca8b9e38ebfaa91893366a175970f1c45599b9`), with a maximum absolute element
difference of `3.814697265625e-06`. The altered wave-local reduction order is
therefore not an exact replacement.

This prototype is rejected before model integration. The artifact preserves
the complete hipcc command and timing JSON for later component work:

```text
/home/david/freetoken-amd/artifacts/fp8-hip-prototype-20260829T133900Z/
```

### System-level performance-policy audit

LAN-223's CPU governor is already `performance`. The Radeon 8060S reports the
standard `auto` GPU performance policy at idle, where shader and SoC clocks
fall to 600 MHz while memory remains at 1,000 MHz. This is not evidence of a
decode throttle: the earlier fixed API workload recorded 100 percent GPU use,
a sustained 2.9 GHz shader clock, and roughly 70 to 90 W graphics package
power during active generation.

This host does not expose the usual amdgpu DPM control files through DRM sysfs,
and `rocm-smi` reports that its power cap is unsupported. An isolated
high-performance-policy test is therefore contingent on interactive sudo
authentication. The port does not change an undocumented platform policy or
claim a clock-based gain without that reversible measurement. The serving
baseline stays on the normal driver policy and remains subject to the same
quality and TPS gates as kernel candidates.

#### DPM-policy measurement contract and setup failure

The DPM experiment uses
`scripts/lan223/run_qwen_dpm_policy_benchmark.sh`. It requires an interactive
sudo credential in the terminal that invokes it because the host caches sudo
authorization per terminal. The script requests a named temporary policy,
records the pre-run policy, delegates the fixed three-sample 256-token Qwen
scheduler workload to the existing harness, and restores the normal `auto`
policy through an `EXIT` trap. It neither reloads FreeToken nor alters the
model, cache, scheduler configuration, llama-swap, or other LAN hosts.

The policy log lives in a newly-created parent evidence directory. The harness
receives a distinct, absent `benchmark` child directory because its immutable
artifact contract intentionally fails if that exact directory already exists.
This separation is enforced by a unit test in
`tests/benchmarks/test_lan223_qwen_benchmark.py`.

An initial manual attempt at `2026-08-29T18:30:35Z` correctly changed the GPU
from `auto` to `high` and restored it to `auto`, but created the harness
artifact directory before invoking the harness. The harness consequently
raised `FileExistsError` before making an API request. It produced no scored
samples, no input TPS, and no output TPS, so it is not a performance result
and must not be compared with the `auto` baseline. The corrected wrapper and
its regression test were added after that attempt. A valid high-policy result
requires a fresh artifact containing the harness manifest, all three scored
sample JSON files, the policy log, and a post-run health check.

#### Valid high-policy result

The corrected wrapper completed a valid high-policy run at
`2026-08-29T22:02:24Z`. It recorded `high` before the workload and `auto`
afterward, completed all three forced 251-generated-token scheduler samples,
and left the isolated OpenAI-compatible Qwen endpoint healthy. The matching
accepted eight-block `auto` baseline used the same 1,212-token prompt,
251 generated tokens, three scored samples, model, server process, cache
configuration, fixed decoding settings, and loopback endpoint.

| GPU policy | Mean output TPS | Median output TPS | Output TPS samples | Mean input TPS | Mean TTFT |
| --- | ---: | ---: | --- | ---: | ---: |
| `auto` | 28.153 | 28.150 | 28.147, 28.150, 28.162 | 2913.096 | 416.255 ms |
| `high` | 28.355 | 28.353 | 28.353, 28.362, 28.349 | 2957.942 | 409.835 ms |

The temporary `high` policy improved fixed-workload output throughput by
0.202 TPS, or 0.72 percent, and raised measured input throughput by 44.846
TPS, or 1.54 percent. Mean first-text latency decreased by 6.420 ms, or 1.54
percent. The output gain exceeds the combined run-to-run standard deviations
of the two three-sample sets, but the sample count is deliberately small, so
this is a measured operating preference rather than a broad claim about every
prompt shape or concurrent load level.

After restoration to `auto`, the live service passed the existing deterministic
AIME quality gate with the required output SHA-1 `0acef4eab6f4`. Its 127-token
quality stream measured 28.421 decode tokens per second and 395.561 ms TTFT.
The quality artifact proves that the run did not leave the model or serving
configuration altered. The policy wrapper changes only driver performance
policy, not model arithmetic, but that post-run gate is not presented as a
separate quality measurement performed while `high` was active.

The complete high-policy evidence is retained at:

```text
/home/david/freetoken-amd/artifacts/qwen-dpm-high-20260829T220224Z/
```
