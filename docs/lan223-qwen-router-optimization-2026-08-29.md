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

The strict cache launch completed the normal serial NVFP4 expert-bank load,
then passed the AIME output gate with the required SHA-1 `0acef4eab6f4` at
28.504 output TPS, 399.99 ms TTFT, and 36.62 ms p99 stream-event gap. It
resolved the same 8,974 MoE cache slots and 2,068 KV pages as the earlier
current-main validation. The startup artifact is retained at:

```text
/home/david/freetoken-amd/artifacts/qwen-reboot-recovery-20260829T120405Z/
```

This cache eliminates FreeToken's C++ and HIP helper JIT for the catalog it
contains. It does not claim to precompile every Triton specialization or a
GGUF kernel: Qwen3.6-35B-A3B-NVFP4 is a safetensors NVFP4 checkpoint, not a
GGUF model, and Triton maintains its own architecture- and source-keyed
persistent cache.
