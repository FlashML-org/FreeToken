# LAN-223 native ROCm validation, 2026-08-28

## Result

This validation passed the first release gate for the AMD port.  FreeToken
served both required MoE models through the OpenAI-compatible API on LAN-223's
Radeon 8060S (`gfx1151`) using a native HIP and ROCm execution path.

This is not a CPU fallback or a Vulkan result.  The serving process uses the
ROCm PyTorch wheel, HIP-compiled native extensions, and Triton GPU kernels.
CUDA graphs were deliberately disabled for this validation because the MVP
needs correctness before graph capture tuning.

## Reproducibility record

| Item | Value |
| --- | --- |
| Host | LAN-223, `david-Gmktec-x2-2` |
| GPU | AMD Radeon 8060S Graphics, `gfx1151`, 40 CUs |
| System ROCm installation | ROCm 10.0 at `/opt/rocm-10.0` |
| PyTorch wheel | `2.13.0+rocm10.0.0` |
| HIP reported by PyTorch | `7.15.26333` |
| FreeToken branch | `amd-rocm-gfx1151` |
| Validation commit | `065d806` |
| API exposure | loopback-only ports, not llama-swap |

The isolated validation layout was `/home/david/freetoken-amd/`; no existing
llama-swap service, model configuration, or production endpoint was changed.

## Models and API evidence

| Model | Source revision | Backend selection | Non-streaming result | Streaming result |
| --- | --- | --- | --- | --- |
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | vendor model snapshot used for this run | Triton attention, MoE offload, native Triton NVFP4, serial expert load | HTTP 200, `AMD ROCm FreeToken ready.` in 1.54 s | HTTP 200, SSE chunks and `[DONE]` |
| `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` | `d1c082be9cf3c8a514acf63b8761f4b41935842e` | Triton attention, MoE offload, serial expert load, HIP GGUF JIT | HTTP 200, `native hip api works` in 341.304 ms | HTTP 200, SSE chunks and `[DONE]` |

Raw evidence remains on LAN-223 in these isolated artifact directories:

```text
/home/david/freetoken-amd/artifacts/qwen36-nvfp4-serial-hip-prefill/
/home/david/freetoken-amd/artifacts/gemma4-q4-rocm-thrust-system/
```

The Gemma telemetry captured immediately after the API tests identified the
same `gfx1151` device, 33 percent GPU utilization, 46 percent allocated VRAM,
and a 40 C edge temperature.  The model uses the APU's shared-memory design;
the tool's VRAM label is therefore only its standard telemetry label.

## Warm single-request throughput

The following measurements use one fixed 733-token prompt, greedy sampling,
and a one-sentence answer that produced 26 completion tokens.  `TTFT` is the
client-observed time to the first non-empty SSE text chunk.  Prompt throughput
is the end-to-end prompt-token count divided by TTFT, so it includes normal
API and scheduler overhead.  Output throughput is completion tokens divided
by the interval from that first chunk through `[DONE]`.

| Model | Prompt tokens | Completion tokens | TTFT | Prompt TPS | Generation interval | Output TPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B NVFP4 | 733 | 26 | 4.976 s | 147.3 | 0.899 s | 28.9 |
| Gemma 4 26B A4B Q4_0 GGUF | 733 | 26 | 3.244 s | 226.0 | 0.581 s | 44.8 |

These are warm, single-request measurements, not concurrency or maximum
throughput claims.  The Qwen configuration uses the native Triton serial
NVFP4 prefill route selected for ROCm correctness.  Its approximately
seven-minute cold initialization is expert-bank preparation and cache
allocation, not inference time.

## Same-model llama.cpp Vulkan comparison

To compare the usable Strix Halo serving baseline rather than an unrelated
model, the exact Gemma GGUF was served by llama.cpp Vulkan build `b10141`
(`0d47ea742`) on a separate loopback port.  Both servers used one slot,
8,192-token context, greedy sampling, and the same repeated scheduler prompt.
The model SHA-256 was
`3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d`.

| Runtime | GPU backend | Prompt tokens | Completion tokens | TTFT | Client prompt TPS | Client output TPS | Runtime prompt TPS | Runtime output TPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FreeToken | ROCm/HIP | 733 | 26 | 3.244 s | 226.0 | 44.8 | not exposed | not exposed |
| llama.cpp `b10141` | Vulkan | 758, 7 template tokens cached | 128 | 0.855 s | 886.7 | 63.2 | 1,078.4 | 61.7 |

For this isolated, single-request Gemma workload, llama.cpp Vulkan reached
first output about 3.8 times sooner, delivered about 3.9 times the
client-observed prompt rate, and delivered about 1.4 times the client-observed
generation rate.  llama.cpp's internal timing excludes ordinary API and
scheduler overhead, so its 1,078.4 prompt TPS and 61.7 output TPS must not be
compared directly with FreeToken's client-observed rates.

The completion lengths differ because llama.cpp exposed Gemma's reasoning
stream and consumed the 128-token cap, whereas FreeToken's parser emitted the
final concise answer and stopped at 26 tokens.  That makes the output-rate
comparison useful as a warm streaming rate, but not a quality or exact
end-to-end task comparison.  The raw llama.cpp evidence is retained under
`/home/david/freetoken-amd/artifacts/llamacpp-vulkan-gemma4-q4-tps/` on
LAN-223.

## Same-model ROCm 10 and HIP comparison

The Vulkan baseline above answers a practical deployment question, but it is
not a backend-for-backend comparison.  This follow-up rebuilt the same
llama.cpp source revision, `b10141` (`0d47ea742`), with HIP for `gfx1151` and
ran it under the same ROCm 10 installation used by FreeToken.  The compiler
was ROCm 10 HIP `7.15.26333` with AMD Clang 23.0.0.  At runtime, llama.cpp's
`libamdhip64`, `libhipblas`, `librocblas`, `libamd_comgr`, and HSA runtime
libraries all resolved from `/opt/rocm-10.0`, not the older ROCm installation.

Both runners used the identical 14 GB Gemma 4 26B A4B Q4_0 GGUF, SHA-256
`3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d`, one
request at a time, an 8,192-token context, greedy sampling, `max_tokens: 128`,
and a 48-times repeated scheduler prompt.  Each measurement used a distinct
nonce, preventing prompt-cache reuse.  The token totals differ by one because
the two runners tokenize and render Gemma's chat template differently.

| Runtime | HIP and ROCm stack | Prompt tokens | Completion tokens | TTFT | Client prompt TPS | Client output TPS | Runtime prompt TPS | Runtime output TPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FreeToken, steady state | PyTorch `2.13.0+rocm10.0.0`, HIP `7.15.26333`, native HIP GGUF extension | 772 | 20 | 2.863 s | 269.6 | 46.1 | not exposed | not exposed |
| llama.cpp `b10141` | ROCm 10 HIP, `gfx1151` | 771 | 128 | 0.850 s | 906.6 | 58.3 | 1,011.6 | 56.2 |

On this uncached, single-request workload, llama.cpp ROCm 10 reached first
text about 3.4 times sooner, supplied about 3.4 times the client-observed
prompt rate, and supplied about 1.3 times the client-observed output rate.
llama.cpp's internal numbers exclude HTTP, SSE, and scheduling overhead and
therefore are only comparable to another internal timing source, not directly
to FreeToken's client values.

The FreeToken request that triggered a fresh GGUF HIP extension build is kept
as a separate cold-start measurement: 768 prompt tokens, 21 completion tokens,
109.938 s TTFT, 6.99 client prompt TPS, and 27.47 client output TPS.  It
contains HIP compilation and must not be presented as inference throughput.
The subsequent steady-state run above was made after the extension completed,
using a fresh nonce and no prompt cache hit.  FreeToken's extension compiler
was `/opt/rocm-10.0/bin/hipcc` targeting `gfx1151`, and its runtime libraries
came from the ROCm 10 PyTorch SDK packages.  Its existing JIT command also
passed `/opt/rocm-7.2.4/include` as a supplemental include path.  That does not
change the ROCm 10 compiler or loaded runtime libraries, but it prevents this
FreeToken build from being described as a strictly ROCm 10-only header build.

The llama.cpp response used all 128 allowed tokens because it exposed Gemma
reasoning text.  FreeToken stopped after a concise 20-token answer.  This
makes the output-rate comparison a useful streaming measurement, but it is
not an exact answer-quality or equal-completion-length evaluation.

Raw artifacts are retained only on LAN-223:

```text
/home/david/freetoken-amd/artifacts/llamacpp-rocm10-gemma4-q4-tps/
/home/david/freetoken-amd/artifacts/freetoken-rocm10-gemma4-q4-tps/
```

## AMD TPS optimization campaign

The first configuration optimization pass used the same warm AIME-25 problem
and a 128-token greedy completion for both FreeToken and the ROCm 10 HIP build
of llama.cpp `b10141`.  Each runner received the identical user message, used
a warm identical request before the measured request, and ran one stream at a
time.  Both rendered 63 prompt tokens; FreeToken's measured request reused 62
prompt tokens and llama.cpp's reused 58.

| Runtime and candidate | Decode TPS | TTFT | Result |
| --- | ---: | ---: | --- |
| FreeToken, offload, eager | 54.89 | 267.9 ms | Baseline |
| FreeToken, offload, HIP graph capture at batch size 1 | 55.73 | 259.3 ms | Best observed safe configuration |
| FreeToken, HIP graph plus experimental `-ffast-math` GGUF extension | 55.65 | 261.6 ms | Rejected: no gain, despite matching output hash |
| FreeToken, final target-specific `gfx1151` GGUF extension plus graph capture | 55.44 | 263.6 ms | Validated shipping configuration; normal run-to-run variation |
| FreeToken, experimental two-row Q4_0 MoE block | 55.30 | 291.6 ms | Rejected: slower with identical output hash |
| FreeToken, experimental Q4_0 MoE two-block residency hint | 55.08 | 294.9 ms | Rejected: slower with identical output hash |
| FreeToken, HIP Q4_0 MoE one-wave/two-row specialization | 55.89 median, 55.91 mean | 262.2 ms mean | Accepted: five independent API runs, identical output hash |
| FreeToken, full 4,096-slot expert cache and pinned 8,320-token KV pool | 60.11 median, 58.61 mean | 260.3 ms mean | Accepted configuration; four of five runs at 60.06 to 60.20 TPS, one host-contention outlier at 52.50 TPS |
| llama.cpp `b10141`, ROCm 10 HIP, earlier matched reference | 60.42 client, 58.88 internal | 128.6 ms | Historical reference |
| llama.cpp `b10141`, ROCm 10 HIP, current-host five-run control | 62.44 median, 62.13 mean client TPS | 111.5 ms mean | Same prompt, greedy decode, five fresh servers, requested 8,320-token context |

The graph configuration removes approximately 1.5 percent of the eager decode
cost.  The capacity-aware resident-expert configuration below then removes the
dominant configuration gap without changing the model, server API, or HIP
kernel arithmetic.  Its uncontended median is within 0.51 percent of the
60.42 client-TPS llama.cpp reference, but its five-run arithmetic mean remains
below that reference because one run experienced external host stalls.  The
criterion of meeting or exceeding llama.cpp is therefore not yet claimed as a
fully repeatable mean result.

### Accepted full-expert-cache and fixed-KV configuration

The original automatic offload configuration sized 3,840 GPU expert slots and
then assigned the remaining memory budget to a very large KV pool.  That pool
is not required by the fixed 8,320-token operating target and lowered the
observed decode rate.  A fixed expert-cache configuration leaves the same
native Q4_0 GGUF, HIP extension, graph-captured decode, OpenAI-compatible API,
and `offload` backend intact while making the capacity choices explicit:

```bash
python benchmarks/bench_decode_moe.py \
  --model /home/david/freetoken-amd/models/Gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf \
  --backend offload --cache 4096 --num-token-override 8320 \
  --mem-ratio 0.50 --decode 128 --greedy
```

`4096` is the complete 32-layer by 128-expert cache domain.  A 3,840-slot
control preserved the fixed KV allocation but produced two severe decode-tail
events, confirming that leaving any of the 4,096 slots uncached can still
exercise the miss path.  The explicit 4,096-slot configuration was therefore
retained.  The new benchmark option maps `--num-token-override` to the public
server flag `--num-tokens`, so experiments can pin KV capacity without a
private wrapper.

Five independent API runs used the fixed 63-token AIME request, 126 measured
decode steps, greedy sampling, `0.50` memory ratio, and the deterministic
output SHA-1 `abeee5e73e89`:

| Run | Decode TPS | ms/token | TTFT | Event p50 / p99 |
| --- | ---: | ---: | ---: | --- |
| 1 | 60.063 | 16.649 | 259.7 ms | 16.929 / 17.841 ms |
| 2 | 52.499 | 19.048 | 261.4 ms | 16.928 / 120.541 ms |
| 3 | 60.203 | 16.610 | 259.6 ms | 16.855 / 17.657 ms |
| 4 | 60.114 | 16.635 | 260.3 ms | 16.937 / 17.619 ms |
| 5 | 60.183 | 16.616 | 260.6 ms | 16.876 / 17.522 ms |
| Aggregate | **58.613 mean, 60.114 median** | 17.112 mean | 260.3 ms mean | 16.928 / 17.657 ms median |

The four normal runs are within 60.063 to 60.203 TPS and have p99 latency at
or below 17.841 ms.  The one low-throughput run kept the same output, VRAM,
TTFT, and p50 latency, but had isolated 120.541 ms decode events.  Kernel logs
recorded `kfd_process_wq_release` holding CPU for more than 10 ms and the host
showed full I/O pressure.  Read-only inspection also found two long-running,
blocked user-owned filesystem scans.  They were not stopped by this campaign.
This is host contention evidence, not a FreeToken numerical or API failure.

Capacity was tested through the public OpenAI-compatible API, not merely at
startup.  A request with 7,619 prompt tokens plus one completion token ran
inside the pinned 8,320-token pool, returned exactly `OK`, and completed in
22.352 seconds.  The server then exited cleanly with no KFD processes.

The retained raw evidence is:

```text
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/full-expert-cache-4096-20260829T010740Z/
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/fixed-expert-cache-3840-control-20260829T011537Z/
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/full-cache-4096-context8320-20260829T012342Z/
```

### Current-host ROCm llama.cpp control

The historical llama.cpp reference was useful for identifying the original
gap, but it was not collected alongside the accepted 4,096-slot FreeToken
configuration.  A new five-run control was therefore run immediately after
that configuration investigation, without changing LAN-223, stopping any
user process, or enabling a production service.  Each trial launched a fresh
`llama-server` from the ROCm 10 `b10141` build with all layers on `gfx1151`,
Flash Attention enabled, one parallel slot, and `-c 8320`.  The server reports
an 8,448-token slot after its own request reserve is added.  This is a llama.cpp
internal allocation detail; the requested application context target was
8,320 tokens in both runners.

Both runners used the same cached AIME-25 problem 0, a warmed streamed
OpenAI-compatible `/v1/chat/completions` request, greedy sampling, and a
128-token completion.  The metric in this table is client-observed decode
throughput: `(completion_tokens - 1)` divided by elapsed time from the first
to last SSE token event.  It includes HTTP and SSE delivery for both runners.

| Runtime | Five client decode TPS | Mean | Median | Mean TTFT | p99 event gap median |
| --- | --- | ---: | ---: | ---: | ---: |
| FreeToken, 4,096 experts, 8,320-token KV pool | 60.06, 52.50, 60.20, 60.11, 60.18 | 58.61 | 60.11 | 260.3 ms | 17.66 ms, excluding the host-stalled run 120.54 ms |
| llama.cpp `b10141`, ROCm 10 HIP | 61.04, 62.03, 62.44, 62.57, 62.56 | 62.13 | 62.44 | 111.5 ms | 16.63 ms |

llama.cpp leads FreeToken by 3.7 percent on median client decode TPS
(`62.44 / 60.11 - 1`) and 5.7 percent on the unfiltered five-run mean
(`62.13 / 58.61 - 1`).  It also has lower warm TTFT.  FreeToken produced the
same deterministic output hash in every measured run; llama.cpp produced the
same deterministic output hash in every one of its own runs.  The hashes are
not compared across runtimes because their tokenizers and chat-template
implementations differ.

This is a close result for decode rate, but it does **not** meet the stated
criterion of meeting or exceeding llama.cpp.  The remaining performance work
is therefore directed at the HIP decode path and the source of the FreeToken
tail stall, rather than a claim of parity.  The raw llama.cpp evidence is
retained on LAN-223 at:

```text
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/llamacpp-current-host-context8320-20260829T013730Z/
```

### Current-upstream rebase and full API revalidation

After the comparison, upstream `main` advanced from `9ef3651` to `a05c265`
with Qwen 3.8 support and engine or cache changes.  The AMD branch was rebased
onto that current upstream revision without a conflict, rather than leaving a
performance result attached to an obsolete upstream base.  The rebased branch
was then installed into the isolated LAN-223 virtual environment so its native
HIP pinned-memory extension was built from the rebased source.  The source
checkout used for that validation was deliberately separate from the earlier
test checkout, preventing an uncommitted working-tree change from becoming
test evidence.

The first complete Gemma launch from the rebased checkout rebuilt the target-
specific GGUF HIP extension and matching graph helper because the source path
is part of their cache identity.  The build used ROCm 10 `hipcc`, `-O3`, and
`--offload-arch=gfx1151`; a later process restart can reuse that cache.  The
server then completed its normal graph capture, exposed `/v1/models`, and
served two streamed OpenAI-compatible chat completions before clean shutdown.

| Check | Observed value |
| --- | --- |
| Upstream revision in branch history | `a05c265` |
| HIP build and ROCm-runtime tests | 4 passed |
| MoE configuration | `offload`, 4,096 expert slots, 8,320 KV tokens, graph batch size 1 |
| Warm streamed API decode | 60.07 client TPS, 16.648 ms/token |
| Warm TTFT | 258.2 ms |
| Prompt and completion tokens | 63 and 127, respectively |
| Output SHA-1 | `abeee5e73e89` |
| Server VRAM | 15.66 GiB |
| Post-run process state | Server shut down; no serving process remained |

The response ended at 127 tokens despite the requested 128-token limit, so
the benchmark harness emitted its explicit token-count warning.  The request
was otherwise successful, deterministic, and had the expected response hash.
This one-run revalidation is evidence that rebasing did not break native HIP
serving.  It is intentionally not folded into the five-run performance score.
Its raw logs and result are retained at:

```text
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/rebased-current-main-api-retry-20260829T014741Z/
```

### Rebased Qwen3.6 NVFP4 API revalidation

The other MoE model retained in the isolated FreeToken inventory is
`Qwen3.6-35B-A3B-NVFP4`.  It was revalidated after the upstream rebase through
the same loopback OpenAI-compatible streaming API, using the native Triton
NVFP4 expert path, graph batch size 1, automatic expert-cache sizing, a 0.35
memory ratio, and greedy 128-token AIME decoding.  This exercised its complete
21.8 GiB parallel expert-bank load, cache allocation, graph capture, warm
request, measured request, and cleanup.

| Check | Observed value |
| --- | --- |
| Resolved expert cache | 9,499 slots and 8,255 KV tokens |
| Warm streamed API decode | 28.93 client TPS, 34.560 ms/token |
| Warm TTFT | 404.7 ms |
| Prompt and completion tokens | 54 and 127, respectively |
| Output SHA-1 | `0acef4eab6f4` |
| Server VRAM | 19.12 GiB |
| Post-run process state | Server shut down; no serving process remained |

As with the rebased Gemma validation, the response ended at 127 tokens and the
harness recorded its explicit limit-warning rather than silently treating it as
a 128-token result.  The API transaction and deterministic response succeeded.
The server also reported that `triton_kernels` was absent and selected the
numerically equivalent pure-PyTorch router fallback.  That is a documented
performance limitation, not a functional failure.  A native ROCm-compatible
fused-router installation must be independently verified before it can be
considered an optimization.

The raw evidence is retained at:

```text
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/rebased-current-main-qwen36-api-20260829T015240Z/
```

### Rejected ROCm vendored-Triton router candidate

The Qwen revalidation exposed a pure-PyTorch router fallback because OpenAI's
`triton_kernels` package contains CUDA-only binaries.  Current upstream also
contains an in-tree Triton router, so it was evaluated as a ROCm-only candidate
before any production use.  On the actual Radeon 8060S it selected the same
expert set as PyTorch; BF16 equal-logit ties can have a different internal
ordering, while FP32 indices matched exactly.  Its selected routing weights
matched PyTorch within `2.98e-8`, and the isolated one-token, 128-expert,
top-8 router time improved from 21.14 us to 14.96 us.

That microbenchmark improvement was insufficient.  The complete Qwen API run
with the candidate reached 30.26 client TPS, but its deterministic greedy
response SHA-1 was `cd580f4978fb`, not the reference `0acef4eab6f4`.  Small
router differences therefore accumulated into a different generated response.
The candidate was reverted and ROCm continues to use the reference PyTorch
router.  This keeps quality behavior stable even though the fused alternative
is faster in isolation.  The fallback warning now explicitly distinguishes
intentional ROCm behavior from a missing CUDA Linux package.

The rejected candidate evidence is retained at:

```text
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/qwen36-vendored-router-api-20260829T015937Z/
```

The restored branch was then revalidated through the full Qwen API path.  It
returned to the reference SHA-1 `0acef4eab6f4` at 28.96 client TPS, with the
same 19.12 GiB VRAM use and clean shutdown.  Its 1,272.2 ms warm TTFT is not a
performance regression claim: the model's 21.8 GiB expert-bank load was
concurrently slowed by the documented host I/O pressure, taking 3 minutes and
37 seconds instead of about 2 minutes.  The final exact-path artifact is:

```text
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/qwen36-router-revert-api-20260829T020626Z/
```

### Accepted HIP Q4_0 one-wave/two-row MoE specialization

The first two-row experiment did not reproduce llama.cpp's execution shape: it
used two independent 32-thread waves.  Commit `d1de602` instead adds a ROCm-only
Q4_0 kernel in which one 32-thread wave accumulates two adjacent output rows.
It preserves FreeToken's flattened token/top-k route IDs, packed expert-bank
layout, Q8_1 activation layout, and BF16 public output contract.  CUDA retains
the established generic path.

The dedicated LAN-223 microbenchmark uses the verified Gemma 4 26B A4B Q4_0
geometry: 128 experts, top-k 8, hidden width 2816, intermediate width 704, and
one decode token.  Five runs with 2,000 timed calls each measured a 73.509 us
baseline median for the gate/up plus down pair and a 64.340 us candidate median,
a 12.5 percent kernel-pair reduction.  ROCprof recorded a 32-thread wave, zero
LDS and scratch allocation, and half the former row-block grid.  The compiler
still allocated 48 VGPRs, so future work must target register pressure
separately rather than claiming it was resolved by this change.

The end-to-end gate was five independent loopback OpenAI-compatible API server
runs, each using the exact Gemma GGUF SHA-256, offload backend, 0.50 memory
ratio, HIP graph capture, greedy AIME-25 problem 0, and 128-token decode
procedure.  All five emitted the original deterministic output SHA-1
`abeee5e73e89` and retained 27.52 GiB server-reported VRAM use.

| Metric | Five-run result |
| --- | --- |
| Decode TPS | 55.713 to 56.071 |
| Decode TPS median / mean | **55.894 / 55.905** |
| Decode ms/token median / mean | **17.891 / 17.887** |
| TTFT mean | 262.2 ms |
| Output SHA-1 | `abeee5e73e89` in every run |
| Compared shipping configuration | 55.44 TPS single verified run |
| Matched llama.cpp ROCm 10 reference | 60.42 client TPS |

The candidate is accepted because it produces a repeatable FreeToken gain of
approximately 0.8 percent over the prior shipping result while preserving the
observable API result.  It remains approximately 7.5 percent below the
matched llama.cpp client-TPS reference, so it is an incremental port
improvement rather than completion of the performance objective.

Artifacts are retained on LAN-223:

```text
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/q4-moe-microbench-20260828T231332Z/
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/q4-moe-two-row-wave-20260828T231950Z/
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/q4-moe-two-row-wave-20260828T231950Z/api-repeats-20260828T232646Z/
```

### Rejected two-row MoE Q8 activation-reuse candidate

The accepted HIP Q4_0 one-wave/two-row MoE kernel computes two adjacent output
rows from the same Q8_1 activation block.  The generic dot helper loads the
four packed Q8 activation words independently for each row.  Commit `684148d`
tested a ROCm-only helper that loads those four words once and supplies them to
both row dot products, while retaining the same Q4 nibble order, DP4A order,
scales, BF16 public-output contract, and all CUDA code.

The shape-accurate Gemma routed-expert microbenchmark improved from about
64.34 us to **61.54 us per gate/up plus down pair**.  That local result did not
translate to a material full-server result.  Five independent OpenAI-compatible
API runs, each using the fixed 63-token prompt and 126 measured decode steps,
all returned greedy output SHA-1 `abeee5e73e89`:

| Run | Decode TPS | ms/token | TTFT | Event p50 / p99 |
| --- | ---: | ---: | ---: | --- |
| 1 | 55.993 | 17.859 | 264.2 ms | 18.133 / 18.753 ms |
| 2 | 55.970 | 17.867 | 262.1 ms | 18.171 / 18.853 ms |
| 3 | 55.958 | 17.871 | 259.9 ms | 18.183 / 18.823 ms |
| 4 | 56.012 | 17.853 | 260.8 ms | 18.086 / 18.990 ms |
| 5 | 56.155 | 17.808 | 262.9 ms | 18.018 / 18.853 ms |
| Aggregate | **56.018 mean, 55.993 median, 0.080 stddev** | 17.851 mean | 262.0 ms mean | 18.133 / 18.853 ms median |

This is only 0.20 percent above the accepted 55.905 TPS mean, materially below
the campaign's repeatable-improvement threshold and far below the 60.42 client
TPS matched llama.cpp ROCm 10 reference.  The candidate was therefore reverted
in `a237b12`; the accepted one-wave/two-row implementation remains active.
The benchmark sequence also ended with no KFD GPU processes, confirming that
the service was torn down cleanly.

The retained raw evidence is:

```text
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/moe-q8-reuse-20260829T005332Z/microbench.json
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/moe-q8-reuse-20260829T005332Z/api-first.jsonl
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/moe-q8-reuse-20260829T005332Z/api-repeats.jsonl
```

### Rejected dense Q4_0 one-wave/two-row specialization

The dense Q4_0 vector path uses the same older one-row scheduling structure as
the routed-expert path.  Commit `b4a53d1` applied the accepted one-wave/two-row
pattern to that dense kernel, while leaving CUDA unchanged.  A new shape-aware
microbenchmark covered the exact Gemma projection dimensions recovered from the
GGUF: 2816x4096, 8192x2816, 4224x2816, and 10240x2816.  In isolation it reduced
the measured GPU event time for every shape, including 10240x2816 from 41.34 us
to 28.12 us.

That synthetic gain did not survive the real graph-captured serving path.  The
fixed loopback API workload compiled the candidate from a fresh HIP extension
cache, returned the exact deterministic output SHA-1 `abeee5e73e89`, and used
the same 27.52 GiB of server-reported VRAM, but measured only **54.71 TPS** or
18.277 ms/token.  This is below the 55.89 TPS accepted MoE-specialization
median and below the prior 55.04 TPS repaired baseline.  The dense candidate
was therefore reverted.  It proves that isolated event timing alone is not an
acceptance metric for graph-captured end-to-end decode.

The retained raw evidence is:

```text
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/dense-q4-microbench-20260828T233506Z/
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/dense-q4-two-row-wave-20260828T233930Z/
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/dense-two-row-wave-api-20260828T234147Z/
```

### Rejected dense Q4_0 FP32-output hypothesis

llama.cpp's corresponding vector kernel stores FP32 values, whereas the
FreeToken GGUF adapter normally returns the input dtype, BF16 for this Gemma
run.  That difference was a plausible explanation for the profiler contrast:
FreeToken's generic Q4_0 dense kernel reported 48 architectural VGPRs and the
llama.cpp reference reported 24.  Commit `e77a44b` added a deliberately
benchmark-only Q4_0 flag that changed only the destination tensor to FP32.
It was never wired to the GGUF model layers, and a source guard ensured the
normal serving call retained its BF16 contract.

The result rejects that explanation.  In the first independent event run, the
four exact Gemma projection geometries measured 18.28 us, 33.59 us, 18.16 us,
and 35.98 us respectively.  The profiler trace showed the FP32 specialization
still at **48 VGPRs**, 128 SGPRs, no LDS, and no scratch.  It therefore did not
match llama.cpp's 24-VGPR code shape.  Its profile-run event values also showed
no consistent gain.  The experiment was reverted in `203062f`; no public or
model-serving API changed.

The retained raw evidence is:

```text
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/dense-q4-fp32-output-20260828T234953Z/
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/dense-q4-fp32-output-rocprof-20260828T235258Z/
```

### Rejected dense HIP launch-bound candidate

Commit `c7009a9` tested the other conspicuous structural difference from the
matched llama.cpp Q4_0 vector kernel: a HIP-only `__launch_bounds__(32, 1)`
constraint for FreeToken's one-wave dense GEMV.  CUDA was unchanged.  The
candidate compiled cleanly for `gfx1151` and kept the same 48 VGPRs, 128
SGPRs, zero LDS, and zero scratch as the generic FreeToken kernel.  It did
improve three of the four isolated Gemma projection measurements, but did not
reduce the compiler resource gap against llama.cpp.

Five independent graph-captured loopback API runs produced **55.90 TPS mean**
and **55.88 TPS median**, with deterministic output SHA-1 `abeee5e73e89` in
every run.  The accepted MoE-only specialization measured 55.91 TPS mean and
55.89 TPS median under the same workload.  The candidate therefore has no
meaningful decode gain and its 274.7 ms mean TTFT was worse than the accepted
candidate's 262.2 ms mean.  It was reverted in `1d555e3`; the upstream-ready
path remains unchanged by this experiment.

The retained raw evidence is:

```text
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/dense-q4-launch-bounds-20260828T235459Z/
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/dense-q4-launch-bounds-rocprof-20260828T235726Z/
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/dense-launch-bounds-api-20260828T235756Z/
```

### Rejected indexed Q4_0 dense HIP kernel

The dominant llama.cpp Q4_0 trace was rechecked before this experiment.  Its
main kernel uses the same 32-thread by 1-row workgroup as FreeToken, but
reports 24 VGPRs versus FreeToken's 48.  Commit `a5e04d1` isolated the remaining
source-level difference: a Q4_0-only HIP kernel that keeps the base weight
pointer and block index separate until the vector-dot helper.  It preserved
the generic FreeToken launch geometry and left CUDA and every non-Q4_0 type
unchanged.

The isolated evidence was favorable but insufficient.  The four exact Gemma
dense projections measured 16.49 us, 28.69 us, 17.83 us, and 35.52 us, and the
profile trace measured 18.16 us, 22.88 us, 13.17 us, and 29.22 us.  The compiler
still used 48 VGPRs, 128 SGPRs, no LDS, and no scratch.  The first full
graph-captured API run preserved the deterministic output SHA-1
`abeee5e73e89`, but collapsed to **15.21 TPS**, 65.73 ms/token, 3014.5 ms TTFT,
and 1674.5 ms p99 event latency.  This is a functional result but a clear
performance failure.  It was reverted in `b281e0e` and must not be retried
without an explanation for the end-to-end stalls.

The retained raw evidence is:

```text
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/dense-q4-indexed-pointer-20260829T000901Z/
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/dense-q4-indexed-pointer-rocprof-20260829T001126Z/
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/dense-indexed-pointer-api-20260829T001156Z/
```

### Rejected scalarized dense Q4_0 dot-product candidate

The source and trace audit found that the historical FreeToken dense Q4_0
helper materializes two packed Q4 words and four Q8 words in short local
arrays before issuing four DP4A operations.  llama.cpp's newer HIP path does
not share FreeToken's old wrapper structure, so a HIP-only candidate replaced
only that dense helper with named scalar values.  It retained the original
packed GGUF layout, four DP4A operations in the same order, scale formula, and
BF16 output contract.  CUDA and the separately accepted routed-MoE kernel were
unchanged.

The candidate compiled for `gfx1151`, passed the targeted HIP build and
attention tests, and produced finite results for all four exact Gemma dense
projection shapes.  Its isolated event times were 17.38 us for 2816x4096,
28.38 us for 8192x2816, 19.20 us for 4224x2816, and 35.84 us for 10240x2816.
That showed useful synthetic movement, especially for the second shape, but
was not enough to accept it.

Five independent graph-captured API runs all returned the deterministic
SHA-1 `abeee5e73e89`, retained 27.52 GiB server-reported VRAM, and left no KFD
process after shutdown.  Their TPS range was 55.812 to 56.155, with **55.946
TPS mean** and **55.919 TPS median**.  Those figures differ from the accepted
Q4_0 MoE baseline by only 0.041 TPS mean and 0.025 TPS median, while mean TTFT
increased from 262.2 ms to 269.7 ms.  This is normal run-to-run noise, not a
repeatable end-to-end improvement, so it was reverted in `d9ce2c5`.

The retained raw evidence is:

```text
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/dense-q4-scalarized-20260829T003720Z/microbench.json
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/dense-q4-scalarized-20260829T003720Z/microbench-rocprof.json
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/dense-q4-scalarized-20260829T003720Z/api-first.jsonl
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/dense-q4-scalarized-20260829T003720Z/api-repeats.jsonl
```

### Rejected Q4_0 MoE route-grouping candidate

The source comparison showed that llama.cpp places the eight routed experts in
separate waves of one multi-wave MoE workgroup, while FreeToken's accepted HIP
specialization uses one workgroup per route.  Commit `dc73e8e` tested that
topology directly: a HIP-only Q4_0 kernel with eight independent 32-lane route
waves in a 256-thread workgroup, retaining the accepted two-output-row
arithmetic inside each wave.  CUDA and all non-Q4_0 formats remained unchanged.

The candidate compiled for `gfx1151` and passed the targeted HIP build tests,
but failed the shape-accurate microbenchmark gate.  For Gemma's eight-route
decode geometry it measured 34.93 us gate/up plus 30.69 us down, or **65.62 us
per pair**, versus the accepted two-row kernel's 64.25 us mean pair time.  Since
the grouped workgroup was slower before the API workload, no server benchmark
was run.  It was reverted in `96c51f9` and the accepted one-wave/two-row MoE
kernel remains active.

The retained raw evidence is:

```text
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/q4-moe-route-group8-20260829T001802Z/
```

### Rejected Triton GQA attention eight-warp candidate

Gemma's sliding decode attention has 16 query heads, 8 KV heads, a 256-wide
head dimension, and a 1,024-token sliding window.  The HIP production path
uses a 16-head padded tile, 32-token KV blocks, and four Triton warps.  A
shape-accurate benchmark tested head tiles of 2, 4, and 8, a 64-token KV
block, and two or eight warps.  All tile and 64-token-block alternatives were
slower.  Eight warps was faster in isolation: 39.51 us versus 41.87 us for
sliding attention, and 49.84 us versus 93.17 us for Gemma's 2-KV-head,
512-wide full-attention geometry.

That microbenchmark win did not survive the full serving workload.  An
otherwise identical graph-captured loopback OpenAI-compatible API run with
eight warps returned the expected deterministic SHA-1 `abeee5e73e89`, but
measured **53.76 TPS**, 18.600 ms/token, 281.2 ms TTFT, and 20.227 ms p99
event latency.  This is below the accepted five-run 55.91 TPS mean.  The
production override was removed, so normal HIP serving remains at four warps;
the benchmark-only probe parameters remain available for future controlled
research.  This result is a second independent example of why isolated GPU
event timings cannot be used as a serving-performance acceptance criterion.

The retained raw evidence is:

```text
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/gqa-attention-blockh2-20260829T002315Z/
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/gqa-attention-blockh4-8-20260829T002334Z/
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/gqa-attention-blockn64-20260829T002442Z/
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/gqa-attention-warps2-8-20260829T002459Z/
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/gqa-attention-global-warps8-20260829T002558Z/
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/attention-warps8-api-20260829T002618Z/
```

The best verified FreeToken command shape is:

```bash
export ROCM_PATH=/opt/rocm-10.0
export HIP_PATH=/opt/rocm-10.0
export TORCH_EXTENSIONS_DIR=/home/david/freetoken-amd/cache/torch_extensions

ft serve --model-path /home/david/freetoken-amd/models/Gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf \
  --attention-backend triton --moe-backend offload --moe-cache-size 4096 \
  --num-tokens 8320 --memory-ratio 0.50 --max-running-requests 1 \
  --max-seq-len-override 8320 --cuda-graph-max-bs 1
```

The port now derives and exports `PYTORCH_ROCM_ARCH=gfx1151` before the GGUF
extension is compiled when the operator did not set an explicit architecture.
This avoids compiling for unnecessary visible targets and makes the extension
cache target-specific.  It does not itself increase steady-state TPS because
the original HIP build already selected `gfx1151` on this single-GPU host.

The remaining gap is not an untested cache or residency setting: Gemma's GGUF
adapter only supports the native Q4_0 offload implementation, and the accepted
configuration keeps all 4,096 routed-expert slots resident while retaining a
verified 8,320-token KV pool.  Closing the gap requires a profile-guided
improvement to the HIP GGUF decode kernels or another proven ROCm attention or
quantized-linear implementation.

The initial direct `rocprofv3` attempts did fail because the host profiler
injected a second LLVM and rocprofiler SDK beside the SDK bundled with the
PyTorch ROCm wheel.  That historical failure is retained in
`rocprof-gfx1151*/` and `rocprof-launch-gfx1151-v2/` under the raw artifact
directory.  It was subsequently repaired by
[`scripts/lan223-rocprof-wheel-sdk.sh`](../scripts/lan223-rocprof-wheel-sdk.sh),
which directs the host profiler front end to the wheel's matching SDK.  The
repaired launch produced FreeToken kernel traces, including the active
`moe_vec_q4_0_hip_two_rows` kernel.  Traces are diagnostic evidence only and
are never used as TPS scoring because profiling changes execution timing.

### Current-source trace and rejected RDNA4 dense Q4_0 eight-wave candidate

After an unprofiled final-source warm run rebuilt the path-specific native HIP
extension, the complete loopback API workload returned the established greedy
Gemma SHA-1 `abeee5e73e89` at **60.16 TPS**, 16.623 ms per token, 259.0 ms
TTFT, 16.877 ms p50 event latency, and 17.931 ms p99 event latency.  It kept
the full 4,096-slot expert cache and 8,320-token KV budget.  This is the
current unprofiled checkpoint for the accepted source path.

The repaired profiler wrapper then traced that already-built final source.
The trace also preserved the output SHA-1, but measured 41.41 TPS and a 216.2
ms p99 because tracing changes dispatch timing.  It is not a performance
result.  Its kernel statistics do identify the next work order: routed Q4_0
MoE vector work consumed 31.05 percent of GPU kernel time, dense Q4_0 vector
work 28.18 percent, and dense Q6_K vector work 15.64 percent.  The active
MoE kernel name was `moe_vec_q4_0_hip_two_rows`, proving that the trace covers
the accepted HIP specialization rather than the earlier generic path.

Current llama.cpp source uses an RDNA4-specific eight-wave policy for simple
one-vector Q4_0 matvecs.  Candidate commit `1bf9489` applied that scheduling
policy only to FreeToken's dense HIP Q4_0 launcher.  It deliberately retained
the generic dot product, Q4_0 and Q8_1 packing, BF16 result contract, CUDA
path, all non-Q4_0 types, and the separately accepted routed-MoE kernel.
This made the candidate distinct from the already rejected dense two-row and
launch-bound experiments.

The four shape-accurate dense microbenchmarks were mixed when rerun with 10
warmups and 100 repetitions: the candidate improved 8,192 by 2,816 from
28.29 to 27.49 microseconds and 4,224 by 2,816 from 26.89 to 17.26
microseconds, but regressed 2,816 by 4,096 from 21.30 to 23.91 microseconds
and 10,240 by 2,816 from 33.04 to 34.18 microseconds.  Because Gemma uses all
four projections, this was insufficient to accept the launch policy.

The full graph-captured API result confirmed rejection.  It returned the
exact established SHA-1, but reached only **59.33 TPS**, 16.856 ms per token,
290.7 ms TTFT, and 18.011 ms p99 event latency.  This is below the current
60.16 TPS final-source checkpoint and below the established accepted five-run
60.11 TPS median.  The candidate remains on its separate branch and is not
part of the upstream-review branch.  Its isolated worktree initially lacked
the unchanged native pinned-memory extension; the test setup copied the
validated extension only after SHA-256 and byte-for-byte equality checks.
That repair affected no source logic and the resulting API run is the only
performance outcome used for this decision.

The retained raw evidence is:

```text
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/gemma-final-path-warm-20260829T021243Z/
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/gemma-final-current-kernel-trace-20260829T021738Z/
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/dense-q4-current-baseline-micro-20260829T022723Z/
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/dense-q4-rdna4-eightwaves-micro-20260829T022528Z/
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/dense-q4-rdna4-eightwaves-api-repaired-20260829T023038Z/
```

### Rejected RDNA4 dense Q6_K eight-wave candidate

The current final-source trace showed dense Q6_K vector work at 15.64 percent
of total traced GPU kernel time.  Gemma uses Q6_K for its tied token embedding
and LM head, and the matching current llama.cpp RDNA4 policy selects eight
waves for one-vector Q6_K matvec.  Candidate commit `ebf3f06` therefore
changed only FreeToken's HIP dense Q6_K wrapper to launch the established
generic Q6_K dot-product kernel with eight independent row waves per block.
It retained Q6_K and Q8_1 packing, reduction arithmetic, the BF16 result
contract, CUDA behavior, Q4_0 dense behavior, and all routed-MoE behavior.

The full graph-captured loopback API workload returned the exact established
Gemma SHA-1 `abeee5e73e89`, used 15.66 GiB VRAM, and reported 259.6 ms TTFT
with a 17.663 ms p99 event latency.  Its decode result was **59.98 TPS** or
16.672 ms per token.  That is close to, but below, the 60.16 TPS accepted
final-source checkpoint.  A one-run result without a TPS improvement does not
justify a second specialized scheduling path, so the candidate remains on its
separate experiment branch and is not part of the upstream-review branch.

The test used the unchanged native pinned-memory extension after SHA-256 and
byte-for-byte equality checks against the validated final source.  The raw
evidence is retained at:

```text
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/dense-q6-rdna4-eightwaves-api-20260829T023720Z/
```

### Current host-interference qualifier

A read-only LAN-223 health capture at 2026-08-29T02:41:15Z found no GPU reset,
thermal problem, or active FreeToken server.  The Radeon 8060S was idle at
30 C after the test.  It did, however, identify two pre-existing user-owned
filesystem scans in uninterruptible `D` state: one scanning `/home/david`,
`/mnt`, and `/data` for large GGUF or SafeTensors files, and one scanning
`/home/david` and `/media/david` for Gemma GGUF files.  At capture time they
had been alive for approximately 8.8 and 6.1 hours respectively.

The same capture reported I/O full-pressure at 0.61 percent over ten seconds
and retained kernel warnings that `kfd_process_wq_release` and
`svm_range_deferred_list_work` had exceeded their CPU workqueue budget.  These
facts do not prove that a particular FreeToken result is invalid, but they
provide a concrete explanation for occasional multi-millisecond dispatch
outliers and the isolated 52.50 TPS baseline run.  They can affect both
FreeToken and llama.cpp under a matched test.

No process priority, service state, kernel option, ROCm installation, or
hardware component was changed by this investigation.  Any decision to stop
or otherwise alter the two user-owned scans requires explicit operator
authorization.  Until then, accepted performance claims remain based on
multiple clean launches and retain raw tail-latency data rather than hiding
the interference.

### Current review-branch static validation

The current upstream-review commit `6c6198b10d9fb6a9c93e0aa94a05ac4144ec061d`
was validated directly on LAN-223 after the I/O evidence capture tooling was
added.  The check completed without starting an inference server or changing
host state:

```text
python -m compileall -q python benchmarks                 passed
pytest -q tests/kernels/test_gguf_hip_build_flags.py \
          tests/utils/test_rocm_runtime.py                4 passed
```

The raw output and commit metadata are retained at:

```text
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/current-review-static-validation-20260829T024456Z/
```

A temporary high-performance DPM governor test could not be run because the
non-root LAN-223 account cannot write `power_dpm_force_performance_level`;
automatic mode was unchanged.

Raw campaign artifacts are retained on LAN-223:

```text
/home/david/freetoken-amd/artifacts/amd-optimization-2026-08-28/
```

## Deep-investigation baseline and profiler repair

The reproducible read-only baseline is captured by
[`../scripts/lan223-capture-baseline.sh`](../scripts/lan223-capture-baseline.sh).
The first baseline was written to:

```text
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/baseline-20260828T220753Z/
```

### Test-checkout repair and revalidated shipping baseline

During the follow-on investigation, the isolated LAN-223 source checkout was
found at `61a1505`.  That commit contained the subsequently rejected
two-block-residency Q4_0 MoE experiment.  The authoritative branch had already
reverted that experiment at `b77825d` and documented the rejection at
`222cbd3`.  Using the stale checkout for another benchmark would have made the
result impossible to attribute to the branch under review.

The checkout was clean, so it was repaired with a fast-forward only update to
`origin/amd-rocm-gfx1151`, reaching `222cbd3`.  No production process,
llama-swap configuration, or other LAN host was touched.  The next run used a
new, dated `TORCH_EXTENSIONS_DIR`, forcing a fresh native HIP binary rather
than reusing the binary compiled from the stale source.

| Item | Revalidated value |
| --- | --- |
| Artifact directory | `/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/repaired-baseline-20260828T224642Z/` |
| Source commit | `222cbd3` |
| Model SHA-256 | `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d` |
| Extension build | Fresh ROCm 10 `hipcc`, `--offload-arch=gfx1151`, `-O3` |
| API and workload | Loopback FreeToken API, greedy AIME-25 problem 0, one warm and one measured request |
| Measured completion | 127 tokens, 126 decode intervals |
| Client decode throughput | **55.04 TPS** or **18.169 ms/token** |
| TTFT | 295.2 ms |
| Event p50 / p99 | 18.454 ms / 19.509 ms |
| Output SHA-1 | `abeee5e73e89`, identical to the earlier shipping-configuration run |
| Post-run ROCm process check | No KFD PIDs |

The single revalidation is consistent with the existing 55.44 TPS shipping
baseline and remains below the 60.42 TPS matched llama.cpp reference.  It is a
provenance repair, not a new performance claim and not a substitute for the
planned repeated candidate measurements.

### Rejected Q4_0 aligned-load candidate

The matched traces showed FreeToken's Q4_0 vector kernels using 48 VGPRs per
thread, while llama.cpp's corresponding generic Q4 vector kernel reported 24
VGPRs.  Both used a 32-thread workgroup with zero LDS and scratch allocation.
As a narrow, low-risk test, commit `b8de163` replaced only the Q4_0 packed-load
helper expressions with the aligned `get_int_b2` and `get_int_b4` expressions
used by the current llama.cpp HIP source.  The dot-product arithmetic, output
type, data layout, model, workload, and launch geometry were otherwise
unchanged.

The target-host HIP build-configuration tests passed, the extension rebuilt
for `gfx1151`, and the output SHA-1 remained `abeee5e73e89`.  However, the
candidate measured 54.99 TPS or 18.185 ms/token, versus 55.04 TPS or 18.169
ms/token for the immediately preceding repaired baseline.  That difference is
well inside normal run variation and does not improve the runner.  The
candidate was therefore reverted by `c1899a0`; it is not part of the shipping
configuration.

The raw candidate evidence is retained at:

```text
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/q4-load-alignment-20260828T225237Z/
```

This eliminates aligned helper spelling as the explanation for the measured
register and throughput gap.  The next candidate must change a more material
component: the Q4_0 vector-kernel execution structure, MoE expert dispatch,
or intermediate BF16 output path.

### Rejected Q4_0 FP32-intermediate candidate

llama.cpp's HIP Q4 vector paths use an FP32 destination, while FreeToken's
normal Q4_0 MoE path returns an activation-typed BF16 tensor after each vector
product.  Commit `5bbe10f` added a deliberately opt-in experiment that used
FP32 only for the two Q4_0 MoE vector-product temporaries, then converted the
final MoE result back to the original BF16 public contract.  It was activated
only with `FREETOKEN_GGUF_MOE_FP32_INTERMEDIATE=1`; normal launches stayed on
the existing dtype-preserving path.  The target-host build-flag and opt-in
contract tests passed before the full-model run.

The first launch under this candidate used the literal placeholder `model`
instead of the local GGUF path and exited during model resolution.  It did not
reach HIP compilation, graph capture, or an API request.  The failed artifact
is retained as a labelled harness error and is excluded from every comparison.
The corrected launch used the exact Gemma GGUF checksum, offload backend,
0.50 memory ratio, graph capture, greedy AIME-25 problem 0, and 128-token
decode procedure used by the repaired baseline.

| Candidate | Decode TPS | ms/token | TTFT | Output SHA-1 | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| Repaired BF16 baseline | 55.04 | 18.169 | 295.2 ms | `abeee5e73e89` | Reference |
| FP32 intermediates | 55.11 | 18.144 | 292.6 ms | `ce247609d76c` | Rejected |

The 0.14 percent TPS change is smaller than the observed run-to-run variation,
does not close the gap to the 60.42 client TPS ROCm 10 llama.cpp reference,
and changes the deterministic greedy response hash.  The candidate was
therefore reverted and is not a shipping option.  Raw evidence remains on
LAN-223 at:

```text
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/fp32-intermediate-20260828T230126Z/
/home/david/freetoken-amd/artifacts/amd-deep-investigation-2026-08-28/fp32-intermediate-retry-20260828T230209Z/
```

It confirms the active device is `gfx1151`, PyTorch is
`2.13.0+rocm10.0.0` with HIP `7.15.26333`, and `/opt/rocm` resolves to
`/opt/rocm-10.0`.  It also records that the system package database retains
ROCm 7.2 development packages.  This alone does not prove an application
runtime conflict, so library maps were collected before changing any host
component.

The maps show that the PyTorch wheel loads its own ROCm SDK, including LLVM 23
and rocprofiler-sdk 1.3.5, from `_rocm_sdk_core` in the virtual environment.
The host `rocprofv3` launch initially injected a second LLVM 23 and profiler
SDK from `/opt/rocm-10.0`, causing `import torch` to abort with duplicate LLVM
registration for `spirv-expand-step`.  The failure was reproduced with a
minimal PyTorch import, so it is not caused by FreeToken.

`scripts/lan223-rocprof-wheel-sdk.sh` repairs the launch path without editing
the host installation.  It keeps the host `rocprofv3` front end but passes
`--rocm-root` for the wheel's `_rocm_sdk_core`, making the profiler use the
same library identities as PyTorch.  The repair was validated by profiling a
small HIP allocation and reduction.  ROCm emitted `kernel_trace.csv` and
`kernel_stats.csv` with the expected GPU dispatches.  Use this wrapper only
for profiling, never for TPS scoring because tracing alters execution time.

The first full FreeToken trace launch passed PyTorch import and model loading,
then reached the GGUF JIT compiler.  The profiler environment is inherited by
that compiler subprocess, so the run was stopped before a request was sent.
The next trace must warm the GGUF extension unprofiled, then profile the
already-built decode path, or explicitly prevent profiler injection into JIT
child processes.  This avoids treating compile activity as token-generation
performance.

## GGUF extension reuse validation

The first Gemma request after the original source change built the native HIP
GGUF extension.  A subsequent complete server restart retained the existing
Torch extension cache.  Its first API request returned HTTP 200 and Ninja
reported `no work to do`, proving the compiled shared module was reused.
Torch still runs a lightweight hipify and dependency check before loading the
cached module; it did not run `hipcc` compilation or shared-library linking.
See the persistent-cache operating procedure in
[`amd-rocm-gfx1151.md`](amd-rocm-gfx1151.md#persistent-gguf-hip-jit-cache).

## Commands used

Qwen was started in the isolated environment with this functional shape:

```bash
ft serve --model-path /home/david/freetoken-amd/models/Qwen3.6-35B-A3B-NVFP4 \
  --served-model-name qwen3.6-35b-a3b-nvfp4-amd --host 127.0.0.1 --port 18501 \
  --attention-backend triton --moe-backend offload --nvfp4-backend triton \
  --expert-load serial --moe-cache-auto --memory-ratio 0.35 \
  --max-seq-len-override 8192 --kv-reserve-tokens 2048 \
  --cuda-graph-max-bs 0 --disable-pynccl --disable-moe-prefill-overlap
```

Gemma used the native GGUF model file and its own loopback port:

```bash
ft serve --model-path /home/david/freetoken-amd/models/Gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf \
  --served-model-name gemma-4-26b-a4b-q4-amd --host 127.0.0.1 --port 18502 \
  --attention-backend triton --moe-backend offload --expert-load serial \
  --moe-cache-auto --memory-ratio 0.50 --max-seq-len-override 8192 \
  --kv-reserve-tokens 2048 --cuda-graph-max-bs 0 --disable-pynccl
```

The API checks used `/v1/models` and `/v1/chat/completions`, both with normal
JSON responses and with `stream: true`.  The front-end port can answer before
the worker finishes loading, so the successful tests waited for the server log
line `API server is ready to serve` before submitting requests.

## AMD-specific corrections verified here

1. ROCm detection is explicit, preventing `gfx1151` from being treated as an
   NVIDIA SM 11.5 capability.
2. CUDA-only optional backends are not selected on HIP.
3. DLPack and fast indexed-copy tensor handling accepts HIP tensors.
4. HIP avoids the unsafe grouped NVFP4 prefill kernel and uses the native
   Triton serial expert implementation instead.  This trades prompt prefill
   speed for correctness on the current Strix Halo stack.
5. The Gemma GGUF JIT discovers a system Thrust include directory when the
   PyTorch wheel omits Thrust.  It passes that path as a compiler system
   include, avoiding an attempted hipify write into the ROCm installation.
6. The same JIT adds a system ROCm library directory only when the wheel SDK
   lacks the unversioned `libamdhip64.so` linker name.  On LAN-223 this allowed
   the native `gfx1151` object and shared module to compile and link.

## Known limitations and follow-up work

- This is a functional API validation, not a performance benchmark.  The
  recorded request timings include the chosen small fixed requests and are not
  tokens-per-second claims.
- CUDA graph capture remains disabled for the HIP MVP.
- Qwen's HIP prefill deliberately uses the safe serial Triton route instead of
  the grouped NVFP4 prefill route that produced an HSA aperture violation on
  this machine.
- The first Gemma request compiles its GGUF HIP extension and has a substantial
  cold-start cost.  Later requests use the cached module.
- llama-swap integration is intentionally outside this release gate.

## Local checks completed

```bash
python -m compileall -q python
git diff --check
```

The port's HIP gate tests are retained under `tests/utils/test_rocm_runtime.py`.
The live end-to-end checks above are the required full-model validation for
this change.
