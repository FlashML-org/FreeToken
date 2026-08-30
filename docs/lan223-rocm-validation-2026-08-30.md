# LAN-223 ROCm validation results, 2026-08-30

## Scope

This report records post-repair validation of the native FreeToken ROCm/HIP port on the LAN-223 Radeon 8060S. It covers the OpenAI-compatible API, Gemma 4 vision correctness, Qwen reliability, a controlled llama.cpp ROCm comparison, and a strict multi-turn endurance run. It is local hardware evidence, not a reproduction of the FreeToken paper's NVIDIA results.

## Reproduction boundary

| Item | Observed value |
| --- | --- |
| Host GPU | AMD Radeon 8060S Graphics, `gfx1151` |
| FreeToken revision | `d6ee8cef479c6e72b2210c24dc848b66cf9da75a` |
| Python | 3.12.13 |
| HIP | 7.15.26333 |
| PyTorch | `2.13.0+rocm10.0.0`, HIP 7.15.26333 |
| Qwen service | `qwen3.6-35b-a3b-nvfp4-amd` on loopback port 1919 |
| Gemma service | `gemma4-26b-q4-amd` on temporary loopback port 1923 |
| llama.cpp control | ROCm 10 build, temporary loopback port 1921 |

All model-server work used the native ROCm/HIP path. No Vulkan runner, CPU fallback, llama-swap route, or other LAN host was used as a substitute.

## Gemma 4 multimodal repair

The first live Gemma controls established that image tensors reached the GPU but the model answered simple colors incorrectly. The repair had two required parts:

1. Preserve `mm_pixel_values` and `mm_image_position_ids` when the tokenizer server forwards a user message to the scheduler.
2. Emit RGB patches in channel-planar order, not pixel-interleaved order. The Gemma projector weight `v.patch_embd.weight` is a convolution kernel with `[output, channel, patch_y, patch_x]` layout, so each patch must contain all red values, then green, then blue.

The fixed path was tested through the real OpenAI `image_url` data-URL contract, decoding, patchification, tensor wire protocol, ROCm vision tower, projector, embedding scatter, and response generation.

| Control | FreeToken result | llama.cpp ROCm result |
| --- | --- | --- |
| solid red | pass | pass |
| solid green | pass | pass |
| left half of red-left/blue-right image | red | red |
| solid blue | pass | pass |
| solid yellow | pass | pass |
| right half of red-left/blue-right image | blue | blue |
| top half of blue-top/yellow-bottom image | blue | blue |

FreeToken passed all seven controls in one extended run, then passed all 21 requests in three complete repetitions. Its 45 to 65 word visual-description control also passed, correctly describing a red left side and blue right side at 53.67 visible output tokens per second. The matching llama.cpp ROCm control passed the same seven deterministic fixtures.

## Qwen API and correctness

The FreeToken Qwen endpoint returned a healthy status before and after every exclusive Gemma or llama.cpp control. `/v1/models` reported the expected model and 8,192-token configured context length.

The deterministic visible-output suite passed on both FreeToken and llama.cpp:

| Check | FreeToken | llama.cpp ROCm |
| --- | --- | --- |
| exact `LAN223` output | pass | pass |
| `17 * 19 = 323` | pass | pass |
| exact JSON fields `status=ok`, `value=7` | pass | pass |

Ten independent three-turn FreeToken conversations also passed every turn: remember `azure-17`, recall it, and transform its numeric component to `23`. The median maximum per-turn TTFT was 0.414 seconds and the worst observed token gap was 39.77 ms.

## Concurrency and long context

The following Qwen API matrix used fixed greedy streaming requests, 128 output tokens, three rounds per level, and retained every raw stream. The first run started from pre-existing swap pressure and is labeled diagnostic rather than clean-memory endurance evidence.

| Concurrent requests | Successful rounds | Mean aggregate TPS | p99 TTFT | p99 token gap |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3 of 3 | 19.96 | 8.60 s | 69.98 ms |
| 2 | 3 of 3 | 28.30 | 0.84 s | 126.80 ms |
| 4 | 3 of 3 | 50.52 | 0.93 s | 139.65 ms |
| 8 | 3 of 3 | 50.78 | 10.56 s | 139.58 ms |

The 8-way result is a saturation result. Aggregate throughput did not improve over four simultaneous requests, while p99 time to first token increased substantially. It is not a recommended interactive concurrency target.

Long-context retrieval used an exact early marker, three samples at each size, and a unique prefix nonce per sample to prevent full-prefix cache reuse. Every sample returned only the required marker.

| Reported prompt tokens | Passed samples | Mean TTFT | Maximum TTFT | p99 token gap |
| ---: | ---: | ---: | ---: | ---: |
| 2,616 | 3 of 3 | 5.58 s | 7.74 s | 38.57 ms |
| 5,176 | 3 of 3 | 9.02 s | 12.97 s | 39.90 ms |
| 7,736 | 3 of 3 | 16.41 s | 17.73 s | 40.47 ms |

## Matched workload comparison with llama.cpp

Both runners executed the same fixed scheduler prompt, 256 requested output tokens, greedy decoding, one concurrent request, one 8,192-token slot, and three measured samples after warmup on LAN-223. The values are decode TPS, not aggregate concurrent throughput.

| Runner | Model format | Successful samples | Median decode TPS |
| --- | --- | ---: | ---: |
| FreeToken ROCm/HIP | Qwen3.6-35B-A3B NVFP4 | 3 of 3 | 28.15 |
| llama.cpp ROCm 10 | Qwen3.6-35B-A3B Q4_K_M GGUF | 3 of 3 | 48.87 |

This is a same-host, same-prompt, same-output-length comparison, but it is not a quantization-equivalent comparison. FreeToken loaded NVFP4 while llama.cpp loaded Q4_K_M GGUF. Therefore it proves the current observed runner outcome for these deployed artifacts, not an intrinsic winner between FreeToken and llama.cpp. The current FreeToken configuration does not meet or exceed the llama.cpp decode figure in this workload.

## Exact-Q4_K_M ROCm comparison

The branch now includes a native FreeToken loader for the same
`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` file used by the llama.cpp control. This path
keeps the GGUF weights packed: dense Q8_0 and Q6_K tensors use the native GGML
HIP operators, routed gate and up experts use Q4_K, routed down experts use
Q5_K or the file's late-layer Q6_K exception, and the Qwen hybrid
Gated-DeltaNet metadata and recurrent-state layout are handled by the native
Qwen3.5 model path.

Before serving, a source-revision-specific gfx1151 helper cache compiled 82
native ROCm/HIP modules and a strict no-JIT verifier loaded all 82. The Q4
server then started on a temporary loopback port with the same 8,192-token
context policy used by llama.cpp: `0.35` memory ratio, 8,192-token KV reserve,
one host, one GPU, greedy sampling, one request, the fixed scheduler prompt,
256 requested output tokens, warmup, and three scored samples. The FreeToken
Q4 server resolved 8,626 MoE cache slots and 8,227 KV pages.

| Runtime | Model file and format | Mean decode TPS | Median decode TPS | Sample standard deviation | Quality suite |
| --- | --- | ---: | ---: | ---: | --- |
| FreeToken ROCm/HIP | Exact Q4_K_M GGUF | 48.444 | 48.450 | 0.0267 | 3 of 3 pass |
| llama.cpp ROCm 10 | Same exact Q4_K_M GGUF | 49.125 | 49.131 | 0.0138 | 3 of 3 pass |

The fresh same-format difference is 0.680 TPS, or 1.39 percent in favor of
the current llama.cpp control. This is the relevant comparison for runner
efficiency because it removes the NVFP4-versus-Q4_K_M weight-format difference.
FreeToken is very close but does not yet meet or exceed llama.cpp in this
strict matched workload.

FreeToken's additional caller-rendered, 512-token raw-prompt control produced
511 visible completion tokens at 48.487 TPS and 433.11 ms TTFT. The standard
visible-output quality suite passed its exact `LAN223`, arithmetic `323`, and
strict JSON controls. A temporary GPU `high` DPM policy was also tested with
the loaded Q4 server, but it reduced mean decode throughput to 47.287 TPS while
quality still passed. The normal `auto` policy therefore remains the accepted
policy for this configuration.

The exact-Q4 evidence is retained on LAN-223 at
`/home/david/freetoken-amd/artifacts/qwen35moe-gguf-full-control-20260830T141438Z/`
and
`/home/david/freetoken-amd/artifacts/qwen35b-llamacpp-rocm10-q4matched-20260830T142002Z-retry/`.

The Q4 server also passed the full cold long-context retrieval control: five
unique-prefix requests at 6,856 reported prompt tokens all returned only
`azure-17`. Mean TTFT was 26.989 seconds, maximum TTFT was 39.088 seconds,
and p99 visible token gap was 23.890 ms. The strict 30-session multi-turn
endurance gate was initially not qualified for the `0.35` memory-ratio
configuration. After the cold long-context run, 3.3 GiB of whole-host swap
usage was observed. A controlled reset returned the host counter to zero and
preserved endpoint health, but 540 KiB reappeared immediately before the first
endurance session, exceeding the original whole-host 64 KiB guard. That
measurement did not identify the process responsible for the swapped pages.
The original high-cache profile remains an active stability investigation
because it can also hit the separate ROCm SVM-resident-memory failure above,
not because of a throughput or quality failure.

## Q4 SVM-resident-memory recovery profile

Follow-up investigation established that the failures above were not an
incorrect answer or an API-contract failure. A Q4 server with the original
`0.35` memory ratio could initialize successfully, but a first decode after a
forced cancellation could stall. The Linux kernel recorded
`amdgpu: SVM mapping failed, exceeds resident system memory limit`; the
associated FreeToken scheduler worker consumed CPU while the request emitted no
response bytes. The test procedure also revealed that stopping only the HTTP
parent leaves its multiprocessing children alive, including a child that keeps
the internal distributed port `1923` bound. All subsequent controls used a
dedicated process group and terminated that full group before another GPU
server was started.

The recovery profile preserves the exact same Q4_K_M GGUF, native ROCm/HIP
path, 8,192-token context policy, four request slots, OpenAI-compatible API,
and automatic MoE cache policy. It changes only the memory ratio from `0.35`
to `0.25`, retains the host's temporary `vm.swappiness=1` test policy, and
starts from a verified zero-swap state. The lower ratio resolved 5,465 MoE
slots and 8,237 KV pages, leaving 23.06 GiB free after initialization instead
of about 17.46 GiB. It is therefore a stability-oriented configuration, not a
claimed decode-speed optimization.

| Control | Result |
| --- | --- |
| visible-output quality suite | 3 of 3 pass |
| multi-turn state retention | 30 of 30 sessions pass, zero KiB verified runner-process-group swap at every session boundary |
| multi-turn p99 maximum turn TTFT | 0.429 s |
| multi-turn p99 visible-token gap | 25.75 ms |
| 6,856-token cold marker retrieval | 5 of 5 pass, 24.733 s mean TTFT, 26.255 s maximum TTFT |
| two simultaneous users | 3 of 3 rounds pass, 45.29 mean aggregate TPS, 8.192 s p99 TTFT |
| four simultaneous users | 3 of 3 rounds pass, 79.00 mean aggregate TPS, 1.561 s p99 TTFT |

The five long-context requests coincided with a 4.864 MiB increase in the
whole-host swap counter, despite the low-swappiness policy. That counter is
useful host telemetry but does not identify the model process, so the
long-context result is a successful quality and latency result, not proof of a
strict zero-swap all-day service state. Later attribution showed that desktop
and monitoring daemons can hold swapped pages while every member of the
verified FreeToken server process group reports `VmSwap: 0 kB`. The ongoing
wall-clock battery therefore records whole-host swap but fails only when the
dedicated FreeToken process group itself has swapped pages.

The same fixed 256-token scheduler workload was rerun from the current
FreeToken Q4 profile and a fresh ROCm 10 llama.cpp control, with the same GGUF
file, prompt, tokenizer, temperature, top-p, top-k, output length, context,
and one request. Both also passed the same deterministic three-case quality
suite.

| Runtime | Mean decode TPS | Median decode TPS | Mean TTFT | Quality suite |
| --- | ---: | ---: | ---: | --- |
| FreeToken Q4 recovery profile | 47.960 | 48.075 | 0.453 s | 3 of 3 pass |
| llama.cpp ROCm 10 current control | 48.831 | 48.832 | 0.062 s | 3 of 3 pass |

The recovery profile is 0.871 TPS, or 1.78 percent, below the fresh llama.cpp
control for that fixed decode workload. It restores full functional
qualification under the memory guard but does not meet or exceed llama.cpp.
The original higher-cache Q4 profile remains the closer decode result, at 1.39
percent below its fresh llama.cpp control, but requires a repair for the SVM
resident-memory limit before it can be recommended as the stable profile.

Retained raw evidence for this recovery investigation is under
`/home/david/freetoken-amd/artifacts/qwen35moe-gguf-memory-ratio-025-20260830T150554Z/`
and the fresh llama.cpp control is under
`/home/david/freetoken-amd/artifacts/qwen35moe-llamacpp-rocm10-current-harness-retry-20260830T151654Z/`.

## Initial clean-memory endurance

The initial 30-session battery reset whole-host swap before starting and
enforced a maximum of 64 KiB at every session boundary. It completed before
the later process attribution work. Its functional and timing results remain
valid, but the whole-host swap limit is superseded by the verified
runner-process-group gate used by the current wall-clock endurance battery.

| Metric | Observed result |
| --- | --- |
| Completed sessions | 30 of 30 |
| Passed sessions | 30 of 30 |
| p95 maximum turn TTFT | 0.596 s |
| p99 maximum turn TTFT | 2.369 s |
| p99 maximum token gap | 39.63 ms |
| Initial swap guard | passed, zero KiB whole-host usage observed after completion |
| Qwen health after run | `status: ok` |
| Final sampled GPU edge temperature | 42 C |

## Process-scoped wall-clock endurance qualification

The corrected endurance battery ran 60 deterministic three-turn conversations
at one-minute cadence, for a full hour of wall-clock observation. It validated
the exact visible answers `ACK`, `azure-17`, and `23` in every session. The
wrapper also resolved the dedicated Q4 HTTP server process group before every
memory sample and rejected a session if any member reported nonzero `VmSwap`.
Whole-host swap was retained as diagnostic telemetry only, because Linux
desktop and monitoring processes can use swap independently of FreeToken.

| Metric | Result |
| --- | --- |
| completed and passed sessions | 60 of 60 |
| runner process-group swap | 0 KiB minimum and maximum |
| maximum-turn TTFT mean | 0.424 s |
| maximum-turn TTFT p95 | 0.414 s |
| maximum-turn TTFT p99 and maximum | 1.184 s |
| maximum visible-token-gap mean | 24.95 ms |
| maximum visible-token-gap p95 | 25.98 ms |
| maximum visible-token-gap p99 and maximum | 27.17 ms |
| whole-host swap telemetry | 33.07 MiB to 38.17 MiB |

The single 1.184-second maximum-turn TTFT observation is retained as an
observed tail outlier, not hidden by a mean-only result. It did not cause an
incorrect answer, runner swapping, process failure, or loss of API service.
The machine was restored after the battery: the temporary Q4 listener on
port 1922 was stopped and `vm.swappiness` was restored to 60. The normal
NVFP4 service on loopback port 1919 required about eight minutes of cold
expert initialization before its readiness log appeared. It was then checked
through the OpenAI-compatible endpoint with a live arithmetic request, which
returned the correct visible answer `4`, and retained its advertised
8,192-token context.

Raw evidence is retained under
`/home/david/freetoken-amd/artifacts/qwen35moe-gguf-process-scoped-endurance-20260830T153333Z/`,
including each request JSON, per-session telemetry, and the machine-generated
`summary.json`. The reusable verifier is
`benchmarks/lan223_qwen/summarize_qwen_gguf_endurance.py`.

## Full-context MoE cache telemetry

The normal Qwen service intentionally leaves MoE counters disabled because the
counter atomics are diagnostic work. A temporary, loopback-only instance was
therefore started with `--moe-collect-stats`, using the same native ROCm/HIP
configuration, `0.35` memory ratio, and 8,192-token KV reservation as the
restored service. The normal no-counter service was restarted immediately after
the test and passed its deterministic AIME output-hash gate.

The diagnostic instance resolved 8,903 MoE cache slots and 8,224 KV pages. Its
fixed scheduler workload completed all three scored samples at 28.035 mean
decode TPS, with only 0.0066 TPS standard deviation. Across 40,800 decode-layer
calls, it selected eight experts per layer and missed 0.586 experts per layer,
for a 7.33 percent MoE cache miss rate. No expert fetches were reported through
the separate fetch counter on this workload.

This result confirms that full 8K context capacity is active while the Qwen
decode rate remains near the accepted 28 TPS baseline. It also supports the
previous rejection of a larger static MoE cache: prior 0.38-memory-ratio
testing reduced misses but did not produce a sustained TPS gain. Cache capacity
alone is therefore not a justified route to closing the current llama.cpp gap.

The telemetry and restoration evidence is retained on LAN-223 at
`/home/david/freetoken-amd/artifacts/qwen-cache-stats-driver-20260830T135236Z/`.
The restored normal service returned the required AIME SHA-1
`0acef4eab6f4`, at 28.60 visible decode TPS, 399.08 ms TTFT, and 38.49 ms p99
stream-event gap.

## Regression tests

The focused regression suite passed 21 tests on LAN-223:

```text
tests/server/test_message_wire.py
tests/tokenizer/test_gemma4_image.py
tests/models/test_gemma4_mmproj_mapping.py
tests/benchmarks/test_lan223_qwen_benchmark.py
```

## Remaining work

1. The quantization-equivalent Qwen control is now complete. The exact Q4_K_M comparison is close but FreeToken remains 1.39 percent below llama.cpp in the fixed single-request decode workload. Any claim to meet or exceed llama.cpp needs a new retained optimization and a fresh matched requalification.
2. Continue kernel-level decode work only from profiler evidence. Existing cache-capacity, graph, copy-grid, DPM-policy, and several dense and NVFP4 kernel candidates did not produce a quality-preserving end-to-end gain. Candidate work must preserve the API, vision, quality, long-context, and endurance gates in this report.
3. The one-hour process-scoped wall-clock endurance workload is complete and qualified. Consider a longer all-day workload only if deployment requires evidence beyond this explicit one-hour qualification.
4. Package sanitized build manifests and selected raw artifacts for the fork and upstream pull request. Do not publish local model files, private host paths, or operational access information.
