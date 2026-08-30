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

## Clean-memory endurance

Before the strict endurance run, diagnostic inspection showed swapped pages belonging primarily to FreeToken multiprocessing workers. With about 18 GiB of RAM available, the existing controlled `swapoff` and `swapon` reset was performed. Qwen remained healthy, swap stayed at zero during a short observation period, and the strict battery was then allowed to start.

The battery ran 30 complete multi-turn sessions and enforced a maximum of 64 KiB swap at every session boundary.

| Metric | Observed result |
| --- | --- |
| Completed sessions | 30 of 30 |
| Passed sessions | 30 of 30 |
| p95 maximum turn TTFT | 0.596 s |
| p99 maximum turn TTFT | 2.369 s |
| p99 maximum token gap | 39.63 ms |
| Swap guard | passed, zero KiB observed after completion |
| Qwen health after run | `status: ok` |
| Final sampled GPU edge temperature | 42 C |

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

1. Add a quantization-equivalent Qwen control before making any broader performance claim. The current NVFP4 versus Q4_K_M result is intentionally labeled non-equivalent. That work requires FreeToken support for the Qwen hybrid GGUF architecture and every tensor encoding used by the reference file, not merely a different launch flag.
2. Continue kernel-level decode work only from profiler evidence. Existing cache-capacity, graph, copy-grid, and several dense and NVFP4 kernel candidates did not produce a quality-preserving end-to-end gain. Candidate work must preserve the API, vision, quality, long-context, and endurance gates in this report.
3. Run a longer wall-clock endurance workload with periodic telemetry if the deployment target requires all-day serving evidence.
4. Package sanitized build manifests and selected raw artifacts for the fork and upstream pull request. Do not publish local model files, private host paths, or operational access information.
