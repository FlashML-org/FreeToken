# LAN-223 Qwen router and cache optimization, 2026-08-29

## Scope

This record covers only the isolated FreeToken server on LAN-223's Radeon 8060S
(`gfx1151`). It did not start, stop, unmask, or reconfigure llama-swap or the
production llama.cpp service. All server instances bound only to `127.0.0.1:1919`.

## Accepted change

`freetoken.moe.fused.fused_topk` now uses FreeToken's vendored Triton softmax
top-k router on ROCm. Qwen3.6 NVFP4 uses 256 experts and selects eight experts
per token. On LAN-223, the candidate matched the PyTorch reference exactly and
reduced router-only latency at the production shape.

| Router microbenchmark | PyTorch reference | HIP Triton | Speedup |
| --- | ---: | ---: | ---: |
| 1 token, 256 experts, top-8 | 0.02449 ms | 0.01512 ms | 1.62x |
| 4 tokens, 256 experts, top-8 | 0.02480 ms | 0.01518 ms | 1.63x |

The focused ROCm test set passed 11 tests, including routing parity cases.

## End-to-end quality and decode result

The Qwen API harness sends greedy sampling and `reasoning_effort=none`. This
is required because otherwise Qwen can stream its reasoning trace until the
output cap before returning a final answer. The exact-answer canary returned
`LAN223` on warmup and scored requests after the routing change.

The sustained-decode workload is a 1,212-prompt-token, repeated scheduler
paragraph with 251 forced generated tokens and three scored samples. It is a
warm cache workload, so its input TPS is not an uncached-prefill claim.

| Configuration | Quality | Mean output TPS | Sample output TPS | First-text TTFT |
| --- | --- | ---: | --- | ---: |
| Reference PyTorch router, 8,990 slots | Passed previously | 26.731 | 26.724, 26.728, 26.740 | not measured correctly by earlier harness |
| HIP Triton router, 8,990 slots, no graph | Passed | **29.186** | 29.201, 29.183, 29.175 | 400 to 405 ms |
| HIP Triton router, 10,006 slots, no graph | Passed | 29.080 | 29.077, 29.078, 29.086 | 359 ms canary |
| HIP Triton router, 8,990 slots, graph batch 1 | Passed | 28.830 | 28.827, 28.823, 28.839 | 398 to 404 ms |

The accepted configuration is 8,990 slots with graph capture disabled. It is
the only row above that combines the best sustained throughput with a passing
exact-answer quality canary.

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
```

The current best configuration is reloading under:

```text
/home/david/freetoken-amd/artifacts/qwen-reboot-recovery-20260829T092725Z/
```

## Remaining gap

The accepted 29.186 client decode TPS is quality-validated but remains below
the FreeToken paper's 39.3 TPS report. The paper's exact prompt sequence,
stop policy, warm-cache state, and source revision remain unrecovered, so this
is not a strict paper-parity comparison. Further work should profile per-layer
NVFP4 expert execution and the Qwen linear-attention path under native HIP,
then repeat this same quality and workload protocol.
