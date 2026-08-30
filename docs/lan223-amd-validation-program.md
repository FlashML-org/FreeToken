# LAN-223 AMD FreeToken validation program

## Purpose

This program establishes what the `amd-rocm-gfx1151` branch proves on LAN-223.
It separates native AMD functionality, LAN-223 performance, local ROCm control
comparisons, and strict replication of FreeToken's NVIDIA paper. A result may
only be labelled with the category its evidence supports.

## Scope and safety contract

- Every executable workload refuses hosts other than LAN-223.
- Candidate servers bind to loopback-only ports and never change llama-swap.
- Every temporary candidate run restores Qwen and waits for `/health` to report
  `status: ok` before success.
- Every artifact directory is immutable. A duplicate run identifier is a
  failure, not permission to overwrite old evidence.
- Timed runs reuse the native HIP extension cache. JIT compilation, swapping,
  thermal throttling, unexpected disk traffic, or failed quality invalidates a
  scored sample.
- The branch is validated only on `gfx1151`; it is not a general AMD claim.

## Evidence categories

| Category | Meaning | Current example |
| --- | --- | --- |
| Native AMD functionality | HIP, ROCm, API, and recovery work correctly | Qwen and Gemma serving on LAN-223 |
| Local control | Same local workload against an AMD control engine | Qwen Q4 FreeToken versus ROCm llama.cpp |
| Paper-inspired | Workload follows paper category but lacks exact paper fields | Future LAN-223 agent suite |
| Strict paper replication | Model, precision, prompts, warmup, policy, metrics, and scoring all match | Not yet available |

## Metric definitions

| Metric | Definition |
| --- | --- |
| Warm TTFT | Client monotonic time from request write to first content-bearing SSE event after warmup |
| Decode TPS | `(generated_tokens - 1) / (last_content_event - first_content_event)`; one-token outputs have no TPS |
| Output token gap | Adjacent client-observed content-bearing SSE timestamp difference |
| p50, p95, p99 token gap | Nearest-rank percentile of raw output-token gaps |
| Tail TTFT | Maximum whole-request TTFT across a named completed workload matrix |
| Quality result | Fixed expected answer, schema, executable test, or visible-output rule recorded with raw response |

## Acceptance sequence

1. Reproducibility and protocol ledger.
2. Native HIP, API, cache-reuse, and recovery regression.
3. Fixed Qwen and Gemma quality suite.
4. Five-sample cold and warm LAN-223 baseline matrix.
5. Paper-inspired agent workloads and tail analysis.
6. Twenty-four-hour endurance and recovery qualification.
7. Larger-model capacity assessment only after Qwen gates pass.
8. Strict NVIDIA comparison only with a reference system and complete paper protocol.

## Prohibited claims

- A Q4 GGUF control is not a replication of NVFP4 or BF16 paper tests.
- A short warm request is not the paper's worst agent-turn TTFT.
- Hidden reasoning text is not a visible OpenAI-compatible answer.
- A larger model does not fit until a complete memory-reserve manifest proves it.

