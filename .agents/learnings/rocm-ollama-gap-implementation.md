# ROCm/Ollama Gap Implementation

## Context

2026-09-01 implementation of `.plans/rocm-ollama-gap` across KV storage, GGUF dispatch,
profiling, fused MoE, and promotion harnesses.

## Hardest decision

Keep q8 KV and gfx1100 GGUF paths explicit and fail-closed: requested CLI settings never count as
observed execution, and candidate code never silently falls back in forced mode.

## Alternatives rejected

- Running Ollama/ROCm benchmarks during implementation — user reserved GPU validation for manual review.
- Reusing model dtype for q8 accounting — allocation, packed scales, attention views, and metadata must share one descriptor.
- Combining sampled and teacher-forced results — Gate A absolute throughput and Gate B q8 replay measure different claims.

## Least confident

HIP/CUDA compilation and real-model replay remain unverified. The b10434 bridge currently enforces
512-column activation alignment and needs manual ABI, numerical, graph-capture, and performance proof.

## Reuse

Read before running `.plans/rocm-ollama-gap` evidence: start with `fixture-manifest.json`, enable
dispatch tracing for per-op records, run q8 with `--kv-type q8_0`, and report Gate A/B separately.

## Validation closure — 2026-09-01

### Hardest decision

Keep legacy GGUF MMVQ as default after exact 512-token q8 greedy output parity but no speed gain:
candidate `69.040` versus legacy `69.194 tok/s`; fused gate/up also produced degenerate output.

### Alternatives rejected

- Promoting gfx1100 MMVQ or fused gate/up — candidate was slower, and fused model output repeated
  `We.` despite direct-kernel tolerance.
- Calling b10434 `predicted_ms` decode timing — forced-token server requests bill their forward pass
  as one-token `prompt_eval_ms`, so replay accounting now uses that field.

### Least confident

Cross-runtime route parity remains unproven because b10434 server has no route instrumentation;
short rocprof q8 graph startup also hit duplicate physical Q8 destinations before request.

### Reuse

Read `.plans/rocm-ollama-gap/notes-results.md` before any promotion or kernel follow-up; preserve
legacy default, q8 opt-in, route-parity requirement, and the warm-offload measured bound.
