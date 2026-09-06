# Unified ROCm optimization references

## Local implementation

- `python/freetoken/utils/arch.py` — backend/target capability records and
  exact candidate matrix.
- `python/freetoken/kernel/backend.py`, `python/freetoken/kernel/utils.py`,
  `python/freetoken/kernel/_toolchain.py` — runtime/build backend and toolchain
  selectors that must share normalized ROCm identity.
- `python/freetoken/kernel/gguf.py` — generic GGUF dispatch, candidate ABI,
  runtime metadata, and target-specific fallback policy.
- `python/freetoken/layers/gguf.py` — packed dense GGUF dispatch.
- `python/freetoken/moe/fused_gguf.py` — gate/up/down operation metadata and
  offload/resident integration.
- `python/freetoken/utils/graph_gate.py` — ROCm graph/BLAS subprocess probes,
  cache identity, and worker environment resolution.
- `python/freetoken/server/launch.py` — pre-worker environment and target
  guard.
- `python/freetoken/server/args.py`, `python/freetoken/engine/engine.py` —
  compatible implementation selectors and worker propagation.
- `python/freetoken/moe/offload_cache.py` — cache slot, hit/miss, copy, and
  stream/event contracts.
- `benchmarks/bench_decode_moe.py` — streamed serving timing.
- `benchmarks/bench_decode_replay.py`, `benchmarks/check_decode_gate.py`,
  `benchmarks/bench_rocm_matrix.py` — evidence identity and promotion gates.

## Prior local plans to reuse

- `../FreeToken/.plans/rocm-parity-next/plan.md` (historical sibling checkout;
  context only) — causal timeline, generic default,
  graph worker policy, and operation-path hypotheses.
- `../FreeToken/.plans/rocm-perf-parity/plan.md` (historical sibling checkout;
  context only) — stage profile, router/graph/MMVQ
  priorities, and CUDA integrity requirements.
- `../FreeToken/.plans/qwen-moe-speed/plan.md` (historical sibling checkout;
  context only) — base decode scope, identity-complete
  baseline, known-negative tuning branches, and 75/80 tok/s target context.
- `../FreeToken/.plans/rocm-ollama-gap/plan.md` (historical sibling checkout;
  context only) — replay schema, oracle, resident phases,
  q8 identity warnings, and incomplete evidence rules.
- `../FreeToken/.plans/amd-gpu-support-2/plan.md` (historical sibling checkout;
  context only) — plain HIP shared-kernel strategy,
  stream-ordered batch copy, and compile-only versus served status.

## External source families

Use only after recording exact repository and commit in implementation notes:

- llama.cpp GGUF MMVQ/MMQ and HIP vendor policy for matching layouts.
- PyTorch ROCm BLAS preference/reporting APIs.
- ROCm Triton and rocprofiler documentation for portable router/profiling
  behavior.
- Composable Kernel fused-MoE examples as research references only; no direct
  integration without matching layout and independent evidence.

## Pinning and provenance rules

- Record upstream repository, commit SHA, file path, license/header obligations,
  and local adaptation before copying code.
- Distinguish GGUF file SHA from Ollama manifest digest.
- Record source hash, compiler, target, ABI, and cache namespace for every
  candidate artifact.
