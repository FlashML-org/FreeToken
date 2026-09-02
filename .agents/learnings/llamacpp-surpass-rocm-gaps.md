# llama.cpp Surpass ROCm Gaps

## Context

2026-08-31: Qwen3.6-35B-A3B GGUF decode work on RX 7900 XTX/gfx1100 added
native Q4_K/Q5_K/Q6_K paths, residency planning, execution gates, and traces.

## Hardest decision

Keep native mixed-K support fail-closed and retain offload defaults: the full
resident budget does not fit this 24-GiB card, while native Q5/Q6 offload only
reached 58.98 tok/s API / 61.19 tok/s scheduler.

## Alternatives rejected

- Disable safety reserve to force `fused` residency — risks late OOM and invalid
  benchmark evidence.
- Enable grouped GGUF prefill by default — real gfx1100 execution failed during
  launch; synthetic ABI success was insufficient.
- Promote gfx1100 rotated-wave or Q5/Q6 path from microbench alone — candidate
  and end-to-end evidence did not beat the incumbent or llama.cpp ROCm.
- Treat Torch profiler CPU ranges as additive wall time — synchronization and
  overlap inflate attribution; no scheduler edit was accepted.

## Least confident

Exact remaining attribution is unresolved because rocprofv3 attach was blocked
by host ptrace policy. Native executor cadence, full resident capacity with a
different context/budget, and a genuinely fused/grouped GGUF kernel remain open.

## Reuse

Read before further Qwen ROCm speed work. Preserve model/fixture hashes,
resident execution evidence, zero fetch/remap gates, graph state, exact output
count, and paired same-file llama.cpp measurements.
