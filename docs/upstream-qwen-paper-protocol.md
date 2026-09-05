# Upstream Qwen 8 GB benchmark protocol evidence

## Confirmed source facts

The FreeToken paper states that its main experiments use Qwen3.6-35B-A3B,
DeepSeek-V4-Flash, and GLM-5.2 on six machines spanning an 8 GB RTX 4060 laptop
through an RTX PRO 6000 workstation. Its RTX 4060 laptop row is a Core
i9-13900H with 20 threads, 32 GiB LPDDR5, 8 GiB RTX 4060 Laptop VRAM, PCIe 4.0
x8, measured 11.8 GB/s expert-transfer bandwidth, and measured 47.5 GB/s
CPU-side MoE bandwidth. The paper states that this laptop serves a 35B model at
39.3 tokens per second.

The 8 GB laptop uses Qwen3.6-35B-A3B's official NVFP4 release. The other Qwen
cross-engine comparisons use BF16 for exact weight-format parity. The paper's
metrics are per-request mean decode throughput and per-request mean TTFT. Its
four workloads are AIME math reasoning, an OpenCode SWE-bench coding agent,
the same issue via Claude Code with concurrent subagents, and a 13-turn
OpenClaw email/calendar agent. The broader comparison includes llama.cpp,
Ollama, KTransformers, and MoE-Infinity.

The source repository identifies `Qwen/Qwen3.6-35B-A3B` and
`nvidia/Qwen3.6-35B-A3B-NVFP4` as known-good Qwen MoE checkpoints. Its backend
documentation defines `offload`, `cpu`, `hybrid`, and `auto`, where the latter
selects offload for MoE and may select hybrid following `ft bench bw`.

Primary sources:

- <https://arxiv.org/abs/2608.16157>
- <https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md>

## Fields the paper summary does not establish

The published HTML establishes the hardware, model format, metric type, and
workload classes. It does not identify the following fields for the 39.3 TPS
row. They must be resolved from released artifacts, the authors, or marked
unavailable before calling the GMKtec EVO-X2 result a strict replication:

| Field | State | Required action |
| --- | --- | --- |
| Checkpoint revision and exact quantization | Unknown | Inspect paper appendix, released benchmark assets, and upstream history. |
| Laptop CPU and RAM | Resolved | Core i9-13900H, 20 threads, 32 GiB LPDDR5. Record OS, driver, CUDA, and FreeToken commit if recovered. |
| Prompt corpus and token count | Workload class resolved | Locate the exact AIME questions, SWE issue, tool harness versions, and rendered token counts. |
| Output length and stop policy | Unknown | Locate benchmark runner defaults and raw results. |
| Warmup procedure and cache state | Partially resolved | Paper says the first request warms the cache normally. Recover the scored-run sequence. |
| TPS definition and reported statistic | Resolved at paper level | Per-request mean decode TPS and per-request mean TTFT. Retain the client-side formula and raw timestamps. |
| Expert cache, KV allocation, CPU thread count, and selected backend | Unknown | Recover the launch configuration or state that parity is approximate. |

## Current GMKtec EVO-X2 comparison status

Existing evidence proves native HIP functional serving for
`nvidia/Qwen3.6-35B-A3B-NVFP4` and a prior controlled warm output rate around
28.9 client TPS. It does not prove paper parity because the model revision,
workload, and policy contract above are incomplete. The new harness records
those differences rather than hiding them.
