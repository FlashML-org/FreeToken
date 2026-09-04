# FreeToken paper benchmark specification and AMD coverage

This document transcribes the benchmark scope stated in the supplied FreeToken
paper and maps each requirement to the evidence currently available for the
GMKtec EVO-X2 Strix Halo port. It is a planning and evidence index. It does
not treat a paper-inspired workload as an exact reproduction unless the model,
fixture, protocol, and measurement definition are all known to match.

## Models named by the paper

| Model | Paper precision and role | AMD status |
|---|---|---|
| Qwen3.6-35B-A3B | BF16; primary model across the four workloads | Bounded Qwen validation and ROCm 10 llama.cpp comparison complete. Exact paper harness parity is not established. |
| DeepSeek-V4-Flash | 284B parameters, 13B active; native MXFP4 routed experts; large-model demonstration | Not reproduced. The exact checkpoint is not present in the current capacity inventory. |
| GLM-5.2 | 753B parameters, approximately 40B active; NVFP4 routed experts; workstation-tier demonstration | Not reproduced. The required 433 GB checkpoint and workstation-class memory are outside the current test inventory. |

The paper also states that FreeToken supports more than 20 MoE models, but the
evaluation section identifies the three models above as the representative
benchmarks. A support claim is not equivalent to a completed benchmark result.

## Paper hardware tiers

The paper reports six discrete-GPU systems:

| System | GPU and VRAM | PCIe | Measured host-to-device bandwidth |
|---|---|---|---:|
| 5090 | RTX 5090, 32 GB | PCIe 5.0 x16 | 52.7 GB/s |
| 4090 | RTX 4090, 24 GB | PCIe 4.0 x16 | 25.1 GB/s |
| 3090 | RTX 3090, 24 GB | PCIe 4.0 x16 | 25.3 GB/s |
| 5090 desktop | RTX 5090, 32 GB | PCIe 5.0 x16 | 49.0 GB/s |
| 4060 laptop | RTX 4060 Laptop, 8 GB | PCIe 4.0 x8 | 11.8 GB/s |
| PRO 6000 | RTX PRO 6000 Blackwell, 96 GB | PCIe 5.0 x16 | 51.5 GB/s |

The GMKtec EVO-X2 is not one of these systems. It uses an integrated Radeon
8060S Strix Halo GPU with unified memory rather than a discrete NVIDIA card
with a separately reported VRAM pool. Its results therefore need a separate
AMD platform label and must not be presented as a direct replication of an
RTX 4060, RTX 5090, or RTX PRO 6000 result.

## Paper workloads

The evaluation defines four scenarios:

1. **W1 math reasoning:** AIME competition problems, long chain-of-thought,
   no tools, single-turn, decode-dominated.
2. **W2 coding agent:** A SWE-bench repository issue solved through OpenCode,
   with real tool execution over three scripted user turns.
3. **W3 native-protocol coding agent:** The same issue driven through Claude
   Code using the Anthropic-compatible endpoint. The harness starts concurrent
   subagents and grows sessions to approximately 56,000 to 65,000 tokens.
4. **W4 email and calendar agent:** Thirteen fixed user turns over a mailbox
   kit through OpenClaw, with an approximately 24,500-token system-context
   floor. The paper disables OpenClaw's 120-second idle watchdog for measurement.

The paper requires the coding runs to produce the reference gold patch and the
W4 run to complete all thirteen turns. Our existing AIME, tool, long-context,
and state-retention tests are useful bounded controls, but they are not exact
W2, W3, or W4 reproductions because the original repository fixtures and agent
clients are not all available in the current evidence set.

## Paper metrics and reported claims

The primary metrics are per-request mean decode throughput and per-request
mean TTFT. The paper separately discusses tail TTFT because availability
timeouts matter for agents. It reports FreeToken at approximately 77 to 83
decode tokens per second on Qwen3.6 and 22 to 25 decode tokens per second on
DeepSeek-V4-Flash on the RTX 5090 setup.

The paper's prefill analysis also reports an 8,192-token Qwen prefill chunk
completing in approximately 1.19 to 1.22 seconds with pipelined full-layer
loading, and approximately 6,700 tokens per second at 16,000 tokens. These
figures are mechanism-analysis results, not a replacement for the four
agent-workload measurements.

## Current AMD evidence and gaps

| Requirement | Current evidence | Classification |
|---|---|---|
| Native ROCm/HIP execution | FreeToken AMD port runs on Strix Halo | Proven |
| Qwen functional quality | Deterministic Qwen matrices and 1,440-session endurance pass | Proven for tested Qwen workload |
| Qwen single-request speed parity | FreeToken approximately 28 decode TPS versus ROCm 10 llama.cpp approximately 47 TPS in the matched control | Gap remains |
| Qwen aggregate concurrency | One four-request control reached approximately 94.8 aggregate decode TPS for both runtimes | Workload-specific parity, not universal parity |
| Gemma 4 bounded operation | Text, vision, concurrency, long-context, and endurance suites pass | Proven for tested Gemma workload |
| Exact W1 to W4 paper reproduction | Fixtures, protocol, and scoring are not all identical | Incomplete |
| DeepSeek-V4-Flash 284B | No checkpoint or measured run in the current inventory | Incomplete |
| GLM-5.2 753B | No checkpoint or workstation-class capacity in the current inventory | Incomplete |

## Next test gates

1. Obtain or reconstruct the exact W2, W3, and W4 fixtures and acceptance
   criteria before calling those workloads reproduced.
2. Qualify the exact DeepSeek-V4-Flash checkpoint and MXFP4 format only after
   a read-only capacity calculation confirms that the test is safe on the
   available unified-memory system.
3. Keep Qwen kernel optimization separate from paper-replication claims. Every
   candidate must pass deterministic quality, long-context, concurrency, and
   recovery gates before it can be compared on TPS.
4. Report AMD results beside, not as replacements for, the paper's discrete-GPU
   results unless the model, workload, and measurement protocol are identical.

