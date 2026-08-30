# LAN-223 AMD FreeToken execution log

This file is append-only. Each entry records UTC time, branch and commit, test
category, command or script, artifact location, quality result, outcome, and
restoration result. Do not replace a failed entry with a later passing entry.

## Baseline record

| UTC date | Evidence | Category | Outcome |
| --- | --- | --- | --- |
| 2026-08-28 | `lan223-rocm-validation-2026-08-28.md` | Native AMD functionality | Qwen NVFP4 and Gemma Q4 served through native ROCm/HIP API paths |
| 2026-08-29 | `lan223-qwen-router-optimization-2026-08-29.md` | Local control and optimization | Rejected quality-changing router candidates; retained a safe configuration |
| 2026-08-30 | `lan223-qwen-q4-raw-control-20260830.md` | Local control | FreeToken Q4 50.63 TPS versus ROCm llama.cpp 50.29 TPS on fixed raw prompt |
| 2026-08-30 | `lan223-gemma4-q4-vision-control-20260830.md` | Native AMD and local control | Text and visible-image controls passed |
| 2026-08-30 | `/home/david/freetoken-amd/artifacts/qwen-nvfp4-tail-baseline-20260830T081500Z/` | LAN-223 warm NVFP4 baseline | Five fixed-length samples passed: 28.76 mean TPS, 363 ms mean TTFT, 37.93 ms p99 gap, 526.95 ms maximum gap |
| 2026-08-30 | `/home/david/freetoken-amd/artifacts/qwen-aime-quality-20260830T082000Z/quality.json` | Qwen deterministic quality | Expected AIME output hash passed: 28.34 TPS, 410 ms TTFT, 37.62 ms p99 gap |
| 2026-08-30 | `/home/david/freetoken-amd/artifacts/qwen-quality-suite-20260830T083000Z/quality-suite.json` | Qwen versioned quality suite | Three visible-output checks passed: exact canary, arithmetic, and JSON fields |
| 2026-08-30 | LAN-223 read-only memory snapshot | Capacity and measurement readiness | Host reports 64 GB total RAM and about 1.4 GB swap in use, mainly Qwen workers; timed acceptance is paused pending clean memory recovery |
| 2026-08-30 | `/home/david/freetoken-amd/artifacts/qwen-reboot-recovery-20260830T081547Z/` | Controlled Qwen recovery | Verified server restart completed only after health returned `status: ok`; cold serial expert loading took about 6 minutes 22 seconds |
| 2026-08-30 | LAN-223 swap-residency reset | Measurement remediation | Temporarily disabled and re-enabled configured swap after verifying 20 GB available RAM and 2.1 GB swapped; swap use returned to zero and Qwen stayed healthy |
| 2026-08-30 | `/home/david/freetoken-amd/artifacts/runtime-manifest-20260830T082300Z/` | Runtime provenance | Captured clean host, ROCm, GPU policy, source, memory, storage, and process state before accepted baseline |
| 2026-08-30 | `/home/david/freetoken-amd/artifacts/qwen-nvfp4-clean-baseline-20260830T082400Z/` | LAN-223 warm NVFP4 baseline | Five samples passed with zero swap: 28.69 mean TPS, 367 ms mean TTFT, 37.89 ms p99 gap, 39.08 ms maximum gap |
| 2026-08-30 | `/home/david/freetoken-amd/artifacts/qwen-nvfp4-clean-scheduler-20260830T082500Z/` | LAN-223 medium scheduler baseline | Three samples passed with zero swap: 27.89 mean TPS, 429 ms mean TTFT, 38.99 ms p99 gap, 71.23 ms maximum gap |
| 2026-08-30 | `/home/david/freetoken-amd/artifacts/qwen-multiturn-state-20260830T083100Z/multiturn.json` | Bounded multi-turn state control | Three dependent turns passed with zero swap: 411 ms mean TTFT, 440 ms worst TTFT, 38.49 ms worst token gap |
| 2026-08-30 | `/home/david/freetoken-amd/artifacts/qwen-long-context-2k-clean-20260830T083657Z/long-context.json` | LAN-223 1.8K-context retrieval control | Five of five exact marker retrievals passed at 1,845 reported prompt tokens with zero swap: 428 ms mean TTFT, 431 ms p99 TTFT, and 40.48 ms p99 token gap. This is a LAN-223 control, not a replication of the paper's 56K to 65K agent sessions. |
| 2026-08-30 | `/home/david/freetoken-amd/artifacts/qwen-long-context-7k-calibration-20260830T083721Z/long-context.json` | Long-context limit discovery | Preserved expected failure: 6,845-token prompt was rejected because the live auto-cache geometry exposed only 2,068 prompt-plus-generation tokens despite `--max-seq-len-override 8192`. The server stayed healthy and swap-free. |
| 2026-08-30 | `/home/david/freetoken-amd/artifacts/qwen-kv-8192-rebuild-20260830T083845Z/` | Reversible cache repair | Idle-only runtime rebuild succeeded: reduced the MoE cache from 8,974 to 8,700 slots and expanded KV pages from 2,068 to 8,192. Cache-budget arithmetic retained about 361 MB more dynamic-cache headroom than the original geometry; server remained healthy. |
| 2026-08-30 | `/home/david/freetoken-amd/artifacts/qwen-long-context-7k-kv8192-rerun-20260830T084010Z/long-context.json` | 6.8K identical-prefix control | Five exact marker retrievals passed at 6,845 reported prompt tokens. The first request had 32.98 s TTFT while repeated identical-prefix requests were about 433 ms, demonstrating prefix-cache reuse. A brief 2.04 MB swap residency was remediated to zero before the next acceptance run. |
| 2026-08-30 | `/home/david/freetoken-amd/artifacts/qwen-long-context-7k-cold-kv8192-20260830T084300Z/long-context.json` | 6.8K forced-cold-prefill control | Five of five exact marker retrievals passed at 6,856 reported prompt tokens with a unique early nonce per sample, preventing long-prefix reuse: 13.506 s mean TTFT, 13.520 s p99 TTFT, 44.43 ms p99 token gap, zero swap, and 38 C post-run GPU temperature. |
| 2026-08-30 | `/home/david/freetoken-amd/artifacts/qwen-kv8192-short-decode-20260830T084448Z/summary.json` | Expanded-KV short decode control | Five 128-token throughput samples passed with zero swap: 28.85 mean TPS, 28.87 median TPS, and 0.071 TPS standard deviation. This is within measurement noise of the earlier 28.69 TPS clean baseline, so the 8K KV profile did not show a short-decode regression. |

## Open work

| ID | Required evidence | State |
| --- | --- | --- |
| P0 | Complete paper protocol fields or explicit unresolved record | In progress |
| P1 | Harness manifest and tail-summary validation | Completed: tail summaries and clean runtime manifest validated |
| P2 | Five-sample Qwen NVFP4 warm and cold baseline | Warm short and medium baselines complete; long-context cache-hit and forced-cold-prefill controls complete; full service-restart request timing remains |
| P3 | Versioned Qwen and Gemma quality suite | Qwen three-case suite completed; Gemma expansion remains |
| P4 | Paper-inspired W1 to W4 agent workloads | Bounded state-retention control completed; full tool-using workloads remain |
| P5 | Tail-latency matrix and 24-hour endurance | Long-context p99 tail controls started; concurrent matrix and endurance remain |
| P6 | 284B capacity manifest | Blocked pending clean-memory assessment; current host has 64 GB RAM, not the paper desktop's 192 GiB system RAM plus 32 GB VRAM |
| P7 | Strict NVIDIA reference run | Blocked on reference hardware and missing paper fields |
