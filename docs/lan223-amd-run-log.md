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

## Open work

| ID | Required evidence | State |
| --- | --- | --- |
| P0 | Complete paper protocol fields or explicit unresolved record | In progress |
| P1 | Harness manifest and tail-summary validation | Tail summary implemented and validated; provenance expansion remains |
| P2 | Five-sample Qwen NVFP4 warm and cold baseline | Warm fixed-length baseline completed; cold baseline remains |
| P3 | Versioned Qwen and Gemma quality suite | Qwen three-case suite completed; Gemma expansion remains |
| P4 | Paper-inspired W1 to W4 agent workloads | Not started |
| P5 | Tail-latency matrix and 24-hour endurance | Not started |
| P6 | 284B capacity manifest | Blocked pending clean-memory assessment; current host has 64 GB RAM, not the paper desktop's 192 GiB system RAM plus 32 GB VRAM |
| P7 | Strict NVIDIA reference run | Blocked on reference hardware and missing paper fields |
