# GMKtec EVO-X2 FreeToken AMD campaign completion audit

## Purpose and scope

This audit is the controlling completion record for the native ROCm and HIP
FreeToken port evaluated on the authorized GMKtec EVO-X2. It separates what
has been proven on that system from paper-inspired evidence, from comparisons
that require an external NVIDIA reference system or unreleased author inputs.
It must be updated from immutable artifacts, not from a plan or an intended
command.

The campaign may claim only GMKtec EVO-X2 results. A second host is outside
the authorized scope, so it cannot be silently substituted for a missing
result or used to claim broader AMD support.

## Completion rules

A row is **proven** only when its named artifact or tracked source demonstrates
the stated condition. A row is **in progress** when a live controller is
collecting the required evidence. A row is **external evidence unavailable**
when the required source, fixture, or hardware is not available to this
campaign. The latter is a documented limitation, never a passing result.

The campaign is not complete while any in-scope proven or in-progress row
lacks its required evidence. The final audit must retain comparison limits
instead of converting a different model format, workload, hardware tier, or
metric boundary into an equal comparison.

## Requirement matrix

| Requirement | Required proof | Current status | Authoritative evidence or next action |
| --- | --- | --- | --- |
| Native ROCm and HIP execution | Native extension build, HIP runtime evidence, and no substitute backend | Proven | [`amd-rocm-gfx1151.md`](amd-rocm-gfx1151.md) and recorded Qwen and Gemma artifacts |
| OpenAI-compatible local API | Model listing plus completed streaming and non-streaming requests | Proven | Qwen and Gemma controls in [`gmktec-evo-x2-amd-run-log.md`](gmktec-evo-x2-amd-run-log.md) |
| Qwen deterministic visible-output quality | Versioned exact canary, arithmetic, JSON, and AIME records with raw responses | Proven for the controlled suite | C139 records canonical AIME SHA1 `3302eda43396`; the run log records the suite boundaries |
| Gemma 4 quality | Text, multimodal fixtures, and bounded visual description with raw outputs | Proven for the controlled suite | Gemma entries in [`gmktec-evo-x2-amd-run-log.md`](gmktec-evo-x2-amd-run-log.md) |
| Gemma 4 performance and stability | Single-request, concurrent, long-context, and bounded endurance controls with raw artifacts | Proven for bounded controls | [`gmktec-evo-x2-gemma4-comparison-report.md`](gmktec-evo-x2-gemma4-comparison-report.md) records FreeToken and ROCm 10 llama.cpp results. A full 1,440-session Gemma campaign is optional publication evidence, not a missing functional gate. |
| Gemma 4 performance comparison | Same model and fixed request contract for single, concurrent, and long-context controls | Proven for bounded controls | [`gmktec-evo-x2-gemma4-comparison-report.md`](gmktec-evo-x2-gemma4-comparison-report.md) records FreeToken and ROCm 10 llama.cpp matrices, including reasoning-off llama.cpp long-context results |
| Qwen Q4_K_M same-format comparison | Same checkpoint, tokenizer, caller-rendered prompt, completion cap, and five warmed requests on both runtimes | Proven for decode parity; TTFT remains a separate boundary | [`gmktec-evo-x2-cross-model-manifest-20260905.json`](gmktec-evo-x2-cross-model-manifest-20260905.json) records five-request artifacts. FreeToken steady decode was 49.4357 TPS versus 49.1772 TPS for llama.cpp, while all-sample means were 48.6028 and 49.1155 TPS respectively. |
| Q5-only four-row optimization correctness | Real-weight component parity and complete API quality gate | Proven | C138 exact component hash and C139 API quality evidence |
| Q5-only four-row performance value | Same configuration baseline comparison, scheduler, C4, and tail metrics | Proven for the stated local Qwen workload | C139 records higher C4 prefill and decode TPS plus lower C4 tails; it separately retains the slight single-request decode reduction |
| ROCm llama.cpp local control | Same host, recorded model format, API shape, quality suite, and timing matrix | Proven as a practical Q4 control; five-sample refresh recorded | C89 remains the earlier four-slot workload control. The 2026-09-04 five-sample refresh is preserved at `/home/david/freetoken-amd/artifacts/qwen35b-llamacpp-rocm10-timeshare-five-20260904T101357Z/`: five of five samples passed, mean decode 46.6625 TPS, median 46.7524 TPS, mean prefill 19,343.40 TPS. Protected-service recovery completed and the paired FreeToken control is preserved at `/home/david/freetoken-amd/artifacts/qwen35b-freetoken-five-20260904T102530Z/`. It is not a same-format NVFP4 equivalence claim. |
| Paper-inspired W1 control | Pinned AIME source, complete local request contract, five samples, raw responses, and quality result | Proven as paper-inspired control | Five raw samples and aggregate evidence are preserved at `/home/david/freetoken-amd/artifacts/w1-paper-inspired-five-sample-20260904T094252`. All five matched output SHA1 `0acef4eab6f4`; the run log records token counts and timing. This remains a reproducible W1-style control, not strict paper replication, because the paper's original prompt, cache policy, and exact runner contract remain unpublished. |
| W2 through W4 strict replication | Authors' exact harnesses, fixtures, versions, policy, and scoring | External evidence unavailable | Public source audit documents that OpenCode SWE-bench, Claude Code, OpenClaw, and raw paper artifacts are not released |
| 24-hour Q5 endurance | All 1,440 minute-cadence sessions, zero candidate and host swap, final summary, restored swap, and real normal-service completion | Proven | C142 artifact `/home/david/freetoken-amd/artifacts/q4-c142-q5-swapdrain-endurance-20260902T222206Z` contains exactly 1,440 valid session JSON files, zero failures, zero candidate and host swap, completed controller evidence, and preserved recovery artifacts. Per-session records measure state correctness, TTFT, token-gap tails, swap, and thermal telemetry. They intentionally do not claim per-session prefill TPS. |
| Normal service recovery | Recovered protected Qwen API produces a real completed response with `finish_reason: stop` | Proven | Read-only probe artifact `/home/david/freetoken-amd/artifacts/qwen-protected-recovery-explicit-20260904T093948` records model `qwen3.6-35b-a3b-nvfp4-amd`, visible response `READY.`, and `finish_reason: stop` after the C142 recovery. |
| 284B capacity claim | Model manifest, reserved-memory evidence, load and quality result on comparable resources | Incomplete, metadata gate rejects full load | [`gmktec-evo-x2-paper-model-capacity-gate.md`](gmktec-evo-x2-paper-model-capacity-gate.md) pins the official release revision and records a reproducible metadata-only `REJECT_FULL_LOAD` result: 155.425 GiB payload versus a 4 GiB authoritative budget after explicit headroom. The new real-shape slice measures transfer only; full-model quality and serving throughput remain unmeasured. |
| Strict NVIDIA paper comparison | Same model, precision, workload, policy, metric boundary, and NVIDIA reference hardware | External evidence unavailable | The paper protocol still lacks exact released inputs and no reference NVIDIA system is in scope |
| Upstream-ready documentation | Reproducible, secret-safe tracked source and current evidence links | Proven for current evidence set | C142, Gemma comparison, Qwen same-format warmed matrix, machine-readable manifest, long-context boundaries, recovery proof, W1 result, and the capacity baseline are tracked. Strict NVIDIA parity and 284B qualification remain explicitly unresolved. |

## Required terminal sequence for C142

1. Confirm exactly 1,440 session JSON records and a passing `summary.json`.
2. Inspect the full latency and swap summaries without discarding the cold
   first-session result.
3. Confirm candidate process-group and whole-host swap remain zero throughout.
4. Confirm the controller restores configured swap before normal recovery.
5. Verify normal Qwen with a real OpenAI-compatible completion ending in
   `finish_reason: stop`.
6. Add a C142 evidence entry, commit only tracked campaign documentation, and
   re-run the documentation and benchmark-tool regression checks.
7. Run the pinned paper-inspired W1 control after normal-service recovery, then
   update this matrix with the observed five-sample evidence and its remaining
   strict-paper limitations.

## Performance metric boundaries

C139 is the qualified complete API performance gate. It records client-observed
prefill TPS, decode TPS, warm TTFT, C4 aggregate prefill and decode TPS, and
C4 tail latency under the exact Q5-only four-row candidate. C142 has a distinct
purpose: it establishes long-duration state, swap, thermal, and tail stability
under minute cadence. Its three-turn state suite has no controlled fixed-size
input throughput interval, so it must not be presented as a prefill-TPS
measurement. The final report must show the C139 TPS results and C142 endurance
results together, with their different measurement boundaries stated plainly.

## Final reporting rule

The final report must state separately: native AMD functionality, controlled
quality, local Q4 control comparisons, paper-inspired controls, strict-paper
limitations, and external hardware limitations. It may not state that a
GMKtec EVO-X2 result equals or exceeds a published NVIDIA result unless every
condition in the strict NVIDIA comparison row is proven.
