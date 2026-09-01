# GMKtec EVO-X2 Q4 hardening execution plan

## Objective

Close the remaining reliability, performance, readiness, endurance, and
publication gaps in the native ROCm/HIP Qwen Q4 path without disrupting the
protected GMKtec EVO-X2 NVFP4 loopback service except during a recorded, reversible
time-share window.

## Non-negotiable controls

1. All candidate servers bind only to `127.0.0.1:1922`; the normal service
   remains `127.0.0.1:1919` and is not added to llama-swap.
2. Before a time-share handoff, verify the port owner, model path, command,
   and complete process group. Stop only that verified group.
3. Every candidate has a new dated artifact directory, fixed model file,
   tokenizer, request suite, runtime versions, and raw log retention.
4. A candidate is accepted only if the API, deterministic quality, long
   context, concurrency, process-scoped swap, and recovery checks pass.
5. Rejected candidates remain documented with their artifacts and are never
   silently promoted to the normal service.

## Work items and acceptance gates

| Item | Execution | Acceptance gate | Rollback or rejection rule |
| --- | --- | --- | --- |
| 1. Correct report terminology | State the stable `0.25` and experimental `0.35` comparisons separately. | The report names 1.78 percent as the stable gap and 1.39 percent as an unstable historical result. | Do not publish a parity claim. |
| 2. Profile before optimizing | Use the wheel-compatible ROCm profiler only on an isolated Q4 workload, then rank kernels by measured end-to-end relevance. | Trace, source revision, command, and kernel aggregate are retained. | Reject profiler-only throughput claims. |
| 3. Repair lifecycle and SVM exposure | Launch recovery and Q4 candidates in dedicated sessions, verify the whole process group on stop, and test forced cancellation only in the candidate window. | No orphan listener or child remains after stop; `/health` reports loading, serving, or failure honestly. | Keep `0.25` as the recommended profile if `0.35` again triggers the SVM resident-memory fault. |
| 4. Make cold readiness explicit | Treat `/health` `status: ok` and `maintenance: serving` as readiness, not the presence of `/v1/models`. | Cold launch emits loading while unavailable and serving only after the engine is ready. | Never score a request that received a loading 503. |
| 5. Requalify performance and quality | Run same-file Q4 FreeToken and llama.cpp controls with fixed prompt, tokens, warmup, and quality suite. | All quality rows pass and FreeToken median TPS is at least the accepted baseline; a parity claim requires a new matched result. | Revert code and preserve evidence if quality, tail latency, or runner swap regresses. |
| 6. Extended endurance | Run a process-scoped 24-hour, 1,440-session three-turn Q4 battery after the Q4 server is qualified. | 1,440 of 1,440 sessions pass, every verified FreeToken process has zero `VmSwap`, and normal service recovers after cleanup. | Stop immediately on a wrong answer, runner swap, process death, or health failure. |
| 7. Sanitized publication package | Capture a redacted host manifest, checksums, source state, benchmark code, and selected public artifacts. | Manifest passes HIP and target checks, checksum verification, and scans free of private host names, addresses, and home paths. | Do not publish any artifact that fails the privacy scan. |

## Current execution state

- Item 1 is complete in commit `cc3a531`.
- Item 2 has a retained native ROCm baseline that identifies dense Q4 GEMV and
  quantization as the measured GPU work to investigate. It is not reported as
  serving throughput.
- Items 3 and 4 are implemented in commit `07b99da`: recovery now launches in
  a dedicated session and exposes lifecycle state through `/health`.
- Item 7 is implemented and live-validated in commits `adc76be`, `5e8448f`,
  and `e592130`. The validated manifest has a redacted host field, native HIP
  metadata, verified checksums, and no detected private host name, address, or
  home path.
- Items 5 and 6 begin only after the current hardened recovery launch reaches
  `/health` serving state. The normal service is restored by the same
  verified-session mechanism after every candidate window.

## Decision rule

The stable Q4 profile is already functionally qualified. The purpose of the
remaining work is to either produce a measured, quality-preserving improvement
or document why the current 1.78 percent llama.cpp gap and high-cache SVM
limit remain. A failed experiment is still a completed investigation when its
cause, raw evidence, and rollback are retained.
