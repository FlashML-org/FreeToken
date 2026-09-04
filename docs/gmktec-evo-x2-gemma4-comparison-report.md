# Gemma 4 AMD comparison report

This report consolidates the Gemma 4 Q4 GGUF evidence collected on the GMKtec
EVO-X2 Strix Halo system.  FreeToken used the native ROCm/HIP path.  The
comparison control used the ROCm 10 llama.cpp build.  Both runs used the same
14 GB Gemma 4 26B A4B Q4_0 GGUF and the same isolated loopback test procedure.

The report distinguishes measured user-visible API behavior from internal or
protocol-limited observations.  It does not claim strict replication of the
FreeToken NVIDIA paper because the paper's complete prompts, fixtures, cache
policy, and reference hardware are not available.

## Executive result

FreeToken produced the stronger interactive result in the tested concurrent
matrix.  Its four-client aggregate decode was 26.53 tokens/s with 363 ms mean
TTFT.  llama.cpp produced 22.42 aggregate tokens/s with 3.47 s mean TTFT and
6.90 s p95 TTFT.  llama.cpp's isolated per-request decode rate was higher, but
its one-slot configuration serialized concurrent requests.

For single fixed-length requests, FreeToken averaged 50.87 decode tokens/s
versus 30.50 for llama.cpp.  llama.cpp had faster client-observed prefill in
that matrix, 478.13 versus 174.27 tokens/s.  These are runtime controls, not a
claim that the two engines have identical scheduler internals.

## Single-request matrix

| Metric | FreeToken native ROCm/HIP | llama.cpp ROCm 10 |
| --- | ---: | ---: |
| Completed samples | 5/5 | 5/5 |
| Prompt tokens | 34 each | 34 each |
| Completion tokens | 127 each | 127 each |
| Mean TTFT | 203.07 ms | 118.71 ms |
| Mean client prefill | 174.27 tokens/s | 478.13 tokens/s |
| Mean decode | 50.87 tokens/s | 30.50 tokens/s |
| Median decode | 53.21 tokens/s | 22.43 tokens/s |
| p95 decode | 53.67 tokens/s | 46.98 tokens/s |
| Aggregate p99 token gap | 132.36 ms | 283.97 ms |

FreeToken artifact:
`/home/david/freetoken-amd/artifacts/gemma4-gguf-text-20260904T150838Z/`

llama.cpp artifact:
`/home/david/freetoken-amd/artifacts/gemma4-llamacpp-vision-20260904T152038Z/`

## Four-client concurrency matrix

| Metric | FreeToken native ROCm/HIP | llama.cpp ROCm 10 |
| --- | ---: | ---: |
| Requests | 12/12 | 12/12 |
| Mean per-request decode | 28.50 tokens/s | 56.76 tokens/s |
| Aggregate decode | 26.53 tokens/s | 22.42 tokens/s |
| Mean TTFT | 363.13 ms | 3.471 s |
| p95 TTFT | 494.80 ms | 6.901 s |
| Mean per-request p99 gap | 46.65 ms | 18.02 ms |
| Aggregate p99 gap | 56.69 ms | 18.54 ms |

The llama.cpp result has a higher isolated decode rate but a lower aggregate
rate because its tested one-slot configuration serialized concurrent work.
FreeToken admitted the four clients with substantially lower TTFT.

FreeToken artifact:
`/home/david/freetoken-amd/artifacts/gemma4-gguf-text-20260904T152716Z/`

llama.cpp artifact:
`/home/david/freetoken-amd/artifacts/gemma4-llamacpp-vision-20260904T161303Z/`

## Long-context behavior

FreeToken used the OpenAI chat protocol and passed the exact `LONG_OK` marker:

| Prompt size | Reported prompt tokens | TTFT | Client prefill | Decode |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 characters | 2,528 | 8.238 s | 306.86 tokens/s | 57.00 tokens/s |
| 8,192 characters | 5,033 | 12.879 s | 390.80 tokens/s | 65.46 tokens/s |

The 16,384-character request failed closed at the configured 8,192-token
context ceiling.

The llama.cpp control did not produce a valid visible answer at 4K or 8K.  A
second parser captured its reasoning channel, but that channel contained only a
short prompt fragment and no `LONG_OK` marker.  The 16K request returned HTTP
400.  Therefore no llama.cpp long-context TTFT or TPS claim is accepted.  The
raw artifacts document this as a response-contract and context-boundary issue.

## Quality and multimodal evidence

FreeToken's extended image suite passed 21 of 21 cases across three repetitions.
The suite checked exact colors, spatial relationships, valid visible outputs,
and a constrained visual description.  The visual description used 309 prompt
tokens and 64 completion tokens, with 1,139 ms TTFT and 52.57 visible decode
tokens/s.

The short Gemma arithmetic control returned the exact `323` answer.  The
llama.cpp wrapper also produced image-quality artifacts, but its long-context
visible-answer contract remained unresolved and is not treated as equivalent
quality evidence.

## Endurance and recovery

FreeToken completed a bounded 30-session Gemma control with 30 exact answers,
zero protocol errors, 216.61 ms mean TTFT, and 234.25 ms p95 TTFT.  Protected
Qwen recovery returned `status: ok` and `maintenance: serving` after teardown.

The Qwen FreeToken path separately completed the full 1,440-session endurance
qualification.  Gemma has not yet completed a 1,440-session or 24-hour
endurance campaign.

## Remaining limitations

1. The llama.cpp Gemma long-context response contract needs a native invocation
   that produces a comparable visible answer before that matrix can be scored.
2. The paper's exact NVIDIA reference conditions remain unavailable, so strict
   paper replication is not proven.
3. The 50 percent AMD speed-improvement target has not been reached by a
   quality-preserving candidate.
4. The 284B capacity claim still requires a clean GPU-visible unified-memory
   manifest and a model-specific capacity test.

## Source evidence

- `gmktec-evo-x2-amd-run-log.md` contains the dated artifact ledger.
- `benchmark_gemma4_gguf_text_matrix.py` defines the single-request metrics.
- `benchmark_gemma4_concurrency.py` defines the concurrent metrics.
- `benchmark_gemma4_long_context.py` defines the long-context quality gate.
- `benchmark_gemma4_endurance.py` defines the bounded endurance gate.
