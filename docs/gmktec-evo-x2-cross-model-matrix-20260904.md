# GMKtec EVO-X2 cross-model benchmark matrix

This matrix consolidates the preserved September 4, 2026 controls for the
native ROCm/HIP FreeToken port and the local ROCm 10 llama.cpp controls. It is
an evidence index, not a claim that every row is a strict apples-to-apples
comparison. Each comparison must retain its model format, prompt contract,
sampling settings, warmup rules, and concurrency boundary.

## Fixed-length text controls

| Model and runtime | Samples | Prompt tokens | Completion tokens | Mean prefill TPS | Mean decode TPS | Mean TTFT | p99 token gap | Quality status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Gemma 4 Q4 FreeToken | 5 | 34 | 127 | 174.58 | 53.08 | 196.33 ms | 21.60 ms | Exact text control passed |
| Gemma 4 Q4 llama.cpp ROCm 10 | 5 | 34 | 128 | 737.00 | 56.83 | 47.67 ms | 18.14 ms | Matched visible prompt contract passed |

The Gemma rows use the matched five-sample artifacts
`gemma4-gguf-text-20260905T084837Z/text-matrix.json` and
`gemma4-llamacpp-vision-20260905T091532Z/text-matrix.json`. The visible prompt
contract is matched; the one-token difference in tokenizer-reported prompt
length is retained as observed telemetry rather than silently normalized.

## Qwen controls

| Model and runtime | Samples | Prompt tokens | Completion tokens | Mean prefill TPS | Mean decode TPS | Tail evidence | Quality status |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Qwen3.6 35B-A3B FreeToken Q4 | 5 | 1,212 | 255 | 2,936.92 | 28.04 | Mean p99 gap 35.38 ms | Passed paired quality gate |
| Qwen3.6 35B-A3B llama.cpp Q4_K_M ROCm 10 | 5 | 1,212 | 256 | 19,343.40 | 46.66 | Mean gap 21.43 ms | Passed control suite |
| Qwen3.6 35B-A3B FreeToken Q5 four-row | 3 scheduler plus 3 C4 rounds | Fixed scheduler contract | Fixed scheduler contract | 3,130.30 | 48.20 single, 94.80 aggregate C4 | p99 TTFT 1.025 s; p99 gap 39.93 ms | Canonical AIME passed |
| Qwen3.6 35B-A3B FreeToken Q4 MMV_Y=4 | 5 | Fixed API contract | Fixed API contract | 2,857.78 | 48.03 | Mean client TTFT about 0.424 s | Quality and API checks passed |

### Same-checkpoint and same-format raw-prompt control

| Model and runtime | Samples | Prompt tokens | Completion tokens | Decode TPS | TTFT | Quality result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3.6 35B-A3B FreeToken Q4_K_M GGUF | 1 | 54 | 255 | 50.0169 | 54.311 s cold request | Expected answer path passed; output hash differs from llama.cpp |
| Qwen3.6 35B-A3B llama.cpp Q4_K_M GGUF ROCm 10 | 1 | 54 | 256 | 49.3875 | 234.0 ms loaded control | Expected answer path passed; output hash differs from FreeToken |
| Qwen3.6 35B-A3B FreeToken Q4_K_M GGUF warmed matrix | 5 | 54 | 255 | 48.6028 all samples; 49.4357 samples 2 to 5 | 982.17 ms all samples; 424.26 ms samples 2 to 5 | All five output hashes match; expected answer path passed |
| Qwen3.6 35B-A3B llama.cpp Q4_K_M GGUF warmed matrix | 5 | 54 | 256 | 49.1155 all samples; 49.1772 samples 2 to 5 | 92.07 ms all samples; 58.83 ms samples 2 to 5 | All five output hashes match; expected answer path passed |

The same 22 GiB Q4_K_M GGUF checkpoint, tokenizer, caller-rendered raw
prompt, and output harness were used. FreeToken was approximately 1.27 percent
faster on decode. TTFT is not a valid parity claim in this pair because the
FreeToken measurement includes its cold model initialization while llama.cpp
was already loaded. The two responses both reached the expected answer path,
but their full output hashes differ, so this run is a performance control and
not proof of bit-identical generation.

The warmed FreeToken follow-up used one loaded server and five consecutive
requests. Its first scored request measured 45.2713 TPS while requests 2 to 5
measured 49.3630, 49.3096, 49.6545, and 49.4156 TPS. This separates cold
startup and first-request effects from the steady request path. A warmed
five-sample llama.cpp matrix is still required before declaring a statistical
same-format winner. The warmed llama.cpp follow-up measured 48.8686 TPS on
the first request and 49.1575, 49.1606, 49.1887, and 49.2019 TPS on requests
2 through 5. FreeToken's samples 2 to 5 mean was 49.4357 TPS, approximately
0.53 percent above llama.cpp's 49.1772 TPS. Across all five samples, llama.cpp
was approximately 1.06 percent faster because FreeToken's first request was
slower. This is a near-parity result, not a material performance lead.

The Q4 rows are practical local controls, not a same-format NVFP4 equivalence
claim. The Q5 four-row row is the currently qualified quality-preserving
optimization and is not directly comparable to the Q4 llama.cpp row without a
matched Q5 control.

## Concurrency and long-context coverage

| Model and runtime | Concurrency | Long-context | Endurance | Current conclusion |
| --- | --- | --- | --- | --- |
| Qwen FreeToken | 1, 2, 4, and 8-client tail controls | 4,856-token W3-style control passed | 1,440 sessions passed | Qwen stability qualification complete |
| Gemma 4 FreeToken | 2, 4, and 8-client matched controls passed | 2,528 and 5,033 prompt-token controls passed; 8,192-token ceiling reached | 30 sessions passed | Aggregate decode is 4.1 percent above llama.cpp at four clients and 25.7 percent above it at eight clients; TTFT is lower under concurrency |
| Gemma 4 llama.cpp ROCm 10 | 2, 4, and 8-client matched controls passed | Reasoning-off 2,528 and 5,033-token controls passed; 8,192-token ceiling reached | Not run | Matched local control, not strict paper replication |

### Matched Gemma concurrency detail

| Clients | FreeToken aggregate decode TPS | llama.cpp aggregate decode TPS | FreeToken mean TTFT | llama.cpp mean TTFT | Quality |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 31.04 | 35.80 | 359.8 ms | 1,264.0 ms | All requests passed |
| 4 | 22.19 | 21.31 | 369.0 ms | 3,678.1 ms | All requests passed |
| 8 | 14.87 | 11.84 | 3,178.3 ms | 8,483.9 ms | All requests passed |

## Missing cells before final campaign closure

1. Run every additional model only after its exact payload and backend are
   admitted. The current active service inventory is already covered by the
   Qwen and Gemma rows; archived models without payloads remain unqualified.
2. Add a format-matched Qwen control using the same checkpoint and
   quantization on both runtimes before making a same-format claim.
3. Consolidate telemetry fields, cold-start policy, and cache state into a
   machine-readable comparison manifest.
4. Keep a full Gemma 1,440-session campaign optional. It is not required for
   the current functional or bounded-performance release gates.
5. Keep strict NVIDIA comparison and 284B capacity as separate unresolved
   work items because their required external evidence is still missing.

