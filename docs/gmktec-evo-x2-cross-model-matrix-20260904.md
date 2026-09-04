# GMKtec EVO-X2 cross-model benchmark matrix

This matrix consolidates the preserved September 4, 2026 controls for the
native ROCm/HIP FreeToken port and the local ROCm 10 llama.cpp controls. It is
an evidence index, not a claim that every row is a strict apples-to-apples
comparison. Each comparison must retain its model format, prompt contract,
sampling settings, warmup rules, and concurrency boundary.

## Fixed-length text controls

| Model and runtime | Samples | Prompt tokens | Completion tokens | Mean prefill TPS | Mean decode TPS | Mean TTFT | p99 token gap | Quality status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Gemma 4 Q4 FreeToken | 5 | 34 | 127 | 174.27 | 50.87 | 203.07 ms | 132.36 ms | Exact text control passed |
| Gemma 4 Q4 llama.cpp ROCm 10 | 5 | 35 | 128 | 478.13 | 30.50 | 118.71 ms | 283.97 ms | Same visible prompt contract passed |

The Gemma rows use the preserved fixed-length artifacts
`gemma4-gguf-text-20260904T150838Z-text-matrix.json` and
`gemma4-llamacpp-vision-20260904T152038Z-text-matrix.json`. The one-token
prompt-count difference and different output token limits are recorded rather
than silently normalized.

## Qwen controls

| Model and runtime | Samples | Prompt tokens | Completion tokens | Mean prefill TPS | Mean decode TPS | Tail evidence | Quality status |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Qwen3.6 35B-A3B FreeToken Q4 | 5 | 1,212 | 255 | 2,936.92 | 28.04 | Mean p99 gap 35.38 ms | Passed paired quality gate |
| Qwen3.6 35B-A3B llama.cpp Q4_K_M ROCm 10 | 5 | 1,212 | 256 | 19,343.40 | 46.66 | Mean gap 21.43 ms | Passed control suite |
| Qwen3.6 35B-A3B FreeToken Q5 four-row | 3 scheduler plus 3 C4 rounds | Fixed scheduler contract | Fixed scheduler contract | 3,130.30 | 48.20 single, 94.80 aggregate C4 | p99 TTFT 1.025 s; p99 gap 39.93 ms | Canonical AIME passed |

The Q4 rows are practical local controls, not a same-format NVFP4 equivalence
claim. The Q5 four-row row is the currently qualified quality-preserving
optimization and is not directly comparable to the Q4 llama.cpp row without a
matched Q5 control.

## Concurrency and long-context coverage

| Model and runtime | Concurrency | Long-context | Endurance | Current conclusion |
| --- | --- | --- | --- | --- |
| Qwen FreeToken | 1, 2, 4, and 8-client tail controls | 4,856-token W3-style control passed | 1,440 sessions passed | Qwen stability qualification complete |
| Gemma 4 FreeToken | 4-client, 12-request control passed | 2,528 and 5,033 prompt-token controls passed; 8,192-token ceiling reached | 30 sessions passed | Full 1,440-session Gemma run remains open |
| Gemma 4 llama.cpp ROCm 10 | 4-client, 12-request control completed | Reasoning-off 2,528 and 5,033-token controls passed; 8,192-token ceiling reached | Not run | Matched local control, not strict paper replication |

## Missing cells before final campaign closure

1. Run every model currently in the in-scope serving inventory through this
   same matrix.
2. Add a format-matched Qwen control using the same checkpoint and quantization
   on both runtimes.
3. Consolidate telemetry fields, cold-start policy, and cache state into a
   machine-readable comparison manifest.
4. Decide whether a full Gemma 1,440-session campaign is required for the
   release, then run it only after the standardized matrix is frozen.
5. Keep strict NVIDIA comparison and 284B capacity as separate unresolved
   work items because their required external evidence is still missing.

