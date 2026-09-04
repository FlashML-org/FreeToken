# GMKtec EVO-X2 real Qwen NVFP4 route matrix

## Purpose

This follow-up expands the real layer-zero checkpoint test across three
deterministic hidden states and routed expert sets. It checks whether the
production Marlin decode path remains numerically equivalent to the retained
LUT-gather baseline when routes are contiguous, widely scattered, or repeated.

## Result

All three cases completed with finite output. The first two route patterns had
zero difference. The repeated-route case differed only by floating-point
reduction order:

| Case | Route pattern | Maximum absolute difference | Mean absolute difference | Marlin mean |
| --- | --- | ---: | ---: | ---: |
| 0 | Experts 0 through 7 | 0.0 | 0.0 | 0.187887 ms |
| 1 | 8, 19, 37, 64, 91, 127, 191, 255 | 0.0 | 0.0 | 0.114801 ms |
| 2 | 3, 3, 3, 11, 42, 42, 200, 201 | 0.00000190735 | 0.00000000483124 | 0.093191 ms |

The route matrix used the actual layer-zero packed weights, FP8 block scales,
and FP16 global scales from the Qwen3.6 NVFP4 checkpoint. Each case used a
different deterministic BF16 hidden vector and normalized routed weights.

## Interpretation

The result is a strong numerical qualification for the Marlin path. Scattered
expert IDs do not cause stale-bank or address-selection errors, and the only
nonzero difference is a sub-two-millionth absolute change in a duplicate-route
reduction. This is far below the production quality tolerance and is expected
from a different accumulation order.

The test remains a layer component test. It does not establish end-to-end
model TPS or prove that a new kernel will improve the complete server.

## Next action

Use the qualified real-bank path in an isolated serving candidate and run the
fixed Qwen API matrix. Compare the candidate to the current production profile
for prefill, decode, TTFT, tail latency, concurrency, deterministic visible
quality, cache misses, and recovery.
