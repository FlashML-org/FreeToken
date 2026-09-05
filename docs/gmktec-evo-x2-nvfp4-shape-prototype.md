# GMKtec EVO-X2 production-shape NVFP4 prototype

## Purpose

This experiment moves the previous fused-expert investigation to the
production NVFP4 tensor layout used by the FreeToken Qwen path. It uses random
device-resident tensors, so no model checkpoint or production service is
involved. The goal is to measure kernel launch behavior at representative
dimensions before attempting a production implementation change.

## Configuration

| Field | Value |
| --- | --- |
| Backend | Native FreeToken Triton NVFP4 Marlin decode kernel through HIP |
| GPU target | AMD `gfx1151` |
| Experts in bank | 8 |
| Hidden size | 1,152 |
| Intermediate size | 512 |
| Routed experts per token | 8 |
| Gate/up packed shape | `[8, 1024, 576]` uint8 |
| Gate/up scale shape | `[8, 1024, 72]` uint8, one scale per 16 values |
| Down packed shape | `[8, 1152, 256]` uint8 |
| Down scale shape | `[8, 1152, 32]` uint8, one scale per 16 values |
| Activation | BF16 input, SiLU gated path |
| Timed samples | 10, after one warmup call |

The production function performs both gate/up and down expert GEMV operations,
the SiLU activation, routed-weight handling, and final expert reduction. The
benchmark uses the production Marlin-style entry point rather than a separate
synthetic CUDA or HIP kernel.

## Result

The isolated run completed successfully and returned finite output:

```text
experts=8 hidden=1152 intermediate=512 top_k=8
samples_ms=[0.305189, 0.184113, 0.176063, 0.165334, 0.107996,
            0.104776, 0.097746, 0.099526, 0.099096, 0.094497]
mean_ms=0.143434 output_finite=true
```

The first timed sample includes residual one-time runtime work. Excluding that
sample, the steady-state mean was **0.125461 ms** for the complete two-GEMV
fused expert operation. The packed gate/up plus down input footprint was
7,077,888 bytes per call, equivalent to approximately **56.4 GB/s** of packed
input traffic at that steady-state mean.

## Interpretation

This is the first format-faithful kernel-path result in the investigation. It
confirms that the production NVFP4 Marlin decode entry point can execute the
representative routed shape on HIP with finite output and sub-millisecond
steady-state latency.

It is not model TPS. The weights are random, the bank has only eight experts,
and this test excludes router execution, attention, KV management, token
scheduling, API overhead, and host-side expert-cache misses. It therefore
cannot be used to claim quality or end-to-end speed improvement.

## Next action

Instrument the same entry point with the actual cache bank and deterministic
Qwen layer inputs, then compare its output hash and latency against the current
production candidate. A replacement kernel must preserve exact quality and
survive the complete API, concurrency, tail-latency, and recovery gates before
it can be promoted.
