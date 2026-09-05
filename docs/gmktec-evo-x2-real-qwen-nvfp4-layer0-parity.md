# GMKtec EVO-X2 real Qwen NVFP4 layer-zero parity

## Purpose

This bounded experiment loads only the first routed-expert layer from the
actual Qwen3.6 NVFP4 checkpoint, then compares the production Marlin decode
kernel with FreeToken's retained LUT-gather baseline. The loader stops when
layer zero is delivered, so later layers are not materialized and the complete
checkpoint is never loaded into the isolated process.

## Configuration

| Field | Value |
| --- | --- |
| Checkpoint | `Qwen3.6-35B-A3B-NVFP4` |
| Layer | 0 |
| Experts in source bank | 256 |
| Hidden size | 2,048 |
| Intermediate size | 512 |
| Routed experts tested | 8 |
| Input | Deterministic BF16 hidden vector |
| Kernel comparison | Marlin NVFP4 versus LUT-gather NVFP4 |
| Runtime | Native HIP on `gfx1151` |

The source loader provided the actual packed NVFP4 tensors, FP8 block scales,
and FP16 per-row global scales from the checkpoint. No synthetic weights were
used in this test.

## Real-bank fingerprints

The captured layer-zero source banks had these SHA-256 fingerprints:

```text
gate_up_packed  fe048d221cddc900220aca2f894ece4c0fbef59f504f1a5e822e67cec586dc13
gate_up_scale   5c2028ffb715de9bb84983f5ba3979872d697a58d746ddd3585cfd2d3a838800
gate_up_global  db32bb8d0ba65259794748e5d5f6d50a9cf0761310fa68e11b7e17c5763d7152
down_packed     8b8db4ac1fc04992189ea4371a763bab0fa0ec536b7a528b0183eb4023179945
down_scale      1b85bd6599f7191e8e54accea54db1c7ac93341fb50f9f8abc5a65b105327306
down_global     877ca8c4575a8c703899dc0dd4ea2432af9a5716eeb4b8fa683464d06fe17815
```

## Result

The two production kernels produced exactly identical output for the same
real checkpoint bytes and deterministic input:

```text
max_abs_diff=0.0
mean_abs_diff=0.0
outputs_finite=true
```

Ten Marlin samples, including first-use effects, were:

```text
[0.422324, 4.516500, 0.316868, 0.285419, 0.282369,
 0.238581, 0.134624, 0.121815, 0.124095, 0.127925] ms
```

The final eight steady-state samples averaged **0.203962 ms**. The eight
routed experts read approximately 12.58 MiB of packed gate/up and down input
per call, equivalent to approximately **61.7 GB/s** of routed packed input
traffic. This is a component result, not end-to-end model TPS.

## Interpretation

This closes the most important numerical uncertainty before a live candidate:
the production Marlin path can consume real Qwen NVFP4 checkpoint data on HIP
and match the baseline exactly for a deterministic routed layer operation.
The unusually high second sample is retained rather than discarded because
it is evidence of first-use or runtime scheduling overhead.

The result does not yet establish a full serving improvement. It excludes the
router, attention, KV management, scheduler, host-side cache misses, API
overhead, and all other decoder layers. A candidate replacement still needs a
complete API quality and throughput gate.

## Next action

Run the same real-bank differential across several deterministic hidden-state
vectors and routed expert sets, then attach the candidate to an isolated Qwen
server. Compare full API prefill, decode, TTFT, tail latency, concurrency,
quality hashes, and recovery against the qualified current configuration.
