# Triton decode output gate

Models using `attention_output * sigmoid(gate)` can call
`BaseAttnBackend.forward_gated`. The default implementation applies the Torch
sigmoid and multiply after the existing backend. Hybrid backends delegate to the
selected prefill/decode backend. Qwen3.5/3.6 full-attention layers use this interface.

## Automatic dispatch

The Triton backend fuses the gate into its existing decode stage2 only for:

- ROCm gfx1201, BF16 query/gate/KV, query shape `[1,16,256]`, two KV heads;
- Graph capture, contiguous query/gate/KV, CSR index buffer capacity exactly 512;
- the default full-attention specification (no sinks, sliding window, custom scale,
  or bidirectional multimodal blocks).

The flag `FREETOKEN_TRITON_FUSE_OUTPUT_GATE=0` disables automatic fusion.
Eager, prefill, larger Graph capacities, other batch sizes, dtypes, GPU architectures and
attention backends use the default implementation. Eligibility is checked before
storing KV, so each route stores the new rows once. This change includes no HIP
extension, native build, model-name check or software-version fingerprint.

Capacity is the host-visible allocated CSR buffer size, not the current effective
length. Setting max sequence capacity to 4096 retains the fallback even for a
short current request. The initial automatic scope is deliberately limited to the
B1/512 configuration with real-model validation. The low-level kernel accepts
broader BF16 shapes for explicit testing, including the B4 kernel benchmark;
these are not automatically enabled by the serving backend.

## Arithmetic

Attention stage1 and the stage2 FP32 reduction are retained. Stage2 rounds the
attention result to BF16, applies the BF16 sigmoid value, then rounds the product
as Torch does. A per-device 128 KiB table covers every BF16 input bit pattern and
is computed by Torch once before Graph capture. It preserves the active runtime's
sigmoid rounding. Explicit integer round-to-nearest-even for the product avoids
compiler narrowing to gfx12 BF16 dot2 arithmetic. An approximate sigmoid or
skipping the intermediate BF16 rounding can change FP8 MoE model logits.

The measured window starts with prepared Q/K/V and CSR metadata and ends at the
gated attention output. QKV projection, rotary, KV writes and output projection
are outside the kernel measurement. Kernel speedup does not imply the same
end-to-end speedup.

## Reproduce

Kernel pairs, including the same buffers and an alternating Graph order:

```bash
PYTHONPATH=python python benchmarks/bench_decode_output_gate.py \
  --baseline-path /path/to/main/python/freetoken/kernel/triton/attention.py \
  --json kernel.json
```

Run the following server command on main and this branch on the same free GPU,
with the same checkpoint, environment and options:

```bash
PYTHONPATH=python python -m freetoken.cli serve \
  --model-path "$MODEL" --dtype bfloat16 --attention-backend triton \
  --moe-strategy offload --moe-cache-mem 20GiB --moe-cache-scope unified \
  --moe-cache-split equal --num-tokens 544 --page-size 1 \
  --max-seq-len-override 512 --max-prefill-length 512 \
  --max-running-requests 1 --cuda-graph-max-bs 1 --max-output-tokens 128 \
  --memory-ratio 0.9 --expert-load serial --cache-type naive \
  --sampling-defaults none --reasoning-parser off --text-model-only \
  --served-model-name gate-bench --host 127.0.0.1 --port 19741
```

From this branch, measure either running server:

```bash
PYTHONPATH=python python benchmarks/bench_decode_output_gate.py \
  --url http://127.0.0.1:19741 --model "$MODEL" \
  --input-tokens 128 --output-tokens 128 --repeats 1 --json server.json
```

The client warms once, checks exact usage, then measures from the first content
SSE to the finish frame for decode throughput. It separately reports TTFT and
request-wide throughput and saves event timestamps and an output hash. For the
larger-capacity control use 4096 for both max lengths, 4128 tokens, and a 512-token
input. Alternate main/branch order and restart the server for each arm.

Tests:

```bash
PYTHONPATH=python pytest tests/kernels/test_triton_attention.py \
  tests/engine/test_attention_backend_arch.py \
  tests/engine/test_attention_backend_matrix.py tests/daemon/test_daemon_import_safety.py
```

## Measured on Radeon AI PRO R9700

Base: `cc4aa085b6eaeda01c0441fb3ecb9cc843b3e465`. gfx1201, 32 GiB;
AMD EPYC 9334; Linux 7.0.0-38-generic, amdgpu 7.1.9.31600000;
Torch 2.14.0+rocm10.1.0, HIP 7.16.26385, Triton 3.8.0, Python 3.13.
Only this GPU platform was measured. These figures do not establish a speedup on
other architectures or checkpoints.

### Kernel window

Two independent rounds; seven alternating samples per round; each Graph contains
four calls, with 50 replays per sample after warmup. Unit: microseconds per call.
The reference loads the upstream kernel module and applies Torch sigmoid/multiply.
Both B1 and B4 outputs match the reference elementwise exactly in these cases.
B4 lengths are 512/495/478/461; B4 is a low-level kernel result, not automatic
serving admission.

| Shape | Main us, round 1 / 2 | Branch us, round 1 / 2 | Speedup, round 1 / 2 |
|---|---:|---:|---:|
| B1 | 14.971 / 14.844 | 10.431 / 10.416 | 1.435 / 1.425x |
| B4 | 15.886 / 15.656 | 11.551 / 11.517 | 1.375 / 1.359x |

### End to end: gate fusion alone

Checkpoint: `Qwen/Qwen3.6-35B-A3B-FP8` (locally converted to FTW), BF16
activations/KV, FP8-block experts. The command above uses a 20 GiB offload expert
cache. Convolution and GDN are unchanged on both arms. Three fresh-process pairs
per capacity, one warmup then one measured request per process, alternating order
main/branch, branch/main, main/branch. CPU affinity 16-23,80-87 (NUMA2), GPU NUMA0;
the host is shared and clocks are not locked.

| Graph capacity | Input/output tokens | Main / branch decode tok/s | Median paired decode gain | Median paired request-wide gain | Main / branch TTFT ms |
|---|---|---:|---:|---:|---:|
| 512 | 128 / 128 | 77.179 / 77.562 | +0.474% | -0.031% | 895.63 / 898.21 |
| 4096 | 512 / 128 | 76.723 / 76.723 | +0.001% | +0.047% | 895.77 / 903.04 |

Throughput and TTFT columns are independent medians. Paired gains are computed
within each round then summarized, so they need not equal ratios of those medians.
All six pairs have equal output text hashes and exactly 128 generated tokens.
The short Graph has a small decode improvement; these measurements do not establish
a reliable request-wide gain. The 4096-capacity control retains the Torch gate.

| Capacity | Round | Decode ratio | Request-wide ratio |
|---|---:|---:|---:|
| 512 | 1 | 1.004735x | 1.006108x |
| 512 | 2 | 1.003748x | 0.999692x |
| 512 | 3 | 1.009045x | 0.999500x |
| 4096 | 1 | 0.997650x | 0.991592x |
| 4096 | 2 | 1.003383x | 1.001867x |
| 4096 | 3 | 1.000008x | 1.000473x |

### Correctness and actual execution

- Attention/backend/import regression tests: 128 passed. They include the Torch
  attention oracle, exact fused-vs-unfused comparison, sinks/window and non-power
  GQA groups, changed Graph replay inputs, fallback, opt-out and Hybrid delegation.
- Runtime cache rebuild and Graph re-capture on `Qwen/Qwen3-1.7B`: one test passed, including growth, rejected oversized targets, and continued generation.
- Per capacity: four unforced QA requests plus 64 matched-prefix logit positions
  (four requests x sixteen positions). All sampled logits are bitwise equal to
  main and finite; max KL is zero; no new first-answer failures.
- Short Graph capture records ten gated attention calls. Its diagnostic first
  replay has 1157 -> 1137 GPU kernels, removing two gate kernels per full-attention
  layer. At capacity 4096, capture records zero fused calls and both traces have
  1157 kernels. These are diagnostic first-replay counts, not a steady-state
  time breakdown.
