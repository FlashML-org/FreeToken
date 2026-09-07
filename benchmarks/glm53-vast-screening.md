# Full GLM-5.3: H200 Serverless qualification and preliminary screening

Measured 2026-09-07. This is full GLM-5.3, not GLM-5.3-Flash.
The H200 NVL run verifies model readiness, authenticated Vast Serverless
routing, correct semantic output, and concurrency throughput. Idle shutdown
and cached restart were not tested before the instance was deleted.

## H200 NVL Serverless qualification

Vast Serverless provisioned one H200 NVL worker in Japan with 143,771 MiB
reported VRAM, 32 effective CPU cores, and a 520 GiB disk. FreeToken 0.1.2
loaded `LibertAIDAI/GLM-5.3-NVFP4`, allocated 8,384 KV tokens, and captured
CUDA graphs for batch sizes 1, 2, 4, and 8. Both worker creation and routing
through the authenticated Serverless endpoint were observed.

The semantic smoke prompt, `What is 17 + 25? Reply with only the number.`,
returned exactly `42` with finish reason `stop` in 10.402 seconds.

The concurrency workload requested a concise Python function for merging two
sorted iterables. It used non-streaming end-to-end timing, including routing,
queueing, and prefill, with `reasoning_effort="low"` and a 128-token output
cap.

| Offered concurrency | Running slots | Successful requests | Output tokens | Wall seconds | Aggregate output tok/s | Mean request seconds | p95 request seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 8 | 8/8 | 1,016 | 57.136 | 17.782 | 57.133 | 57.135 |
| 16 | 8 | 16/16 | 2,032 | 112.747 | 18.023 | 85.600 | 112.738 |

All 24 requests succeeded and produced assistant output. Each reached the
fixed 128-token cap, so these runs measure throughput rather than completed
answer quality. At c16, eight requests ran while the other eight queued behind
the worker's eight active-request slots. The nearly unchanged aggregate rate
and nearly doubled wall time place practical saturation at c8 for this worker
configuration.

The [qualification record](results/glm53-full-h200-serverless-qualification-20260907.json),
[raw c8/c16 results](results/glm53-full-h200-serverless-c8-c16-20260907.json),
and [semantic smoke response](results/glm53-full-h200-serverless-smoke-20260907.json)
preserve the machine-readable evidence.

## Preliminary RTX PRO 6000 screening

## Configuration

- Checkpoint: `LibertAIDAI/GLM-5.3-NVFP4`, 282 indexed shards,
  464,822,689,680 bytes. File structure was validated, not independent hashes.
- Runtime: FreeToken 0.1.2, commit
  `55120d9114eb18d90e4dfe2e1454f25f0098723e`.
- GPU: one RTX PRO 6000 Blackwell Max-Q, 97,887 MiB reported VRAM,
  300 W power cap; NVIDIA driver 595.84.
- Vast machine 104296, India; CPU quota 36.48 effective cores;
  container memory limit 2,057,523,167,232 bytes.
- MoE offload, Triton NVFP4, DSA attention, tensor parallelism 1,
  eight running request slots, max sequence length 32,768, 48 CPU threads,
  memory fraction 0.95. Expert cache: 3,461 slots; KV capacity: 8,239 tokens.
- Direct loopback HTTP, not the Vast serverless routing path.
- OS: Ubuntu 24.04.3 LTS. Host CPU reports `Genuine Intel(R) CPU 0000%@`
  across two sockets; no more specific processor SKU was exposed.

The running process was inspected with this exact command line:

```bash
/workspace/freetoken/.venv/bin/ft serve \
  --model /workspace/models/GLM-5.3-NVFP4 \
  --served-model-name glm-5.3-nvfp4 --host 127.0.0.1 --port 1919 \
  --moe-backend auto --moe-cpu-threads 48 --memory-ratio 0.95 \
  --max-seq-len-override 32768 --max-running-requests 8 \
  --disable-moe-prefill-overlap
```

### Initial results

The prompt requested a concise Python function merging two sorted iterables,
with a request index appended. Each request used a 32-token output cap and
default reasoning. The model was loaded and a smoke completion had succeeded.

| Offered concurrency | Running slots | Output tokens | Wall seconds | Aggregate output tok/s | Mean request seconds | Mean request output tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 8 | 248 | 62.991 | 3.937 | 62.983 | 0.492 |
| 16 | 8 | 496 | 78.767 | 6.297 | 58.434 | 0.603 |

All 24 requests returned without transport errors, but all stopped at the
output cap while still reasoning. None supplied a final code answer. These
are output-token measurements, not successful coding-task throughput.
Both results are below the requested 15 aggregate output tok/s c8 gate.
Per-request rates include queueing and prefill; they are not steady decode
rates. Prefix-cache conditions differ between the two sequential levels.

A separate arithmetic smoke request produced the correct answer, 42, in
40.255 seconds (10 reported output tokens). With `reasoning_effort="none"`,
the unpatched runtime exposed a closing thinking tag in its answer. The
checkpoint template always opens a thinking block; disabling parsing does
not disable reasoning computation. A separately prepared parser regression fix
has local test coverage but was not deployed for these measurements.

## Remaining lifecycle checks

The H200 run verifies worker creation, model readiness, authenticated
Serverless routing, and a correct semantic response. Repeated concurrency
runs, idle shutdown, and cached restart remain outstanding. Do not describe
the result as a complete Serverless lifecycle qualification. The earlier
[raw screening report](https://github.com/earlvanze/FreeToken/blob/0182c7e1a4509c79906b40b0ca29ccdc44b084a1/benchmarks/results/glm53-full-c8-c16-screen-20260907.json)
retains every request result. The
[qualification record](https://github.com/earlvanze/FreeToken/blob/0182c7e1a4509c79906b40b0ca29ccdc44b084a1/benchmarks/results/glm53-full-vast-50116782-qualification-20260907.json)
records the RTX PRO 6000 hardware, checkpoint validation, and original
screening limitations.
