# Full GLM-5.3: preliminary Vast screening

Measured 2026-09-07. This is full GLM-5.3, not GLM-5.3-Flash.
These short runs do not qualify this host for production or serverless use.

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

## Initial results

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

## Outstanding qualification

Longer output runs, repeated c1/c2/c4/c8/c16 measurements, generated-answer
checks, actual serverless routing, idle shutdown, and cached restart remain
required. Do not describe an endpoint configuration or direct-worker result
as a verified serverless lifecycle. The
[raw screening report](https://github.com/earlvanze/FreeToken/blob/0182c7e1a4509c79906b40b0ca29ccdc44b084a1/benchmarks/results/glm53-full-c8-c16-screen-20260907.json)
retains every request result. The
[qualification record](https://github.com/earlvanze/FreeToken/blob/0182c7e1a4509c79906b40b0ca29ccdc44b084a1/benchmarks/results/glm53-full-vast-50116782-qualification-20260907.json)
records hardware, checkpoint validation, and outstanding lifecycle gates.
