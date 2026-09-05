# Reproducibility and independent extension

This guide is the public counterpart to the recorded evidence workflow. It is
for contributors who want to build, validate, benchmark, or extend the native
ROCm/HIP FreeToken port on their own AMD system. It does not require the evaluated
host, an internal address, or a production service.

## Scope

The public artifact proves only the experiment it records. It must not be used
to infer strict paper replication, general AMD performance, or quality parity
outside the stated model, tokenizer, prompt, runtime, and measurement contract.

The exact source revision, model provenance, runtime versions, raw streaming
timestamps, and output-quality evidence are required for any reported result.
Never report a profiler trace as an unprofiled throughput result.

## Prerequisites

- Linux system with a native ROCm-capable AMD GPU.
- Git checkout of this repository.
- Python environment containing HIP-enabled PyTorch, Triton, and FreeToken.
- ROCm tools including `rocminfo`; `rocm-smi` is optional but recommended.
- A model obtained directly from its original publisher under its license.

Confirm that `torch.version.hip` is non-empty and that `torch.cuda.is_available()`
returns true before building FreeToken. HIP maintains the `torch.cuda` namespace
for compatibility, so a successful import alone is not sufficient evidence of
native ROCm execution.

## Capture a host manifest

From the repository root, use a new artifact directory for every run:

```bash
bash scripts/reproduce/collect_host_manifest.sh \
  --source-dir "$PWD" \
  --artifact-dir "$PWD/artifacts/host-$(date -u +%Y%m%dT%H%M%SZ)" \
  --python /path/to/venv/bin/python \
  --expected-gfx gfx1151
```

Omit `--expected-gfx` only when the work is intentionally cross-architecture.
The script requires a native HIP PyTorch device before it creates an artifact.
It never starts or stops a server, changes a clock policy, clears a cache,
changes swap, or records a shell environment. It redacts the host name by
default and deliberately omits process lists, serial numbers, and network
addresses. Each bundle includes a `SHA256SUMS` file covering the raw report
files and manifest.

The collector was live-validated on the evaluated system on 30 August 2026 using native
ROCm PyTorch and `--expected-gfx gfx1151`. It emitted a redacted host field,
identified `gfx1151`, and produced only the documented non-sensitive artifact
files. This validates the collector itself, not a broader performance claim.

## Build and functional validation

Install only into an isolated environment. Do not install CUDA-only packages
such as CUDA-indexed PyTorch wheels, FlashInfer, or NVIDIA kernel wheels into a
ROCm environment. With a verified ROCm Python runtime, build from the checkout:

```bash
python -m pip install -e . --no-build-isolation --no-deps
python -m unittest tests.utils.test_rocm_runtime
```

Then run model-specific functional controls before any throughput run. Record
the model publisher, revision, file byte count, SHA-256, tokenizer revision,
chat template, request JSON, output, and scorer result in the artifact. Do not
redistribute model weights unless the model license explicitly allows it.

## Benchmark an already-running local server

`benchmarks/reproduce/run_local_api_benchmark.py` is the public client for a
server that the operator has already started on a loopback endpoint. It accepts
only `localhost`, `127.0.0.1`, or `::1`; it cannot send benchmark traffic to a
LAN or public address. It neither starts nor stops a server. Quality mode
requires an explicit expected visible response, while throughput mode requires
a fixed generation length of at least two tokens.

```bash
python benchmarks/reproduce/run_local_api_benchmark.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model your-served-model-name \
  --tokenizer /path/to/original/checkpoint \
  --prompt-file /path/to/quality-prompt.txt \
  --expected-text EXPECTED_VISIBLE_ANSWER \
  --samples 5 --warmup \
  --artifact-dir artifacts/quality-$(date -u +%Y%m%dT%H%M%SZ)
```

Use a separate throughput artifact after the quality gate passes. Retain the
same prompt and model representation, opt into `--mode throughput`, set a fixed
`--max-tokens` value, and explain any difference between server-side generation
tokens and tokenizer-counted visible text. The client writes a manifest, an
immutable JSON artifact for each request, and a summary. It does not claim that
the server is correct merely because it streamed successfully.

## Benchmark contract

For each row, retain raw artifacts before generating a summary table:

1. A cold-start result and a warm-server result, labelled separately.
2. At least five independently started scored samples for a performance claim.
3. Client-observed TTFT, prompt throughput, decode throughput, and p50, p95,
   and p99 content-token gaps.
4. GPU clock, temperature, power policy, memory use, host memory pressure,
   swap state, and competing I/O or compute activity.
5. The exact model representation. Do not compare NVFP4 and GGUF performance
   as if they were a format-neutral engine comparison.
6. A quality gate that is separate from fixed-length throughput mode.

Reject or rerun samples affected by unexpected compilation, swapping, thermal
transition, stale service processes, active model copies, or unexplained I/O
contention. Preserve rejected samples and state why they were rejected.

## Extension policy

Contributions should target one measured bottleneck at a time. Suitable work
includes a unified-memory-aware cache policy, a profile-ranked HIP kernel,
additional GPU-host manifests, or an expanded quality suite. Every change must
retain its baseline, raw evidence, quality comparison, full API result, and an
accept-or-reject decision. A faster isolated kernel is not an accepted runtime
optimization until it preserves model output and improves the full serving
workload.

## Publication package

For an external release, publish a pinned source tag, this guide, `CITATION.cff`,
`.zenodo.json`, the safe host manifests, workload and scorer code, sanitized raw
results, and scripts that regenerate paper tables. Archive the tagged release
with Zenodo and cite its version DOI. Keep private model files, credentials,
internal addresses, serial numbers, and unrelated service logs out of the
release.
