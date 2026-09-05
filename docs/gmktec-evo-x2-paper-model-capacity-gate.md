# GMKtec EVO-X2 paper-model capacity gate

This is a read-only capacity gate for deciding whether to attempt the
FreeToken paper's large-model demonstrations on the GMKtec EVO-X2 Strix Halo.
It records the live host state and does not download, load, or alter a model.

## Live host observation

The observation was collected on 2026-09-04 from the configured GMKtec EVO-X2
using `free -h`, `swapon --show --bytes`, `rocm-smi`, and a bounded model-file
inventory.

| Resource | Observed value |
|---|---:|
| GPU | AMD Radeon 8060S Graphics, gfx1151 |
| ROCm-reported VRAM | 2 GiB total, approximately 352 MiB used at observation time |
| System memory | 59 GiB total, 18 GiB available |
| Swap | 127 GiB total, approximately 2.1 GiB used |
| Root filesystem | 1.9 TiB total, 769 GiB available |
| GPU temperature | 31 C |
| GPU power | 12 W |
| GPU load | 0 percent |

The 59 GiB system-memory figure is not a promise that all 59 GiB is available
to model weights. The live `MemAvailable` value was approximately 18 GiB, and
the ROCm device reports a separate 2 GiB VRAM aperture. Unified-memory
allocation, runtime buffers, KV cache, and the protected service must be
accounted for before any model load.

## Installed model evidence

The bounded inventory found the following relevant payloads:

- Qwen3.6-35B-A3B Q4 GGUF: approximately 22.1 GB.
- Gemma 4 26B Q4 model GGUF: approximately 14.4 GB.
- Gemma 4 projector GGUF: approximately 1.2 GB.
- No DeepSeek-V4-Flash checkpoint.
- No GLM-5.2 checkpoint.

The current official DeepSeek repository metadata at commit
`7872f01b1d1fe23eabc4c98b48bffcef5a386062` lists 48 safetensors shards.
Read-only HTTP `HEAD` requests to every shard reported a combined
`Content-Length` of 166,886,535,336 bytes, or approximately 155.43 GiB for the
model payload alone. This excludes the tokenizer, runtime allocations,
expert-cache policy, KV cache, allocator slack, and any duplicate conversion
buffers. The measurement was refreshed on 2026-09-05.

The official `config.json` reports 43 hidden layers, 256 routed experts, one
shared expert, and six routed experts active per token. The hidden size is
4,096. This confirms that the 13B activated-parameter figure does not reduce
the storage requirement to 13B parameters: the complete routed-expert pool is
still part of the 148.66 GiB checkpoint and must be streamed, cached, or
otherwise retained by the serving system.

## Paper-model decision

The FreeToken paper identifies DeepSeek-V4-Flash as a 284B-parameter model
with 13B activated parameters and mixed FP4 plus FP8 deployment. The current
official `DeepSeek-V4-Flash-0731` model page is a later release that reports
304B parameters and BF16, I64, F32, F8_E4M3, and I8 tensor types. Its raw
configuration still confirms FP4 expert storage, 256 routed experts, six
experts active per token, and 43 layers, but the release identity is not
automatically the same as the paper's 284B demonstration. The paper's
prefill discussion describes roughly 140 GB of routed expert weights. The
paper describes GLM-5.2 as a 753B-parameter model with a 433 GB checkpoint.
Neither payload is installed on this host, and the live available-memory
observation is far below either stated payload scale.

The paper names `deepseek-ai/DeepSeek-V4-Flash-0731` as its official
checkpoint, and the repository history exposes a release commit
`9e165c30e2704aec5d9d593cce3eebd58bbef1cb` that predates the current model-card
metadata update. That release commit still contains the same 48-shard payload
size measured above. We must pin that revision in any reproduction record and
report the paper's 284B label separately from the current model-card 304B label;
the two labels are not enough by themselves to prove an exact parameter-count
match.

As an additional identity check, `config.json` is byte-for-byte identical at
the release commit and the current model-card commit. Its SHA-256 is
`6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`.

## Reproducible metadata-only gate

The repository includes
[`deepseek_capacity_gate.py`](../scripts/gmk-evo-x2/deepseek_capacity_gate.py).
Using the pinned payload size, 18 GiB of observed `MemAvailable`, a 2 GiB
ROCm-reported aperture, and explicit reserves of 8 GiB for the OS, 2 GiB for
runtime state, 2 GiB for KV cache, and 2 GiB for recovery, it produced:

```text
decision: REJECT_FULL_LOAD
authoritative model budget: 4.000 GiB
payload: 155.425 GiB
authoritative deficit: 151.425 GiB
optimistic deficit even counting the VRAM aperture: 135.425 GiB
```

The machine-readable result is
[`gmktec-evo-x2-deepseek-capacity-gate-result-20260905.json`](gmktec-evo-x2-deepseek-capacity-gate-result-20260905.json).
This is a metadata-only rejection. No model files were downloaded, and no
service or model process was changed.

## Expert-slice metadata

The pinned safetensors index and a bounded header range from shard 2 provide
enough metadata to size a production-shaped slice without downloading tensor
payloads. Each core routed expert has three I8 matrices and three scale
arrays totaling 13,369,344 bytes, or 12.75 MiB. The 43-layer, 256-expert core
pool is approximately 137.0625 GiB. Six active experts per layer across all
43 layers would touch approximately 3.22265625 GiB before attention, shared
experts, KV cache, runtime buffers, or allocator overhead.

This makes a bounded transfer and packing experiment worthwhile, but it does
not make full serving feasible. The exact derived values are preserved in
[`gmktec-evo-x2-deepseek-expert-slice-metadata-20260905.json`](gmktec-evo-x2-deepseek-expert-slice-metadata-20260905.json).

The executable next-stage harness is
[`deepseek_expert_slice_benchmark.py`](../scripts/gmk-evo-x2/deepseek_expert_slice_benchmark.py).
Its default selection is one layer and six experts, approximately 76.5 MiB of
core routed expert weight bytes before scales and other model state. It
requires a locally staged safetensors directory, PyTorch with ROCm, and the
`safetensors` package. It touches no API port and records
`protected_service_touched: false` in its output. It must be run only as an
isolated candidate after the normal service is verified healthy. The
`--metadata-only` mode was validated locally against the six-expert fixture in
`tests/fixtures/deepseek_expert_index`; it selected all 36 expected tensors
without importing GPU libraries.

## Real-shape ROCm slice result

The isolated harness was run on the GMKtec EVO-X2 using the pinned shard and
the native ROCm environment. It transferred 80,216,064 bytes, or 76.5 MiB,
covering all six experts and all six tensors per expert for layer 0. Five
round trips were recorded. The first H2D sample was cold at 0.863 GiB/s,
consistent with initial mapping and page-fault overhead. The final three H2D
samples averaged 76.645 GiB/s, while all four post-cold samples averaged
73.987 GiB/s. D2H averaged 64.073 GiB/s across the four post-cold samples.
Using decimal units, the final-three H2D result is approximately 82.30 GB/s
and the post-cold D2H result is approximately 68.80 GB/s. That is consistent
with, but slightly more realistic than, the earlier contiguous synthetic bound
of 79.79 GB/s H2D and 70.24 GB/s D2H.

The protected Qwen health endpoint remained healthy after the run, reporting
`status: ok` and `maintenance: serving`. ROCm reported 28 C, 13.041 W, and
zero GPU utilization at the post-run check. The raw result is preserved in
[`gmktec-evo-x2-deepseek-expert-slice-result-20260905.json`](gmktec-evo-x2-deepseek-expert-slice-result-20260905.json).
This is evidence for the AMD transfer path only, not a full-model serving or
quality result.

A second isolated run expanded the same layer to 16 experts, or 204.0 MiB and
96 tensors. The final three H2D samples averaged 77.561 GiB/s, all four
post-cold H2D samples averaged 77.346 GiB/s, and post-cold D2H averaged 64.762
GiB/s. The protected service again returned `status: ok` after the run. This
larger slice shows no material H2D collapse as the transfer batch grows, but
it remains a single-layer transfer test rather than a model-serving result.
Its raw output is preserved in
[`gmktec-evo-x2-deepseek-expert-slice-16-result-20260905.json`](gmktec-evo-x2-deepseek-expert-slice-16-result-20260905.json).

Finally, a two-layer slice selected experts 0 through 5 from layers 0 and 1,
spanning both shard 2 and shard 3. It transferred 153.0 MiB across 72
tensors. The final three H2D samples averaged 77.976 GiB/s, all four
post-cold H2D samples averaged 77.686 GiB/s, and post-cold D2H averaged
64.622 GiB/s. Cross-shard loading completed successfully, and the protected
service returned `status: ok` afterward. This strengthens the transfer-path
result across layer and shard boundaries, but it remains a transfer-only
experiment. Raw output is preserved in
[`gmktec-evo-x2-deepseek-expert-slice-2layer-result-20260905.json`](gmktec-evo-x2-deepseek-expert-slice-2layer-result-20260905.json).

Therefore the large-model demonstrations are **not currently actionable** on
this host. A model download must not be treated as the next step. Before any
attempt, we need the exact checkpoint, quantization, required host-resident
weights, KV-cache budget, and an explicit policy for whether swap-backed
execution qualifies as interactive. A successful allocation alone would not
reproduce the paper's claim.

## Next gate

1. Obtain the exact DeepSeek-V4-Flash checkpoint metadata and file layout.
2. Compute weight, expert-cache, runtime, and KV-cache requirements from that
   metadata before downloading anything.
3. If the calculated working set exceeds available unified memory, classify the
   paper demonstration as capacity-incomplete rather than forcing a swap-heavy
   run that cannot meet the paper's interactive criterion.
4. Keep Qwen and Gemma performance optimization independent from this capacity
   gate.
