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
