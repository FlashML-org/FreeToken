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

## Paper-model decision

The official DeepSeek model card identifies DeepSeek-V4-Flash as 284B total
parameters and 13B activated parameters, with FP4 plus FP8 mixed precision.
The paper's prefill discussion describes roughly 140 GB of routed expert
weights. The model card also lists the safetensors repository as approximately
291B parameters and identifies the official local deployment path. The paper
describes GLM-5.2 as a 753B-parameter model with a 433 GB checkpoint. Neither
payload is installed on this host, and the live available-memory observation
is far below either stated payload scale.

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
