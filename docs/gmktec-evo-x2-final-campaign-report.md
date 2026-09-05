# GMKtec EVO-X2 native ROCm FreeToken campaign report

## Executive result

The native ROCm and HIP port is functional and quality-qualified on the
GMKtec EVO-X2 with an AMD Radeon 8060S `gfx1151` GPU. The port serves Qwen
text and Gemma 4 GGUF workloads through an OpenAI-compatible local API. The
controlled Qwen Q4_K_M same-format decode result is effectively at parity with
the ROCm 10 llama.cpp control. Gemma FreeToken is slower than llama.cpp for
isolated single-request decode and long-prefill work, but it has lower TTFT and
higher aggregate throughput at the tested four- and eight-client loads.

This report does not claim strict parity with the published NVIDIA results.
The paper does not expose every required fixture and policy field, and no
reference NVIDIA system is part of this campaign.

## Platform and build

| Field | Value |
| --- | --- |
| Host | GMKtec EVO-X2 |
| GPU | AMD Radeon 8060S |
| GFX target | `gfx1151` |
| ROCm | 10.0 |
| HIP | 7.15.26333 |
| PyTorch | `2.13.0+rocm10.0.0` |
| Execution mode | Native ROCm/HIP, no CUDA compatibility fallback |
| API | OpenAI-compatible local HTTP API |
| llama.cpp control | ROCm 10 build on the same host |

## Qualified functionality

- Native HIP extension build and import succeeded.
- CUDA-only capability detection and launch options are gated away on HIP.
- Qwen text streaming and non-streaming requests passed.
- Gemma 4 text and multimodal image controls passed.
- Deterministic text, JSON, multi-turn, long-context, and visual checks passed
  within their documented scopes.
- The protected normal Qwen service was restored and health-checked after
  every isolated candidate run.

## Performance evidence

### Qwen Q4_K_M same-format control

Both runtimes used the same Q4_K_M GGUF checkpoint, tokenizer, caller-rendered
54-token raw prompt, and 256-token completion cap.

| Runtime | Mean decode all samples | Mean decode samples 2 to 5 | Mean TTFT samples 2 to 5 |
| --- | ---: | ---: | ---: |
| FreeToken ROCm/HIP | 48.6028 TPS | **49.4357 TPS** | 424.26 ms |
| llama.cpp ROCm 10 | **49.1155 TPS** | 49.1772 TPS | 58.83 ms |

FreeToken is 0.53 percent faster on the warmed requests 2 through 5. Across
all five samples, llama.cpp is 1.06 percent faster because FreeToken's first
request is slower. This is decode near-parity, not a material FreeToken lead.
The TTFT values have different cache behavior and are not an apples-to-apples
latency claim.

Evidence:

- FreeToken: `/home/david/freetoken-amd/artifacts/qwen-gguf-warm-matrix-20260905T122817Z`
- llama.cpp: `/home/david/freetoken-amd/artifacts/qwen-llama-warm-matrix-20260905T124007Z`

### Gemma 4

The five-sample matched text control measured 53.0762 TPS for FreeToken and
56.8293 TPS for llama.cpp. The long 544-token prompt control measured 48.1441
TPS for FreeToken and 54.3743 TPS for llama.cpp. FreeToken's concurrent
aggregate decode was 4.1 percent higher at four clients and 25.7 percent
higher at eight clients, with substantially lower mean TTFT in both cases.

The exact values, prompt contracts, quality status, and raw artifact paths are
in [`gmktec-evo-x2-cross-model-manifest-20260905.json`](gmktec-evo-x2-cross-model-manifest-20260905.json).

## Reliability and endurance

The Qwen Q5 endurance campaign completed exactly 1,440 minute-cadence session
records with zero candidate and host swap, valid JSON, passing deterministic
state checks, and successful normal-service recovery. This campaign measures
state correctness, swap, thermal state, TTFT, and token-gap behavior. It does
not claim per-session prefill TPS.

Gemma has completed bounded 30-session endurance, long-context, multimodal,
and concurrency controls. A full 1,440-session Gemma campaign remains
optional publication evidence and is not required for the current functional
release gate.

## Rejected optimization candidates

The following candidates were tested and not promoted because they regressed
quality, stability, or the primary throughput target:

- Grouped Q4 and Q5 numerical paths with non-identical real-weight output.
- Python-level prefill overlap.
- Larger Gemma memory ratio.
- NVFP4 Marlin tile, warp-count, and staging variants that changed the
  deterministic AIME result.
- One-fetch hybrid expert transfer.
- Six-request and eight-request scheduler alternatives as universal defaults.

All rejected candidates retain raw artifacts and recovery evidence.

## Unresolved claims

### Strict NVIDIA comparison

Not proven. The paper does not publish every input fixture, harness policy,
trace, and scoring detail required for strict reproduction, and no reference
NVIDIA hardware is in scope.

### 284B interactive serving

Not qualified. The exact 284B checkpoint is not present in the configured
model directory. The live capacity audit recorded 59 GiB system memory, 18
GiB available at capture, 2 GiB dedicated VRAM reported by ROCm, and no 284B
payload. The primary paper describes DeepSeek-V4-Flash as a 284B model with
about 13B active parameters, six selected experts from 256, and roughly 140 GB
of FP4 expert weights, demonstrated on a 32 GB RTX 5090-class GPU. Those facts
make the exact model payload, host-memory budget, and bandwidth behavior
mandatory evidence before a Strix Halo reproduction can be claimed. See the
[primary paper](https://arxiv.org/abs/2608.16157) and the local
[`284B capacity manifest`](gmktec-evo-x2-284b-capacity-manifest-20260904.md).

The paper reports approximately 53.8 GB/s host bandwidth on its 16-core
desktop reference. Our local transfer prototypes measured 79.79 GB/s for
contiguous 64 MiB copies, 5.009 GB/s for serialized random 64 KiB blocks, and
12.84 GB/s for grouped expert-like rounds after CPU staging. These are useful
transport bounds, but they are not evidence that the full 284B model will fit
or reach paper throughput. The scattered and staged measurements show why a
real checkpoint and production-shaped expert access pattern are still required.

There is also a checkpoint identity gate. The current official
`deepseek-ai/DeepSeek-V4-Flash-0731` model card lists 304B parameters and
BF16, I64, F32, and F8_E4M3 tensor types. It is not automatically the same
artifact as the paper's 284B FP4 deployment. A valid reproduction therefore
needs the exact 284B checkpoint or an authoritative conversion recipe before
any capacity or throughput claim is made.

As a separate current-release reference, a read-only metadata refresh of
`DeepSeek-V4-Flash-0731` at repository commit
`7872f01b1d1fe23eabc4c98b48bffcef5a386062` found 48 safetensors shards totaling
166,886,535,336 bytes, approximately 155.43 GiB. This is a capacity reference
for the later official release, not proof that it is the paper's exact 284B
artifact.

### Additional archived model identifiers

Archived routing names without an admitted payload are not treated as failed
FreeToken benchmarks. Each would require an exact checkpoint, backend
admission, deterministic quality fixture, TPS matrix, and recovery evidence.

## Reproduction index

- [`gmktec-evo-x2-upstream-handoff-checklist.md`](gmktec-evo-x2-upstream-handoff-checklist.md)
- [`gmktec-evo-x2-campaign-completion-audit.md`](gmktec-evo-x2-campaign-completion-audit.md)
- [`gmktec-evo-x2-cross-model-matrix-20260904.md`](gmktec-evo-x2-cross-model-matrix-20260904.md)
- [`gmktec-evo-x2-cross-model-manifest-20260905.json`](gmktec-evo-x2-cross-model-manifest-20260905.json)
- [`gmktec-evo-x2-amd-run-log.md`](gmktec-evo-x2-amd-run-log.md)
- [`gmktec-evo-x2-284b-capacity-manifest-20260904.md`](gmktec-evo-x2-284b-capacity-manifest-20260904.md)

## Review status

The source branch containing the native ROCm/HIP implementation and this
evidence set is the branch proposed in upstream PR #260. The PR remains open.
Merge status and external maintainer review are separate from the completed
local AMD qualification gates. The latest upstream page still shows the PR as
open. The two Copilot build-detection findings are marked outdated and the
branch records their fix in commit `54d6ab2`; no new actionable review request
was visible during the latest handoff check.
