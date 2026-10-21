# Optional ARK FP8 decode kernels on Radeon

`FREETOKEN_USE_ARK=1` adds native HIP candidates for block-FP8 dense and MoE
**decode** on Linux gfx1201. The default is `0`. Other operations retain FreeToken.

The packaged profile is specific to R9700, Torch 2.14.0+rocm10.1.0, HIP
7.16.26385, Triton 3.8.0, the compiler, host platform and source fingerprints.
An unmatched environment or unlisted shape uses FT. This is not an all-Radeon
support declaration. Recalibrate on another software stack; do not strip the
environment check.

## Integration

`ArkFp8MoEKernel` and `ArkFp8LinearKernel` inherit the existing Triton kernels'
packing, layouts and CPU formats. MoE consumes the current `ExpertView` directly,
including resident, materialized and cache-slot banks. The FTW format remains
`fp8_block`; reverse lookup still returns the original `triton` kernel. CPU expert
execution, hybrid routing, shared experts, cache management and upstream locked
expert fetching are owned by FT.

- MoE: BF16 activations, E4M3 weights, BF16 padded 128x128 block scales,
  contiguous INT32 slots/FP32 routing weights, M=1 and plain SiLU. It is W8A16.
  Other activations, router-weight placement, dtypes, shapes and prefill use FT.
  Inputs are preserved and the BF16 result is a new tensor.
- Dense: flattened M=1 uses W8A16, preserving leading dimensions and optional
  bias. M>1, including batched decode, remains the original W8A8 FT path.
  Unsupported layouts, scales and shapes use FT.
- `supports_inactive_slots=False` preserves FT's safe slot-zero/zero-weight
  contract on hybrid routes. Native negative slots are covered only by direct
  boundary tests; the default integration does not advertise that capability.
- The original primitives are independent fallback targets. No FT function is
  overwritten by the production adapters.
- Every MoE Graph shape remains FT. A prior Qwen3 matched-prefix model probe
  exceeded its 0.05-nat KL gate with native MoE Graph; operator tolerance alone
  was insufficient. Native Graph correctness tests are separate from permission
  to enable it in a model. Dense Graph also remains FT: its three-round model A/B did not demonstrate a benefit.
  Native Graph correctness tests are retained for future qualification.

## Build and use

In an existing ROCm FreeToken environment, use `uv pip install -e .` (or install
the wheel). HIP/C++ sources and the profile are included in the package.
The first enabled use builds the two HIP translation units and Torch bridge in
`$FREETOKEN_ARK_CACHE_DIR`, or `$XDG_CACHE_HOME/freetoken/ark`. Build keys include
native source, build recipe, compiler version and Torch/HIP versions. The exact
ARK Python/native and baseline source fingerprints also gate the profile.
No experiment directory, external `common` package or prebuilt ARK library is
required. The engine loads the library before Graph capture.

```bash
export ROCR_VISIBLE_DEVICES=0
export HIPCC=/opt/rocm/bin/hipcc
export FREETOKEN_ARK_CACHE_DIR=/tmp/freetoken-ark
FREETOKEN_USE_ARK=1 ft serve --model-path /path/to/Qwen3-30B-A3B-FP8-FTW --attention-backend triton --moe-strategy offload --cuda-graph-max-bs 0
FREETOKEN_USE_ARK=0 ft serve --model-path /path/to/Qwen3-30B-A3B-FP8-FTW --attention-backend triton --moe-strategy offload --cuda-graph-max-bs 0
```

Expose one R9700 (the profile requires one visible GPU).

Existing explicit FT backend requests, e.g. `moe.fp8_block=triton`, keep selecting
FT. `FREETOKEN_ARK_TRACE=/tmp/dispatch-{pid}.json` writes operator, shape, backend,
mode and Python call counts. Graph capture counts are not replay counts; inspect
a GPU trace for replay execution. Formal performance runs omit this tracing.

## Reproduce validation and profile selection

```bash
python -m pytest tests/kernels/test_ark_fp8.py -q
python -m benchmarks.ark.calibrate --output /tmp/ark-calibration.json --write-profile python/freetoken/kernel/ark/common/validated_shapes.json
python -m benchmarks.ark.model_ab --model-path /path/to/Qwen3-30B-A3B-FP8-FTW --output /tmp/ark-eager --gpu-pci 0000:23:00.0 --memory-nodes 2-3 --rounds 3
python -m benchmarks.ark.model_ab --model-path /path/to/Qwen3-30B-A3B-FP8-FTW --output /tmp/ark-graph-candidate --profile docs/benchmarks/ark-fp8-decode-20261021/graph-candidate-profile.json --gpu-pci 0000:23:00.0 --memory-nodes 2-3 --rounds 3 --graph 1
python -m benchmarks.ark.model_ab --model-path /path/to/Qwen3-30B-A3B-FP8-FTW --output /tmp/ark-probe --gpu-pci 0000:23:00.0 --memory-nodes 2-3 --rounds 1 --probe
```

Use an idle GPU and appropriate CPU/memory affinity for your host. The calibration
command intentionally forces native candidates for measurement. Serving uses
`FREETOKEN_ARK_POLICY=validated` (default); `native` is a calibration/debug override,
not a validated serving policy. `baseline` forces FT fallbacks. An external
`FREETOKEN_ARK_PROFILE` can hold a separately measured table. The generator keeps all Graph entries disabled by default. `--graph-candidates`
can emit dense-only microbenchmark candidates for a separate model evaluation;
it does not qualify them for production. MoE Graph always remains disabled.

The tests cover numerical error, dtype/shape/alias and input preservation,
repeated/zero/inactive routes, padded scale strides, finite E4M3 codes, bias,
leading dimensions, non-default streams, Graph replay with changed inputs/routes,
actual method selection/packing/Kernel.apply, FTW compatibility, explicit FT
selection, disabled mode and environment mismatch.

The model harness uses fresh off/on services, alternating off/on order across rounds,
warmup per context, no prefix reuse, 512/32768 input tokens and fixed 256 output
tokens. Its server command, SSE events, usage, per-round metrics and resources
are saved. Decode speed is 255 divided by the span between first and last nonempty
text events; if events differ from tokens, the record labels this a client timing
estimate. TTFT includes request/frontend work. These are concurrency-one client
measurements, not aggregate throughput.

`--probe` is separate and untimed: it records real GPU kernel symbols, dispatch
counts, same-input per-layer comparisons and eight bounded generation checks.
It is a diagnostic wrapper; production code does not monkeypatch FT methods.
Numerical tolerance and bounded generation checks are not a standardized model
accuracy certification.

Source provenance: [AMD-AIM/amd_radeon_kernels](https://github.com/AMD-AIM/amd_radeon_kernels)
revision `975e8cd9918ec1365d3f275d1259d215315f505a`; see the packaged ARK `NOTICE`.
Related: [AMD roadmap #541](https://github.com/FlashML-org/FreeToken/issues/541).

## This calibration (2026-10-21)

Fresh build against upstream `fe5a5ec9c5644e266527de5afb9f1156b9bdcabb`.
GPU `23:00.0` is R9700 32 GiB, PCIe 32 GT/s x16. CPU affinity selects one
EPYC 9334 socket (`0-31,64-95`) of a dual-socket host, memory interleave `2-3`.
Ubuntu 24.04.1, kernel 7.0.0-38, amdgpu 7.1.9.31600000; remaining environment
fingerprints are in [runtime.json](benchmarks/ark-fp8-decode-20261021/runtime.json).

Each entry retains five alternating-order paired repeats. Eager timings include
Python dispatch, checks, allocation and the complete GPU operation. Graph timings
include the captured operation, using GPU events over replay; Python capture cost
is excluded. No activation quantization is omitted: these M=1 paths are W8A16.

| Operator / shape | Eager FT / native, us | Eager speedup | Graph FT / native, us | Deployed Graph selected? |
|---|---:|---:|---:|---|
| dense_fp8_decode / N5120_K2048 | 42.737 / 14.336 | 2.981x | 15.091 / 12.952 | FT (model veto) |
| dense_fp8_decode / N2048_K4096 | 43.586 / 16.359 | 2.664x | 13.675 / 16.654 | FT |
| dense_fp8_decode / N151936_K2048 | 581.642 / 500.559 | 1.162x | 580.391 / 499.153 | FT (model veto) |
| moe_decode_fp8 / C32_H2048_I768 | 105.468 / 45.588 | 2.314x | 53.397 / 44.586 | FT |
| moe_decode_fp8 / C4550_H2048_I768 | 104.505 / 43.965 | 2.377x | 54.333 / 40.295 | FT |

All five cases passed the numerical gate (NRMSE < 0.03, cosine > 0.999 and a
bounded maximum error). Observed NRMSE was 0 for QKV, output projection and both
MoE bank sizes; the lm-head-shaped case was 0.00000534. The tests include bias and
boundary cases separately. The lm-head shape is an operator benchmark, not a
claim that this model's lm head actually selected native FP8.

[Raw calibration](benchmarks/ark-fp8-decode-20261021/calibration.json) includes
all repeats and the negative Graph result. Its `enabled` fields are the initial
operator-level candidates; subsequent model evaluation vetoed all Graph entries.
The packaged profile is the deployed decision table. No speedups from the original
experimental report are reused. A different R9700 was used because the prior
measurement GPU was busy; the new off/on model comparison uses this same card.

## Model results and real-call evidence

Three-round medians, fixed 256 output tokens, same model/prompt/settings per pair.
Baseline uses the original FT compute path with `FREETOKEN_USE_ARK=0`.
Formal timing excludes instrumentation. The Graph rows evaluate the **rejected
candidate profile**, not the deployed all-FT Graph policy.

| Mode | Input tokens | FT / ARK tok/s | Change | Paired wins | FT / ARK TTFT, s |
|---|---:|---:|---:|---:|---:|
| eager | 512 | 29.454 / 34.838 | +18.28% | 3/3 | 0.6517 / 0.6866 |
| eager | 32768 | 28.366 / 36.337 | +28.10% | 3/3 | 10.8668 / 10.8626 |
| Graph candidate (rejected) | 512 | 100.180 / 100.099 | -0.08% | 1/3 | 0.7239 / 0.7942 |
| Graph candidate (rejected) | 32768 | 65.221 / 64.977 | -0.37% | 0/3 | 10.9913 / 10.9634 |

**The recommended enabled scope is eager. Both deployed Graph tables are empty.**
512-token Graph samples vary between about 53 and 101 tok/s; 32K candidate
Graph was slightly slower in all three pairs. No Graph model gain is claimed.
The initial operator-only candidate profile is preserved in
[graph-candidate-profile.json](benchmarks/ark-fp8-decode-20261021/graph-candidate-profile.json).
The model harness's `--profile` flag reproduces that candidate without changing
the packaged production table.

- Final focused and related tests: **41 passed, 42 skipped** (23 ARK cases;
  skips mostly require other checkpoint configurations/weights).
- Real model eager trace contains `dense_gemv_fp8_rows<4>`, `gate_up<true, 4>`
  and `down_reduce<true, 4>`. In each context's eight-token diagnostic request,
  seven decode steps hit native dense **672** times and native MoE **336** times.
  Native shapes are dense `(M,N,K)=(1,5120,2048)/(1,2048,4096)` and MoE
  `(M,H,I,bank_rows,top_k)=(1,2048,768,4550,8)`.
- **144** real-input per-layer shadows passed, maximum NRMSE **0.000734784**.
- Deployed Graph replay probe: **zero ARK GPU kernels**, and eight-token outputs match
  FT exactly for both input contexts. The server's eager warmup still hits native
  kernels; these are not Graph capture/replay hits. Full fallback is not counted as ARK gain.
- Corrected bounded generation suite: FT **7/8**, ARK **7/8**, no new failure.
  Both incorrectly answer `0.11` for the decimal comparison. Initial 64-token
  thinking-mode probes were truncated and excluded; the scored run uses the
  checkpoint's non-thinking template, 128 output tokens and the first answer.
- Standalone wheel passed with `python -I`, the repo removed from `sys.path`,
  new native/Triton caches and validated policy. All 17 ARK packaged files match
  this tree; the enabled dense kernel was hit and NRMSE was **0**.

[Per-round model samples](benchmarks/ark-fp8-decode-20261021/model-samples.csv)
and [validation evidence](benchmarks/ark-fp8-decode-20261021/validation.json)
retain positive/negative outcomes and exact native symbol names. Complete raw
SSE, resource records, traces and layer/answer comparisons are produced by the
commands above. Validate raw timing with `python -m benchmarks.ark.audit PATH`;
for the generation checks use:

```bash
python -m benchmarks.ark.model_ab --model-path /path/to/Qwen3-30B-A3B-FP8-FTW --output /tmp/ark-quality --gpu-pci 0000:23:00.0 --memory-nodes 2-3 --rounds 1 --probe --quality-only
python -m benchmarks.ark.audit_quality /tmp/ark-quality
```

The CPU is shared; all slow samples are retained. FT and native generate different
continuations, changing subsequent expert-cache traffic. These are configuration
A/B measurements, not fixed-token kernel attribution or a universal speed guarantee.
Prefill compute is unchanged, so no TTFT acceleration is claimed. Only this FP8
Qwen3 offload model was measured end to end. Full-model resident/hybrid performance,
other checkpoints, standardized accuracy and NVIDIA execution remain unverified;
operator/bank tests do not substitute for those results.
