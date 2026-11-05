# Optional HIP FP8 decode kernels on R9700

`FREETOKEN_USE_ARK=1` adds two native HIP candidates on Linux gfx1201:
block-FP8 dense decode and block-FP8 routed MoE decode. The default is `0`.
Both candidates retain independent FT fallbacks. The packaged profile enables
only measured eager shapes; Graph entries remain empty.

## Operator contracts

- Dense: flattened M=1, BF16 activations, E4M3 weights and BF16 128x128 block
  scales. W8A16 preserves leading dimensions and optional bias. M>1, including
  batched decode, retains FT W8A8.
- MoE: M=1, plain SiLU, BF16 activations, E4M3 weights, padded BF16 block
  scales, contiguous INT32 slots and FP32 routing weights. W8A16 preserves
  inputs and returns a new BF16 tensor. Prefill and unsupported configurations
  retain FT.
- The `LinearKernel` / `MoEKernel` subclasses inherit FT packing, layouts and
  CPU formats and consume existing `ExpertView` banks. FTW remains compatible
  with `fp8_block` / `triton` banks. CPU experts, hybrid routing, shared experts
  and cache ownership remain with FT.
- `supports_inactive_slots=False` preserves FT's safe slot-zero/zero-weight
  routing. Explicit requests for the `triton` backend retain FT.

## Build and selection

Install in an existing ROCm environment with `uv pip install -e .`. The package
includes both HIP translation units, the shared header, Torch bridge and shape
profile. First enabled use compiles into `$FREETOKEN_ARK_CACHE_DIR`, defaulting
to `$XDG_CACHE_HOME/freetoken/ark` (or `~/.cache/freetoken/ark`). Build keys include
native source, compiler, build recipe and Torch/HIP versions. No external ARK
workspace, generic `common` package or precompiled ARK binary is required.

Selection checks GPU/architecture, visible device count, Torch/HIP/Triton,
compiler, platform, ARK source and FT primitive fingerprints. Unmatched
settings, shapes, dtypes, layouts and modes retain FT. This profile applies to
the measured R9700 environment, not to every Radeon device or software stack.

```bash
export ROCR_VISIBLE_DEVICES=0
export HIPCC=/opt/rocm/bin/hipcc
export FREETOKEN_ARK_CACHE_DIR=/tmp/freetoken-ark
FREETOKEN_USE_ARK=1 ft serve --model-path /path/to/Qwen3-30B-A3B-FP8-FTW --attention-backend triton --moe-strategy offload --cuda-graph-max-bs 0
```

Set `FREETOKEN_USE_ARK=0` for the FT control. Default validated selection uses
`common/validated_shapes.json`; `FREETOKEN_ARK_PROFILE` can name a separately
calibrated profile. `FREETOKEN_ARK_POLICY=native` bypasses the measured table for
operator validation and is not the production selection policy.

## Validation and recalibration

```bash
python -m pytest tests/kernels/test_ark_fp8.py -q
python -m benchmarks.ark.calibrate --output /tmp/ark-calibration.json --write-profile python/freetoken/kernel/ark/common/validated_shapes.json
```

The focused tests verify outputs, dtype/shape, input and alias contracts,
repeated/zero/inactive routes, padded scales, bias, nondefault streams, updated
inputs during direct Graph replay, FT fallback, bank packing, `Kernel.apply`
and FTW selection. Native-hit tests observe the actual Torch operator call.
Direct Graph API tests do not enable Graph in the model profile.

The calibration benchmark covers only these two operators, records every
paired eager/Graph repeat and checks numerical error against FT. Eager entries
require at least 1.05x median speedup and every paired repeat to be faster.
Graph entries stay empty; operator timing alone cannot qualify a model path.
Changing source or runtime requires recalibration rather than removing checks.

Model validation uses Qwen/Qwen3-30B-A3B-FP8 (local FTW), BF16 activations/KV,
one R9700, offload, 20GiB expert cache, 512/32768 input tokens and 256 generated
tokens. Updated A/B results and exact validation commands are in the PR body.
This PR does not claim a Graph or sustained-CPU hybrid speedup.
