# GMKtec EVO-X2 NVFP4 Marlin warp-8 candidate rejection

## Candidate

This isolated candidate kept the production Marlin output tile at `BLOCK_N=16`
and changed only the decode launch configuration from four warps to eight
warps. It used the same Qwen3.6 NVFP4 checkpoint, native HIP runtime, cache
policy, API workload, and deterministic quality gates as the qualified
configuration.

## Performance observation

Five fixed API samples completed without protocol errors:

```text
mean decode TPS   30.244579
median decode TPS 30.238741
stdev             0.014834
```

This is retained as an observation only because the candidate failed quality.

## Quality result

The exact canary, arithmetic, and JSON checks passed. The deterministic AIME
check failed:

```text
expected output SHA-1: cd580f4978fb
observed output SHA-1: 1cae5bae914f
observed decode TPS:   30.764005
observed TTFT:         389.508 ms
```

The raw artifacts are preserved under:

```text
/home/david/freetoken-amd/artifacts/nvfp4-marlin-api-candidate-20260904T182532Z/
```

## Recovery verification

The isolated process stopped normally. The recovery launcher restored the
production setting:

```text
_DECODE_MARLIN_WARPS = 4
```

The protected Qwen service returned `status: ok` and
`maintenance: serving` after recovery.

## Decision

**Rejected.** Changing only the warp count still changed the deterministic
model output, so the observed 30.2446 TPS cannot be accepted as a
quality-preserving improvement. Future candidates must retain both tile and
warp geometry unless their numerical effects are explicitly understood and
the complete quality suite passes.
