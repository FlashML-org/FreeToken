# GMKtec EVO-X2 NVFP4 Marlin tile-8 candidate rejection

## Candidate

The isolated candidate changed the production Marlin decode output-row tile
from `BLOCK_N=16` to `BLOCK_N=8`. It used the same native ROCm/HIP runtime,
Qwen3.6 NVFP4 checkpoint, offload policy, cache sizing, and API benchmark as
the qualified control. The protected service was stopped only after its
identity and health were verified, and the launcher restored it in its exit
path.

## Performance observation

Five fixed API samples completed without protocol errors:

```text
mean decode TPS   29.835737
median decode TPS 29.828223
stdev             0.013950
```

These numbers are retained as an observation only. They are not an accepted
performance result because the candidate failed the required quality gate.

## Quality result

The exact canary, arithmetic, and JSON checks passed. The deterministic AIME
check failed:

```text
expected output SHA-1: cd580f4978fb
observed output SHA-1: 1cae5bae914f
observed decode TPS:   30.398296
observed TTFT:         396.048 ms
```

The candidate stopped after the failure, preserving the raw benchmark and
quality artifacts under:

```text
/home/david/freetoken-amd/artifacts/nvfp4-marlin-api-candidate-20260904T180912Z/
```

## Recovery verification

The launcher restored the original source file, leaving:

```text
_DECODE_MARLIN_BLOCK_N = 16
```

The protected Qwen service then returned:

```json
{"status":"ok","maintenance":"serving"}
```

## Decision

**Rejected.** The tile-8 candidate is not eligible for API promotion. The
throughput observation is useful for diagnosis, but deterministic quality is a
hard gate and the changed output hash demonstrates that this tile configuration
cannot be used as a like-for-like optimization.
