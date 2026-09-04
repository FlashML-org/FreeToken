# GMKtec EVO-X2 NVFP4 Marlin warp-count candidate rejection

## Candidate

This isolated candidate retained the production Marlin output tile
(`BLOCK_N=16`) and changed only the decode launch warp count from four to eight.
It used the same Qwen3.6 NVFP4 checkpoint, native HIP runtime, offload cache,
API workload, and deterministic quality gates as the qualified control.

## Performance observation

Five fixed API samples completed without protocol errors:

```text
mean decode TPS   30.244579
median decode TPS 30.238741
stdev             0.014834
```

The AIME request itself reported 30.764005 decode TPS and 389.508 ms TTFT.
These numbers are retained as diagnostic observations only.

## Quality result

The exact canary, arithmetic, and JSON checks passed. The deterministic AIME
hash failed:

```text
expected output SHA-1: cd580f4978fb
observed output SHA-1: 1cae5bae914f
```

The raw artifacts remain under:

```text
/home/david/freetoken-amd/artifacts/nvfp4-marlin-api-candidate-20260904T182532Z/
```

## Recovery

The candidate process stopped normally. The recovery launcher restored the
production setting:

```text
_DECODE_MARLIN_WARPS = 4
```

The protected Qwen service returned `status: ok` and
`maintenance: serving` after recovery.

## Decision

**Rejected.** Changing only the warp count changes the deterministic model
output. The measured speed increase is not admissible as a quality-preserving
optimization. Future work must preserve the current launch geometry and target
memory scheduling or orchestration overhead instead.
