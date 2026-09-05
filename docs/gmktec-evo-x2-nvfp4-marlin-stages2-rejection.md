# GMKtec EVO-X2 NVFP4 Marlin `num_stages=2` candidate rejection

## Candidate

This isolated candidate kept the Marlin tile (`BLOCK_N=16`), warp count (4),
and reduction expression unchanged. It changed only Triton's launch staging
parameter by adding `num_stages=2` to the production Marlin kernel launch.

## Performance observation

Five fixed API samples completed without protocol errors:

```text
mean decode TPS   30.037302
median decode TPS 30.036226
stdev             0.006263
```

The values are retained as diagnostic observations only because the candidate
failed the deterministic quality gate.

## Quality result

The exact canary, arithmetic, and JSON checks passed. The deterministic AIME
hash failed:

```text
expected output SHA-1: cd580f4978fb
observed output SHA-1: 1cae5bae914f
```

Raw artifacts are preserved under:

```text
/home/david/freetoken-amd/artifacts/nvfp4-marlin-api-candidate-20260904T184547Z/
```

## Recovery

The candidate stopped normally. The wrapper restored the original kernel file,
and the protected Qwen service returned `status: ok` with
`maintenance: serving`.

## Decision

**Rejected.** Even a launch-staging-only change altered deterministic output.
The current four-warp, default-staging Marlin configuration remains the
qualified baseline. Future optimization must avoid changing Triton execution
schedule unless its numerical consequences are fully controlled.
