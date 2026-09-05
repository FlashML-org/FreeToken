# Upstream handoff checklist for the native ROCm/HIP port

This checklist is for reviewing the AMD `gfx1151` port in PR #260. It keeps
the implementation review, local reproducibility, quality evidence, and
performance claims separate.

## Source and build review

- [ ] Check out branch `amd-rocm-gfx1151` from the contributor fork.
- [ ] Use PyTorch `2.13.0+rocm10.0.0` with HIP `7.15.26333`.
- [ ] Confirm the active device reports AMD `gfx1151`.
- [ ] Confirm `torch.version.hip` selects the ROCm build path even when a CUDA
      toolkit is installed on the same host.
- [ ] Build the native host extensions and confirm they link against
      `libamdhip64` rather than `libcudart`.
- [ ] Confirm CUDA-only launch options, PTX paths, and NVIDIA capability
      probes are gated away on HIP.
- [ ] Confirm the GGUF JIT path discovers ROCm Thrust headers and the HIP
      runtime without requiring a CUDA toolkit.

## Functional validation

- [ ] Start the local OpenAI-compatible API with a supported Qwen model.
- [ ] Complete streaming and non-streaming text requests.
- [ ] Verify deterministic canary, arithmetic, JSON, multi-turn, and state
      retention controls.
- [ ] Start the Gemma 4 GGUF path in an isolated candidate process.
- [ ] Complete the arithmetic text gate and the documented image fixtures.
- [ ] Restore the normal Qwen service after candidate teardown.
- [ ] Confirm the normal service returns `status: ok` and a real completion
      with `finish_reason: stop`.

## Performance evidence boundaries

- [ ] Use the machine-readable manifest in
      `gmktec-evo-x2-cross-model-manifest-20260905.json`.
- [ ] Preserve prompt length, completion cap, warmup policy, concurrency, and
      cache state for every comparison.
- [ ] Treat client-observed prefill, decode, TTFT, and token-gap metrics as
      separate measurements.
- [ ] Do not compare cold-start TTFT with a loaded-runtime TTFT.
- [ ] For the Qwen Q4_K_M same-format control, use the exact checkpoint and
      tokenizer recorded in the manifest.
- [ ] Report the warmed requests 2 through 5 separately from the first
      request.
- [ ] Treat the current Qwen result as decode near-parity, not a material
      universal performance lead.
- [ ] Report Gemma single-client and concurrent results separately because the
      runtime ranking changes with concurrency.

## Reliability evidence

- [ ] Review the completed Qwen 1,440-session endurance artifact.
- [ ] Confirm candidate and host swap telemetry remained zero for the accepted
      endurance run.
- [ ] Confirm malformed JSON, failed quality markers, and incomplete sessions
      are absent from the accepted artifact.
- [ ] Review the bounded Gemma endurance, long-context, and concurrency
      artifacts.
- [ ] Do not reinterpret the endurance suite as a prefill-TPS measurement.

## Claim boundaries

- [ ] Keep strict NVIDIA paper comparison marked unresolved because the exact
      fixtures, policies, and reference hardware are unavailable.
- [ ] Keep 284B interactive serving marked unresolved because the exact model
      payload is absent and the available memory configuration is materially
      different from the paper system.
- [ ] Keep archived model routing names separate from admitted model payloads.
- [ ] Do not add llama-swap integration to the ROCm MVP unless separately
      reviewed and requested.
- [ ] Keep CUDA graph capture disabled on the HIP MVP unless a new quality and
      stability qualification is completed.

## Evidence index

- `gmktec-evo-x2-final-campaign-report.md`
- `gmktec-evo-x2-campaign-completion-audit.md`
- `gmktec-evo-x2-cross-model-matrix-20260904.md`
- `gmktec-evo-x2-cross-model-manifest-20260905.json`
- `gmktec-evo-x2-amd-run-log.md`
- `gmktec-evo-x2-284b-capacity-manifest-20260904.md`
- `gmktec-evo-x2-paper-model-capacity-gate.md`
- `gmktec-evo-x2-deepseek-capacity-gate-result-20260905.json`
- `gmktec-evo-x2-deepseek-expert-slice-metadata-20260905.json`
- `scripts/gmk-evo-x2/deepseek_expert_slice_benchmark.py`
- `gmktec-evo-x2-deepseek-expert-slice-result-20260905.json`
- `gmktec-evo-x2-deepseek-expert-slice-16-result-20260905.json`

The checklist is a review aid. The raw benchmark artifacts remain the
authoritative evidence for measured claims.

## Local handoff audit

The final local audit passed on 2026-09-05:

- The branch worktree is clean and synchronized with its remote branch.
- `git diff --check` passed.
- The machine-readable manifest parses as valid JSON.
- Every relative evidence link in this checklist and the final campaign report
  resolves to a tracked local file.
