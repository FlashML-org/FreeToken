# Q4 MMV_Y=4 promotion record

This record documents the reproducible FreeToken Q4 candidate measured on the
GMKtec EVO-X2. It is an evidence record, not a claim that the candidate has
already replaced the protected service configuration.

## Candidate identity

- Candidate source revision: `fb4e0232dbd7804b7d86c1ddd2dd366e2b0c05a7`.
- Model file: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
- Model SHA-256: `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`.
- GPU: AMD Radeon 8060S Graphics, architecture `gfx1151`.
- PyTorch: `2.13.0+rocm10.0.0`.
- HIP runtime reported by PyTorch: `7.15.26333`.
- Python: `3.12.13`, Clang-backed environment.
- Reusable extension cache: `/home/david/freetoken-amd/cache/torch_extensions-q8-api-y4`.

## Runtime configuration

The candidate was started on an isolated loopback port with the following
performance-affecting settings:

```text
FREETOKEN_GGUF_MMV_Y=4
FREETOKEN_GGUF_Q8_MMV_WARPS=1
PYTORCH_ROCM_ARCH=gfx1151
ROCM_HOME=/opt/rocm-10.0
ROCM_PATH=/opt/rocm-10.0
HIP_PATH=/opt/rocm-10.0
--attention-backend triton
--moe-backend offload
--nvfp4-backend triton
--expert-load serial
--moe-cache-auto
--memory-ratio 0.25
--max-seq-len-override 8192
--kv-reserve-tokens 8192
--cuda-graph-max-bs 0
--disable-pynccl
--disable-moe-prefill-overlap
```

The Q8 warp count remains one because the current implementation deliberately
rejects multiwarp Q8 values. The accepted experiment changes `MMV_Y`, which is
the supported dense activation launch geometry.

## Acceptance evidence

The first five-sample API run is preserved at:

`/home/david/freetoken-amd/artifacts/qwen-q4-mmv-y4-api5-20260905T075002Z`

Its mean decode rate was 48.03 TPS, with five successful samples and a
standard deviation of 0.054 TPS. The independent repeat is preserved at:

`/home/david/freetoken-amd/artifacts/qwen-q4-mmv-y4-repeat-20260905T082425Z`

The repeat measured 47.86 TPS across five successful samples, with a standard
deviation of 0.092 TPS. The matched ROCm10 llama.cpp control measured 48.75
TPS, so the repeat was approximately 1.8 percent slower in decode.

The quality and state evidence is preserved at:

`/home/david/freetoken-amd/artifacts/qwen-q4-mmv-y4-quality-20260905T080132Z`

The deterministic suite passed its exact, arithmetic, and JSON cases. The
three-turn state suite passed acknowledgment, recall, and transformation.

The long-context and resource evidence is preserved at:

`/home/david/freetoken-amd/artifacts/qwen-q4-mmv-y4-longctx-20260905T081222Z`

Five nonce-varied 6,056-token prompts passed exact marker retrieval. Available
memory changed from 19 GiB to 18 GiB. Swap use decreased from 2.1 GiB to 710
MiB. GPU temperature changed from 35 C to 54 C, GPU use reached 91 percent,
and measured power reached 110 W.

## Promotion decision

The candidate is reproducible and passes the current functional, quality,
long-context, and resource gates. It should be treated as the leading Q4
optimization candidate, but not silently installed as the protected default.
Before permanent promotion, rerun the documented commands from a clean shell,
verify the reusable extension cache contents, and attach the resulting build
log and final source diff to the upstream review record.

## Clean-shell reproduction

The fresh-cache reproduction was completed from a clean remote shell using the
same source revision, ROCm 10.0 paths, `gfx1151` target, `MMV_Y=4`, and Q8
one-wave guard. The native build invoked `/opt/rocm-10.0/bin/hipcc` and emitted
the expected `-DGGML_CUDA_MMV_Y=4` and `--offload-arch=gfx1151` flags.

- Artifact: `/home/david/freetoken-amd/artifacts/qwen-q8-mmv-y4-clean-20260905T084057Z`.
- Real-weight Q8 screen: 25.983 microseconds mean over 300 repetitions.
- Device and software identity matched the candidate record.
- Build and benchmark output is preserved in `build-and-bench.log`.

## Known limitations

The client-visible prefill rate remains far below the llama.cpp control even
though decode throughput is nearly matched. The long-context test is a local
6,056-token retrieval control and is not a reproduction of the paper's agent
workload. The quality suite is deterministic and bounded; it does not replace
the paper's full tool-using evaluation.
