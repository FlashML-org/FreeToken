# LAN-223 Qwen Q4 raw-prompt control, 2026-08-30

This report records an apples-to-apples ROCm 10 comparison between the AMD
FreeToken port and llama.cpp. It is a quality and steady-state decode control,
not a throughput claim for cold startup or a production service benchmark.

## Host and runtime

- Host: LAN-223, AMD Strix Halo `gfx1151`, 56 GiB unified GPU memory.
- FreeToken runtime: native ROCm 10 and HIP execution path, Triton attention,
  offload MoE backend, serial expert loading, Q4_K_M GGUF.
- llama.cpp runtime: ROCm 10 `llama-server`, full GPU layer offload, Flash
  Attention enabled, Q8_0 K and V cache.
- Model file: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
- Prompt renderer: the checkpoint's native Qwen tokenizer, not either server's
  chat-template implementation.

## Control contract

Each server received the exact same UTF-8 string at `/v1/completions` with
`temperature=0`, `top_p=1`, `top_k=-1`, streaming enabled, and a 1024-token
generation ceiling. The prompt SHA-256 was
`224f02631165a176e660363fefeb8eb58e5a150271fed72bdc1f90fa39448523` and each
server reported 54 prompt tokens. The shared AIME answer is `70`.

This test exists because the earlier GGUF fast-tokenizer conversion split Qwen's
`<think>` marker into three normal pieces. FreeToken now restores GGUF CONTROL
and USER_DEFINED token entries as atomic special tokens, preserving their
original vocabulary IDs and matching the checkpoint tokenizer's 54-token input.

## Results

| Engine | Prompt tokens | Generated tokens | Steady decode TPS | Quality evidence |
| --- | ---: | ---: | ---: | --- |
| FreeToken AMD | 54 | 1023 | 47.12 | Derives `b + 7` divides `56`; verifies `b=21` and `b=49` |
| llama.cpp ROCm 10 | 54 | 1024 | 50.29 | Derives the same divisibility condition and the same two bases |

FreeToken's steady decode rate is 6.3% below llama.cpp on this matched Q4
control. It must not be described as meeting or exceeding llama.cpp until a
subsequent optimization produces a measured improvement under this same
contract.

The response from each engine remained inside Qwen's verbose reasoning trace at
the 1024-token ceiling, so neither emitted the requested boxed final line. This
is not treated as a quality pass based only on formatting. The recorded math
explicitly proves the two valid bases, whose sum is 70, matching the fixed
ground truth. A future quality gate should either provide a larger token budget
or use a prompt that requests a concise answer after the reasoning trace.

## Evidence locations on LAN-223

- FreeToken: `/home/david/freetoken-amd/artifacts/qwen-gguf-raw-20260830T032253Z/raw-quality.json`
- llama.cpp: `/home/david/freetoken-amd/artifacts/qwen-llama-raw-20260830T033324Z/raw-quality.json`

The two self-restoring control runners are
`scripts/lan223/run_qwen_gguf_raw_control.sh` and
`scripts/lan223/run_qwen_llamacpp_raw_control.sh`. They reserve the GPU only
temporarily and invoke the production recovery helper on exit.
