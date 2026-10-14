# Opt-in layer-major prefill

`--prefill-layer-major` changes the loop order of a long Qwen3.8 text prefill:
each decoder layer processes every internal token chunk before the next layer
starts. GPU offload can then reuse a streamed expert layer across those chunks.
The flag is disabled by default.

## Supported configuration

The initial implementation requires Qwen3.8/Qwen4Exp, TP=1, NVFP4 experts,
`--moe-strategy offload`, and **`--ple-backend pinned`**. A single pending text
request is eligible when its new-token length exceeds `--max-prefill-length`
and is at most 32,768. The existing prefill length remains the internal chunk
size. This limit bounds the activations retained between layers; it is not a
promise that every supported configuration fits every GPU.

For an existing local checkpoint, add these options to the otherwise identical
serve command used for a comparison:

```sh
--moe-strategy offload --ple-backend pinned \
--quant-backend moe.nvfp4=triton --max-prefill-length 4096 \
--prefill-layer-major
```

The default disk PLE backend is not supported by this execution path: its host
staging is sized for a token chunk. Disk or other PLE modes, multimodal inputs,
multiple pending requests, TP>1, other models or expert formats, short prefills,
and prefills above the limit retain ordinary chunked execution.

## State and numerical behavior

The outer request owns checkpoint planning. Internal chunks keep absolute
cached positions and project that plan without choosing new checkpoint slots.
Conv, recurrent and PLE state advance in chunk order within their layer. The
outer batch context is restored after normal completion and exceptions.

Numerical comparisons must fix the internal chunk structure, model, quantized
kernel and sampling parameters. This flag does not promise bitwise equivalence
between different chunk structures, unsupported runtimes, or concurrent runs.
An eligible long prefill is one scheduling turn; this implementation does not
interleave other requests' decoding between its internal chunks.

## Measuring the change

Compare `main` and the candidate with the same pinned-PLE configuration,
checkpoint, prompt bytes, 4K chunk size and output length. Record first-token
latency with its exact streaming definition, expert H2D traffic, peak VRAM,
generated token IDs and finish reason. Keep any NVFP4 kernel patch identical
on both sides; kernel speedups are not layer-major speedups.

This implementation is informed by [Layered Prefill](https://arxiv.org/abs/2510.08055).
The repeated expert-streaming problem is also described in
[#579](https://github.com/FlashML-org/FreeToken/issues/579).
[#578](https://github.com/FlashML-org/FreeToken/pull/578) is a separate kernel
optimization and is not included here.
