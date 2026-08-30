# LAN-223 Gemma 4 Q4 GGUF vision control

## Scope

This control proves the AMD ROCm/HIP path for the locally available
`gemma-4-26B_q4_0-it.gguf` plus its sibling
`gemma-4-26B-it-mmproj.gguf` projector. It is isolated from the protected Qwen
service: the candidate binds only `127.0.0.1:1923`, and the runner restarts
Qwen on `127.0.0.1:1919` on every exit path.

## Build and runtime contract

- Host: LAN-223, Radeon 8060S (`gfx1151`) unified-memory GPU.
- Backend: native ROCm/HIP and Triton. No CUDA compatibility path was used.
- Text GGUF: `gemma-4-26B_q4_0-it.gguf`.
- Vision projector: sibling `gemma-4-26B-it-mmproj.gguf`.
- Opt-in: `FREETOKEN_LOAD_VISION=1`.
- Vision geometry recovered from the projector and Gemma 4 release contract:
  27 layers, hidden width 1152, 16 heads, MLP width 4304, 16-pixel patches,
  10,240 position entries, 3 by 3 pooling, and at most 280 soft tokens per
  image.
- Image API: OpenAI-compatible `messages[].content[]` with `type: image_url`.
  The initial local-safe implementation accepts `data:image/...;base64,...`
  values. It intentionally rejects remote URLs, preventing the serving process
  from becoming an arbitrary LAN or Internet fetch client.

## Evidence

Latest artifact directory on LAN-223:

`/home/david/freetoken-amd/artifacts/gemma4-gguf-vision-20260830T045559Z`

The runner completed both controls before it shut down the candidate and
started Qwen recovery.

| Control | Result | Prompt tokens | Completion tokens | Observed latency or rate |
| --- | --- | ---: | ---: | --- |
| Text arithmetic | `323` | 30 | 4 | TTFT 2471.37 ms, 45.52 decode tok/s across two decode steps |
| Solid red PNG data URL | `red` | 284 | 2 | 1.89 s end-to-end request time |
| Solid green PNG data URL | `green` | 284 | 2 | 1.08 s end-to-end request time |
| Red-left, blue-right PNG | `red` for the left half | 282 | 2 | 1.06 s end-to-end request time |

The image prompts had 282 to 284 tokens because the processor produced 256
image soft tokens, plus the rendered text/template tokens. All three controls
returned their expected one-word answer. The spatial split-color control shows
that the path preserves image position rather than merely detecting a dominant
global color. Together they verify decoding, resizing, patchification, shaped
inter-process tensor transport, ROCm vision-tower execution, projector
execution, image-token replacement, and OpenAI response formatting.

## Reproduction

From the isolated checkout on LAN-223, first ensure the protected server health
is exactly `status: ok`, then run:

```bash
bash scripts/lan223/run_gemma4_gguf_text_control.sh \
  /home/david/freetoken-amd/validation-qwen-gguf-d1dd473 vision
```

The control runner saves `quality.json` for the text control and
`image-quality.json` for the OpenAI image control before its cleanup trap
restarts Qwen. The image verifier is also independently callable against an
already-running isolated candidate:

```bash
PYTHONPATH=python /home/david/freetoken-amd/.venv/bin/python \
  scripts/lan223/verify_gemma4_gguf_image.py \
  --base-url http://127.0.0.1:1923 \
  --model gemma4-26b-q4-amd \
  --artifact /tmp/gemma4-image-quality.json
```

## Boundaries

This is a functionality and short-control measurement, not a long-output
throughput benchmark. The next performance phase must use a fixed visual task,
multiple repetitions, warmup exclusion, server telemetry, and matched
llama.cpp controls before making any TPS comparison.
