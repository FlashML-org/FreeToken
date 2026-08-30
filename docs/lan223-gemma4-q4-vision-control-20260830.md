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

## Matched ROCm 10 llama.cpp control

The matched llama.cpp runner used the same text GGUF, sibling projector,
ROCm 10 installation, loopback-only OpenAI API contract, and deterministic
image fixtures. Its artifact is:

`/home/david/freetoken-amd/artifacts/gemma4-llamacpp-vision-20260830T051736Z`

| Control | FreeToken AMD ROCm/HIP | llama.cpp ROCm 10 | Result |
| --- | --- | --- | --- |
| Text arithmetic | `323`, 47.97 decode tok/s, 1687.71 ms TTFT | `323`, 30.99 decode tok/s, 204.23 ms TTFT | Both correct. FreeToken decoded this two-step short control 54.8% faster, while llama.cpp had lower first-token latency. |
| Solid red image | `red`, 284 prompt and 2 completion tokens, 2.27 s wall time | `red`, 82 prompt and 92 completion tokens, 56.10 generated tok/s, 1.96 s wall time | Both correct. llama.cpp emitted 91 reasoning tokens before its visible answer. |
| Solid green image | `green`, 284 prompt and 2 completion tokens, 1.08 s wall time | `green`, 82 prompt and 84 completion tokens, 56.00 generated tok/s, 1.81 s wall time | Both correct. llama.cpp emitted 83 reasoning tokens before its visible answer. |
| Red-left, blue-right image | `red`, 282 prompt and 2 completion tokens, 1.06 s wall time | `red`, 79 prompt and 121 completion tokens, 56.16 generated tok/s, 2.44 s wall time | Both correct. llama.cpp preserved spatial information but emitted 120 reasoning tokens first. |

This is a real OpenAI-compatible quality comparison, not an equivalence claim
for visual TPS. The runtimes tokenize image inputs differently, and llama.cpp
deliberately exposes a long `reasoning_content` trace on this Gemma template.
That makes its reported 56 tok/s an internally useful decode measurement but
not directly comparable to FreeToken's two-token user-visible response. On the
user-visible contract FreeToken completed the green and split-image controls
faster; on the first cold red request llama.cpp was faster.

The text result is directly comparable because both runners used the same
caller-rendered prompt and returned the same four completion tokens. It shows
that the current native FreeToken ROCm/HIP path exceeds the matched llama.cpp
decode rate for that bounded control, but it does not establish a general
long-output advantage.

## Recovery-contract result

The final isolated FreeToken vision run is
`/home/david/freetoken-amd/artifacts/gemma4-gguf-vision-20260830T053317Z`.
It passed all three image controls (`red`, `green`, and spatial `red`), then
shut down the candidate and restored the protected Qwen server. Qwen reported
the authoritative `status: ok` after about eight minutes and twenty seconds;
the control runner then exited cleanly. This verifies that the runner now
handles the real serial-NVFP4 recovery envelope without a false success while
Qwen is still loading or a false failure from re-running its launcher.

## Boundary and next measurement

The image controls are functionality and short-request measurements, not a
long-output visual throughput benchmark. The next performance phase should use
a fixed visual-description task, a quality rubric that scores both visible and
reasoning channels separately, warmup exclusion, multiple repetitions, and
streaming telemetry. That will measure prompt TPS, first-token latency, and
decode TPS without rewarding a runtime merely for emitting more hidden
reasoning tokens.
