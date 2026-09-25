# CLI reference

```
ft <command> [args]
```

| Command | Purpose |
|---|---|
| `ft serve` | Start the API server (OpenAI `/v1/*`, Anthropic `/v1/messages`, Responses) |
| `ft shell` | Chat with a server in the terminal |
| `ft ctl` | Query and manage a running server over HTTP |
| `ft launch` | Configure and launch a coding agent against a server |
| `ft checkpoint` | Convert an HF checkpoint to the FTW fast-load format |
| `ft bench bw` | Benchmark CPU vs PCIe bandwidth to calibrate the MoE backend |

`ft --version` prints the installed version (torch-free; nightly wheels carry a
`+g<sha>` build stamp, tagged releases a bare version). Every command supports
`--help`.

## ft serve

```bash
ft serve --model <path-or-hf-id> [options]
```

`--model` is the only required flag — dtype, attention backend, MoE backend,
MoE cache size, KV capacity, CUDA-graph sizes and the tool-call/reasoning
parsers all resolve automatically from the checkpoint and the GPU.

### Model

| Flag | Default | Meaning |
|---|---|---|
| `--model-path`, `--model` | required | Local dir, HF repo id, or an FTW dir (auto-detected) |
| `--served-model-name` | basename of `--model` | Model id reported by `/v1/models` |

### Server & runtime

| Flag | Default | Meaning |
|---|---|---|
| `--host` | 127.0.0.1 | Bind address |
| `--port` | 1919 | Bind port |
| `--gpu` | GPU 0 | GPU to run on: a UUID from `nvidia-smi -L` or an `nvidia-smi` index; see [below](#choosing-a-gpu) |
| `--max-running-requests` | 4 | Max concurrently running requests |
| `--max-output-tokens` | 32768 | Default output budget for requests that omit one |
| `--max-seq-len-override` | from checkpoint | Max sequence length |
| `--max-prefill-length` | 8192 | Chunked-prefill chunk size in tokens |
| `--cuda-graph-max-bs`, `--graph` | = max running requests | Max batch size captured as CUDA graphs |
| `--decode-log-interval` | 40 | Scheduler status line every N decode steps |

### Choosing a GPU

For example, a machine with an RTX 5090 and an RTX 3060 Ti:

```console
$ nvidia-smi -L
GPU 0: NVIDIA GeForce RTX 3060 Ti (UUID: GPU-2f3a9b1c-8d7e-4a05-b6c1-0e5f9a3d7b42)
GPU 1: NVIDIA GeForce RTX 5090 (UUID: GPU-9e8d7c6b-5a49-4f13-8207-c1b0a4e6d3f5)
```

```bash
ft serve --model ... --gpu 1             # by nvidia-smi index -- the 5090
ft serve --model ... --gpu GPU-9e8d7c6b  # the same card by UUID (a unique prefix is enough)
```

### KV cache & memory

| Flag | Default | Meaning |
|---|---|---|
| `--memory-ratio` | 0.9 | Fraction of free VRAM the engine may use (weights + MoE cache + KV) |
| `--num-pages` / `--num-tokens` | auto | KV capacity override in pages / tokens (mutually exclusive; auto sizes from VRAM left after weights and MoE cache) |
| `--page-size` | 1 | KV page size; DSV4 forces 128, the TRTLLM backend needs 16/32/64, SWA models require 1 |
| `--cache-type` | radix | `radix` (prefix reuse; SWA/GDN-aware variants picked automatically) or `naive` |
| `--attention-backend`, `--attn` | auto | `trtllm`/`fi`/`fa`/`triton`/`dsv4_sparse`/`dsa`; `prefill,decode` pair allowed; auto picks per model + GPU |

### MoE offload

See [models.md](models.md#moe-strategies) for what each strategy does.

| Flag | Default | Meaning |
|---|---|---|
| `--moe-strategy` | auto | `fused`/`offload`/`cpu`/`hybrid`; auto → offload, or hybrid with a `ft bench bw` profile. `--moe-backend` is the deprecated old spelling |
| `--quant-backend` | auto | Kernel per quantized layer type, `layer[.kind]=name` entries: `linear=marlin,moe=b12x` or `moe.nvfp4=triton`. A layer-level entry applies to every kind whose table lists the name |
| `--nvfp4-backend` | — | Deprecated: stands in for `--quant-backend moe.nvfp4=<marlin\|b12x\|triton>` (`flashinfer` means b12x); cannot be combined with `--quant-backend` |
| `--moe-cache-size` / `--moe-cache-rate` / `--moe-cache-auto` | auto | GPU expert-cache size as slots / fraction of all experts / sized from free VRAM (mutually exclusive; auto is enabled by default for offload-family strategies) |
| `--kv-reserve-tokens` | 8192 | KV token floor reserved before `--moe-cache-auto` fills experts |
| `--moe-cpu-threads` | physical cores | CPU worker threads for the cpu/hybrid executor |
| `--moe-cpu-layers` | all on GPU | With `offload`: which MoE layers decode on CPU (`3,7,11`, a count, a fraction, or `auto`). `auto` is for Windows/WSL only, where CUDA pinned memory is capped; every value needs an expert format the CPU executor serves (bf16, nvfp4, mxfp4), so fp8 experts cannot use it |
| `--moe-hybrid-max-fetch` | auto | With `hybrid`: max experts fetched over PCIe per layer per step; rest computed on CPU |
| `--moe-prefill-hit-d2d` | off | Prefill: copy cache-hit experts device-side, stream only misses (CUDA >= 13) |
| `--disable-moe-prefill-overlap` | overlap on | Disable the two-buffer prefill copy overlap |

### API behaviour

| Flag | Default | Meaning |
|---|---|---|
| `--sampling-defaults` | model | Fill unspecified sampling params from the checkpoint's `generation_config.json` (`none` = framework defaults) |
| `--tool-call-parser` | auto | Tool-call format; auto-inferred from the model family |
| `--reasoning-parser` | auto | Splits chain-of-thought into `reasoning_content`; auto-inferred; `off` disables |
| `--enable-cache-report` | off | Report prefix-cache hits in each response's usage block |
| `--enable-metrics-report` | off | Serve a per-request `metrics` object next to usage ([below](#per-request-performance-metrics)) |

### Per-request performance metrics

`--enable-metrics-report` adds a `metrics` object to each response on `/v1/chat/completions`,
`/v1/messages` and `/v1/responses`. It is not part of any of those protocols, which is why it is
off by default; clients that do not read it are unaffected.

```json
{
  "usage": { "prompt_tokens": 8192, "completion_tokens": 256, "total_tokens": 8448 },
  "metrics": {
    "ttft_ms": 320.5,
    "prefill_tokens": 2048,
    "cached_prompt_tokens": 6144,
    "prefill_time_ms": 410.0,
    "prefill_tokens_per_second": 4995.12,
    "decode_tokens": 256,
    "decode_time_ms": 5120.8,
    "decode_tokens_per_second": 49.8,
    "total_time_ms": 5480.2
  }
}
```

| Field | Meaning |
| --- | --- |
| `ttft_ms` | Time to the first generated token, from the point the request entered the generation path |
| `prefill_tokens` | Prompt tokens actually forwarded: `prompt_tokens - cached_prompt_tokens` |
| `cached_prompt_tokens` | Prompt tokens served from the prefix cache instead of recomputed |
| `prefill_time_ms` | The scheduler's own prefill span: admission to the token sampled off the last prefill chunk |
| `prefill_tokens_per_second` | `prefill_tokens / prefill_time_ms` |
| `decode_tokens` | Generated tokens (same as `usage.completion_tokens`) |
| `decode_time_ms` | First generated token to the last |
| `decode_tokens_per_second` | `(decode_tokens - 1) / decode_time_ms`; the first token comes out of prefill, so it spans no decode interval |
| `total_time_ms` | Request latency up to the terminal engine reply, excluding HTTP framing |

Reading the numbers:

- `prefill_time_ms` is measured inside the scheduler, which is the point of the flag: a client
  timing SSE frames cannot see when prefill starts or ends. The others are measured at the API
  layer and carry the same IPC hops a client's own timestamps would.
- `ttft_ms` is larger than `prefill_time_ms` by the time the request spent queued before
  admission. Under load that gap is the queue, not the model.
- Every span is **this request's share of shared work**. Its prefill chunk is co-scheduled with
  other prompts and its decode steps are batched with other requests, so the throughputs
  describe this request under that load, not isolated engine benchmarks. For those,
  send one request at a time.
- `cached_prompt_tokens` here always reports the real prefix-cache hit, whether or not
  `--enable-cache-report` is set; that flag governs the billing fields in `usage`. Without it
  `prefill_tokens_per_second` would be computed over tokens that were never forwarded.

Streaming responses carry `metrics` on the same final message as usage, so the request has to ask
for usage too: `stream_options: {"include_usage": true}` on `/v1/chat/completions`. On
`/v1/messages` it rides `message_delta`; on `/v1/responses`, `response.completed`.
`/v1/completions` does not serve `metrics`: it does not go through the shared generation core.

### Image input

Experimental. Needs a checkpoint whose family registers a vision encoder ([models.md](models.md#image-input) lists them and how each one
maps the flags below); a request carrying images is rejected otherwise. Images are accepted on all three protocols (OpenAI `image_url`,
Anthropic `image` blocks, Responses `input_image`) as an http(s) URL or base64. Images inside a tool
result (an Anthropic `tool_result` block from Claude Code's Read, a Responses `function_call_output`
from Codex's view_image) are moved to the user turn that follows the tool message, as vLLM does,
because chat templates render tool messages as plain text.
`GET /v1/stats` reports what the server accepts as `model.input_modalities` (`["text"]` or `["text", "image"]`),
so a client can gate its attachment controls without reading the checkpoint config.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--text-model-only` | off | Serve a multimodal checkpoint text-only: no encoder tower is built (its VRAM goes to the KV/expert pools) and every multimodal input is rejected. Same as `--mm-disable` with every encoder kind |
| `--mm-disable` | none | Encoder towers to leave unbuilt (`vision`, `audio`); every input they would serve is rejected |
| `--mm-encoder-weights` | host | Where the encoder tower's block weights live. `host` streams them from pinned host banks two blocks at a time behind the compute, so the GPU holds two blocks instead of the whole tower; small images pay the copy time, large ones hide it behind the compute. `gpu` keeps them resident. An encoder without a block stack stays resident either way |
| `--image-min-tokens`, `--image-max-tokens` | processor defaults | Per-image token budget: the image processor resizes every image to take between these many tokens, converted to the family's own units by its processor. A family with fixed budgets honors the maximum only and refuses one below its smallest budget at start-up |
| `--mm-processor-kwargs` | none | JSON object of extra keyword arguments for the checkpoint's image processor call, for knobs the token budget does not cover; applied after the budget, so an explicit key wins |
| `--mm-embed-cache-device` | cpu | Where encoded image embeddings live between prefill chunks. `cpu` keeps them out of the VRAM budget; `cuda` skips the copy back |
| `--allowed-media-domains` | any | Comma-separated hostname allowlist for image URLs; requests for other domains are rejected with a 400. Empty allows any domain |
| `--allowed-local-media-path` | off | Directory `file://` image refs may be read from; unset rejects local files |

## ft shell

```bash
ft shell                                    # attach to a running server
ft shell --model ~/models/Qwen3.6-35B-A3B   # serve + chat in one process
```

- Attach mode talks to `--server URL` (default `http://127.0.0.1:1919`)
- `/help` inside the shell lists the commands (`/think`, `/cache`, `/reset`).

## ft ctl

```bash
ft ctl [--base-url http://127.0.0.1:1919] [--timeout 10] [--json] <subcommand>
```

| Subcommand | Endpoint | Purpose |
|---|---|---|
| `health` | `GET /health` | Server status, model, load progress |
| `stats` | `GET /v1/stats` | Throughput, latency, VRAM, pool occupancy, accepted input modalities |
| `generate [prompt] [--max-tokens N] [--ignore-eos]` | `POST /generate` | Raw completion smoke test (no chat template) |
| `cache` | `GET /v1/cache/status` | Cache pool table |
| `cache --moe N \| --kv N \| --mamba N \| --swa N [--wait 300]` | `POST /v1/cache/rebuild` | Live pool resizing without a restart (`k`/`m` suffixes; `--kv`/`--swa` in tokens) |
| `requests [--since N] [--limit N]` | `GET /v1/requests` | Recent request ring |

## ft launch

```bash
ft launch {claude,codex,dsh,hermes,openclaw,opencode} [options] [-- <agent args>]
```

Discovers the served model via `/v1/models`, writes the agent's provider
config, installs the agent CLI if missing, then launches it. Cloud API keys
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …) are cleared from the child
environment so the agent cannot silently fall back to a paid endpoint.
When `/v1/stats` reports `image` among `model.input_modalities`, the written
config declares the model image-capable, which Codex, OpenCode, OpenClaw and
dsh require before their image tools and attachments send anything; Claude
Code and Hermes need no declaration.

| Flag | Meaning |
|---|---|
| `--server URL` | Server to point the agent at (default `http://127.0.0.1:1919`) |
| `--dry-run` | Print the planned config changes and command, touch nothing |
| `-y`, `--yes` | Approve install/config prompts |
| `--config` | Configure without launching |
| `--install-only` | Just install the agent CLI (needs no server) |
| `--force-reinstall` | Re-run the agent installer |
| `-- <args>` | Forwarded verbatim to the agent |

## ft checkpoint

```bash
ft checkpoint --model <hf_dir> --out <ftw_dir> [--dtype bfloat16] [--moe-backend offload] [--quant-backend moe.nvfp4=b12x] [--shard-gib 8] [--gpu <uuid-or-index>]
```

Converts an HF safetensors checkpoint to FTW, FreeToken's self-contained
fast-load format; point `ft serve --model` at the output dir. `--moe-backend
offload` (default) packs experts into offload banks; `--moe-backend triton`
keeps them dense for resident serving. See the FTW caveats in
[models.md](models.md#notes); FTW files from older builds can be repaired with
[scripts/ftw_hotfix.py](ftw-hotfix.md) instead of reconverting.

## ft bench bw

```bash
ft bench bw                       # once per GPU
ft bench bw --dtype nvfp4,bf16    # only the formats you serve
ft bench bw --gpu 1               # a specific GPU (UUID or nvidia-smi index, as for ft serve)
```

Measures host-RAM vs PCIe bandwidth with the real cpu/offload MoE kernels and writes a
profile that `ft serve --moe-strategy auto` and `--moe-hybrid-max-fetch -1` then read.

- One profile per GPU, at `~/.cache/freetoken/benchbw/<gpu-uuid>.json`.
- Keyed on expert format + GPU, so a profile from other hardware is ignored rather than
  misapplied. An older single `benchbw.json` still counts if its GPU name matches.
- What to measure: `--dtype`, `--model`, `--formats`, `--isa`.
- `--threshold` (default 2.0) sets the call: recommend hybrid when CPU bandwidth beats PCIe
  by that factor.

