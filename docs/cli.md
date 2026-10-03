# CLI reference

```
ft <command> [args]
```

| Command | Purpose |
|---|---|
| `ft serve` | Start the API server (OpenAI `/v1/*`, Anthropic `/v1/messages`, Responses) |
| `ft info` | Forecast a model's GPU memory for a serve configuration, without loading it |
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
| `--hf-overrides` | — | JSON object applied to the checkpoint's config as vLLM's `--hf-overrides`: a nested config section updates key by key, any other value is replaced whole. A YaRN `rope_parameters` override serves `original_max_position_embeddings * factor` positions |

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
| `--skip-preflight` | off | Load the weights even when the pre-load memory forecast says the configuration cannot fit; see [ft info](#ft-info) |
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

## ft info

```bash
ft info <model> [ft serve flags] [--json] [--gpu-memory-gib N | --gpu-free-gib N]
```

Forecasts what `ft serve <model> [flags]` would put on the GPU, and whether it
fits, without reading a single weight: it resolves the configuration the way the
engine does, builds the model on PyTorch's meta device, and sizes every pool
with the engine's own sizing code. It runs in a few seconds, needs no CUDA, and
takes every `ft serve` flag, so "what if" combinations can be tried before a
multi-minute load:

```bash
ft info ~/models/Qwen3.8-27B-NVFP4 --text-model-only
ft info Qwen/Qwen3-30B-A3B --gpu-memory-gib 24 --max-running-requests 1
```

The report lists:

- the architecture, the format each layer type is stored in on the GPU, and the
  checkpoint's dtypes;
- the weights resident on the GPU per category (attention, linear attention,
  MLP, routed experts, embeddings / lm_head, vision encoder, MTP heads, rope
  tables) next to what the checkpoint stores, so a skipped vision tower
  (`--text-model-only`), MTP heads the engine does not load, or offloaded experts
  living in host RAM are visible;
- the budget: `--memory-ratio` x free memory, minus weights, the MoE expert slot
  cache, the GDN state pool (slots x bytes), and the KV cache (bytes per token x
  tokens);
- what lives outside the budget: the attention backend's workspace, the page
  table, the CUDA graphs, the prefill activations of one `--max-prefill-length`
  chunk, and `--vram-reserve-mb`; the "free after init" line is the number the
  engine logs as `Free memory after initialization`;
- a verdict -- `fits`, `tight` or `does not fit` -- with the max context and the
  number of concurrent requests, and tips: flag changes with their predicted
  effect, plus the cheapest combination that fits.

Free memory comes from NVML for the local GPU (`--gpu` picks the card), minus
the CUDA context the engine creates first. `--gpu-memory-gib N` plans for an
empty card of N GiB instead, `--gpu-free-gib N` for N GiB free as `nvidia-smi`
reports it. Without a GPU and without these flags the weights and per-token
costs are still reported. `--json` prints the same data for tools; the exit code
is 1 when the configuration does not fit, else 0.

Weights come from the meta-built model, whose parameters are exactly what the
loader fills (tensors the family skips are absent, re-quantized ones already
have their stored size); the checkpoint column reads only the safetensors
headers -- from the local directory, the Hugging Face cache, or the Hub through
HTTP range requests. The load overhead (kernel modules, library handles), the
prefill activations and the CUDA graph memory are estimates and are labelled so.

`ft serve` (and the offline `LLM` class) runs the same forecast right before it
loads the weights, against the free memory it has just measured. A configuration
whose KV cache cannot be sized even before those estimates is refused within
seconds, with the forecast and the tips in the log, instead of failing with
`Not enough memory for KV cache` after the load. A tight one only logs a warning
and continues. `--skip-preflight` loads regardless.

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

