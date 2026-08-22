# CLI reference

```
ft <command> [args]
```

| Command | Purpose |
|---|---|
| `ft serve` | Start the API server (OpenAI `/v1/*`, Anthropic `/v1/messages`, Responses) |
| `ft serve-metal` | Start the same API surface over an Apple Silicon Metal backend (mlx or llama.cpp) |
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
| `--max-running-requests` | 4 | Max concurrently running requests |
| `--max-output-tokens` | 32768 | Default output budget for requests that omit one |
| `--max-seq-len-override` | from checkpoint | Max sequence length |
| `--max-prefill-length` | 8192 | Chunked-prefill chunk size in tokens |
| `--cuda-graph-max-bs`, `--graph` | = max running requests | Max batch size captured as CUDA graphs |
| `--decode-log-interval` | 40 | Scheduler status line every N decode steps |

### KV cache & memory

| Flag | Default | Meaning |
|---|---|---|
| `--memory-ratio` | 0.9 | Fraction of free VRAM the engine may use (weights + MoE cache + KV) |
| `--num-pages` / `--num-tokens` | auto | KV capacity override in pages / tokens (mutually exclusive; auto sizes from VRAM left after weights and MoE cache) |
| `--page-size` | 1 | KV page size; DSV4 forces 128, the TRTLLM backend needs 16/32/64, SWA models require 1 |
| `--cache-type` | radix | `radix` (prefix reuse; SWA/GDN-aware variants picked automatically) or `naive` |
| `--attention-backend`, `--attn` | auto | `trtllm`/`fi`/`fa`/`triton`/`dsv4_sparse`/`dsa`; `prefill,decode` pair allowed; auto picks per model + GPU |

### MoE offload

See [models.md](models.md#moe-backends) for what each backend does.

| Flag | Default | Meaning |
|---|---|---|
| `--moe-backend` | auto | `fused`/`offload`/`cpu`/`hybrid`; auto → offload, or hybrid with a `ft bench bw` profile |
| `--moe-cache-size` / `--moe-cache-rate` / `--moe-cache-auto` | auto | GPU expert-cache size as slots / fraction of all experts / sized from free VRAM (mutually exclusive; auto is enabled by default for offload-family backends) |
| `--kv-reserve-tokens` | 8192 | KV token floor reserved before `--moe-cache-auto` fills experts |
| `--moe-cpu-threads` | physical cores | CPU worker threads for the cpu/hybrid executor |
| `--moe-cpu-layers` | all on GPU | With `offload`: which MoE layers decode on CPU (`3,7,11`, a count, or a fraction) |
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

## ft serve-metal

`ft serve-metal` runs the same OpenAI `/v1/*`, Anthropic `/v1/messages`, and
Responses API surface, but backs it with an **Apple Silicon Metal runtime**
instead of FreeToken's CUDA scheduler. It does not import FreeToken's CUDA
stack, so it works standalone on a Mac (no triton/flashinfer/sglang-kernel, no
`flashlib`) by proxying to a Metal engine that already implements the protocol.

```bash
ft serve-metal --model mlx-community/Qwen3-0.6B-4bit --backend mlx --port 1919
ft serve-metal --model ~/models/MyModel.Q4_K_M.gguf --backend llama --port 1919
```

The API surface (chat/completions/models/health, streaming included) is
bit-for-bit the same FreeToken wire, so the desktop app, `ft shell`, `ft launch`
agents, and OpenAI/Anthropic clients all work unchanged.

| Flag | Default | Meaning |
|---|---|---|
| `--model` | required | Model path or HF id for the Metal engine |
| `--backend` | `auto` | `mlx` (Apple's `mlx_lm.server`) or `llama` (llama.cpp `llama-server`); `auto` prefers mlx |
| `--host` | 127.0.0.1 | FreeToken API bind address |
| `--port` | 1919 | FreeToken API port (user-facing) |
| `--metal-port` | 0 | Upstream Metal engine port; a free loopback port is chosen when `0` |

Notes:

- **mlx** requires the model loadable by `mlx-lm` (HF-converted or MLX weights,
  e.g. `mlx-community/*`). `mlx_lm` uses the Apple GPU (Metal) directly.
- **llama** requires the `llama-server` binary on PATH and a GGUF model (the
  same format FreeToken already parses). It runs with the Metal backend
  (`-ngl 999`).
- `ft serve --backend mlx|llama` also switches between these backends on a CUDA
  box, but still imports the CUDA config stack; `ft serve-metal` is the
  standalone, CUDA-free form for Apple Silicon.

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
| `stats` | `GET /v1/stats` | Throughput, latency, VRAM, pool occupancy |
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
ft checkpoint --model <hf_dir> --out <ftw_dir> [--dtype bfloat16] [--moe-backend offload] [--shard-gib 8] [--device cuda:0]
```

Converts an HF safetensors checkpoint to FTW, FreeToken's self-contained
fast-load format; point `ft serve --model` at the output dir. `--moe-backend
offload` (default) packs experts into offload banks; `--moe-backend triton`
keeps them dense for resident serving. See the FTW caveats in
[models.md](models.md#notes).

## ft bench bw

```bash
ft bench bw                       # once per machine
ft bench bw --dtype nvfp4,bf16    # only the formats you serve
```

Measures host-RAM vs PCIe bandwidth with the real cpu/offload MoE kernels and
writes a profile (`~/.cache/freetoken/benchbw.json`) that `ft serve
--moe-backend auto` and `--moe-hybrid-max-fetch -1` read. Profiles are keyed on
expert format + GPU name, so a profile from different hardware is ignored
rather than misapplied. Selection flags: `--dtype`, `--model`, `--formats`,
`--isa`; decision rule: `--threshold` (default 2.0 — recommend hybrid when CPU
bandwidth > 2× PCIe).

