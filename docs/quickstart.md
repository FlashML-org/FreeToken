# Quick start

Assumes FreeToken is installed — see [install.md](install.md).

## Launch a server

```bash
ft serve --model ~/models/Qwen3.6-35B-A3B
```

`--model` also takes a Hugging Face repo id. Everything else — dtype, attention
and MoE backends, cache sizes, tool-call and reasoning parsers — resolves from
the checkpoint and the GPU; see [cli.md](cli.md) for the flags. The server is
ready when the log reaches `API server is ready to serve on 127.0.0.1:1919`.

### macOS (Apple Silicon)

```bash
ft serve-metal --model mlx-community/Qwen3-0.6B-4bit --backend mlx
```

`ft serve-metal` serves the same API surface as `ft serve`, backed by MLX or
llama.cpp on Metal (see [install.md](install.md#method-3-macos--apple-silicon-metal-backend)).
`--model` takes an MLX/HF repo id (`mlx-community/*`) or, with `--backend llama`,
a local GGUF file. The rest of this page (`/v1/models`, chat completions, `ft shell`,
`ft launch`) works unchanged against it — including `ft shell --model <id>`, which
starts the Metal engine and chats in one process on a Mac, and the `/model <id>`
shell command to switch models without restarting.

## Send a request

Check what is being served:

```bash
curl http://127.0.0.1:1919/v1/models
```

Then use that id as the `model` field:

```bash
curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.6-35B-A3B",
    "messages": [{"role": "user", "content": "What is a Mixture-of-Experts model?"}],
    "max_tokens": 256,
    "stream": true
  }'
```

FreeToken serves the OpenAI API (`/v1/chat/completions`, `/v1/responses`,
`/v1/models`) and the Anthropic API (`/v1/messages`,
`/v1/messages/count_tokens`), so a client library for either works by pointing
its base URL at the server. 

## Chat in the terminal

A simple TUI to interact with the server:

```bash
ft shell                                    # attach to the server above
ft shell --model ~/models/Qwen3.6-35B-A3B   # start an engine and chat, one process
```

`/help` lists the in-shell commands. Attach mode needs no GPU, so it also drives
a server on another machine (`--server URL`). On a Mac, `ft shell --model <mlx/hf id>`
starts the Metal backend instead (see [quickstart.md](quickstart.md)); `/model <id>`
switches the served model on servers that support it (Metal).

## Use a coding agent

```bash
ft launch claude   # claude / codex / dsh / hermes / openclaw / opencode
```

Writes that agent's provider config, installs its CLI if missing, and starts it
against your server. `--dry-run` previews the changes.
