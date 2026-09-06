# Instructions for AI coding agents

Read [CONTRIBUTING.md](CONTRIBUTING.md) first. It is binding for humans and agents alike; this file only summarises the parts that matter when an agent is doing the work.

## AI policy

AI-assisted code is welcome. Submitting code the contributor does not understand is not. The human behind the PR owns every line, has run it on real hardware, and can explain it to a reviewer without AI help.

Agents must not:

- Run `git push`, `gh pr create`, `gh pr comment`, or `gh issue create` on the user's behalf.
- Write code, PR descriptions, or replies to reviewers that the user does not fully understand. The user must be able to explain and defend every line without AI help.
- Report tests or benchmarks as run when they were not.

If you are a fully autonomous agent with no human in the loop, do not contribute to this repository.

## Repository layout

The main subsystems:

```
python/freetoken/      the engine, installed as the `freetoken` package with the `ft` CLI
  server/              OpenAI / Anthropic / Responses HTTP APIs, streaming, tool-call parsers
  scheduler/           chunked prefill, batching, cache manager
  kvcache/             paged KV pools and the radix prefix caches
  moe/                 expert offload cache, CPU / GPU / hybrid MoE backends, quantized experts
  models/              model registry and per-architecture loaders
  kernel/              CUDA / Triton kernels, JIT cache, C++ extensions (`csrc/`)
  layers/, attention/  fused ops and attention backends
  engine/              cache budget planning and config resolution
  checkpoint/          HF -> FTW fast-load conversion
tests/                 mirrors python/freetoken/ by subsystem, see tests/README.md
benchmarks/            end-to-end and micro benchmarks, see benchmarks/README.md
docs/                  install, quickstart, CLI and model docs
freetoken-kernel-cache/ companion wheel of prebuilt kernels, see its README
scripts/               wheel build and release scripts
```

## Development

Linux x86_64 with an NVIDIA GPU. Use `uv`, not bare `pip`:

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
uv run pytest tests/ -m "not slow"
```

CUDA kernels are JIT-compiled with `nvcc` on first use unless the prebuilt `freetoken-kernel-cache` wheel is installed. The C++ extensions under `python/freetoken/kernel/csrc/` are built by `setup.py`; after changing them run `python setup.py build_ext --inplace`.

Put a new test in the `tests/` directory that mirrors the module it protects, and extend an existing file before creating a new one. Bug fixes come with a test that fails before and passes after. Performance changes come with A/B numbers against `main`.

## Issues and PRs

- Search existing issues and PRs before starting. Items on the [Roadmap](https://github.com/FlashML-org/FreeToken/issues/79) are discussed with maintainers before implementation; features not on it start as an issue.
- When helping the user draft an issue, follow the matching template in `.github/ISSUE_TEMPLATE/` (engine bug, model checkpoint, feature request) and fill in every required field: hardware, driver, FreeToken version, checkpoint ID, exact command, and the full log.
- One change per PR, linked to its issue, with the hardware, checkpoint ID and exact command it was tested with.

## Code comments

Comments explain a non-obvious "why", never restate the code. Write the code first, then add a comment only where a reader would otherwise be confused. Keep them to one or two lines. Configuration files get no comments. Use ASCII: `-` not em-dash, `->` not arrows.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/), one line, imperative, lowercase, no trailing period:

```
fix(kvcache): size the SWA radix pool for chunked prefill
```

PRs are squash-merged, so the PR title follows the same format. The subject line is usually enough; add a body only when the change needs a why that the diff does not show, and keep it to a few lines. Only commit when the user asks. If the user wants attribution, use `Assisted-by: <agent name>`, not `Co-authored-by`.
