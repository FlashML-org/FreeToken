# Contributing to FreeToken

Thanks for helping make FreeToken better. This page covers how to report issues and how to submit pull requests.

## Getting help

- [FAQ](https://github.com/FlashML-org/FreeToken/issues/84): kept up to date; most install and runtime problems are answered there.
- [Roadmap](https://github.com/FlashML-org/FreeToken/issues/79): what we are working on next.
- [Developer Slack](https://join.slack.com/t/flashml/shared_invite/zt-3zpdh5j10-9dwTXrgLiqpVxizhA9KVbA) for development discussion; [Community Discord](https://discord.gg/xzwSnMdsX) or [Community WeChat](https://github.com/FlashML-org/FreeToken/blob/main/assets/freetoken-wechatgroup.png) for usage questions.

## Reporting issues

1. **Check the FAQ and the Roadmap first.** If your request is already on the Roadmap, please do not open a new issue for it. Search [existing issues](https://github.com/FlashML-org/FreeToken/issues?q=is%3Aissue); if one matches, comment there.
2. **Describe the problem clearly.** A report we cannot reproduce is a report we cannot fix. Include your hardware (GPU and VRAM, CPU, system RAM), OS, NVIDIA driver version, FreeToken version (`ft --version`, or the version shown in the Desktop app), the exact checkpoint (the Hugging Face or ModelScope ID, e.g. `Qwen/Qwen3.6-35B-A3B-FP8`, not just the model name), the command, and the full log, not a screenshot of the last line. Mention anything else that may matter: other GPUs in the machine, WSL, proxies.
3. **Desktop users: make sure you are on the latest version.** The Desktop app updates itself; restart it to pick up the latest version and check whether the problem still happens. Attach the output of **Logs → Server status → Copy**, which copies the full engine log.
4. **Building from source: check against `main`.** Many problems are already fixed on `main`. Pull, rebuild, confirm the problem still happens, and include the commit you tested (`git rev-parse --short HEAD`).

Issues that do not follow these guidelines may be closed until the missing information is provided.

## Pull requests

### AI policy

AI-assisted contributions are welcome. You are responsible for everything in your PR, however it was produced.

Pure agent PRs are not accepted. A human must understand what the agent did and why, have run the code on real hardware, and be able to explain the change to a reviewer without AI help. If you are a fully autonomous agent operating without human oversight, do not open PRs in this repository.

If a PR contains AI hallucinations (code that does not do what the description says, references to APIs, files or behaviour that do not exist, or test and benchmark results that were not actually run), it will be closed without review, and PRs from the same author will be treated with less trust afterwards.

### Roadmap features

If you want to implement something on the [Roadmap](https://github.com/FlashML-org/FreeToken/issues/79), please join the [Developer Slack](https://join.slack.com/t/flashml/shared_invite/zt-3zpdh5j10-9dwTXrgLiqpVxizhA9KVbA) and discuss it with the maintainers before starting. It avoids duplicate work and makes sure the design fits the engine.

### What a PR needs

- One change per PR. Unrelated fixes go in separate PRs.
- Link the issue it addresses. Features that are not on the Roadmap should start as an issue, not a PR.
- State what you tested on: GPU, CPU, driver, the checkpoint's Hugging Face / ModelScope ID, and the exact command.
- For performance changes, include A/B end-to-end results: the same model, prompt and settings on `main` and on your branch, with tokens/s (and TTFT if prefill is affected) for both, and the commands used.
- For bug fixes, add a test that fails before the change and passes after, where the code allows it.

## Development setup

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv pip install -e ".[accel]"
```

See [docs/install.md](docs/install.md) for requirements. CUDA kernels are JIT-compiled on first use and need a CUDA 13 toolkit with `nvcc` on `PATH`.

Run the tests with `pytest`. Most tests need an NVIDIA GPU. `-m "not slow"` skips the long kernel sweeps; tests marked `needs_weights` are off unless you point them at a local checkpoint (see [tests/README.md](tests/README.md)).

## Commit messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>
```

- `type`: `feat`, `fix`, `perf`, `refactor`, `build`, `ci`, `docs`, `test`, `chore`.
- `scope`: the module you touched, e.g. `server`, `engine`, `moe`, `kvcache`, `kernel`, `launch`, `gemma4`, `deps`, `README`. Optional.
- `subject`: imperative, lowercase, no trailing period.
- Breaking changes: add `!` after the scope and explain in the body.

Examples from the history:

```
feat(server): enable reasoning_effort on /v1/chat/completions
perf(fp8): run per-tensor fp8 as W8A8 via scaled_mm
fix(server): stop API server when backend dies
build(deps): pin apache-tvm-ffi, flashlib, triton to exact versions
```

PRs are squash-merged, so the **PR title** must follow the same format, because it becomes the commit subject.

## License

By contributing to FreeToken, you agree that your contributions will be licensed under the [LICENSE](LICENSE) in the root of this repository.
