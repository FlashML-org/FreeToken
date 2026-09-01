# FreeToken test suite

Directories mirror the `python/freetoken/` subsystem a test primarily exercises —
put a new test next to the module it protects. There is no per-model directory:
a model-specific test lives in the subsystem it stresses, and says which model
in its filename (`kvcache/test_dsv4_pool.py`, `models/test_glm4_nvfp4.py`).

| directory    | subsystem under test |
|--------------|----------------------|
| `daemon/`    | `freetoken.daemon` — torch-free supervisor (import safety, serve manager) |
| `server/`    | `freetoken.server` — OpenAI/Anthropic/Responses APIs, streaming, parsers, accounting, maintenance state |
| `scheduler/` | `freetoken.scheduler` — chunked prefill, cache manager, commit/window locking, status reporting, KV usage, cost accounting |
| `kvcache/`   | `freetoken.kvcache` — paged pools (incl. DSV4's), rebuild, cache-cost accounting; the three prefix caches live in `kvcache/radix/` behind a shared reference model (see its README) |
| `tokenizer/` | `freetoken.tokenizer` — tokenize/detokenize request plumbing, thinking-mode resolution |
| `engine/`    | `freetoken.engine` — cache budget planning, config resolution, backend gating |
| `runtime/`   | `freetoken.runtime.gpu` — AMD/NVIDIA vendor detect, iGPU hide, gfx arch (CPU mocks) |
| `kernels/`   | `freetoken.kernel` / `freetoken.layers` — attention, rope, fused ops, JIT cache, pinned memory |
| `moe/`       | `freetoken.moe` — offload cache, CPU/GPU expert kernels, quantized backends (fp8/nvfp4/mxfp4/q4_0) |
| `models/`    | `freetoken.models` — the registry, and the loading machinery every model shares (sharding, qkv/expert merge, streaming layers into banks) |
| `e2e/`       | real-server gates: AIME 24/25/26 generation, runtime cache rebuild |

The CLI surface itself is not unit-tested. `ft` dispatch, `ft ctl`, `ft launch` and
`install.sh` are thin, they change often, and a test written after the fact only
restates whatever the code currently does — running the command is the real gate.
What does live here is the logic reachable *through* those commands when it has its
own failure mode: `server/test_parser_auto_selection.py` covers the architecture ->
parser inference behind `--tool-call-parser auto`.

## Running

```bash
uv run pytest tests/                 # full suite, ~2-4 min on a GPU box
uv run pytest tests/ -m "not slow"   # skip the handful of tens-of-seconds tests
uv run pytest tests/kvcache/         # one subsystem
```

GPU-dependent tests skip themselves when CUDA is unavailable. Marlin NVFP4 tests
skip unless `vllm` is importable (dedicated venv with `vllm>=0.14,<0.15`).

AMD/HIP unit tests (`tests/runtime/test_gpu.py`, `tests/kernels/test_hip_compat.py`,
`tests/kernels/test_pinned_hip_fallback.py`) are CPU-only mocks. They do **not**
stand in for an R9700 e2e run — see [docs/amd.md](../docs/amd.md) for the on-box
checklist. Do not treat a green CI as “we ran on gfx1201”.

`needs_weights`-marked tests skip unless the env var pointing at a real local
checkpoint is set:

| env var                    | used by |
|----------------------------|---------|
| `FREETOKEN_TEST_MODEL`     | `e2e/test_aime.py` — local model directory |
| `FREETOKEN_AIME_SERIES`    | `e2e/test_aime.py` — `aime24` \| `aime25` \| `aime26` (default `aime25`) |
| `FREETOKEN_AIME{24,25,26}_JSONL` | `e2e/test_aime.py` — jsonl for the selected series |
| `FREETOKEN_AIME_REQ`       | `e2e/test_aime.py` — 1-based problem id, comma list (`1,3,7`), or `all` (default `1`) |
| `FREETOKEN_AIME_MAX_TOKENS`| `e2e/test_aime.py` — per-sample token budget (default `16384`) |
| `FREETOKEN_AIME_SAMPLES`   | `e2e/test_aime.py` — pass@N sample count when sampling (default `3`) |
| `FREETOKEN_AIME_MIN_FREE_GIB` | `e2e/test_aime.py` — required free GPU memory (default `70`) |
| `FREETOKEN_TEST_MOE_CACHE_SIZE` | `e2e/test_aime.py` — >0 switches to the offload MoE backend with this cache size |
| `FREETOKEN_TEST_MEM_RATIO` | `e2e/test_aime.py` — offload-mode memory_ratio (default `0.9`) |
| `FREETOKEN_REBUILD_TEST_MODEL` | `e2e/test_cache_rebuild.py` — a SMALL local model dir; boots a real server (falls back to `FREETOKEN_TEST_MODEL`) |
| `FREETOKEN_GEMMA4_GGUF_GLOB` | `models/test_gemma4_gguf_rope.py` — glob matching a local gemma-4 GGUF file |

`test_aime.py` takes its sampling protocol from the checkpoint's own
`generation_config.json` (pass@N at the recommended temperature, or a single greedy
sample when the checkpoint recommends greedy). To run an fp8 / offload checkpoint,
point `FREETOKEN_TEST_MODEL` at the fp8 dir and set `FREETOKEN_TEST_MOE_CACHE_SIZE=8192`;
`FREETOKEN_AIME_REQ=2 FREETOKEN_AIME_MAX_TOKENS=32768` is the offload recipe (req 2 with
a 32k budget).

Budget the token allowance for sampling, not for greedy: on Qwen3.5-35B-A3B-FP8 the three
recommended-sampling samples of aime25 req 1 ran 4.1k / 10.7k / 10.0k tokens against greedy's
~4k, and req 2 needs ~21k even greedy — hence the 32k budget for the harder problems.

Run these before opening a PR — they are cheap, and a gate nobody trips rots silently:

```bash
FREETOKEN_REBUILD_TEST_MODEL=<small model dir> uv run pytest tests/e2e/test_cache_rebuild.py
```

## What earns a place here

A test must be able to fail on a plausible regression of production code.

Weigh that against how the bug would be found otherwise. Application-facing
surfaces — the terminal UI, the access-log filter, the request ring, the status
and health endpoints, the CLI — announce their own breakage the moment anyone
uses them, and they change shape often enough that a unit test mostly restates
today's implementation. Those belong to `e2e/` and to actually running the
thing. What earns a unit test is logic that fails *silently*: a wrong number
that still looks like a number, a model family that quietly stops having its
tool calls parsed, a cache that returns the wrong slot. Prefer covering that
where it lives rather than through the surface that happens to expose it.

Writing a test in the same commit as the feature is right; what matters is where
its expectations come from. A test that checks the result against something
independent — a PyTorch reference, a round-trip, a CPU mirror, a live registry —
keeps working when the implementation is rewritten, and can disagree with it.
A test whose assertions restate the branch structure the author just wrote can
only ever agree, so it never fails on anything except a deliberate change, and
then it is edited to match. If you cannot name what the expectation is checked
*against*, the test is unlikely to earn its place.
Bug-repro tests should drive the current call pattern of the fixed code, not the
pre-fix one; wire-format and kernel tests should assert against an independent
reference (round-trip, dequant oracle, CPU mirror), not against the
implementation's own output. Registry-style tests should iterate the live
registry rather than hand-maintained whitelists, so new entries are covered
automatically.
