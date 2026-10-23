# FreeToken test suite

Directories mirror the `python/freetoken/` subsystem a test primarily exercises —
put a new test next to the module it protects. A test specific to one model
lives in `models/<model>/`, mirroring `python/freetoken/models/<model>/`
(`models/deepseek_v41/`, `models/qwen4_exp/`). Older model-specific tests still
sit flat in `models/` (`models/test_glm5_next_model.py`), in subsystem
directories (`kvcache/test_dsv4_pool.py`) or in `dsv4/`, and will move there.

| directory    | subsystem under test |
|--------------|----------------------|
| `daemon/`    | `freetoken.daemon` — torch-free supervisor (import safety, serve manager) |
| `server/`    | `freetoken.server` — OpenAI/Anthropic/Responses APIs, streaming, parsers, accounting, maintenance state |
| `scheduler/` | `freetoken.scheduler` — chunked prefill, cache manager, commit/window locking, status reporting, KV usage, cost accounting |
| `kvcache/`   | `freetoken.kvcache` — paged pools (incl. DSV4's), rebuild, cache-cost accounting; the three prefix caches live in `kvcache/radix/` behind a shared reference model (see its README) |
| `tokenizer/` | `freetoken.tokenizer` — tokenize/detokenize request plumbing, thinking-mode resolution |
| `engine/`    | `freetoken.engine` — cache budget planning, config resolution, backend gating |
| `kernels/`   | `freetoken.kernel` / `freetoken.layers` — attention, rope, fused ops, JIT cache, pinned memory |
| `attention/` | `freetoken.attention` — attention backends against eager references |
| `layers/`    | `freetoken.layers` — whole layer modules (mHC); a layer's kernels and fused ops go in `kernels/` |
| `moe/`       | `freetoken.moe` — offload cache, CPU/GPU expert kernels, quantized backends (fp8/nvfp4/mxfp4/gguf) |
| `models/`    | `freetoken.models` — the registry, and the loading machinery every model shares (sharding, qkv/expert merge, streaming layers into banks); one directory per model for tests specific to it |
| `mm/`        | `freetoken.mm` — multimodal processor, placeholder expansion, encoder cache |
| `checkpoint/` | `freetoken.checkpoint` — FTW read/write and replay |
| `utils/`     | `freetoken.utils` — HF config loading, ROCm arch detection |
| `dsv4/`      | DeepSeek-V4 tests that predate `models/<model>/` |
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
| `FREETOKEN_QWEN36_MODEL`   | `tokenizer/test_mm_tokenize.py` — local Qwen3.6 checkpoint (image processor + tokenizer) |
| `FREETOKEN_QWEN3VL_MODEL`  | `tokenizer/test_mm_tokenize.py` — local Qwen3-VL checkpoint (same expansion, DeepStack family) |
| `FREETOKEN_GEMMA4_MODEL`   | `models/test_gemma4_vision.py`, `tokenizer/test_mm_tokenize.py` — local Gemma-4 `gemma4` checkpoint (tower parity, boi/eoi expansion; the tokenizer test also takes the 12B `gemma4_unified` checkpoint) |
| `FREETOKEN_GEMMA4_UNIFIED_MODEL` | `models/test_gemma4_vision.py` — local gemma-4-12B `gemma4_unified` checkpoint (embedder parity) |
| `FREETOKEN_GLM53_MODEL`    | `tokenizer/test_mm_tokenize.py` — local GLM-5.3-Flash checkpoint (image-token expansion) |
| `FREETOKEN_DSV4_MODEL` | `tokenizer/test_tokenize.py` - local DSV4 checkpoint with its native `encoding/encoding_dsv4.py`; prefix stability and reminder retention, tokenizer only (no GPU or model weights loaded) |
| `FREETOKEN_INLINE_GLM_MODEL` | `tokenizer/test_tokenize.py` - local GLM-5.3 tokenizer/template; inline-system preserve policy, prefix stability and reasoning retention (no weights loaded) |
| `FREETOKEN_INLINE_QWEN_MODEL` | `tokenizer/test_tokenize.py` - local Qwen3.6 tokenizer/template; inline-system fold policy, prefix stability and reasoning retention (no weights loaded) |
| `FREETOKEN_MUSE_MODEL`     | `models/test_muse_glimmer_vision.py`, `tokenizer/test_mm_tokenize.py`, `tokenizer/test_tokenize.py` - local Muse-Glimmer-30B checkpoint (vision tests; tokenizer-only inline-system regressions) |
| `FREETOKEN_MINIMAX_M3_MODEL` | `tokenizer/test_mm_tokenize.py` — local MiniMax-M3 checkpoint (start/end expansion, token budget) |
| `FREETOKEN_TEST_MOE_CACHE_SIZE` | `e2e/test_aime.py` — >0 switches to the offload MoE backend with this cache size |
| `FREETOKEN_TEST_MEM_RATIO` | `e2e/test_aime.py` — offload-mode memory_ratio (default `0.9`) |
| `FREETOKEN_REBUILD_TEST_MODEL` | `e2e/test_cache_rebuild.py` — local checkpoint directory or GGUF file; boots a real server (falls back to `FREETOKEN_TEST_MODEL`) |
| `FREETOKEN_REBUILD_EXTRA_ARGS` | `e2e/test_cache_rebuild.py` — extra server flags, e.g. `--moe-strategy hybrid --moe-cache-size 1024 --max-running-requests 1` for offloaded MoE |
| `FREETOKEN_GEMMA4_GGUF_GLOB` | `models/test_gemma4_gguf_rope.py` — glob matching a local gemma-4 GGUF file |
| `FREETOKEN_TEST_MODELS_ROOT` | `models/test_quant_config.py` — directory whose checkpoints the stored-tensor scheme scan checks (default `/mnt/nvme/models`) |
| `FREETOKEN_DSV4_VL_MODEL` | `tokenizer/test_mm_tokenize.py` - local DeepSeek-V4 vision checkpoint |
| `FREETOKEN_DSV4_VISION_CKPT` | `models/test_dsv4_vision.py` - the same checkpoint, for the parity tests against its `inference/` reference code |

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

A test must be able to fail on a plausible regression of production code: name
the future change that would turn it red. If the only answer is editing the
test, do not write it. A test earns its place by guarding one of:

- A bug a normal run would not reveal: a wrong number that still looks like a
  number, a wrong cache slot, tool calls that quietly stop parsing, a crash only
  one interleaving or one checkpoint hits.
- A property that took reasoning to get right, checked against something
  independent of the code (see below).
- Bookkeeping that is easy to forget: every registry entry names exports its
  package has, every registered architecture gets a parser the factories know.

Not admitted: happy-path tests that repeat what the code obviously does,
assertions that restate the branches just written, config tests that echo
fixture values back, stress loops that cannot hit the bug, and tests whose only
check is `assert_called*` or "no exception".

Weigh that against how the bug would be found otherwise. Application-facing
surfaces — the terminal UI, the access-log filter, the request ring, the status
and health endpoints, the CLI — announce their own breakage the moment anyone
uses them, and they change shape often enough that a unit test mostly restates
today's implementation. Those belong to `e2e/` and to actually running the
thing. What earns a unit test is logic that fails *silently*, like the bugs
listed above. Prefer covering that where it lives rather than through the
surface that happens to expose it.

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

## Adding, sizing and removing tests

- **Extend before you create.** Add a case to an existing `parametrize` or
  table first, then a function in the existing file for that module. In an
  existing file, reuse its fixtures unless they stand in for FreeToken objects;
  build new setup with real objects (below).
  A new file, directory, fixture or helper needs a reason.
- **One strong case beats several weak ones.** Each extra case must guard a
  different failure. If the test diff dwarfs the change, cut setup and
  duplicate cases, not the independent check.
- **New models.** A model's own tests go in `models/<model>/` and meet the same
  bar: parity against its reference implementation, and properties specific to
  it (chunked prefill against single-shot, its vision tower against the
  reference tower). What every model needs comes from shared tests: the model
  adds rows where they keep a table (a new tool-call format in
  `server/test_streaming_model_matrix.py`, a new attention kind in
  `engine/test_attention_backend_matrix.py`), and live-registry tests such as
  `server/test_parser_auto_selection.py` and `models/test_models_registry.py`
  cover it with no change. Do not copy config, registry or smoke tests into the
  model's directory.
- **Remove what a change makes redundant.** If a change makes a test wrong,
  update it. If it makes a test redundant, propose deleting it: a case can go
  when deleting it lets no regression past the remaining tests.

## Writing a test

- **Real objects on the lightest real path.** Build FreeToken's own objects
  (requests, batches, pools) with their real constructors; do not stand them in
  with `SimpleNamespace` or `MagicMock`, set private attributes, or patch methods
  to reach a state. Objects that need a model to construct (the scheduler, the
  engine) may be stood in for, limited to the attributes the code under test
  reads. Fakes are also fine for what comes from outside the engine: an HF
  config, a checkpoint file, an HTTP peer. A kernel or layer test does not need
  a server or a model.
- **Show it catches the bug.** Revert only the fix's logic and check that the
  test fails on the assertion that guards the bug, not on an import or
  signature error; then restore the fix and check it passes. Put both runs in
  the PR.
- **Deterministic and tight.** Unit and kernel tests use greedy sampling or a
  fixed seed (the AIME gate samples on purpose, see above). Assert equality
  where the result is exact, and keep tolerances as tight as the math allows.
  Order a race with events and wait on a condition with a timeout, never a
  fixed `sleep`.
- **Never loosen a test to make it pass.** No wider `atol`/`rtol`, lower
  thresholds, retries, or skips to turn a red test green; skipping on missing
  hardware, packages or weights is fine. Find out whether the failure comes
  from your change or also happens on `main`.
