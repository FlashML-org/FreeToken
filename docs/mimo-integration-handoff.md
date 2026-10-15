# MiMo Upstream Integration Handoff

## Merge Provenance

Worktree: `C:\DockerContainers\free-token-mimo-2.6-flash\integration-main`.
Branch: `integrate-mimo-upstream-main`.
Pre-commit HEAD / required first parent: `8f767225f99c5c1c89a77366a7dd87674175669d`.
Pinned upstream / required second parent: `f02fbb16a213f5c81094e4017c988d41f0164141`.
Merge base: `0d652e73a452d014ac5441a15baa75348e9fcb0a` (12 baseline-only / 29 upstream-only commits).

Independent final review reports no remaining findings; the user has authorized
a provisional local merge commit after assessing the independent tests. Native
validation remains deferred. No subagents were used by the execution agent.
This record is prepared before commit; retrieve its resulting SHA, date and parents:

```bash
git show --no-patch --format="%H%n%cI%n%P%n%s" integrate-mimo-upstream-main
```

The user authorized only the upstream relocation of `kvcache/dsv4_cost_model.py`
to `kvcache/dsv4/v4_cost_model.py` and `kvcache/dsv4_paged_pool.py` to
`kvcache/dsv4/v4_pool.py`. No other file deletion is authorized.

## Integration Changes

- Resolve shared-table RoPE with positive even strides <= 2048 and keep the
  FlashInfer width whitelist only on dispatch, so MiMo 192/64 uses Triton.
- Combine ROCm exclusion with real-import FlashInfer diagnostics; keep both
  MiMo and upstream DeepSeek-V4.1 parser inference tests.
- Preserve declared dense dtypes and FP32 sinks/routing bias. Select the resident
  expert method during fused FTW conversion, rather than its BF16 offload layout.
- Cast unquantized resident FTW rows directly from stored dtype to requested model
  dtype. Quantized bank codes/scales remain unconverted.
- Reject unavailable/stale shared-IO native ABI before explicit CPU/Hybrid
  allocation; automatic CPU admission returns false. Native C++/CUDA code is
  unchanged from the pinned upstream.
- Migrate resident state-dict assumptions to bank shapes, row contents, dtype,
  dense-prefix-adjusted bank IDs, and the engine's actual bank attachment method.
  Keep exact FP16 65504 / FP32 1.001953125 sentinel assertions and tied-head tests.
- Add packing-only resident ModelOpt NVFP4 coverage and CPU backend/RoPE guards.
  Parser literal policy and pinned template bytes are not rewritten.

## Compatibility Checks

`docs/integration_compatibility.py` verifies source imports/hashes and syntax,
then either runs pytest (if already installed) or a production-only `--smoke`
harness. It is local integration tooling, not upstream native validation.

Existing image:
`sha256:b59ae52e1d477ea5451a508e31f2d45f9f9db2d1e7d899eea5f372ffacdc700e`.
Observed: Python 3.12.3, torch 2.11.0+cu130, Triton 3.6.0, flashlib 0.3.0,
Transformers 5.16.1, apache-tvm-ffi 0.1.13.post3. Pytest is absent.

All new containers have network disabled, 8 GiB memory/swap limits, 4 CPUs,
256 PID limit, no device requests, no ports or models. Integration is the only
bind mount, read-only at `/workspace`. Source-only imports are verified there;
no installed native extension is injected into the source package.

The attempted pytest selection was blocked before collection. It requested
the five parser/API suites, three MiMo config/numerics/weight suites, FTW dtype
conversion, backend, and rotary tests. No pytest pass/skip count is claimed.

The standard-library smoke harness checks six source/FTW dtype cases, direct
FP32 FTW -> FP16 bank replay, sparse/grouped-routing rejection, resident NVFP4
packing, dual partial RoPE dispatch/frequencies, literal tool values with fragmented
delivery, opaque reasoning markers, the six-case authored template/factory matrix,
and unavailable native ABI rejection. It never executes GPU kernels.

Logs and generated tiny safetensors/FTW artifacts remain inside the named
containers; `docker logs <name>` retrieves logs. No containers were removed.
The first probe had a command-quoting syntax error. Smoke r1-r4 retained
development failures (oracle arithmetic order, Jinja filter setup, bypassed
EngineConfig quant binding, and native import preceding its availability gate).
No regression tolerance was relaxed to address them.

Final smoke container: `mimo-integration-cpu-20261014-smoke-r6`, exit code 0.
Its retained synthetic artifacts are `/tmp/integration-smoke-v_dm4eid`.
The final run mixes FP32 gate pieces with BF16 up/down pieces before packing;
the preceding r5 run also passed with all-FP32 expert source tensors.
Retrieve final output with `docker logs mimo-integration-cpu-20261014-smoke-r6`.

Exact final command (PowerShell):

```powershell
docker run --name mimo-integration-cpu-20261014-smoke-r6 --network none --memory 8g --memory-swap 8g --cpus 4 --pids-limit 256 -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/workspace/python:/workspace -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e XDG_CACHE_HOME=/tmp/integration-cache -e TRITON_CACHE_DIR=/tmp/integration-triton -e TORCHINDUCTOR_CACHE_DIR=/tmp/integration-inductor -v "C:\DockerContainers\free-token-mimo-2.6-flash\integration-main:/workspace:ro" -w /workspace --entrypoint /opt/freetoken/venv/bin/python sha256:b59ae52e1d477ea5451a508e31f2d45f9f9db2d1e7d899eea5f372ffacdc700e docs/integration_compatibility.py --smoke
```

The named container is retained; choose a new unused name for any later run.

## Independent Review Follow-Up

The initial ABI guard covered the executor constructor and viability helper, but
independent review found that the real recommending-profile path bypassed it and
bank/cache allocations preceded explicit rejection. Those P1 findings are fixed:

- `_adjust_config` now uses full ABI/format/activation viability for profile-driven
  auto -> hybrid, and checks explicit CPU/Hybrid/nonzero CPU-layer choices before
  model-weight loading. Zero CPU layers do not require a native executor.
- `_init_offload_moe_cache` defensively checks resolved CPU needs before pin-budget
  checks, host-bank loading, or GPU-cache construction, including callers that
  bypass config adjustment. MXFP4 admission checks its actual `mxfp4_triton` registry
  entry rather than assuming format support.
- `tests/engine/test_cache_budget.py` drives a usable v5 recommending profile with
  absent/stale ABI, missing format, unsupported activation, and a positive current
  case. It also asserts early-failure ordering for explicit CPU/Hybrid/CPU layers.

The P2 hybrid-fetch coverage loss is fixed in `tests/moe/test_hybrid_fetch.py`:
the oracle exhaustively minimizes overlap time, without copying the kernel's
integer-neighbor formula. Real `ensure_experts_hybrid` calls are checked against
timestamp-LRU policy and complementary route outputs across fixed/fraction caps,
repeated experts, hits/misses, both clamp modes and both fetch-priority policies.
GPU route graph replay alternates CPU-work/no-work routing. The existing native
CPU graph test also alternates mixed/fully inactive rows and asserts exact zeros
on CPU-free replays without relaxing its numerical tolerance. Native/GPU execution
is deferred; the native replay test explicitly skips when its ABI is absent.

`docs/cli.md` now explains the resident-only FP16/FP32 FTW `bf16`-label limitation
and deliberate reconversion to a separate BF16 offload output, without automatic
repacking or reinterpretation.

Follow-up compatibility container: `mimo-integration-cpu-reviewfix-20261014-r1`,
exit code 0; artifacts: `/tmp/integration-smoke-phgupz6e` inside it. It uses the
same capped, no-GPU/no-network image command above with only the container name
changed. Logs remain available through `docker logs`.
The expanded harness exercises the actual v5 profile path, early allocation
ordering with forbidden allocation sentinels, and the pure balance-oracle
self-test outside pytest. Pytest remains absent: no suite pass/skip counts or
GPU/native replay results are claimed. Earlier smoke evidence is retained, not
substituted for the independent review finding.

The local harness and handoff remain staged for coordinator evidence. Their final
repository disposition is still an owner decision; neither file was deleted.

## Review Gates

- Obtain a pytest-capable environment without silently installing or upgrading
  the baseline image. Re-run the changed and existing regressions; the smoke
  harness is not a substitute for pytest.
- Native validation requires a separately authorized Python 3.12 Linux stack:
  torch 2.14.1/cu130, torchvision 0.29.x, Triton 3.8, flashlib 0.4.0,
  sglang-kernel 0.4.9 when selected, and rebuilt C++20/multi-source extensions.
  FlashInfer remains 0.6.18.post1 with matching optional cache companions.
- Review FTW dtype semantics. FP16/FP32 unquantized resident bank entries retain
  the legacy `bf16` format label but carry exact per-entry dtypes. They are not
  directly interchangeable with a BF16 offload/CPU layout: upstream cache schema
  checks reject mismatches. No automatic large host repack or legacy dense-expert
  FTW compatibility layer was added; any expansion needs an owner decision.
- Run the unchanged tiny production full/SWA cached-attention regression against
  upstream's new mask/tile/log2-softmax paths, including sink/window mutation
  sensitivity. No GPU, scheduler-reuse, or graph-replay evidence exists here.
- Test CPU/Hybrid eager and graph replay with fresh native binaries, including
  route partitioning, inactive slots, skipped CPU work, shared IO, synchronization,
  and ROCm rejection/fallback behavior. The Python ABI probe checks exported
  capabilities, not numerical correctness or provenance of a compiled binary.
- Keep the validated baseline branches, detached baseline worktree, stash
  `c433bbeb67ac84f30e1f296306819b9b9db2fe49`, and live port-8080 server untouched.

No installs, builds, upgrades, commits, pushes, Git configuration changes,
existing-container exec/restarts, large checkpoint loads, or GPU allocations
were performed in this phase.

## Independent Pytest Validation

The preceding pytest-unavailable, no-install, and smoke-only statements describe
the earlier integration/review-fix phases. They remain truthful historical records.
This subsequent independent phase obtained pytest with explicit user permission
inside NEW isolated containers only. It supersedes the earlier pytest availability
blocker for the focused old-stack selections below, not the fresh-native/GPU review
gates. The user separately authorized appending this report; no staging or commit
was performed by the validation agent.

Verified HEAD: `8f767225f99c5c1c89a77366a7dd87674175669d`.
Verified MERGE_HEAD: `f02fbb16a213f5c81094e4017c988d41f0164141`.
The tests exercised the pending-merge working tree, not a committed merge.
Docker UTC timestamps are October 15, 2026; container names retain the October 14
session date.

### Results

| Selection | Passed | Failed | Skipped | Deselected | Collection errors | Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 MiMo config/numerics/weight and FTW dtype conversion | 74 | 0 | 0 | 0 | 0 | 0 |
| R1 RoPE/MRoPE and backend/architecture matrix | 100 | 0 | 5 | 0 | 0 | 0 |
| R1 parser/reasoning/streaming/wire/API/effort/accounting | 743 | 0 | 0 | 0 | 0 | 0 |
| R1 engine cache-budget/admission and hybrid oracle/profile | 42 | 0 | 31 | 0 | 0 | 0 |
| R1 GPU/native selection | 0 | 0 | 26 | 0 | 0 | 0 |
| R2 tokenizer/DSV4 reasoning/Responses API/Triton fallbacks | 106 | 0 | 17 | 13 | 0 | 0 |
| Aggregate | 1065 | 0 | 79 | 13 | 0 | 0 |

R1 additionally ran the unchanged `docs/integration_compatibility.py --smoke`
harness successfully. Its assertions are separate smoke evidence, not additional
pytest passes. Both containers exited 0 and were inspected as not OOM-killed.

Skip/deselection breakdown:

- 5 MRoPE cases: `needs CUDA`.
- 28 hybrid policy cases: `hybrid ensure requires CUDA and the matching Triton/Flashlib stack`.
- 2 hybrid replay cases: `hybrid ensure graph replay requires CUDA and the matching kernel stack`.
- 1 cache-budget case: `WSL caps pinning`.
- 1 MiMo cached-attention case: `production Triton cached attention and KV store require CUDA`.
- 2 MiMo SWA decode cases: `store_kv drives the Triton/CUDA store kernel; SWA decode numeric test needs CUDA`.
- 23 native CPU MoE cases: `needs CUDA`; the module-level CUDA skip precedes native execution.
- 17 Triton fallback cases (9 activation, 6 norm, 2 sampling): `a CUDA or ROCm GPU is required`.
- 13 checkpoint-dependent tokenizer cases: deselected by `not slow and not needs_weights`.

No dependency-incompatible collection or executed code failure occurred. The
separate source-package probe found `freetoken.kernel._cpu_moe` spec `None` and
`compiled_extension_current() == False`: native execution is unavailable, not a
validated native failure or success. The CUDA skip counts must not be relabeled
as native-ABI skips. Warnings were existing FlashInfer/CUTLASS deprecations and
FastAPI/Starlette TestClient's httpx deprecation; no packages were changed to
suppress them.

### Isolation And Dependencies

Image for both runs:
`sha256:b59ae52e1d477ea5451a508e31f2d45f9f9db2d1e7d899eea5f372ffacdc700e`.
The image was inspected before execution. Retained NEW containers:

- `mimo-independent-pytest-20261014-r1`, ID `efc1dca215f82752ef0b01402ead9ccc2297c7746d20e491f7b5afa49fa1670c`, UTC `00:22:34` through `00:23:56` on October 15.
- `mimo-independent-pytest-20261014-r2`, UTC `00:24:35` through `00:24:52` on October 15.

Each used 8 GiB memory and memory/swap cap, 4 CPUs, 256 PIDs, no GPU/device
requests, no published ports, no model mounts, and exactly one bind mount:
the integration working tree read-only at `/review-src`. The inherited image
metadata exposes port 8080, but no port was published. Existing containers and
the baseline server were not entered, restarted, or changed. No container was
removed; `--rm` was not used.

Bridge networking enabled the small pure-Python testing downloads. HF and
Transformers offline flags remained enabled during tests. The following exact
command ran inside each NEW container, targeting only its `/tmp` directory:

```bash
uv pip install --target /tmp/validation-deps --no-deps --only-binary :all: pytest==8.4.2 iniconfig==2.1.0 packaging==25.0 pluggy==1.6.0 pygments==2.19.2 pytest-asyncio==1.2.0
```

No application dependency resolution, CUDA/PyTorch/Triton/FlashInfer/Flashlib
installation or upgrade, or native build was performed. Source and tests were
not modified or injected with new global mocks. Pytest plugin autoload was
disabled; R2 explicitly loaded `pytest_asyncio.plugin`. R1 tests use their
existing synchronous/`asyncio.run` call patterns and passed without that plugin.

Observed versions (R2 uses direct `importlib.metadata.version` lookups):

| Package | Version |
| --- | --- |
| Python | 3.12.3 |
| pytest | 8.4.2 |
| iniconfig | 2.1.0 |
| packaging | 25.0 |
| pluggy | 1.6.0 |
| Pygments | 2.19.2 |
| pytest-asyncio | 1.2.0 |
| torch | 2.11.0+cu130 |
| torchvision | 0.26.0+cu130 |
| triton | 3.6.0 |
| flashlib | 0.3.0 |
| transformers | 5.16.1 |
| apache-tvm-ffi | 0.1.13.post3 |
| flashinfer-python | 0.6.18.post1 |
| sglang-kernel | 0.4.5+cu130 |
| fastapi | 0.142.1 |
| starlette | 1.7.0 |
| httpx | 0.28.1 |
| safetensors | 0.8.0 |
| Jinja2 | 3.1.6 |

R1's complete `PACKAGES` log enumerates both installed distributions and the
temporary overlay. Its dictionary overwrites duplicate distribution names with
baseline entries for packaging/Pygments; it is not the effective-import version
oracle. The pinned install output and R2 direct lookups confirm the overlay
versions above. The native stack remained the image's existing stack.

### Container Commands

The exact `-c` Python runner arguments are retained in each container's
`Config.Cmd`, encoded with base64 to survive PowerShell quoting. The runner
installs the six pinned packages, verifies provenance, prints commands and
results, and runs subprocesses. It does not edit repository files. The following
PowerShell reconstruction retrieves that exact existing runner payload; replace
the replay name with an unused name before any separately authorized rerun:

```powershell
$record = (docker inspect mimo-independent-pytest-20261014-r1 | ConvertFrom-Json)[0]
$payload = $record.Config.Cmd[1]
docker run --name mimo-independent-pytest-REPLAY-r1 --network bridge --memory 8g --memory-swap 8g --cpus 4 --pids-limit 256 -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/review-src/python:/review-src -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e XDG_CACHE_HOME=/tmp/validation-cache -e UV_CACHE_DIR=/tmp/validation-uv -e TRITON_CACHE_DIR=/tmp/validation-triton -e TORCHINDUCTOR_CACHE_DIR=/tmp/validation-inductor -e OMP_NUM_THREADS=4 -v "C:\DockerContainers\free-token-mimo-2.6-flash\integration-main:/review-src:ro" -w /review-src --entrypoint /opt/freetoken/venv/bin/python sha256:b59ae52e1d477ea5451a508e31f2d45f9f9db2d1e7d899eea5f372ffacdc700e -c $payload
```

The actual run used name `mimo-independent-pytest-20261014-r1`. R2 used the
identical outer flags with name `mimo-independent-pytest-20261014-r2` and its own
`Config.Cmd[1]` payload. `docker inspect` preserves all inherited image environment
variables, the complete runner, mount, limits, state, and image provenance;
`docker logs` preserves `INSTALL`, `ENV` (R1), `VERSIONS` (R2), `PROVENANCE`,
`RUN`, `RESULT`, smoke output, and final outcomes.

R1 created a container-local `/workspace -> /review-src` symlink so the unchanged
harness's hardcoded `/workspace` root worked. Before imports it changed
`PYTHONPATH` to `/tmp/validation-deps:/workspace/python:/workspace` and prepended
those paths to its interpreter's `sys.path`; subprocesses inherited that value.
R1 subprocess working directory was `/workspace`. R2 used
`PYTHONPATH=/tmp/validation-deps:/review-src/python:/review-src`, prepended those
paths to `sys.path`, and ran from `/review-src` without the symlink. Both asserted
`not torch.cuda.is_available()`. All temporary/cache paths were container-local.

### Exact Test Commands

These are the full executed pytest subprocess commands, including explicit file
selections. R1 ran each selection in a fresh Python subprocess:

```bash
/opt/freetoken/venv/bin/python -m pytest tests/models/test_mimo_v2_config.py tests/models/test_mimo_v2_numerics.py tests/models/test_mimo_v2_weight.py tests/checkpoint/test_convert_dense_dtypes.py -q -o addopts= -p no:cacheprovider --basetemp=/tmp/validation-mimo --junitxml=/tmp/validation-mimo.xml -m "not slow and not needs_weights" -ra

/opt/freetoken/venv/bin/python -m pytest tests/kernels/test_rotary.py tests/kernels/test_mrope.py tests/kernels/test_backend.py tests/engine/test_attention_backend_arch.py tests/engine/test_attention_backend_matrix.py -q -o addopts= -p no:cacheprovider --basetemp=/tmp/validation-rope-backend --junitxml=/tmp/validation-rope-backend.xml -m "not slow and not needs_weights" -ra

/opt/freetoken/venv/bin/python -m pytest tests/server/test_function_call_parser.py tests/server/test_reasoning_parser_all_models.py tests/server/test_parser_auto_selection.py tests/server/test_streaming_model_matrix.py tests/server/test_message_wire.py tests/server/test_openai_api.py tests/server/test_anthropic_api.py tests/server/test_api_key.py tests/server/test_effort_dialect.py tests/server/test_generation_accounting.py tests/tokenizer/test_effort.py -q -o addopts= -p no:cacheprovider --basetemp=/tmp/validation-parser-api --junitxml=/tmp/validation-parser-api.xml -m "not slow and not needs_weights" -ra

/opt/freetoken/venv/bin/python -m pytest tests/engine/test_cache_budget.py tests/moe/test_hybrid_fetch.py -q -o addopts= -p no:cacheprovider --basetemp=/tmp/validation-engine-hybrid --junitxml=/tmp/validation-engine-hybrid.xml -m "not slow and not needs_weights" -ra

/opt/freetoken/venv/bin/python -m pytest tests/models/test_mimo_v2_cached_attention.py tests/models/test_mimo_v2_swa_decode.py tests/moe/test_cpu_moe.py -q -o addopts= -p no:cacheprovider --basetemp=/tmp/validation-gpu-native --junitxml=/tmp/validation-gpu-native.xml -m "not slow and not needs_weights" -ra

/opt/freetoken/venv/bin/python docs/integration_compatibility.py --smoke
```

R2 executed:

```bash
/opt/freetoken/venv/bin/python -m pytest tests/tokenizer/test_tokenize.py tests/server/test_reasoning_parser_dsv4.py tests/server/test_responses_api.py tests/kernels/test_triton_fallbacks.py -q -o addopts= -p no:cacheprovider -p pytest_asyncio.plugin --basetemp=/tmp/validation-extra --junitxml=/tmp/validation-extra.xml -m "not slow and not needs_weights" -ra
```

### Import Provenance And Artifacts

R1 imported the following production modules and asserted their resolved paths
were under `/review-src/python`. Their logged `__file__` paths used the
`/workspace/python` symlink. The unchanged smoke subprocess verified the same
paths/hashes independently. No installed native extension was injected into the
source package. R2 separately checked the resolved `freetoken` package location.

| Module | SHA-256 of imported source |
| --- | --- |
| freetoken | 095e182f33349d2b2e7ea32694b3de3ca403ceed972342f3cab7d2dd0419e1ee |
| freetoken.engine.engine | 7e81433ac7d4d2fd87c73d98acb6024438be6cfd3bf0a9ce83a795974f03fc8b |
| freetoken.models.mimo_v2.model | 075489cb502f615f5ce4cb860a0af332a82890410a186f79a3adca0cb20c319a |
| freetoken.server.function_call_parser | f163a501a6c100cef23ff253f4760eb9c0ff958a742ceba2e47832db2f485cc2 |
| freetoken.server.reasoning_parser | 7ff3b00a72ae7c15b0f36e91d7c87975728fa02799f8fd1ebec312c5ab83cacc |
| freetoken.checkpoint.convert | f19cc89841cde83b8dc8d4d66bd924b6a65ab9281d34a8e83c50c521ae0a7426 |
| freetoken.moe.cpu_executor | b98ee90fa87125b9d0c4b6fa9f549460e3cb365b664275d188e52b23495a0546 |

Retained in R1:

- `/tmp/validation-mimo.xml`, `/tmp/validation-rope-backend.xml`, `/tmp/validation-parser-api.xml`, `/tmp/validation-engine-hybrid.xml`, `/tmp/validation-gpu-native.xml`.
- Matching `/tmp/validation-<selection>` pytest temporary directories, including tiny synthetic safetensors/FTW fixtures where generated.
- Smoke synthetic artifacts: `/tmp/integration-smoke-2o0ztx6j`.

Retained in R2: `/tmp/validation-extra.xml` and `/tmp/validation-extra`.
Installation/cache paths in both: `/tmp/validation-deps`, `/tmp/validation-uv`,
`/tmp/validation-cache`, `/tmp/validation-triton`, `/tmp/validation-inductor`
(cache directories are used only as needed). Logs and exact launch records:

```powershell
docker logs mimo-independent-pytest-20261014-r1
docker logs mimo-independent-pytest-20261014-r2
docker inspect mimo-independent-pytest-20261014-r1
docker inspect mimo-independent-pytest-20261014-r2
```

### Assessment And Remaining Gates

No reproducible test defect was found in the executed cases. Newly added FTW
conversion/replay, resident bank attachment, exact FP16/FP32 sentinel, tied-head,
engine early-allocation ordering, ROCm exclusion, and RoPE cases were examined
and passed without changing assertions or production code. Parser literal-value,
fragmentation, reasoning-marker, and authored/pinned-template cases passed.

The independent exhaustive hybrid balance oracle and real profile readers passed
on CPU. Production `ensure_experts_hybrid` asserts CUDA input and has no CPU
execution path; the 28 policy/kernel comparisons and 2 hybrid graph-replay cases
only collected/skipped. The native CPU-work/no-work replay test likewise did not
execute. These results do not establish cache eviction/routing, native shared-IO
synchronization, GPU attention numerics, scheduler reuse, or graph-replay parity.

This is focused compatibility evidence on Torch 2.11, NOT final validation for
the new upstream stack. The separately authorized Torch 2.14.1/cu130,
torchvision 0.29.x, Triton 3.8, flashlib 0.4.0, selected sglang-kernel 0.4.9,
and rebuilt native extensions remain required. Real ROCm hardware behavior,
GPU/native selections, checkpoint-dependent deselections, and the complete
repository suite remain unvalidated here. The old-stack focused pytest blocker
is resolved; the fresh-stack/native/GPU review gates above remain open.

## Final Auto-Resolution Admission Fix

Further rereview found that `auto` with explicit nonzero CPU layers escaped the
configuration-time admission check on discrete GPUs. The same check now runs
after auto strategy resolution, before any weight loading. Unified-memory auto
resolution can still select fused and discard CPU-layer flags without requiring
the native executor. No other strategy or activation policy was changed.

`test_explicit_cpu_paths_fail_before_bank_and_cache_allocation` now includes auto
with explicit CPU layers and forces the discrete-memory path. The new
`test_auto_unified_memory_discards_cpu_layers_before_native_admission` checks
that fused selection clears those flags and never invokes native admission.

The independent pytest report above is preserved in full. Its 1065 passes, zero
failures, 79 skips and 13 deselections describe the tree before this final edit,
not validation of the relocated check or new cases. The next validation agent
must rerun focused real pytest. No pytest, manual smoke, container operation,
installation, build, or commit was performed in this final-edit phase.

## Independent Final-Tree Admission Validation

This subsequent independent validation reran the final auto-resolution admission
change above with explicit user permission. The preceding request to rerun pytest
and the final-edit phase's no-test/no-container statements are historical and are
now followed by the actual results below. The earlier 1065-pass aggregate remains
evidence for the pre-final-edit tree; it is NOT combined with this rerun or claimed
as a final-tree full-suite result.

HEAD and MERGE_HEAD were independently reverified as
`8f767225f99c5c1c89a77366a7dd87674175669d` and
`f02fbb16a213f5c81094e4017c988d41f0164141`. Tests ran against the pending-merge
working tree. No subagents, production/test edits, staging, or commits were used.

### Behavior Review

The validator reviewed `_adjust_config` and the changed test call patterns:

- The unchanged native admission predicate now runs after auto strategy resolution. Discrete-memory auto resolves to an offload-family strategy; explicit nonzero CPU layers therefore require the current ABI before weight/bank/cache loading.
- Explicit CPU and Hybrid choices still require admission. Zero CPU layers remain exempt. The five recommending-profile cases still distinguish absent/stale ABI, wrong format, unsupported activation, and a viable current executor.
- For the supported unquantized/silu unified-memory fixture, auto selects resident fused and clears the CPU-layer flag before the native guard. The new test uses a fail-on-call admission sentinel and asserts both `moe_strategy == "fused"` and `moe_cpu_layers is None`.
- The extended discrete-memory parameterization drives both `_adjust_config` and direct `_init_offload_moe_cache` with absent/stale ABI and forbidden allocation/pinning sentinels. Both added auto cases passed, as did the unified-memory case.
- The separate earlier activation-policy gate was not moved; this change and its new test do not assert that unsupported expert activations are newly admitted on unified memory.

No reproducible code/test defect was found in the reviewed change or executed
cases. No new global mocks or weakened assertions were introduced by validation.

### Exact Execution

NEW retained container: `mimo-independent-final-admission-20261014-r1`.
Docker inspection confirmed exit 0, `OOMKilled=false`, UTC start
`2026-10-15T00:30:25.336339775Z` and finish
`2026-10-15T00:30:50.872255618Z`. The inspected image was exactly
`sha256:b59ae52e1d477ea5451a508e31f2d45f9f9db2d1e7d899eea5f372ffacdc700e`.
Memory and memory/swap were each 8589934592 bytes, CPU limit was 4, PID limit
256, device requests were null, port bindings empty, and AutoRemove false.
Only the integration working tree was mounted, read-only at `/review-src`.
No GPU/model mounts, published ports, native installs/upgrades/builds, existing
container operations, baseline-server access, or container removals occurred.

Exact installation command inside this NEW container:

```bash
uv pip install --target /tmp/validation-deps --no-deps --only-binary :all: pytest==8.4.2 iniconfig==2.1.0 packaging==25.0 pluggy==1.6.0 pygments==2.19.2 pytest-asyncio==1.2.0
```

Bridge networking was enabled for these small pure-Python testing downloads;
HF/Transformers offline flags stayed enabled. After installation the runner set
`PYTHONPATH=/tmp/validation-deps:/review-src/python:/review-src`, prepended the
same paths to `sys.path`, and ran from `/review-src`. No `/workspace` symlink
was needed. It asserted `not torch.cuda.is_available()`, verified source imports,
and compared the engine and both selected test-file hashes before/after execution.

Exact FULL-file pytest commands, each in a fresh Python subprocess, with no
`-k`, marker, or node-selection filter:

```bash
/opt/freetoken/venv/bin/python -m pytest tests/engine/test_cache_budget.py -v -o addopts= -p no:cacheprovider --basetemp=/tmp/validation-final-cache-budget --junitxml=/tmp/validation-final-cache-budget.xml -ra

/opt/freetoken/venv/bin/python -m pytest tests/moe/test_hybrid_fetch.py -v -o addopts= -p no:cacheprovider --basetemp=/tmp/validation-final-hybrid-fetch --junitxml=/tmp/validation-final-hybrid-fetch.xml -ra
```

The hybrid file's three CPU cases executed; its 30 GPU cases collected and
skipped themselves. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` was used; these tests
require no async plugin. The runner's exact base64-encoded Python `-c` payload
is preserved in the retained container's `Config.Cmd[1]`. The following records
the actual outer invocation/environment, retrieving that payload for provenance
(this is a record/reconstruction, not an instruction to restart the container):

```powershell
$record = (docker inspect mimo-independent-final-admission-20261014-r1 | ConvertFrom-Json)[0]
$payload = $record.Config.Cmd[1]
docker run --name mimo-independent-final-admission-20261014-r1 --network bridge --memory 8g --memory-swap 8g --cpus 4 --pids-limit 256 -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/review-src/python:/review-src -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e XDG_CACHE_HOME=/tmp/validation-cache -e UV_CACHE_DIR=/tmp/validation-uv -e TRITON_CACHE_DIR=/tmp/validation-triton -e TORCHINDUCTOR_CACHE_DIR=/tmp/validation-inductor -e OMP_NUM_THREADS=4 -v "C:\DockerContainers\free-token-mimo-2.6-flash\integration-main:/review-src:ro" -w /review-src --entrypoint /opt/freetoken/venv/bin/python sha256:b59ae52e1d477ea5451a508e31f2d45f9f9db2d1e7d899eea5f372ffacdc700e -c $payload
```

The existing name is retained and cannot be reused for a future run. Inspection
preserves the complete encoded runner and inherited image environment. Logs
preserve `INSTALL`, `ENV`, `VERSIONS`, `PROVENANCE`, `FINAL_TREE_HASHES`, full
verbose test node outcomes, `RUN`, `RESULT`, and `HASHES_UNCHANGED`.

### Versions And Provenance

Direct metadata lookups reported Python 3.12.3; pytest 8.4.2; iniconfig 2.1.0;
packaging 25.0; pluggy 1.6.0; Pygments 2.19.2; pytest-asyncio 1.2.0;
torch 2.11.0+cu130; torchvision 0.26.0+cu130; Triton 3.6.0; flashlib 0.3.0;
Transformers 5.16.1; apache-tvm-ffi 0.1.13.post3;
flashinfer-python 0.6.18.post1; sglang-kernel 0.4.5+cu130.
No native-stack dependency was installed or changed.

All five imported production modules were asserted to resolve beneath
`/review-src/python`; source SHA-256 hashes:

| Module | SHA-256 |
| --- | --- |
| freetoken | 095e182f33349d2b2e7ea32694b3de3ca403ceed972342f3cab7d2dd0419e1ee |
| freetoken.engine.engine | 9d6d1db066961acd47b85e71139d49e83b93bddb0e4c483ae41262426a8b9db1 |
| freetoken.engine.cache_budget | fd91b3f018ccf01d41df02d4048baa2f83c30a887dd41643351cfe362fcc85c4 |
| freetoken.moe.bench_profile | 7498a25b8b1e965e1d511be09b419997d0420166d6558870bfa0e9c51524130c |
| freetoken.moe.cpu_executor | b98ee90fa87125b9d0c4b6fa9f549460e3cb365b664275d188e52b23495a0546 |

The engine hash above and these selected test hashes were unchanged before/after
both subprocesses, identifying the actual final-tree files tested:

| Test file | SHA-256 |
| --- | --- |
| tests/engine/test_cache_budget.py | ef55e820050948fd4b906a15b6a205cca6ef44e171db91d3e46a79f8f8c49c15 |
| tests/moe/test_hybrid_fetch.py | c372842d2a834e397ea932c0b684dc7d740981858be81e3bace8c852f01f5ebf |

The native spec probe again returned `None` for
`freetoken.kernel._cpu_moe`; `compiled_extension_current()` returned false.
No installed native extension was injected. Mocked capability cases validate
Python admission policy only, not native computation.

### Results And Artifacts

| Full file | Collected | Passed | Skipped | Failed | Errors | Deselected | Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tests/engine/test_cache_budget.py | 43 | 42 | 1 | 0 | 0 | 0 | 0 |
| tests/moe/test_hybrid_fetch.py | 33 | 3 | 30 | 0 | 0 | 0 | 0 |
| This final-tree rerun only | 76 | 45 | 31 | 0 | 0 | 0 | 0 |

Cache-budget's sole skip: `WSL caps pinning` at line 635. Hybrid skips:
28 at line 83, `hybrid ensure requires CUDA and the matching Triton/Flashlib stack`;
2 at line 161, `hybrid ensure graph replay requires CUDA and the matching kernel stack`.
There were no collection/dependency errors. The cache-budget run emitted 10
existing FlashInfer/CUTLASS deprecation warnings; hybrid emitted none.

Retained inside `mimo-independent-final-admission-20261014-r1`:

- `/tmp/validation-final-cache-budget.xml` and `/tmp/validation-final-cache-budget`.
- `/tmp/validation-final-hybrid-fetch.xml` and `/tmp/validation-final-hybrid-fetch`.
- Testing dependency overlay `/tmp/validation-deps` and isolated download/cache locations `/tmp/validation-uv`, `/tmp/validation-cache`, `/tmp/validation-triton`, `/tmp/validation-inductor` as used.

```powershell
docker logs mimo-independent-final-admission-20261014-r1
docker inspect mimo-independent-final-admission-20261014-r1
```

The final relocation and new cases have focused real pytest evidence on the
old Torch 2.11 stack. The previous 1065-pass aggregate is historical and is not
double-counted with these 45 passes. No other suites or docs smoke were rerun in
this final-tree phase. Fresh Torch 2.14.1/native-stack validation, real unified
memory/ROCm hardware, GPU routing/cache policy and graph replay remain open gates.

## Final Approval And Scope

The independent reviewer signed off with no remaining findings and judged a
provisional local integration commit defensible. The user explicitly authorized
the merge commit `feat(mimo): integrate pinned upstream main`. This is not a
production deployment, fresh-native-stack validation, upstream submission, or
claim of large-checkpoint numerical parity. The local harness and all independent
validation appendices are retained as coordinator evidence.

Historical compatibility aggregate: 1065 passed, zero failed, 79 skipped and
13 deselected before the last admission edit. Final-tree full-file rerun:
cache-budget 42 passed / 1 skipped; hybrid-fetch 3 passed / 30 skipped, with zero
failures or collection errors. These overlapping selections are not added to
the historical aggregate. Source/test code is unchanged after that rerun; only
this handoff incorporates final approval and the external container-stop note.

Repository dependency changes relative to the validated baseline:

| Dependency | Baseline | Integration |
| --- | --- | --- |
| Torch runtime/root build/kernel-cache build | `>=2.11,<2.12` | `>=2.14,<2.15` (selected SGL requires `2.14.1`) |
| torchvision | `>=0.26,<0.27` | `>=0.29,<0.30` |
| Triton on Linux | `>=3.6,<3.8` | `>=3.6,<3.9` (CUDA Torch 2.14 stack selects 3.8) |
| flashlib | `==0.3.0` | `==0.4.0` |
| optional sglang-kernel | `==0.4.5` | `==0.4.9` |
| NumPy | `>=2.0,<2.5` | `>=2.0,<3` |

Unchanged: FlashInfer `[cu13]==0.6.18.post1`, apache-tvm-ffi `==0.1.13.post3`,
Transformers `>=5.16,<5.17`. The native build moves from C++17 to C++20 and adds
per-format sources, runtime-specific GPU synchronization and `gpu_sync_io.cu`.
The Torch 2.11 image/native caches do not validate that new build. No such build
or native dependency upgrade was performed during integration or finalization.

## External Baseline Container Stop

Before finalization the preserved baseline container was observed exited,
unlike the earlier running-state snapshot. The user explicitly reported that
they accidentally stopped `mimo-final-smoke-120k-8080-20261014-1203` and authorized
finalization while leaving it untouched. This is a user-reported external action,
not an inferred diagnosis of the exit code.

- Container ID: `6c4869f431689d364a1ab0a5498cb8c0c6e86ba5e0c55af6f70e60d6bc323c56`.
- Observed status: `exited`; exit code `137`; Docker `OOMKilled=false`; restart count `0`.
- Started: `2026-10-14T21:34:43.066621443Z`; finished: `2026-10-15T00:25:31.245324114Z`.

The execution agent did not stop, restart, enter, or modify this container and
did not diagnose it further. Do not claim it remained running. Protected Git
branch pointers, the detached baseline worktree and protected stash remain
subject to the original preservation checks. Nothing is pushed and no PR is
created as part of this local finalization.
