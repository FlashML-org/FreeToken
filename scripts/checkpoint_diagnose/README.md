# checkpoint_diagnose

Find out why FreeToken does not load a checkpoint, sort the failure into one kind, and repair the checkpoints whose own files are non-standard.
Run everything from the repo root with `PYTHONPATH=python`. Nothing here needs a GPU or the checkpoint's weights.

> [!WARNING]
> The diagnosis is heuristic and can be wrong: the failure kind, the findings and the suggested fixes are a best guess from the headers and configs, not a proof. Check them against the problems it prints before acting on them.

> [!IMPORTANT]
> For agents: explain the diagnosis to the user in plain terms first: what fails, which kind it is, and what the fix would change. Ask the user before you run a fix script or open an issue. Do not change the user's checkpoint on your own.

| Script | Does |
|---|---|
| `diagnose.py` | checks a checkpoint dir or HF repo id, prints the failure kind and the next step |
| `fix_config.py` | level 1 fix: edits only `config.json` / `hf_quant_config.json` |
| `rename_tensors.py` | level 2 fix: renames tensors (shard headers, index, quant skip lists); tensor data is kept byte for byte |
| `fetch_headers.py` | header-only stub of an HF repo; `diagnose.py` calls it for a repo id |
| `ckpt_edit.py` | the edit primitives both fix scripts use, for a case they do not cover |

## Step 1: which format

`diagnose.py` routes by itself, but know where each format goes:

| The checkpoint has | Go to |
|---|---|
| `*.gguf` files | `scripts/gguf_check/` (its README), then the GGUF section below |
| `freetoken_weight.json` (FTW, made by `ft checkpoint`) | diagnose the HF checkpoint it was converted from; FTW repairs are in `docs/ftw-hotfix.md` |
| root `*.safetensors` shards | this directory, from step 2 |
| only `*.bin` / `*.pt`, or MLX weights | not served: FreeToken reads safetensors and GGUF |

### GGUF: three kinds of failure

`scripts/gguf_check/check.py` prints the problems of each file; the first one tells the kind.

| Kind | check.py says | Do |
|---|---|---|
| a ggml type has no kernel | `<layer kind> <type>: no <ops> kernel` (needs `--kernels` from `probe_kernels.py`) | the build or GPU lacks kernels for that quant type: pick a file with another type, or report it with the `probe_kernels.py` table |
| the model's loader does not read the file | `unexpected tensor ...`, `N tensors never loaded`, `tensor ... yielded twice`, `<tensor>: got ..., model has ...`, `MoE layer N ... bank is ..., the model needs ...`, or a `NotImplementedError` from the family's config parse | a FreeToken gap for that layout: report it with the check.py record |
| the GGUF file itself | `is not a GGUF file`, `duplicate GGUF key / tensor`, `runs past the end of the file`, `bad general.alignment`, `tokenizer: no EOS token`, `tokenizer: no chat template`, `chat template: <error>` | the file is broken or incomplete: download it again, or use another file of the repo |

A file whose architecture FreeToken does not serve shows as `skip`, not as a failure.

## Step 2: diagnose a safetensors checkpoint

```bash
python scripts/checkpoint_diagnose/diagnose.py <ckpt dir | HF repo id[@revision]> --out report.jsonl
```

- A local dir is read as is: the headers and JSON files, so a 400 GB checkpoint takes seconds. A weight reader that needs tensor values gets the real bytes it asks for.
- A repo id is fetched as a header-only stub into `~/.cache/freetoken/checkpoint-stubs/` (a few MB). Nothing is downloaded beyond that, and no `.py` file: a repo whose config is its own code (`auto_map`) is skipped, diagnose a local download of it instead.
- A remote repo is only diagnosed: its record has no fixes, and the fix scripts refuse a stub. When it says `fixable`, download the checkpoint and diagnose the local directory to get the fix commands.
- Add `--text-model-only` when the checkpoint will be served with `--text-model-only`.
- `--out` appends one JSON record per checkpoint. The fix scripts read their fixes from it with `--from-report`.

It first checks the files themselves (data offsets cover each shard exactly, the index agrees with the shard headers), then runs the server's own load path on the CPU with meta tensors: config parse, model build, dense weight load through the model's `load_state_dict`, and expert-bank fill for MoE models with the reader the engine picks.
When that fails, it also checks the tensor names against transformers' own model class and its checkpoint conversion rules, and lints the files for the known non-standard forms.

Output, per checkpoint: `PASS`, or `FAIL at <stage>: [<kind>] ...` with the problems, what transformers makes of the names, the findings, and the next step.

## Step 3: the kind of failure

Each failure gets one kind. `diagnose.py` tries, in this order: broken files (`checkpoint`), `arch`, `quant_method`, a broken prefix (`checkpoint`), `quant_model`, names transformers cannot map or another non-standard finding (`checkpoint`), `packing`; anything left is `loader`.

| Kind | Means | Do |
|---|---|---|
| `arch` | the architecture is not registered in `python/freetoken/models/register.py` | not served; a new architecture starts as a feature request |
| `quant_method` | FreeToken has no reader for the quantization format (GPTQ, AWQ, MLX, bitsandbytes, an unsupported compressed-tensors scheme, ...) | not served; use a release in a format `python/freetoken/layers/quantization/configs/` reads |
| `quant_model` | the format is read, but this model or layer kind has no method for it (`no quant method for moe.<kind>`), or the reader does not produce the scales the method needs | a FreeToken gap: report it |
| `loader` | the model and format are served, and transformers loads these names, but FreeToken's reader does not | a FreeToken gap: report it. A finding marked `loader` may offer a fix that works around it |
| `packing` | the expert banks do not fill, and transformers repacks these expert tensors (for example one tensor per expert where the family's release stacks them) | a FreeToken gap: report it. Repacking tensor data is not offered here |
| `checkpoint` | the checkpoint's own files are broken or non-standard: an incomplete shard or index, names transformers would not load either, or a config that contradicts what the shards store | fix it, step 4 |

To report a FreeToken gap, use the "model checkpoint" issue template and paste the `diagnose.py` output.

How to tell `loader` / `packing` from `checkpoint`: the line `transformers ... would not load it either` means transformers' model class and conversion rules cannot map the names either, so the checkpoint is at fault. `transformers loads these names` means the names are a form transformers accepts, so FreeToken should read them too.

## Step 4: fix a non-standard checkpoint

Fixes apply to a local copy of the checkpoint only, never to a remote repo or its stub.

Rules:

1. Fix into a new directory (`--out <new dir>`). Unchanged files there are symlinks to the original, so it costs only the rewritten shards. Use `--in-place` only when the user asks for it. It keeps a backup, and `--restore` undoes the latest in-place edit.
2. One level per round. If the record has a `rename` fix, apply that first, then run `diagnose.py` on the new directory, then apply the config fixes of that new record. A rename changes the names the config findings talk about.
3. Read the plan before applying it. Every run prints the plan first and changes nothing until it is confirmed. Check that:
   - each rule renames the tensors you expect, and the example `old -> new` lines are right;
   - nothing is renamed that should not be. A `warning: ... matches are inside the name` line means the pattern is not anchored;
   - the config edits name the right modules;
   - the shard costs are what you expect. `rewritten, copies N GiB` needs that much free disk.

   If anything in the plan is not what you expected, do not apply it: change the rule or the edit and plan again.
4. Confirming: in a terminal the script asks `[y/N]`. Without a terminal (an agent), pass `--plan <file>`. The first run saves the plan there and changes nothing. Read it, show it to the user, and run the exact same command again to apply it only after the user agrees. If the command or the checkpoint changed in between, the second run saves the new plan instead of applying anything.
5. After every fix, run `diagnose.py` on the result, until it passes or nothing fixable is left.

The usual round, for an agent:

```bash
D=scripts/checkpoint_diagnose
python $D/diagnose.py /models/x --out r0.jsonl
python $D/rename_tensors.py /models/x --out /models/x-fixed --from-report r0.jsonl --plan plan.json   # prints and saves the plan
python $D/rename_tensors.py /models/x --out /models/x-fixed --from-report r0.jsonl --plan plan.json   # after the user agrees to the plan: applies it
python $D/diagnose.py /models/x-fixed --out r1.jsonl
python $D/fix_config.py /models/x-fixed --in-place --from-report r1.jsonl --plan plan.json            # twice, as above
python $D/diagnose.py /models/x-fixed
```

`--from-report` applies the fixes `diagnose.py` found for that checkpoint; anything else is a manual edit, below.

### Fix levels

| Level | Script | Changes | Typical use |
|---|---|---|---|
| 1 | `fix_config.py` | `config.json`, `hf_quant_config.json` | the quantization skip lists, compressed-tensors targets, a config key |
| 2 | `rename_tensors.py` | shard headers, `model.safetensors.index.json`, and the skip-list entries / targets that are module names or globs (a `re:` entry is not rewritten; the plan warns about every entry left naming the old modules) | broken prefixes, legacy tensor names |
| - | not offered | tensor data | repacking experts, dropping tensors, dequantizing: re-export the checkpoint, or report the FreeToken gap |

### Manual edits

When no automatic fix fits, write the edit yourself. It goes through the same plan and confirmation.

```bash
# rename: re.sub(PATTERN, REPL, name), rules in order; anchor with ^ and escape dots
python $D/rename_tensors.py /models/x --out /models/x-fixed --rule '^model\.language_model\.visual\.' 'model.visual.'

# config key: dotted path, JSON value
python $D/fix_config.py /models/x --out /models/x-fixed --set text_config.tie_word_embeddings false

# quantization skip lists, in the dialect's syntax:
#   ModelOpt: glob (model.language_model.layers.*.mlp.gate)
#   compressed-tensors: module name or re:<regex> (re:^model\.visual\..*)
#   fp8: module name (model.language_model.layers.0.linear_attn.in_proj_a)
python $D/fix_config.py /models/x --out /models/x-fixed --skip-add 'model.language_model.layers.*.mlp.gate'
python $D/fix_config.py /models/x --out /models/x-fixed --skip-remove 'lm_head'
```

A rule that renames nothing, and a rename that gives two tensors the same name, are refused. So is an `--out` directory that already has files in it.

### Known non-standard forms

The findings `diagnose.py` prints, and what they fix. `kind` is the failure kind the finding points to.

| Finding | The checkpoint | Level | Fix |
|---|---|---|---|
| `prefix` | nests the text tower more than once (`model.language_model.language_model.`) | 2 | rename to `model.language_model.` |
| `prefix` | stores the vision tower under the text tower (`model.language_model.visual.`) | 2 | rename to `model.visual.` |
| `prefix` | uses vLLM's internal names (`language_model.model.`, `language_model.lm_head.`) | 2 | rename to `model.language_model.` / `lm_head.` |
| `quant_declared_but_dense` | stores modules in bf16 that its quantization_config does not skip (vision tower, lm_head) | 1 | add them to the skip lists |
| `quant_skipped_but_quantized` | stores modules quantized that its skip list covers, often through a container entry (`...linear_attn`) | 1 | remove the entry, and list its dense children instead |
| `ct_target_vllm_names` | writes compressed-tensors targets with vLLM's names, so they match no tensor | 1 | rewrite the targets to `model.language_model.` |
| `quant_unlisted_dense_router` (kind `loader`) | stores in bf16, without listing them as skipped, modules the family serves in bf16 anyway (the Qwen3.5 routers) | 1 | list them; works around the reader |
| `kv_output_scale` (kind `loader`) | stores fp8 KV scales as `k_proj.output_scale` / `v_proj.output_scale` (only reported for the families whose reader drops `k_scale`) | 2 | rename to `k_scale` / `v_scale`, which the reader drops |
| `expert_tensors_incomplete` | lacks some expert's tensors in some layer | - | download it again, or ask its uploader |
| (stage `files`) | has a shard whose tensors do not cover the file, or an index that disagrees with the shards | - | download it again |
| `no_vision_tower` | is text-only under a multimodal architecture | runtime | serve with `--text-model-only` |
| `no_processor_config` | has a vision tower but no `preprocessor_config.json` (image requests fail with 400) | file | copy that file from the base model |
| `fp8_without_config` | stores fp8 weights with no quantization_config | - | not fixable here; the scales and block size are unknown |
| `mlx` (kind `quant_method`) | is MLX-quantized (`.scales` / `.biases`) | - | not served; use a non-MLX release |

## Step 5: check the fixed checkpoint

`diagnose.py` passing means: the names, shapes, quantization methods and expert-bank layout all line up with what the model builds.
It does not read weight values. On a real checkpoint, finish with `ft serve --model-path <fixed dir>` and a short request. For a checkpoint with a vision tower, also send an image request.

## Using the primitives

```python
import sys; sys.path.insert(0, "scripts/checkpoint_diagnose")
from ckpt_edit import Checkpoint, confirm, plan_lines_for_save

ck = Checkpoint("/models/x")
ck.rename_regex([(r"^old\.", "new.")])
ck.quant_skip_add(["model.visual.*"])
print("\n".join(ck.log + plan_lines_for_save(ck, "/models/x-fixed")))
ck.save(out="/models/x-fixed")      # ck.save() edits in place, with a backup
```

`ck.tensors()` gives every tensor's shard, dtype and shape from the headers. `ck.quant_configs()` gives every quantization config the checkpoint carries.
