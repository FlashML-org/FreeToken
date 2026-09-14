# Repairing FTW checkpoints from older builds

An FTW converted by an older FreeToken build may fail to load on the current one. The fix is to
reconvert it with `ft checkpoint`. When the source checkpoint is not on disk, run
`scripts/ftw_hotfix.py` on the FTW dir instead:

```bash
# old Qwen3.6-27B-NVFP4 FTW: fetch the missing input_scale values from the Hub, patch in place
python scripts/ftw_hotfix.py --ftw ~/models/Qwen3.6-27B-NVFP4-FTW --repo nvidia/Qwen3.6-27B-NVFP4

# old DeepSeek-V4 FTW: only renames, nothing to download
python scripts/ftw_hotfix.py --ftw ~/models/DeepSeek-V4-Flash-0731-FTW

# old Qwen3.8-Flash-Next FTW: add the 47.7 GiB PLE table from a local copy of the checkpoint, into a new dir
python scripts/ftw_hotfix.py --ftw ~/models/Qwen3.8-Flash-Next-NVFP4-FTW --source ~/models/Qwen3.8-Flash-Next-NVFP4 --out ~/models/Qwen3.8-Flash-Next-NVFP4-FTW-fixed

# just show what would change
python scripts/ftw_hotfix.py --ftw ~/models/GLM-5.2-NVFP4-FTW --dry-run
```

The script needs the installed `freetoken` package. It downloads only the tensors it needs, by
byte range, never the whole checkpoint; `--revision` pins the Hub revision to read from.

## What breaks and how it is repaired

| Error at load | Checkpoints | Repair | Download |
|---|---|---|---|
| `KeyError: '...input_scale'` | ModelOpt NVFP4 exports with FP8 attention: nvidia/Qwen3.6-27B-NVFP4, RadixArk/Qwen3.8-27B-NVFP4, nvidia/Qwen3.6-35B-A3B-NVFP4 | add the missing `input_scale` scalars | a few KiB |
| `KeyError: 'model.embed.weight'` | deepseek-ai/DeepSeek-V4-Flash-0731 | rename the index entries | none |
| `RuntimeError: Unexpected keys ... .weight_scale` | nvidia/GLM-5.2-NVFP4 | dequantize the old runtime-fp8 weights back to bf16 | none |
| `PLE shard indices are not contiguous 0..N-1: []` | Qwen3.8-Flash-Next (FTWs converted before #420) | write the PLE table as `ple-table-*.safetensors` | 47.7 GiB |

The script decides by itself: it builds the current model from the FTW's `config.json`, compares
the FTW index with the tensors the model declares, and applies only the repairs that are needed.
Before writing anything it checks that the FTW matches the model (shapes, dtypes, byte counts,
shard files) and refuses one that does not. An FTW that loads as is is left untouched.

## What it writes

- In place by default. New tensors go into one appended shard, then the index is replaced
  atomically. Shards left with replaced or dropped entries (GLM-5.2) are compacted one at a time
  into new shard files, each step ending in another atomic index swap, so the FTW is loadable at
  every moment and an interrupted run is finished by running the same command again. While no
  shard has been replaced, the previous index stays as `freetoken_weight.json.bak`.
- `--out <dir>` writes a fresh, compact FTW dir (the dir must be new or empty) and leaves the
  original untouched. Shard files that an interrupted run left behind are removed by the next
  in-place repair and left alone otherwise.
- The PLE table is judged as the engine loads it. A leftover `model.safetensors.index.json` in the
  FTW dir would make the engine look for the table through that file, so the script asks you to
  remove it before it writes the table.
- The plan prints the disk space the run needs. The last line re-checks the result: declared
  tensors still missing, structural problems and shards with dead bytes must all be 0, and the PLE
  table (Flash-Next) must be `complete`. Every shard and index write is synced to disk before the
  index is replaced, so a power loss leaves either the old or the new state.
  Long phases show a progress bar; `-v` prints every step instead.

Not covered: FTWs converted from GGUF, and checkpoints outside [models.md](models.md).
