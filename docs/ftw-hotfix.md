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
byte range, never the whole checkpoint.

## What breaks and how it is repaired

| Error at load | Checkpoints | Repair | Download |
|---|---|---|---|
| `KeyError: '...input_scale'` | ModelOpt NVFP4 exports with FP8 attention: nvidia/Qwen3.6-27B-NVFP4, RadixArk/Qwen3.8-27B-NVFP4, nvidia/Qwen3.6-35B-A3B-NVFP4 | add the missing `input_scale` scalars | a few KiB |
| `KeyError: 'model.embed.weight'` | deepseek-ai/DeepSeek-V4-Flash-0731 | rename the index entries | none |
| `RuntimeError: Unexpected keys ... .weight_scale` | nvidia/GLM-5.2-NVFP4 | dequantize the old runtime-fp8 weights back to bf16 | none |
| `PLE shard indices are not contiguous 0..N-1: []` | Qwen3.8-Flash-Next (FTWs converted before #420) | write the PLE table as `ple-table-*.safetensors` | 47.7 GiB |

The script decides by itself: it builds the current model from the FTW's `config.json`, compares
the FTW index with the tensors the model declares, and applies only the repairs that are needed.
An FTW that loads as is is left untouched.

## What it writes

- In place by default. When tensors are only added, one shard is appended and the existing shards
  are not touched. When tensors are replaced or dropped (GLM-5.2), the live entries are rewritten
  into fresh shards so the FTW holds no dead bytes; this needs free disk space equal to the FTW.
  As long as the old shards are kept, the previous index stays as `freetoken_weight.json.bak`, so
  restoring it (and deleting the appended shard) undoes the patch.
- `--out <dir>` writes a fresh, compact FTW dir and leaves the original untouched.
- The last line of the output reports how many declared tensors are still missing; it must be 0.
  Long phases show a progress bar; `-v` prints every step instead.

Not covered: FTWs converted from GGUF, and checkpoints outside [models.md](models.md).
