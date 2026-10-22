# Sigmoid router experiment evidence

> 最新 main 整合复验：121 项路由测试通过；分支/main 相关回归均 92 通过、1 跳过；四组消融重跑完成。见 [本轮报告](main-integration/README.md)。下文保留历史结果。

> 2026-10-22 WSL 后续：Linux 环境与 FreeToken 编译成功，117 项局部测试和 46 项相关回归通过；真实权重路由复验完成。见 [Linux 验证记录](linux/README.md)。下文的 Windows 实验保留原始环境与数值；完整模型 A/B 仍未完成。


Latest review entry (2026-10-22): [implementation, tie audit, ablation, reproducibility and blockers](REVIEW.md).
The expanded targeted suite now has 117 passes; the original 111-pass logs below remain historical evidence.

This directory contains local evidence for issue #730. It is an ordinary directory,
not an archive. No PR has been created or uploaded. The full-model end-to-end
experiment is **not completed**: see `full-model-preflight.json`.

## Evidence map

- `history/final-targeted-tests.log`: 111 passing tests through the normal repository fixtures.
- `history/related-regression-tests.log`: 43 passes and three Windows FTW failures.
- `history/baseline-ftw-tests.log`: the same three failures on the unchanged base.
- `history/extended-benchmark.json`: all five timing samples per implementation for
  20 post-logit configurations, not generation throughput.
- `history/real-glm-router-results.json`: real router weights with synthetic hidden states.
- `history/real-prompt-results.json`: three real prompt prefixes, 137 tokens;
  first-MoE selections and last-input-token reference-combined expert outputs.
- `real-prefix/`: saved activations, model configuration, and SHA256 tensor-source manifest.
- `reproduction/`: rerun using the portable script included here.
- `environment-current.txt`: exact installed packages for the latest reproduction.
- `reports/`: experimental reports; older report entries retain historical status.
- `inventory.json`: copied-file provenance and hashes; `checksums.json` covers the
  final directory, including the portable scripts and this README.

All original small records are retained, including failed installation/import/test
attempts. Their timestamps and status must not be read as the final outcome.
Historical Python scripts retain their original absolute paths as provenance;
use the portable script below for the real-prefix experiment. The local reports
and logs may also contain original Windows paths.

## Reproduce the real-prefix experiment

Use an environment with CUDA PyTorch, a compatible Triton, Transformers 5.18.0,
huggingface-hub, safetensors, flashlib and the repository's import dependencies.
The exact Windows environment is recorded, not a portable Linux lock file.
The checkpoint revision is pinned in `checkpoint_slices.py`. A first run downloads
only the required tensor ranges and tokenizer files, not the complete model.

From the repository root, in PowerShell:

```powershell
$env:FREETOKEN_VALIDATION_CACHE = 'D:/AI/FreeToken-validation/GLM-4.5-Air-prefix'
$env:HF_HOME = 'D:/AI/FreeToken-validation/hf-cache'
$env:FREETOKEN_VALIDATION_OUTPUT = 'sigmoid-prefix-results'
python benchmarks/sigmoid_router_validation/reproduce_prefix.py
```

On another machine, change these paths. Without the variables, tensor cache
defaults to `~/.cache/freetoken-sigmoid-prefix`, Hugging Face uses its own default,
and results go to `./sigmoid-prefix-results`.

The experiment uses Transformers for the true BF16 prefix, this branch's
FreeToken routing function, and a reference expert combination. It does not run
the complete FreeToken serving engine or produce a generated answer. Comparison
with HF aligns weights by expert ID because HF uses unsorted top-k. Saved `.pt`
files contain tensors only and can be loaded using `torch.load(..., weights_only=True)`.

Run kernel tests and microbenchmarks using the commands in the repository's PR
draft/report. The benchmark accepts `--output` to preserve raw timing samples.

## Full-model experiment: pending protocol, not results

Required: a supported Linux NVIDIA runtime, a complete checkpoint from an affected
family, and enough host/device memory for that checkpoint and its chosen backend.
The checked local Windows machine has approximately 31 GiB RAM and 8 GiB VRAM;
WSL and nvcc are unavailable. The selected GLM-4.5-Air BF16 checkpoint is too large
for the available RAM/VRAM. No whole-model launch, tokens/s, TTFT or quality score
is claimed in this evidence set. Disk capacity alone does not establish that the
engine can load its host expert banks.

When suitable hardware is available:

1. Pin the exact main commit and branch source; use separate checkouts with the
   same runtime, checkpoint revision, quantization, backend, cache budget and GPU.
2. Run the normal tests on Linux and confirm complete model loading/generation.
3. Reuse `benchmarks/bench_decode_moe.py` in each checkout, with the same local
   prompt JSONL, `--greedy`, `--backend offload`, `--decode 256`, and identical
   explicit cache/memory settings. For example (not executed here):

   ```bash
   PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_decode_moe.py \
     --model /absolute/path/to/checkpoint --aime /absolute/path/to/prompts.jsonl \
     --problem 0 --backend offload --decode 256 --greedy --json /absolute/path/to/run.jsonl
   ```

4. Alternate main/branch order across at least five repetitions, using short and
   longer prompts. Keep all server logs and exact commands, report distributions
   and medians, and disclose the harness's warmed request/prefix-cache behavior.
   Its reported VRAM is a live post-request reading, **not peak memory**. Obtain
   peak-memory evidence separately. Do not label warm TTFT as cold prefill latency.
5. Save complete generated texts or token IDs for a fixed small evaluation set;
   the existing benchmark's output hash and truncated preview are insufficient
   for a quality evaluation. Check divergence and task scores; if feasible also
   compare teacher-forced loss under identical inputs. Report the sample size.

## What is intentionally not included

Checkpoint weight binaries, embedding rows, virtual environments, executables,
baseline repository copies, and compiler/model caches are not PR attachments.
The model ID/revision, range-download code, and tensor checksums allow recovery.
Experimental result tensors and all small raw logs are included. Do not commit
the checkpoint cache when running the experiment again. Do not publish old
failed runs as successful measurements.
