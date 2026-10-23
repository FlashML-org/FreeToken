# Recursive Flint

FreeToken serves the Flint decoder with ModelOpt NVFP4 safetensors and the official
NVFP4 GGUF. Flint uses 49 layers (two dense, then 47 MoE), sigmoid top-6 routing
over 256 experts, an ungated shared expert, QK RMSNorm, learned attention sinks,
and hybrid full/sliding attention. Full-attention layers use NoPE; local layers
use RoPE and a 513-token window including the current token.

Checkpoints:

- [FutureMLS-Lab/flint-nvfp4-coldgptq](https://huggingface.co/FutureMLS-Lab/flint-nvfp4-coldgptq):
  community ModelOpt export, approximately 25.16 GB (23.43 GiB) of tensors. Routed
  experts are NVFP4; other weights remain BF16.
- [RecursiveSI/flint-nvfp4-gguf](https://huggingface.co/RecursiveSI/flint-nvfp4-gguf):
  official approximately 22.9 GB GGUF, with more resident matrices quantized.

`dgpp_mixed` FP8/NVFP4 exports and MLX-packed checkpoints use different layouts
and are not supported by this loader. The HF reader also accepts the original
BF16 layout, but full BF16 inference requires substantially more host memory and
was not validated on the 32 GB host used for the NVFP4 checks.

## Run

Start with one request and a short context on a 12 GB GPU. The routed expert
banks need about 20 GiB of host RAM, in addition to the process and resident
weights. The ModelOpt path uses `offload`, `cpu` or `hybrid` experts. These recipes
target TP=1.

```bash
FLASHINFER_USE_CUDA_NORM=1 ft serve \
  --model FutureMLS-Lab/flint-nvfp4-coldgptq \
  --gpu 0 --moe-strategy hybrid --moe-hybrid-max-fetch 1 \
  --moe-cache-size 1024 --num-tokens 8192 \
  --max-seq-len-override 8192 --max-prefill-length 2048 \
  --max-running-requests 1 --host 127.0.0.1 --port 1919
```

For the GGUF, download the file and use its path as `--model` with the same flags:

```bash
hf download RecursiveSI/flint-nvfp4-gguf flint-nvfp4.gguf --local-dir models/flint-gguf
```

The reference SGLang kit explicitly selects FlashInfer's CUDA normalization
implementation. The environment variable above preserves that choice. Ampere
GPUs use FreeToken's compatible NVFP4 paths; they do not have Blackwell's native
W4A4 instructions. Memory fit and throughput therefore need local measurement.

Recommended sampling defaults come from the checkpoint: temperature 1.0,
top-p 0.95 and top-k 20. The documented 262144-token context is a model limit;
the memory required to reserve it is separate from loading the weights.

## Conversation protocol

Use the checkpoint's chat template. The server selects the `glm47` tool parser and `flint` reasoning
parser automatically for HF and GGUF checkpoints. Flint chooses whether to emit
`<think>...</think>` on each turn; the parser never assumes the turn starts inside
reasoning, even when generic thinking flags or tools are present. Replay returned
`reasoning_content` and tool calls in the conversation history.

Tool calls use the GLM-style `<tool_call>` / `<arg_key>` / `<arg_value>` envelope.
This integration uses FreeToken's existing parser and API behavior. It does not
port the separate Flint-specific XGrammar backend distributed in the SGLang kit.

The GGUF pre-tokenizer is `flint`, with the original three regex splits and byte
level encoding. Its explicit EOG array contains four IDs: 150001, 150003, 150005,
and 150018. All four must stop generation, including the role boundaries.

## Validate a local change

The model tests compare fused sigmoid routing against independent tensor math and
check weight fusion across shards. With local checkpoints they also compare GGUF
tokenization, chat rendering and configuration against the original HF files.

```bash
FREETOKEN_FLINT_MODEL=/path/to/flint-modelopt \
FREETOKEN_FLINT_GGUF=/path/to/flint-nvfp4.gguf \
uv run pytest tests/models/test_flint.py tests/server/test_parser_auto_selection.py \
  tests/server/test_reasoning_parser_all_models.py \
  tests/server/test_streaming_model_matrix.py
```
