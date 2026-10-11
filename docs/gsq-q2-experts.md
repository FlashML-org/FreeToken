# GSQ Q2_0 routed experts

Qwen3.8-Flash-Next can use an HF checkpoint for the trunk, shared experts,
tokenizer and PLE, with a separate GSQ Q2_0 GGUF file for routed experts.
This path reads the complete expert set in bounded chunks and keeps native
packed bytes in the host bank. It does not load a complete GGUF model or
support IQ3_XXS, Q2_K or TQ2_0.

GSQ Q2_0 (GGML type 42) stores 64 weights in 18 bytes: one little-endian
FP16 scale and sixteen bytes of adjacent 2-bit codes. The decoded value is
`(code - 1) * scale`. The GPU kernel rounds weights and activations to BF16,
then uses FP32 accumulation and output-side routing weights.

The initial implementation requires TP=1, SiLU, BF16 activations and pure
GPU offload. CPU, hybrid and resident expert execution are unsupported.
It overrides routed-expert quantization only; the HF dialect still decides
all trunk and shared-expert schemes. It uses the existing offload cache and
CUDA graph path, without new cache policy or prefill optimization.

## Source contract

Provide `freetoken_expert_source` in the HF config or through `--hf-overrides`:

```json
{
  "freetoken_expert_source": {
    "format": "q2_0",
    "scope": "routed_experts_only",
    "path": "experts-q2.gguf",
    "sha256": "<verified 64-character lowercase SHA256>",
    "size": 0,
    "verified_mtime_ns": 0,
    "layers": 48,
    "experts_per_layer": 512,
    "hidden": 2560,
    "intermediate": 640
  }
}
```

The numbers above illustrate this model's geometry. Set `size` and
`verified_mtime_ns` from the actual file after verifying its complete SHA256
against a trusted checksum. A relative path resolves against the HF model
directory. Keep the source immutable after verification. Runtime checks the
receipt's size/mtime and the complete GGUF expert directory, including tensor
types, geometry, alignment, bounds and overlap; it does not rehash the entire
file on each load. Invalid directories are rejected before bank allocation.

Use an HF index containing only non-expert tensors. Verify those tensors and
tokenizer independently against the intended HF revision; this feature does
not convert, download or certify a mixed checkpoint. Do not substitute pruned
experts or assume the GGUF trunk is interchangeable with the HF trunk.

An illustrative launch after preparing and verifying both sources is:

```bash
ft serve --model /models/verified-hf-trunk --text-model-only \
  --dtype bfloat16 --moe-strategy offload --expert-load serial \
  --ple-backend disk --hf-overrides "$(cat expert-overrides.json)"
```

Choose expert cache and KV capacity for the actual device. Expert host bytes
are `3 * layers * experts * hidden * intermediate / 64 * 18`; the illustrated
geometry requires 31.640625 GiB for the complete compressed bank, plus other
runtime memory. This launch is documentation, not an end-to-end test claim.

## Tests

```bash
PYTHONPATH=python python -m pytest tests/moe/test_q2_0.py tests/moe/test_q2_source.py -q
```

Tests use a real miniature GGUF and an independent Torch numerical reference.
They cover expert order, incomplete/corrupt directories, native bank packing,
signed and zero scales, slot remapping and CUDA graph replay. GPU cases skip
when CUDA is unavailable.
