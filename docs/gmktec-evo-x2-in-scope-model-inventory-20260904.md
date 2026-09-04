# GMKtec EVO-X2 in-scope model inventory

This inventory separates the model that is active now from models found in
archived model-routing qualification configurations. It is based on a
read-only process and filesystem inspection on September 4, 2026. An archived
configuration is evidence that a model was previously considered or tested;
it is not evidence that the model is currently loaded or production-routed.

## Active native FreeToken service

| Model identity | Runtime state | Existing evidence |
| --- | --- | --- |
| `qwen3.6-35b-a3b-nvfp4-amd` using `/home/david/freetoken-amd/models/Qwen3.6-35B-A3B-NVFP4` | Active native ROCm/HIP service | Qwen quality suite, Q5 four-row qualification, ROCm 10 comparison, W1 to W4 bounded controls, recovery, and 1,440-session endurance |

The active command line uses the native Triton attention path, the offload MoE
backend, automatic expert-cache sizing, serial expert loading, and an 8,192
token context override. No llama-swap process was found active during the
inventory probe.

## Payload admission result

The read-only model-directory probe found the Qwen3.6 35B-A3B safetensors and
NVFP4 payloads plus the Gemma 4 Q4 GGUF and vision projector. It did not find a
Qwen3.8 27B, GLM-4.7, KAT-Coder, Laguna, or Ornith payload under the FreeToken
model directory. Therefore the first queued Qwen3.8 qualification cannot begin
yet: downloading or copying a checkpoint is a separate authorized staging step,
and a load test must not be improvised with a missing artifact.

The source tree does contain model code for Qwen3.8-Flash-Next (`qwen4_exp`)
and GLM-4.7 parser support. Source support alone does not prove that the
archived routed checkpoints are compatible with the current AMD path.

## Archived text-model routing entries

These identifiers were found in archived model-routing configuration files and
remain candidates for a deliberate FreeToken qualification decision:

| Model identifier | Modality | FreeToken qualification status |
| --- | --- | --- |
| `lan223-qwen38-27b` | Text and image input, text output | Not yet qualified through the current standardized FreeToken matrix |
| `lan223-qwen36-27b-control` | Text and image input, text output | Qwen family control exists, but this specific routed artifact needs an explicit matrix record |
| `lan223-glm47-flash` | Text | Not yet qualified through the current standardized FreeToken matrix |
| `KAT-Coder-V2.5-Dev-Q8_0` | Text and tools | Not yet qualified through the current standardized FreeToken matrix |
| `lan223-laguna-xs21` | Text and tools | Not yet qualified through the current standardized FreeToken matrix |
| `lan223-ornith15-9b` | Text and tools | Not yet qualified through the current standardized FreeToken matrix |
| `lan223-ornith15-35b-a3b` | Text and tools | Not yet qualified through the current standardized FreeToken matrix |
| `lan223-glm47-ggml-q4k` | Text and tools | Archived llama-swap artifact-comparison entry; no FreeToken AMD matrix record |
| `lan223-glm47-unsloth-q4km` | Text and tools | Archived llama-swap artifact-comparison entry; no FreeToken AMD matrix record |
| `lan223-glm47-bartowski-q4km` | Text and tools | Archived llama-swap artifact-comparison entry; no FreeToken AMD matrix record |

The existing Qwen3.6 35B-A3B NVFP4 result is not a substitute for these rows.
Each row requires its own exact checkpoint, tokenizer, quantization, prompt
contract, quality result, TPS measurements, and recovery evidence.

## Archived non-text or multimodal entries

The following entries were also found, but they are not ordinary text MoE
serving targets and therefore require a separate backend-admission decision:

| Model identifier | Function | Current decision |
| --- | --- | --- |
| `lan223-whisper-large-v3-turbo` | Audio transcription | No native FreeToken text-generation qualification claim |
| `lan223-qwen3-tts-0.6b-base` | Text-to-speech | No native FreeToken text-generation qualification claim |
| `lan223-qwen-image-2512-gguf` | Image generation or editing | No native FreeToken text-generation qualification claim |
| `lan223-flux2-klein-4b` | Image generation or editing | No native FreeToken text-generation qualification claim |

These models must not be counted as missing FreeToken tests until their
backend, input and output contract, and AMD implementation scope are defined.

## Required qualification order

1. Reconfirm the exact current serving inventory before changing any service.
2. Qualify the text MoE rows first, beginning with Qwen3.8 and GLM-4.7-Flash.
3. Add KAT-Coder, Laguna, and Ornith only after model-format support and
   deterministic quality fixtures are confirmed.
4. Treat image, audio, and speech entries as separate backend projects.
5. Run every admitted model through the cross-model matrix in
   `gmktec-evo-x2-cross-model-matrix-20260904.md`.
