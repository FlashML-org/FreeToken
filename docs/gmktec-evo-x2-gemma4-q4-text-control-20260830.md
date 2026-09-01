# GMKtec EVO-X2 Gemma4 Q4 text control, 2026-08-30

The native ROCm/HIP FreeToken GGUF path was qualified against the on-host
`gemma-4-26B_q4_0-it.gguf` text model. The test was isolated on port 1923 and
the Qwen production recovery helper ran when the temporary process exited.

The caller rendered Gemma's embedded canonical chat template once, submitted
the resulting raw prompt to `/v1/completions`, and disabled extra special-token
insertion. The server and local GGUF tokenizer agreed on 30 prompt tokens.

| Field | Result |
| --- | --- |
| Prompt hash | `0f65acd07a4f57b2644f7720b725d7795999406b90a9f91486da5effa39bb95d` |
| Question | `What is 17 times 19? Reply with only the decimal number.` |
| Expected output | `323` |
| Actual output | `323` |
| Server prompt tokens | 30 |
| Completion tokens | 4 |
| Steady decode TPS | 57.05 |

The preserved GMKtec EVO-X2 evidence is
`/home/david/freetoken-amd/artifacts/gemma4-gguf-text-20260830T035542Z/quality.json`.
This proves text-only loader, template, OpenAI-compatible completion API,
token accounting, and a deterministic basic quality response. It does not yet
qualify image input through the matching multimodal projector.
