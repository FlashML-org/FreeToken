# OpenAI-compatible API: endpoints and parameters

What `ft serve` accepts on its OpenAI-shaped routes, what it ignores, and what it refuses,
next to the same surface in vLLM, SGLang and llama.cpp's server. "honoured" means the value
changes the generation; "accepted" means the request is not rejected but the value has no
effect; "400" means the server refuses the request with a clear message.

## Endpoints

| route | FreeToken | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| `POST /v1/chat/completions` | yes | yes | yes | yes |
| `POST /v1/completions` | yes | yes | yes | yes |
| `GET /v1/models`, `GET /v1/models/{id}` | yes | yes | yes | yes |
| `POST /v1/responses` (+ get, cancel) | yes | yes | yes | yes |
| `POST /v1/messages` (Anthropic), `count_tokens` | yes | yes | no | yes |
| `POST /tokenize`, `POST /detokenize` (also under `/v1/`) | yes | yes | yes | yes (`/tokenize`, `/detokenize`) |
| `GET /metrics` (Prometheus) | yes | yes | yes | yes |
| `GET /version` | yes | yes | yes | no (`/props`) |
| `GET /health` | yes | yes | yes | yes |
| `GET /v1/stats`, `/v1/cache/*`, `/v1/requests`, `/v1/admin/*` | yes (FreeToken control plane) | `/metrics` only | `/get_server_info`, `/flush_cache` | `/slots`, `/props`, `/metrics` |
| `POST /v1/embeddings`, `/rerank`, `/score`, `/pooling` | no (generative models only) | embedding / reranker models | embedding / reranker models | yes, any model with pooling |
| `POST /v1/audio/*`, `/v1/files`, `/v1/batches` | no | transcriptions, batch runner (offline) | transcriptions, files, batches | no |
| `POST /infill`, `/apply-template`, `/completion` (non-OpenAI) | `suffix` on `/v1/completions`; `/tokenize` with `messages` renders the template | no | no | yes |

## Chat completions: request

| parameter | FreeToken | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| `model`, `messages` (text + `image_url` parts) | honoured | honoured | honoured | honoured |
| `max_tokens`, `max_completion_tokens` | honoured (default 32k) | honoured | honoured | `max_tokens` |
| `temperature`, `top_p`, `top_k` | honoured | honoured | honoured | honoured |
| `min_p` | honoured (prob. below `min_p` x top prob dropped) | honoured | honoured | honoured |
| `presence_penalty`, `frequency_penalty` | honoured, over generated tokens | honoured | honoured | honoured |
| `repetition_penalty` | honoured, over prompt + generated tokens (HF semantics) | honoured | honoured | `repeat_penalty` |
| `logit_bias` | honoured (token id -> bias, clamped to [-100, 100]) | honoured | honoured | honoured |
| `min_tokens` | honoured: no EOS / stop token before this many generated tokens | honoured | `min_new_tokens` | no |
| `stop` (strings), `stop_token_ids` | honoured | honoured | honoured | `stop` only |
| `include_stop_str_in_output` | honoured | honoured | `no_stop_trim` | no |
| `skip_special_tokens` | honoured; default off here, the reasoning and tool parsers read the tags | honoured (default on) | honoured (default on) | n/a |
| `n` | honoured, 1..16, one generation per choice (the prefix cache shares the prompt) | honoured | honoured | `n_cmpl` |
| `stream`, `stream_options.include_usage` | honoured | honoured | honoured | honoured |
| `logprobs`, `top_logprobs` | 400 on this route (see #224); `/v1/completions` `logprobs` 400 | honoured | honoured | `n_probs` |
| `tools`, `tool_choice` (`auto` / `none` / `required` / named), `parallel_tool_calls` | honoured | honoured | honoured | honoured |
| `reasoning_effort`, `thinking`, `chat_template_kwargs` | honoured | `chat_template_kwargs` | honoured | honoured |
| `continue_final_message` | honoured (the last assistant message is continued, no generation prompt) | honoured | honoured | no |
| `response_format` `json_object` / `json_schema`, guided decoding | 400 (no constrained decoding) | honoured (xgrammar / guidance) | honoured (xgrammar / outlines / llguidance) | `grammar`, `json_schema` |
| `seed` | accepted, not honoured (one batched sampling kernel, no per-request generator) | honoured | `sampling_seed` | honoured |
| `user`, `metadata`, `store`, `service_tier` | accepted, ignored | accepted, ignored | accepted, ignored | accepted, ignored |
| `request_id` | honoured: becomes the response id | honoured | `rid` | `id_slot` (different meaning) |
| `function_call` (legacy) | 400 | 400 | 400 | no |
| `best_of`, `use_beam_search`, `length_penalty`, `prompt_logprobs`, `echo` (chat), `truncate_prompt_tokens`, `priority`, `cache_salt`, `lora` | no | honoured | partly | partly |
| `ignore_eos` | honoured (benchmarking) | honoured | honoured | honoured |

## Legacy completions: request

`prompt` as a string or a list of strings (token-id prompts are refused); the same sampling
parameters as above; plus:

| parameter | FreeToken | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| `echo` | honoured (the prompt leads the text; not with `logprobs`) | honoured | honoured | no |
| `suffix` | honoured on models with fill-in-the-middle tokens (`<|fim_prefix|>` ... `<|fim_middle|>`): the prompt becomes prefix-suffix-middle | 400 | 400 | `/infill` |
| `n` | honoured, prompt-major choice order (`prompt i`, sample `k` -> `index i*n+k`) | honoured | honoured | no |
| `logprobs` | 400 (see #224 for the sampled-token logprobs) | honoured | honoured | `n_probs` |

## Responses

| field | FreeToken | notes |
|---|---|---|
| `id` | `chatcmpl-<uid>` / `cmpl-<uuid>`, or `<prefix>-<request_id>` when the client sent one | |
| `created`, `model`, `object` | yes | |
| `system_fingerprint` | `null` | vLLM and SGLang also send `null` |
| `choices[].message.reasoning_content` | yes, when the model reasoned | vLLM `reasoning_content`, llama.cpp `reasoning_content` |
| `choices[].message.tool_calls` | yes | |
| `choices[].finish_reason` | `stop`, `length`, `tool_calls` | |
| `usage.prompt_tokens`, `completion_tokens`, `total_tokens` | yes | for `n > 1` the prompt is counted once, completions summed |
| `usage.prompt_tokens_details.cached_tokens` | with `--enable-cache-report` | vLLM, SGLang: always |
| `usage.completion_tokens_details.reasoning_tokens` | yes, when the model emitted a reasoning end tag | vLLM, SGLang |
| streaming: one `[DONE]`, an optional final usage chunk | yes | |
| llama.cpp `timings` | no; `/v1/stats` and `/metrics` carry the server-side rates | |

## Tokenize / detokenize

```
POST /tokenize      {"prompt": "text", "add_special_tokens": true, "return_token_strs": false}
POST /tokenize      {"messages": [...], "add_generation_prompt": true, "chat_template_kwargs": {}}
                    -> {"count": N, "max_model_len": L, "tokens": [...], "token_strs": [...]?}
POST /detokenize    {"tokens": [...], "skip_special_tokens": false} -> {"prompt": "text"}
```

`messages` are rendered through the model's chat template exactly as a generation would
render them, so `count` equals the `prompt_tokens` that generation reports.

## Not implemented, on purpose

- Constrained decoding (`response_format` json / json_schema, regex, grammar): needs a
  grammar engine on the sampling path.
- Prompt logprobs and `echo` with `logprobs`: need the prefill logits.
- Per-request `seed`: the batched sampling kernel draws from one generator.
- Embeddings, reranking, scoring, audio: not generation.
