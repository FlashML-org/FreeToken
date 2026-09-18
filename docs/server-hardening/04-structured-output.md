# JSON Schema contract; enforcement intentionally unavailable

Chat `response_format: {type: "json_schema", json_schema: {name, schema, strict?}}`
and Responses `text.format: {type: "json_schema", name, schema, strict?}` normalize
through the shared generation layer. JSON Schema meta-validation uses jsonschema,
with an explicitly recognized dialect (Draft 2020-12 when absent), without fetching
user references. The independent schema copy lives on GenSpec and then on the
SamplingParams nested in TokenizeMsg and UserMsg across both IPC codecs.

The current sampler has greedy argmax and temperature/top-k/top-p paths but no
grammar logits processor. The scheduler-owned capability gate therefore raises
NotImplementedError with an xgrammar TODO. Shared submission runs that gate before
uid allocation/SSE headers and reports an honest HTTP 400; the actual scheduler
also guards direct IPC requests and sends a terminal error before allocating KV.
No route advertises or silently simulates constrained generation. Plain text and
detector-based tool parsing retain their current paths; legacy text completions
continue to reject constrained output.

[xgrammar](https://github.com/mlc-ai/xgrammar) supplies tokenizer-aware schema/EBNF
compilation, request matchers, token acceptance and bitmask application.
[vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/v1/structured_output/backend_xgrammar.py)
and [SGLang](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/constrained/xgrammar_backend.py)
compile schemas centrally and maintain per-request grammar state before sampling.
FreeToken adopts their contract boundary but defers compiler/mask integration until
both sampling paths, overlap, stop tokens and abort cleanup can be enforced.
TensorRT-LLM's serving configuration/model routing does not provide this missing
engine capability, so adding a separate configuration layer would not solve it.
No grammar engine or new optional xgrammar dependency is introduced here.

Validation: schema rejection before stream headers/admission, both IPC round trips
(including reserved serialization keys), direct scheduler rejection, and unchanged
plain-text/tool paths. This PR does not implement or claim constrained decoding.
