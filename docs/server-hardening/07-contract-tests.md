# Parser, SDK and lifecycle contracts

Each of the 11 concrete tool detectors has its own test class for a complete
call, one-character streaming and an interrupted call. A coverage assertion
compares the classes against both the live registry and concrete subclasses,
so adding a format cannot silently omit these contracts. Recovery is deliberately
format-aware: `recover_truncated_call()` salvages buffered MiniMax M3 calls;
already-incremental detectors return no duplicate and finish arguments from
emitted fragments or their existing argument ledger. These tests preserve that
contract instead of requiring a new buffering architecture.

The six concrete reasoning parsers have the same complete/character/truncated
matrix. Their EOS API is `flush()`, not `recover_truncated_call()`. Generation
mode tests cover Qwen3/GLM defaults and explicit toggles, all Gemma4 thinking
aliases, and MiniMax M3 adaptive/enabled/disabled modes, both with and without
tools. Accounting tests exercise retry after a late terminal acknowledgement,
concurrent prepare-stop calls, abort transport failure, unidentified active
requests, invalid transitions and cancelled drains. Supervisor tests change the
shutdown flag after readiness and kill a non-first worker on a later poll.

Protocol tests use real OpenAI and Anthropic SDKs with strict response validation
over FastAPI's real routes and the shared generation primitive. Only backend
IPC is scripted; character replies exercise reasoning, tool arguments, SSE
assembly, finish reasons, usage and SDK error classes for Chat Completions,
Responses and Messages. Dependencies are required rather than import-skipped.
The SDK tests exposed a missing `signature` string in Anthropic thinking-start
blocks. Adding the same empty unsigned signature used by full responses fixes
strict clients without changing reasoning content or claiming cryptographic
signing. SDKs with httpx and httpx2 use their corresponding ASGI transport.

## Comparison and scope

[vLLM's protocol models](https://github.com/vllm-project/vllm/blob/v0.14.0/vllm/entrypoints/openai/protocol.py)
and [SGLang's detector boundary](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/function_call/base_format_detector.py)
separate wire contracts from model-format parsing; [TensorRT-LLM's server](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/serve/openai_server.py)
similarly adapts its engine to client-facing protocols. FreeToken keeps its
existing GenSpec/generation event path and tests these two boundaries separately,
then composes them through actual SDK clients. It adds neither a second parser
implementation nor a second scheduler. This CPU suite proves deterministic
conversion and state transitions; real model generation, TCP timing, disconnects,
GPU execution, throughput and multi-rank behavior require their own integration runs.

Run with project development dependencies installed:
`PYTHONPATH=python pytest -q tests/server`.
