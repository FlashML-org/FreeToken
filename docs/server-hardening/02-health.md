# Liveness and readiness

Use `/healthz` for HTTP process liveness and `/readyz` for routing readiness.
Readiness returns 200 only while admission is serving and no fatal worker error
is latched; loading, rebuilding, failed and stopping return 503. Its response body
uses the existing lifecycle document, including startup phase/progress.
The desktop's `/health` status code and body remain unchanged in every state.

[vLLM](https://github.com/vllm-project/vllm/blob/v0.14.0/vllm/entrypoints/openai/api_server.py)
checks engine health through its engine client, while
[SGLang](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py)
and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/serve/openai_server.py)
also provide generation-health probes. FreeToken already has an authoritative
admission state driven by worker supervision and cache maintenance, so readiness
reads that state without submitting a synthetic generation. HTTP liveness does
not touch the backend, preventing slow weight loading from causing restart loops.

Validation: `PYTHONPATH=python pytest -q tests/server/test_health.py`.
These are CPU HTTP lifecycle tests without a GPU backend.
