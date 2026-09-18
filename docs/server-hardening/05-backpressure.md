# Opt-in frontend admission capacity

`--max-concurrent-requests 0` (default) preserves unlimited desktop behavior.
A positive value caps outstanding frontend ack/event maps; `-1` explicitly uses
the scheduler's `max_running_req`. The later default-OFF requirement takes
precedence over automatically enabling that scheduler-derived cap.

Admission checks capacity without an await before allocating uid/maps/accounting,
so adapters cannot race for the final slot. Saturated chat, Messages, Responses,
legacy completions and /generate return 429 with Retry-After: ceil(existing request
ring p95 milliseconds / 1000), or one second when no latency sample exists. This
is an estimate, not a scheduler guarantee. Terminal ack receipt frees frontend
maps before yielding to a consumer that may break; cancellation retains the
existing abort/accounting barrier (abort dispatch is not GPU completion).

[vLLM](https://github.com/vllm-project/vllm/blob/v0.14.0/vllm/entrypoints/openai/serving_engine.py)
and [SGLang](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py)
have engine/tokenizer-level scheduling and load management. This smaller frontend
bounds its own retained requests at its existing common admission point, leaving
scheduler policy unchanged. TensorRT-LLM's batch-size/config-file settings concern
engine capacity; FreeToken keeps frontend memory protection a separate, explicit
CLI choice. This is not a per-client request-rate limiter.

Validation covers default unlimited behavior, automatic/explicit caps, no allocation
on rejection, release before terminal yield, abort cleanup, and every adapter's
buffered/streaming 429 plus Retry-After. No throughput improvement is claimed.
