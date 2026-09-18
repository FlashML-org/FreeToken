# Prometheus metrics from the existing tracker

`/metrics` exports the requested `freetoken:` histograms, counters and gauges,
labelled with `model_name`. Counters read StatsTracker's cumulative admitted prompt
and sampled generation totals directly, including late replies after disconnect.
KV usage is a fraction (0..1), consistent with the requested legacy `_perc` name;
unknown pool capacity is omitted. Each tracker owns its registry and timing
histograms; scraping cannot count a request twice or reset history on ring eviction.

The queue dependency is an additive scheduler -> tokenizer -> frontend IPC
snapshot, emitted only when running/queued counts change, including abort/idle
transitions. Running and waiting use the same manager counts as scheduler status
logging. Until that snapshot arrives, gauges and stats values are unknown rather
than fabricated zeros. Offline inference does not emit these online control messages.

Timing observes the exact RequestRecord produced by the shared generation layer
for chat, Messages and Responses. The existing desktop contract is preserved:
non-streaming records have no TTFT, so TTFT/TPOT histograms cover measured streams;
E2E includes buffered requests and failures. TPOT is the per-request mean
`(duration - TTFT) / (output_tokens - 1)`, omitting one-token/incomplete streams.
Legacy raw `/generate` and text completions still contribute engine token counters;
their middleware-only timings are excluded because stream completion is not known.

[vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/loggers.py)
collects engine timing and now distinguishes inter-token latency from per-request
TPOT; [SGLang](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/metrics)
also instruments its scheduler. FreeToken retains the requested legacy metric
names but projects existing accounting instead of maintaining duplicate counters.
Dashboard queries must replace the engine prefix, and timing coverage above is
explicit rather than claiming identical engine-side measurement semantics.
TensorRT-LLM's generator performance collection is not needed for this frontend
projection. No GPU throughput or latency improvement is claimed.

Validation: `PYTHONPATH=python pytest -q tests/server tests/scheduler/test_queue_metrics.py`.
