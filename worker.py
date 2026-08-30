"""Single-request Vast PyWorker proxy for a loopback FreeToken server."""

from __future__ import annotations

import os
from typing import Any, Mapping


def _positive_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _enabled(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _text_size(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(_text_size(item) for item in value)
    if isinstance(value, Mapping):
        return sum(_text_size(item) for item in value.values())
    return 0


def completion_workload(payload: Mapping[str, Any]) -> float:
    prompt_tokens = max(1, (_text_size(payload.get("prompt", "")) + 3) // 4)
    return float(prompt_tokens + _positive_int(payload.get("max_tokens"), 16))


def chat_workload(payload: Mapping[str, Any]) -> float:
    prompt_tokens = max(1, (_text_size(payload.get("messages", [])) + 3) // 4)
    return float(prompt_tokens + _positive_int(payload.get("max_tokens"), 16))


def benchmark_payload() -> dict[str, Any]:
    return {
        "model": os.environ.get("TEKIZAI_SERVED_MODEL", "glm-5.3-flash-nvfp4"),
        "prompt": "Reply with exactly: Vast FreeToken worker ready",
        "max_tokens": 16,
        "temperature": 0,
        "stream": False,
    }


def build_worker_config() -> Any:
    from vastai import (  # type: ignore[import-not-found]
        BenchmarkConfig,
        HandlerConfig,
        LogActionConfig,
        WorkerConfig,
    )

    port = _positive_int(os.environ.get("TEKIZAI_FREETOKEN_PORT"), 1919) or 1919
    allow_parallel = _enabled(os.environ.get("TEKIZAI_ALLOW_PARALLEL_REQUESTS"))
    benchmark_concurrency = (
        _positive_int(os.environ.get("TEKIZAI_BENCHMARK_CONCURRENCY"), 1) or 1
    )
    common = {
        "allow_parallel_requests": allow_parallel,
        "max_queue_time": 180,
    }
    return WorkerConfig(
        model_server_url="http://127.0.0.1",
        model_server_port=port,
        model_log_file=os.environ.get(
            "TEKIZAI_FREETOKEN_LOG", "/workspace/logs/freetoken-glm53.log"
        ),
        handlers=[
            HandlerConfig(
                route="/v1/completions",
                **common,
                workload_calculator=completion_workload,
                benchmark_config=BenchmarkConfig(
                    generator=benchmark_payload,
                    runs=2,
                    concurrency=benchmark_concurrency,
                ),
            ),
            HandlerConfig(
                route="/v1/chat/completions",
                **common,
                workload_calculator=chat_workload,
            ),
        ],
        log_action_config=LogActionConfig(
            on_load=["FREETOKEN_SERVER_READY"],
            on_error=["FREETOKEN_SERVER_EXITED", "Traceback"],
        ),
    )


def main() -> None:
    from vastai import Worker  # type: ignore[import-not-found]

    Worker(build_worker_config()).run()


if __name__ == "__main__":
    main()
