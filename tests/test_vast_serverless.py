import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import worker

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def fake_vast_sdk():
    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    module = types.SimpleNamespace(
        BenchmarkConfig=Config,
        HandlerConfig=Config,
        LogActionConfig=Config,
        WorkerConfig=Config,
    )
    previous = sys.modules.get("vastai")
    sys.modules["vastai"] = module
    try:
        yield
    finally:
        if previous is None:
            del sys.modules["vastai"]
        else:
            sys.modules["vastai"] = previous


def test_vast_worker_defaults_to_serial():
    with mock.patch.dict(os.environ, {}, clear=True), fake_vast_sdk():
        config = worker.build_worker_config()
    assert all(not handler.allow_parallel_requests for handler in config.handlers)
    benchmark = next(
        handler.benchmark_config
        for handler in config.handlers
        if hasattr(handler, "benchmark_config")
    )
    assert benchmark.concurrency == 1


def test_vast_worker_allows_bounded_parallelism():
    env = {
        "TEKIZAI_ALLOW_PARALLEL_REQUESTS": "true",
        "TEKIZAI_BENCHMARK_CONCURRENCY": "2",
    }
    with mock.patch.dict(os.environ, env, clear=True), fake_vast_sdk():
        config = worker.build_worker_config()
    assert all(handler.allow_parallel_requests for handler in config.handlers)
    benchmark = next(
        handler.benchmark_config
        for handler in config.handlers
        if hasattr(handler, "benchmark_config")
    )
    assert benchmark.concurrency == 2


def test_vast_launcher_uses_validated_request_limit():
    source = (ROOT / "scripts/vast_glm53_start.sh").read_text(encoding="utf-8")
    assert 'max_running_requests="${TEKIZAI_MAX_RUNNING_REQUESTS:-1}"' in source
    assert '--max-running-requests "$max_running_requests"' in source
