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


def test_vast_provisioner_installs_jit_headers_and_verifies_commit():
    source = (ROOT / "scripts/vast_glm53_provision.sh").read_text(encoding="utf-8")
    assert "libcurand-dev-13-0" in source
    assert "TEKIZAI_FREETOKEN_EXPECTED_COMMIT" in source
    assert 'git -C "$checkout" rev-parse HEAD' in source


def test_vast_provisioner_has_validated_fast_resume_path():
    source = (ROOT / "scripts/vast_glm53_provision.sh").read_text(encoding="utf-8")
    assert 'FREETOKEN_PROVISION_STAGE=fast_resume' in source
    assert '[[ -s "$model_dir/config.json" ]]' in source
    assert 'checkout_matches_expected_commit' in source
    assert 'printf \'%s\\n\' "$provision_marker_value" >"$provision_marker"' in source


def test_vast_provisioner_supports_persistent_ftw_conversion():
    source = (ROOT / "scripts/vast_glm53_provision.sh").read_text(encoding="utf-8")
    assert 'model_source_dir="${TEKIZAI_MODEL_SOURCE_PATH:-$model_dir}"' in source
    assert 'convert_ftw="${TEKIZAI_CONVERT_FTW:-0}"' in source
    assert 'FREETOKEN_PROVISION_STAGE=ftw_conversion' in source
    assert '"$freetoken_dir/.venv/bin/ft" checkpoint' in source
    assert '--model "$model_source_dir"' in source
    assert '--out "$model_dir"' in source
    assert 'TEKIZAI_PROVISION_MARKER' in source
    assert '--dtype "$model_bench_dtype"' in source


def test_deepseek_v4_vast_provisioner_sets_model_contract():
    source = (ROOT / "scripts/vast_deepseek_v4_provision.sh").read_text(
        encoding="utf-8"
    )
    assert "deepseek-ai/DeepSeek-V4-Flash-0731" in source
    assert "TEKIZAI_MODEL_BENCH_DTYPE" in source
    assert "ds_fp4" in source
    assert "TEKIZAI_SERVED_MODEL" in source
    assert "raw.githubusercontent.com/earlvanze/FreeToken" in source
    assert "TEKIZAI_FREETOKEN_REF" in source
    assert "TEKIZAI_MODEL_SOURCE_PATH" in source
    assert "DeepSeek-V4-Flash-0731-ftw" in source
    assert 'TEKIZAI_CONVERT_FTW="${TEKIZAI_CONVERT_FTW:-1}"' in source
    assert 'exec "$shared_provisioner"' in source
