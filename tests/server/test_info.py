"""ft info (server/info.py): argument handling, the safetensors header view, JSON and exit codes."""

import json
import math

import pytest

from freetoken.engine.forecast import GiB, tensor_category
from tests.engine.test_forecast import (
    _DTYPE_BYTES, KV_PER_TOKEN, TINY_QWEN3, _analyze, tiny_qwen3_tensors, write_checkpoint,
)


@pytest.fixture
def tiny_qwen3(tmp_path):
    return write_checkpoint(tmp_path / "tiny-qwen3", TINY_QWEN3, tiny_qwen3_tensors())


def test_split_argv_takes_the_model_positionally():
    from freetoken.server.info import split_argv

    opts, rest = split_argv(["Qwen/Qwen3-0.6B", "--json", "--max-running-requests", "1", "--gpu-memory-gib", "24"])
    assert opts.json and opts.gpu_memory_gib == 24 and opts.gpu_free_gib is None
    assert rest == ["--model", "Qwen/Qwen3-0.6B", "--max-running-requests", "1"]
    _, rest = split_argv(["--model", "m", "--text-model-only"])
    assert rest == ["--model", "m", "--text-model-only"]
    with pytest.raises(SystemExit):
        split_argv(["m", "--gpu-memory-gib", "24", "--gpu-free-gib", "20"])


def test_hub_headers_without_download(monkeypatch):
    import huggingface_hub
    from types import SimpleNamespace

    from freetoken.server.info import checkpoint_tensors

    def not_cached(*a, **k):
        raise FileNotFoundError

    meta = SimpleNamespace(files_metadata={"model.safetensors": SimpleNamespace(tensors={
        "model.embed_tokens.weight": SimpleNamespace(dtype="BF16", data_offsets=(0, 4096)),
        "model.layers.0.self_attn.q_proj.weight": SimpleNamespace(dtype="F8_E4M3", data_offsets=(4096, 6144)),
    })})
    monkeypatch.setattr(huggingface_hub, "snapshot_download", not_cached)
    monkeypatch.setattr(huggingface_hub, "get_safetensors_metadata", lambda repo: meta)
    tensors, source = checkpoint_tensors("org/not-cached")
    assert tensors == {"model.embed_tokens.weight": ("BF16", 4096),
                       "model.layers.0.self_attn.q_proj.weight": ("F8_E4M3", 2048)}
    assert "HTTP range" in source


def test_checkpoint_view_reads_headers_only(tiny_qwen3):
    r = _analyze(tiny_qwen3)
    expected: dict[str, int] = {}
    for name, (dtype, shape) in tiny_qwen3_tensors().items():
        n = _DTYPE_BYTES[dtype]
        for d in shape:
            n *= d
        expected[tensor_category(name)] = expected.get(tensor_category(name), 0) + n
    assert r.checkpoint == expected
    assert r.checkpoint_source == "local safetensors headers"


def test_json_report_round_trips(tiny_qwen3, capsys):
    from freetoken.server.info import main

    rc = main([tiny_qwen3, "--gpu-free-gib", "1", "--json", "--max-running-requests", "2"])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert doc["architecture"] == "Qwen3ForCausalLM"
    assert doc["resolved"]["max_running_req"] == 2
    assert doc["forecast"]["verdict"] in ("fits", "tight")
    assert doc["forecast"]["kv_bytes_per_token"] == KV_PER_TOKEN
    assert doc["checkpoint"]["bytes_by_dtype"] == {"BF16": sum(
        2 * math.prod(shape) for _, shape in tiny_qwen3_tensors().values())}


def test_cli_exit_code_flags_no_fit(tiny_qwen3, capsys):
    from freetoken.server.info import main

    assert main([tiny_qwen3, "--gpu-free-gib", "0.01"]) == 1
    assert "DOES NOT FIT" in capsys.readouterr().out


def test_gpu_memory_option_plans_for_another_card(tiny_qwen3, monkeypatch):
    from freetoken.engine.forecast import CUDA_CONTEXT_BYTES
    from freetoken.server.args import parse_args
    from freetoken.server.info import gpu_info, split_argv

    opts, argv = split_argv([tiny_qwen3, "--gpu-memory-gib", "24"])
    config, _ = parse_args(argv, prog="ft info")
    info = gpu_info(opts, config)
    assert info.free_before == 24 * GiB - CUDA_CONTEXT_BYTES
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    opts, _ = split_argv([tiny_qwen3])
    assert gpu_info(opts, config).free_before is None


def test_no_gpu_still_reports_weights(tiny_qwen3, monkeypatch, capsys):
    from freetoken.server.info import main

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert main([tiny_qwen3]) == 0
    out = capsys.readouterr().out
    assert "GPU free memory unknown" in out and "attention" in out


def test_skip_preflight_flag_parses(tiny_qwen3):
    from freetoken.server.args import parse_args

    assert parse_args(["--model", tiny_qwen3])[0].skip_preflight is False
    assert parse_args(["--model", tiny_qwen3, "--skip-preflight"])[0].skip_preflight is True


def test_header_fallback_when_the_meta_build_fails(tiny_qwen3, monkeypatch):
    import freetoken.models

    def broken(model_config):
        raise NotImplementedError("no meta build")

    monkeypatch.setattr(freetoken.models, "create_model", broken)
    r = _analyze(tiny_qwen3)
    stored = sum(2 * math.prod(shape) for _, shape in tiny_qwen3_tensors().values())
    assert r.inputs.weights.gpu_total == stored
    assert r.forecast.kv_bytes_per_token == KV_PER_TOKEN
    assert any("meta device" in n for n in r.notes)


def test_config_errors_are_reported_without_a_traceback(tiny_qwen3, capsys, monkeypatch):
    import freetoken.engine.engine
    from freetoken.server.info import main

    def bad_config(config):
        raise ValueError("--moe-strategy fused cannot hold nvfp4 experts")

    monkeypatch.setattr(freetoken.engine.engine, "_adjust_config", bad_config)
    assert main([tiny_qwen3, "--gpu-free-gib", "8"]) == 2
    assert "nvfp4 experts" in capsys.readouterr().err
