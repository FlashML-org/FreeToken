"""CLI cache sizing is validated before loading the checkpoint."""

import pytest

from freetoken.server.args import parse_args


@pytest.mark.parametrize("ratio", ["0.01", "0.02", "0.2", "1"])
def test_window_pool_ratio_reaches_serving_config(ratio):
    config, _ = parse_args([
        "--model", "/models/local", "--dtype", "bfloat16",
        "--tool-call-parser", "llama3", "--reasoning-parser", "off",
        "--swa-full-tokens-ratio", ratio,
    ])
    assert config.swa_full_tokens_ratio == float(ratio)


@pytest.mark.parametrize("ratio", ["0", "-0.1", "1.01", "nan", "inf", "invalid"])
def test_invalid_window_pool_ratio_fails_before_model_lookup(ratio, monkeypatch):
    def unexpected_lookup(*args, **kwargs):
        pytest.fail("invalid ratio reached checkpoint loading")

    monkeypatch.setattr("freetoken.utils.cached_load_hf_config", unexpected_lookup)
    with pytest.raises(SystemExit) as error:
        parse_args(["--model", "/models/local", "--swa-full-tokens-ratio", ratio])
    assert error.value.code == 2
