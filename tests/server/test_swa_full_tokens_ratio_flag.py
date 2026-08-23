"""`--swa-full-tokens-ratio`: the SWA radix-cache window/full ratio, set at startup.

Exposes the existing ``EngineConfig.swa_full_tokens_ratio`` (previously reachable only via a runtime
``/v1/cache/rebuild``) as a load-time CLI flag, so a sliding-window model can size its window pool
up front -- e.g. large enough to retain a fixed prompt prefix for cross-request reuse. The value
must be in ``(0, 1]``, mirroring the ``/v1/cache/rebuild`` validation.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from freetoken.server.args import ServerArgs, parse_args

ANON_PATH = "/models/anon"


class _Config:
    def __init__(self, data: dict) -> None:
        self._data = data

    def to_dict(self) -> dict:
        return self._data


def _parse(*extra):
    # Mock the checkpoint config so parse_args' auto-parser cascade resolves without a real model.
    config = _Config({"architectures": ["LlamaForCausalLM"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: config):
        args, _run_shell = parse_args(["--model", ANON_PATH, *extra])
    return args


def test_defaults_to_the_engine_config_value_when_absent():
    assert _parse().swa_full_tokens_ratio == ServerArgs.swa_full_tokens_ratio


def test_an_explicit_value_in_range_is_accepted():
    assert _parse("--swa-full-tokens-ratio", "0.3").swa_full_tokens_ratio == 0.3
    assert _parse("--swa-full-tokens-ratio", "1").swa_full_tokens_ratio == 1.0


@pytest.mark.parametrize("bad", ["0", "1.5", "-0.1", "nan", "abc"])
def test_out_of_range_or_non_numeric_is_rejected_at_parse_time(bad):
    # argparse validates the arg type before any checkpoint config is loaded, so no mock is needed:
    # the reject must not depend on reaching the model.
    with pytest.raises(SystemExit):
        parse_args(["--model", ANON_PATH, "--swa-full-tokens-ratio", bad])
