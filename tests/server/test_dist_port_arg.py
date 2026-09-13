"""Regression test for #301: the TP rendezvous port must be independently configurable
from --port, so two instances on adjacent ports don't have one's API port collide with
the other's rendezvous port.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from freetoken.server.args import parse_args

ANON_PATH = "/models/anon"


class _Config:
    def to_dict(self) -> dict:
        return {"architectures": ["LlamaForCausalLM"], "torch_dtype": "bfloat16"}


def _parse(argv: list[str]):
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: _Config()):
        return parse_args(["--model", ANON_PATH, *argv])


def test_dist_port_defaults_to_server_port_plus_one():
    args, _ = _parse(["--port", "8081"])
    assert args.distributed_port == 8082
    assert args.distributed_addr == "tcp://127.0.0.1:8082"


def test_dist_port_override_is_independent_of_server_port():
    args, _ = _parse(["--port", "8082", "--dist-port", "9000"])
    assert args.distributed_port == 9000
    assert args.distributed_addr == "tcp://127.0.0.1:9000"


def test_dist_port_rejects_out_of_range_value():
    with pytest.raises(SystemExit):
        _parse(["--dist-port", "70000"])
