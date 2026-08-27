from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from freetoken.server.args import parse_args


class _Config:
    def to_dict(self) -> dict:
        return {"architectures": ["DeepseekV4ForCausalLM"], "torch_dtype": "bfloat16"}


def _parse(extra: list[str]):
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: _Config()):
        return parse_args(["--model", "/models/anon", *extra])


def test_tls_certificate_and_key_are_parsed_as_a_pair():
    args, run_shell = _parse(
        ["--ssl-certfile", "/certs/fullchain.pem", "--ssl-keyfile", "/certs/privkey.pem"]
    )

    assert run_shell is False
    assert args.ssl_certfile == "/certs/fullchain.pem"
    assert args.ssl_keyfile == "/certs/privkey.pem"


@pytest.mark.parametrize(
    "single_flag",
    [
        ["--ssl-certfile", "/certs/fullchain.pem"],
        ["--ssl-keyfile", "/certs/privkey.pem"],
    ],
)
def test_tls_rejects_an_incomplete_certificate_pair(single_flag):
    with pytest.raises(SystemExit, match="2"):
        _parse(single_flag)


def test_tls_rejects_shell_mode():
    with pytest.raises(SystemExit, match="2"):
        _parse(
            [
                "--shell-mode",
                "--ssl-certfile",
                "/certs/fullchain.pem",
                "--ssl-keyfile",
                "/certs/privkey.pem",
            ]
        )


def test_tls_is_forwarded_to_uvicorn():
    from freetoken.server.api_server import _uvicorn_tls_kwargs

    config = SimpleNamespace(
        ssl_certfile="/certs/fullchain.pem",
        ssl_keyfile="/certs/privkey.pem",
    )

    assert _uvicorn_tls_kwargs(config) == {
        "ssl_certfile": "/certs/fullchain.pem",
        "ssl_keyfile": "/certs/privkey.pem",
    }


def test_plain_http_keeps_uvicorn_tls_disabled():
    from freetoken.server.api_server import _uvicorn_tls_kwargs

    config = SimpleNamespace(ssl_certfile=None, ssl_keyfile=None)

    assert _uvicorn_tls_kwargs(config) == {}
