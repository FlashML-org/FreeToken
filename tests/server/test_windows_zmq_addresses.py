from __future__ import annotations

import os

import torch

from freetoken.distributed import DistributedInfo
from freetoken.server import api_server
from freetoken.server.args import ServerArgs


def test_server_internal_zmq_addresses_are_supported_on_current_platform() -> None:
    args = ServerArgs(
        model_path="test-model",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        server_port=1927,
        num_tokenizer=1,
    )
    addresses = [
        args.zmq_backend_addr,
        args.zmq_detokenizer_addr,
        args.zmq_scheduler_broadcast_addr,
        args.zmq_frontend_addr,
        args.zmq_tokenizer_addr,
    ]

    assert len(set(addresses)) == len(addresses)
    if os.name == "nt":
        assert addresses == [f"tcp://127.0.0.1:{port}" for port in range(1929, 1934)]
    else:
        assert all(address.startswith("ipc:///tmp/freetoken_") for address in addresses)


def test_uvicorn_uses_a_zmq_compatible_loop_on_windows() -> None:
    if os.name == "nt":
        assert api_server._uvicorn_loop().endswith(":windows_selector_loop_factory")
        loop = api_server.windows_selector_loop_factory()
        try:
            assert isinstance(loop, __import__("asyncio").SelectorEventLoop)
            assert hasattr(loop, "add_reader")
        finally:
            loop.close()
    else:
        assert api_server._uvicorn_loop() == "auto"
