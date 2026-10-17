"""The ZMQ endpoints every worker process connects through: one set per server, on either transport."""

from __future__ import annotations

import os
import pickle
import re

import pytest
import torch
import zmq

from freetoken.distributed import DistributedInfo
from freetoken.scheduler.config import ZMQ_LINK_COUNT, SchedulerConfig, _choose_zmq_links

LOOPBACK = re.compile(r"^tcp://127\.0\.0\.1:(\d+)$")


@pytest.fixture(params=[True, False], ids=["ipc", "no-ipc"])
def transport(request, monkeypatch) -> bool:
    monkeypatch.setattr(zmq, "has", lambda capability: request.param if capability == "ipc" else True)
    return request.param


def test_without_ipc_every_link_is_a_loopback_port(monkeypatch):
    monkeypatch.setattr(zmq, "has", lambda capability: capability != "ipc")
    ports = [LOOPBACK.match(link) for link in _choose_zmq_links()]
    assert all(ports), "a link that is not loopback TCP cannot bind where libzmq lacks ipc"
    assert len({int(port.group(1)) for port in ports}) == ZMQ_LINK_COUNT


def test_with_ipc_the_links_are_per_process_socket_paths(monkeypatch):
    monkeypatch.setattr(zmq, "has", lambda capability: True)
    assert _choose_zmq_links() == tuple(
        f"ipc:///tmp/freetoken_{link}.pid={os.getpid()}" for link in range(ZMQ_LINK_COUNT)
    )


def test_a_pickled_config_keeps_its_links(transport):
    config = SchedulerConfig(
        model_path="unused", tp_info=DistributedInfo(rank=0, size=1), dtype=torch.float16
    )
    spawned = pickle.loads(pickle.dumps(config))
    assert (spawned.zmq_backend_addr, spawned.zmq_detokenizer_addr, spawned.zmq_scheduler_broadcast_addr) == (
        config.zmq_backend_addr,
        config.zmq_detokenizer_addr,
        config.zmq_scheduler_broadcast_addr,
    )
