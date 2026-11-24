"""The ZMQ endpoints every worker process connects through: one set per server, on either transport."""

from __future__ import annotations

import os
import pickle
import re
import socket

import pytest
import torch
import zmq

from freetoken.distributed import DistributedInfo
from freetoken.scheduler.config import ZMQ_LINK_COUNT, SchedulerConfig, _choose_zmq_links

LOOPBACK = re.compile(r"^tcp://(?P<host>127(?:\.\d{1,3}){3}):(?P<port>\d+)$")


@pytest.fixture(params=[True, False], ids=["ipc", "no-ipc"])
def transport(request, monkeypatch) -> bool:
    monkeypatch.setattr(zmq, "has", lambda capability: request.param if capability == "ipc" else True)
    return request.param


def test_without_ipc_every_link_is_a_loopback_port(monkeypatch):
    monkeypatch.setattr(zmq, "has", lambda capability: capability != "ipc")
    links = [LOOPBACK.match(link) for link in _choose_zmq_links()]
    assert all(links), "a link that is not loopback TCP cannot bind where libzmq lacks ipc"
    assert len({int(link["port"]) for link in links}) == ZMQ_LINK_COUNT


def test_a_connect_to_a_link_leaves_from_another_address(monkeypatch):
    # A connect made before its link binds can be handed the link's port as its own and connect to itself, a peer of
    # its own socket type, which ZMQ never retries; a source address other than the link's rules that out.
    monkeypatch.setattr(zmq, "has", lambda capability: capability != "ipc")
    link = LOOPBACK.match(_choose_zmq_links()[0])
    with socket.socket() as listener, socket.socket() as client:
        listener.bind((link["host"], int(link["port"])))
        listener.listen(1)
        client.connect((link["host"], int(link["port"])))
        assert client.getsockname()[0] != link["host"]


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


def test_the_scheduler_binds_its_links_before_the_engine_loads(monkeypatch):
    # A port nothing holds for a whole load is one another program can take, and a ZMQ connect that meets a socket of
    # the wrong type there never retries.
    import freetoken.engine
    from freetoken.scheduler.scheduler import Scheduler

    class EngineLoading(Exception):
        pass

    monkeypatch.setattr(zmq, "has", lambda capability: capability != "ipc")
    config = SchedulerConfig(model_path="unused", tp_info=DistributedInfo(rank=0, size=1), dtype=torch.float16)
    backend = LOOPBACK.match(config.zmq_backend_addr)
    held_during_the_load: list[bool] = []

    class LoadingEngine:
        def __init__(self, _config):
            with socket.socket() as stranger:
                try:
                    stranger.bind((backend["host"], int(backend["port"])))
                    held_during_the_load.append(False)
                except OSError:
                    held_during_the_load.append(True)
            raise EngineLoading

    monkeypatch.setattr(freetoken.engine, "Engine", LoadingEngine)
    scheduler = Scheduler.__new__(Scheduler)
    with pytest.raises(EngineLoading):
        scheduler.__init__(config)
    for link in ("_recv_from_tokenizer", "_send_into_tokenizer"):
        queue = getattr(scheduler, link, None)
        if queue is not None:
            queue.stop()
    assert held_during_the_load == [True]
