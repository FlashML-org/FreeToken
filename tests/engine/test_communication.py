"""The TP process group of one rank meets no other process, so it must start whatever holds the rendezvous port."""

from __future__ import annotations

import contextlib
import errno
import glob
import os
import socket
import tempfile
from types import SimpleNamespace
from typing import Iterator

import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine.engine import Engine
from freetoken.scheduler.config import SchedulerConfig
from freetoken.server.args import ServerArgs


def _hold(sock: socket.socket, port: int) -> None:
    try:
        sock.bind(("127.0.0.1", port))
        sock.listen()
    except OSError as exc:
        # another process holding it is the condition under test too
        if exc.errno not in (errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", errno.EADDRINUSE)):
            raise


@contextlib.contextmanager
def _single_rank_group(config) -> Iterator[torch.distributed.ProcessGroup]:
    # the gloo branch sizes pynccl's buffer from the model; a single rank never builds it
    object.__setattr__(config, "model_config", SimpleNamespace(hidden_size=8))
    try:
        yield Engine._init_communication(SimpleNamespace(dtype=torch.bfloat16), config)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def test_a_single_rank_server_starts_while_the_port_after_its_own_is_taken():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        _hold(holder, 0)
        config = ServerArgs(model_path="unused", tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
                            server_port=holder.getsockname()[1] - 1)
        with _single_rank_group(config) as group:
            assert torch.distributed.get_world_size(group) == 1


def test_a_single_rank_offline_engine_starts_while_the_fixed_rendezvous_port_is_taken():
    config = SchedulerConfig(model_path="unused", tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16, offline_mode=True)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        _hold(holder, int(config.distributed_addr.rsplit(":", 1)[1]))
        with _single_rank_group(config) as group:
            assert torch.distributed.get_world_size(group) == 1


def test_a_formed_single_rank_group_works_without_a_store_file():
    pattern = os.path.join(tempfile.gettempdir(), "freetoken_store_*")
    before = set(glob.glob(pattern))
    config = SchedulerConfig(model_path="unused", tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16, offline_mode=True)
    with _single_rank_group(config) as group:
        assert set(glob.glob(pattern)) - before == set()
        summed = torch.ones(2)
        torch.distributed.all_reduce(summed, group=group)
        assert summed.tolist() == [1.0, 1.0]
    assert set(glob.glob(pattern)) - before == set()
