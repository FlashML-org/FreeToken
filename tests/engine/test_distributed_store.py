"""Regression test for #301: the TP rendezvous store must not fall back to a wildcard
bind. torch's tcp:// rendezvous handler ignores the host it's given when binding the
master's listening socket and always listens on every interface, so
init_method="tcp://127.0.0.1:PORT" looks loopback-only but isn't -- see
Engine._make_distributed_store. This drives the real store-construction path across two
spawned local ranks (CPU/gloo) and checks the actual bind address, not just the URL.
"""

from __future__ import annotations

import multiprocessing as mp
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist

from freetoken.distributed import DistributedInfo
from freetoken.engine.engine import Engine

WORLD_SIZE = 2
TIMEOUT = timedelta(seconds=15)


def _config(rank: int, port: int) -> SimpleNamespace:
    return SimpleNamespace(
        tp_info=DistributedInfo(rank, WORLD_SIZE),
        distributed_timeout=TIMEOUT.total_seconds(),
        distributed_port=port,
    )


def _run_rank(rank: int, port: int, result_q: mp.Queue) -> None:
    engine = Engine.__new__(Engine)  # only _make_distributed_store is under test
    store = engine._make_distributed_store(_config(rank, port))
    dist.init_process_group(
        backend="gloo", rank=rank, world_size=WORLD_SIZE, timeout=TIMEOUT, store=store
    )
    total = torch.tensor([float(rank)])
    dist.all_reduce(total)
    bind_host = engine._distributed_listen_socket.getsockname()[0] if rank == 0 else None
    result_q.put((rank, total.item(), bind_host))
    dist.destroy_process_group()


def test_two_ranks_rendezvous_with_loopback_only_store():
    ctx = mp.get_context("spawn")
    port = 29511
    result_q = ctx.Queue()
    procs = [ctx.Process(target=_run_rank, args=(rank, port, result_q)) for rank in range(WORLD_SIZE)]
    for p in procs:
        p.start()

    results = {}
    for _ in procs:
        rank, total, bind_host = result_q.get(timeout=30)
        results[rank] = (total, bind_host)
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0

    # both ranks completed the same all_reduce over the store -- rendezvous worked
    assert results[0][0] == results[1][0] == 1.0  # sum(0, 1)
    # the master's listening socket was pre-bound to loopback, not the wildcard
    # address the C10d TCPStore server binds to on its own
    assert results[0][1] == "127.0.0.1"
