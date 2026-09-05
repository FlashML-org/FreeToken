from __future__ import annotations

import multiprocessing as mp
import os
import queue
import socket
import traceback

import pytest
import torch

from freetoken.distributed import DistributedInfo
from freetoken.scheduler.config import _local_zmq_addr
from freetoken.server.args import ServerArgs
from freetoken.utils import ZmqPullQueue, ZmqPushQueue


def _server_args(server_port: int, *, num_tokenizer: int = 0) -> ServerArgs:
    return ServerArgs(
        model_path="unused-by-transport-tests",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        server_port=server_port,
        num_tokenizer=num_tokenizer,
    )


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (0, "ipc:///tmp/freetoken_0.pid=test"),
        (1, "ipc:///tmp/freetoken_1.pid=test"),
        (2, "ipc:///tmp/freetoken_2.pid=test"),
        (3, "ipc:///tmp/freetoken_3.pid=test"),
        (4, "ipc:///tmp/freetoken_4.pid=test"),
    ],
)
def test_posix_ipc_addresses_remain_unchanged(index: int, expected: str) -> None:
    assert (
        _local_zmq_addr(
            index,
            unique_suffix=".pid=test",
            server_port=1919,
            platform_name="posix",
        )
        == expected
    )


@pytest.mark.parametrize("index", range(5))
def test_windows_addresses_are_deterministic_loopback_tcp(index: int) -> None:
    assert _local_zmq_addr(
        index,
        unique_suffix=".ignored-on-windows",
        server_port=1919,
        platform_name="nt",
    ) == f"tcp://127.0.0.1:{1921 + index}"


def test_windows_address_range_rejects_overflow() -> None:
    with pytest.raises(ValueError, match="server port.*too high"):
        _local_zmq_addr(
            4,
            unique_suffix=".pid=test",
            server_port=65530,
            platform_name="nt",
        )


@pytest.mark.skipif(os.name != "nt", reason="native Windows ServerArgs contract")
def test_windows_server_args_cover_all_channels_without_collisions() -> None:
    shared = _server_args(1919, num_tokenizer=0)
    separate = _server_args(1919, num_tokenizer=1)

    assert shared.distributed_addr == "tcp://127.0.0.1:1920"
    assert shared.zmq_backend_addr == "tcp://127.0.0.1:1921"
    assert shared.zmq_detokenizer_addr == "tcp://127.0.0.1:1922"
    assert shared.zmq_scheduler_broadcast_addr == "tcp://127.0.0.1:1923"
    assert shared.zmq_frontend_addr == "tcp://127.0.0.1:1924"
    assert shared.zmq_tokenizer_addr == shared.zmq_detokenizer_addr
    assert separate.zmq_tokenizer_addr == "tcp://127.0.0.1:1925"

    all_ports = {
        shared.server_port,
        int(shared.distributed_addr.rsplit(":", 1)[1]),
        int(shared.zmq_backend_addr.rsplit(":", 1)[1]),
        int(shared.zmq_detokenizer_addr.rsplit(":", 1)[1]),
        int(shared.zmq_scheduler_broadcast_addr.rsplit(":", 1)[1]),
        int(shared.zmq_frontend_addr.rsplit(":", 1)[1]),
        int(separate.zmq_tokenizer_addr.rsplit(":", 1)[1]),
    }
    assert all_ports == set(range(1919, 1926))

    assert shared.tokenizer_create_addr is True
    assert shared.backend_create_detokenizer_link is False
    assert shared.frontend_create_tokenizer_link is False
    assert separate.tokenizer_create_addr is False
    assert separate.backend_create_detokenizer_link is True
    assert separate.frontend_create_tokenizer_link is True


def _reserve_test_port_block() -> int:
    """Return the first seven-port loopback block available in a deterministic test range."""
    for server_port in range(41000, 60000, 7):
        reservations: list[socket.socket] = []
        try:
            for port in range(server_port, server_port + 7):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                if os.name == "nt":
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                sock.bind(("127.0.0.1", port))
                reservations.append(sock)
            return server_port
        except OSError:
            pass
        finally:
            for sock in reservations:
                sock.close()
    raise RuntimeError("No seven-port loopback block is available for the ZMQ smoke test")


def _pull_once(addr: str, ready: mp.Queue, result: mp.Queue) -> None:
    receiver = None
    try:
        receiver = ZmqPullQueue(addr, create=True, decoder=lambda payload: payload)
        ready.put(("ready", addr))
        result.put(("ok", receiver.get()))
    except BaseException:
        ready.put(("error", traceback.format_exc()))
    finally:
        if receiver is not None:
            receiver.stop()


@pytest.mark.skipif(os.name != "nt", reason="native Windows spawn/ZMQ smoke")
def test_windows_spawn_can_bind_and_exchange_on_all_five_internal_endpoints() -> None:
    server_port = _reserve_test_port_block()
    args = _server_args(server_port, num_tokenizer=1)
    addresses = [
        args.zmq_backend_addr,
        args.zmq_detokenizer_addr,
        args.zmq_scheduler_broadcast_addr,
        args.zmq_frontend_addr,
        args.zmq_tokenizer_addr,
    ]
    assert len(set(addresses)) == 5

    ctx = mp.get_context("spawn")
    for index, addr in enumerate(addresses):
        ready = ctx.Queue()
        result = ctx.Queue()
        process = ctx.Process(target=_pull_once, args=(addr, ready, result))
        process.start()
        try:
            ready_status, ready_value = ready.get(timeout=15)
            assert ready_status == "ready", ready_value
            sender = ZmqPushQueue(addr, create=False, encoder=lambda payload: payload)
            try:
                payload = {"endpoint_index": index, "addr": addr}
                sender.put(payload)
                result_status, result_value = result.get(timeout=15)
                assert result_status == "ok", result_value
                assert result_value == payload
            finally:
                sender.stop()
        except queue.Empty as exc:
            pytest.fail(f"Timed out waiting for spawned ZMQ worker at {addr}: {exc}")
        finally:
            process.join(timeout=15)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        assert process.exitcode == 0
