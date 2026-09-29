from __future__ import annotations

from dataclasses import dataclass, field

from freetoken.engine import EngineConfig


ZMQ_LINK_COUNT = 5


def _free_loopback_ports(count: int) -> list[int]:
    import socket

    held = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(count)]
    try:
        for sock in held:
            sock.bind(("127.0.0.1", 0))
        return [sock.getsockname()[1] for sock in held]
    finally:
        for sock in held:
            sock.close()


def _choose_zmq_links() -> tuple[str, ...]:
    """The ZMQ endpoints, chosen once in the parent so every spawned worker reads the same ones;
    loopback TCP ports on Windows, where libzmq has no ipc transport."""
    import os

    import zmq

    if zmq.has("ipc"):
        return tuple(f"ipc:///tmp/freetoken_{link}.pid={os.getpid()}" for link in range(ZMQ_LINK_COUNT))
    return tuple(f"tcp://127.0.0.1:{port}" for port in _free_loopback_ports(ZMQ_LINK_COUNT))


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False

    # networking config
    _zmq_links: tuple[str, ...] = field(default_factory=_choose_zmq_links)

    @property
    def zmq_backend_addr(self) -> str:
        return self._zmq_links[0]

    @property
    def zmq_detokenizer_addr(self) -> str:
        return self._zmq_links[1]

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return self._zmq_links[2]

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
