from __future__ import annotations

import os
from dataclasses import dataclass, field

from freetoken.engine import EngineConfig


def _get_pid_suffix() -> str:
    return f".pid={os.getpid()}"


_ZMQ_CHANNELS = (
    "backend",
    "detokenizer",
    "scheduler_broadcast",
    "frontend",
    "tokenizer",
)
_WINDOWS_ZMQ_FIRST_PORT_OFFSET = 2  # API is +0; distributed process group is +1.


def _local_zmq_addr(
    channel_index: int,
    *,
    unique_suffix: str,
    server_port: int | None,
    platform_name: str | None = None,
) -> str:
    """Return one process-shared local endpoint without changing POSIX behavior.

    Windows libzmq does not support FreeToken's ``ipc:///tmp`` addresses.  All spawned
    workers already receive the same ServerArgs, so deriving five loopback-only TCP ports
    from the API port keeps bind/connect ownership deterministic across ``spawn`` without a
    racy parent-side random-port handoff.
    """
    if not 0 <= channel_index < len(_ZMQ_CHANNELS):
        raise ValueError(f"unknown FreeToken ZMQ channel index: {channel_index}")
    if (os.name if platform_name is None else platform_name) != "nt":
        return f"ipc:///tmp/freetoken_{channel_index}{unique_suffix}"
    if server_port is None:
        raise ValueError("server port is required for Windows loopback ZMQ endpoints")
    port = server_port + _WINDOWS_ZMQ_FIRST_PORT_OFFSET + channel_index
    if server_port < 1 or port > 65535:
        raise ValueError(
            f"server port {server_port} is too high for the five Windows loopback ZMQ "
            f"endpoints; expected 1..{65535 - _WINDOWS_ZMQ_FIRST_PORT_OFFSET - len(_ZMQ_CHANNELS) + 1}"
        )
    return f"tcp://127.0.0.1:{port}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    def _zmq_addr(self, channel_index: int) -> str:
        return _local_zmq_addr(
            channel_index,
            unique_suffix=self._unique_suffix,
            server_port=getattr(self, "server_port", None),
        )

    @property
    def zmq_backend_addr(self) -> str:
        return self._zmq_addr(0)

    @property
    def zmq_detokenizer_addr(self) -> str:
        return self._zmq_addr(1)

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return self._zmq_addr(2)

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
