from __future__ import annotations

import os
from dataclasses import dataclass, field

from freetoken.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    def _zmq_addr(self, ipc_path: str, tcp_port: int) -> str:
        """IPC on POSIX; loopback TCP on Windows, where libZMQ ships no ipc://
        transport (and the '.pid=NNN' instance suffix is invalid after a port).
        Single instance per port -- multi-instance serving must override."""
        if os.name == "nt":
            return f"tcp://127.0.0.1:{tcp_port}"
        return ipc_path + self._unique_suffix

    @property
    def zmq_backend_addr(self) -> str:
        return self._zmq_addr("ipc:///tmp/freetoken_0", 50)

    @property
    def zmq_detokenizer_addr(self) -> str:
        return self._zmq_addr("ipc:///tmp/freetoken_1", 51)

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return self._zmq_addr("ipc:///tmp/freetoken_2", 52)

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
