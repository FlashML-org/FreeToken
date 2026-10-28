from __future__ import annotations

from dataclasses import dataclass, field
import os

from freetoken.engine import EngineConfig


def _get_pid_suffix() -> str:
    return f".pid={os.getpid()}"


def _zmq_addr(index: int, suffix: str) -> str:
    if os.name != "nt":
        return f"ipc:///tmp/freetoken_{index}{suffix}"
    try:
        owner_pid = int(suffix.rsplit("=", 1)[1])
    except (IndexError, ValueError):
        owner_pid = os.getpid()
    port = 30000 + (owner_pid % 10000) * 5 + index
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

    @property
    def zmq_backend_addr(self) -> str:
        return _zmq_addr(0, self._unique_suffix)

    @property
    def zmq_detokenizer_addr(self) -> str:
        return _zmq_addr(1, self._unique_suffix)

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return _zmq_addr(2, self._unique_suffix)

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
