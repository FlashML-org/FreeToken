from __future__ import annotations

from dataclasses import dataclass, field

from freetoken.engine import EngineConfig
from freetoken.engine.config import prefill_tail


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

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/freetoken_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/freetoken_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/freetoken_2" + self._unique_suffix

    @property
    def prefill_tail_tokens(self) -> int:
        """Tokens past --max-prefill-length that a prompt's last chunk takes with it where the scheduler allows it."""
        return prefill_tail(self.max_extend_tokens)

    @property
    def max_forward_len(self) -> int:
        # the longest prefill forward: a chunk and the rest a last chunk may take (scheduler/prefill.py)
        return self.max_extend_tokens + self.prefill_tail_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
