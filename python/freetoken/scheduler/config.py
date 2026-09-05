from __future__ import annotations

import socket
from dataclasses import dataclass, field

from freetoken.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


def _ipc_supported() -> bool:
    """Whether ZMQ can bind ``ipc://`` transports.

    libzmq is built without the IPC transport on Windows (there is no AF_UNIX
    equivalent it can use), so every ``ipc://`` bind raises "Protocol not
    supported". Probing the build is cheaper and more honest than sniffing the
    platform.
    """
    try:
        import zmq
    except ImportError:  # zmq missing entirely -> nothing to bind anyway
        return False
    return bool(zmq.has("ipc"))


def _reserve_port_base(count: int = 8) -> int:
    """Reserve a contiguous, currently-free localhost port block for the TCP fallback.

    The workers are separate processes, so they cannot negotiate ports at runtime:
    the parent picks a base once and every address derives from it deterministically,
    and the resolved base travels to the children inside the (pickled) config. Bind
    to port 0 to let the OS pick a free port, then step past the block we intend to
    use so a second server on the same host lands elsewhere.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        base = probe.getsockname()[1]
    # Keep the block inside the ephemeral range; wrap rather than overflow.
    if base + count > 65535:
        base -= count
    return base


def _default_ipc_base_port() -> int:
    return _reserve_port_base() if not _ipc_supported() else 0


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)
    # Base of the localhost port block used when ipc:// is unavailable (Windows).
    # 0 means "ipc:// works, no ports needed". Resolved once in the parent so the
    # workers inherit the same addresses through the config they are handed.
    _ipc_base_port: int = field(default_factory=_default_ipc_base_port)

    def _socket_addr(self, index: int) -> str:
        """One inter-process socket address. ``index`` must be unique per socket.

        ``ipc://`` where the ZMQ build supports it (the POSIX path, unchanged);
        otherwise a fixed offset into the reserved localhost port block. Loopback
        TCP is visible to other local processes, unlike a filesystem socket, so
        the block is bound to 127.0.0.1 only.
        """
        if self._ipc_base_port == 0:
            return f"ipc:///tmp/freetoken_{index}{self._unique_suffix}"
        return f"tcp://127.0.0.1:{self._ipc_base_port + index}"

    @property
    def zmq_backend_addr(self) -> str:
        return self._socket_addr(0)

    @property
    def zmq_detokenizer_addr(self) -> str:
        return self._socket_addr(1)

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return self._socket_addr(2)

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
