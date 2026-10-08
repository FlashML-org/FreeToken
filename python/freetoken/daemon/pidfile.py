"""Single-instance lock + persisted serve state.

``SingleInstance`` is a flock-held pidfile: an advisory ``LOCK_EX | LOCK_NB`` on an fd we keep
open for the daemon's whole life. flock (not a bare port-bind) avoids TIME_WAIT races and is
released automatically if the daemon dies, so a crashed daemon never wedges its own restart.

``ServeStateStore`` persists ``{model, port, pid, args, starttime, log_path, api_key}`` as JSON on
every lifecycle change, so a restarted daemon can re-adopt a still-running serve. ``starttime`` +
``args`` are what make re-adoption PID-reuse-safe and config-exact. The file is owner-only
because it holds the serve's API key."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field


class AlreadyRunning(RuntimeError):
    """Another daemon holds the single-instance lock."""


class SingleInstance:
    def __init__(self, path: str) -> None:
        self.path = path
        self._fd: int | None = None

    def acquire(self) -> None:
        import fcntl  # POSIX-only; the daemon's reference platform is Linux/WSL

        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise AlreadyRunning(f"another ft daemon holds {self.path}") from exc
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.fsync(fd)
        self._fd = fd

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)  # closing drops the flock
            except OSError:
                pass
            self._fd = None

    def __enter__(self) -> "SingleInstance":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


@dataclass
class ServeState:
    model: str
    port: int
    pid: int
    args: list[str]
    starttime: int | None = None
    log_path: str | None = None
    api_key: str | None = field(default=None, repr=False)


class ServeStateStore:
    def __init__(self, path: str) -> None:
        self.path = path

    def save(self, state: ServeState) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = f"{self.path}.tmp"
        # unlink first: O_CREAT applies the 0o600 mode only to a file it creates
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as fh:
            json.dump(asdict(state), fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)  # atomic swap — never a torn/partial state file

    def clear(self) -> None:
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass
        except OSError:  # pragma: no cover - defensive
            pass

    def load(self) -> ServeState | None:
        try:
            with open(self.path) as fh:
                doc = json.load(fh)
        except (FileNotFoundError, ValueError):
            return None
        except OSError:  # pragma: no cover - defensive
            return None
        try:
            return ServeState(
                model=doc["model"],
                port=int(doc["port"]),
                pid=int(doc["pid"]),
                args=list(doc.get("args") or []),
                starttime=doc.get("starttime"),
                log_path=doc.get("log_path"),
                api_key=doc.get("api_key"),
            )
        except (KeyError, TypeError, ValueError):
            return None
