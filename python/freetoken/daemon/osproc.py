"""Thin, defensive OS/`/proc` helpers for the supervisor: spawn-group signalling, footprint,
orphan detection. Every function is best-effort: on a non-Linux host (no ``/proc``) or a vanished
pid it returns a safe default instead of raising, so the daemon never goes down over an
environment quirk ("degraded start"). Imports stdlib only."""

from __future__ import annotations

import errno
import os
import signal
import time
from typing import Iterable

_HAS_PROC = os.path.isdir("/proc")


def pid_alive(pid: int) -> bool:
    """True iff a process with this pid currently exists. ``kill(pid, 0)`` is the portable probe:
    ESRCH → gone, EPERM → exists but not ours (still alive)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:  # pragma: no cover - defensive
        return exc.errno != errno.ESRCH
    return True


def _read_proc(pid: int, name: str) -> str | None:
    try:
        with open(f"/proc/{pid}/{name}", "rb") as fh:
            return fh.read().decode("utf-8", "replace")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    except OSError:  # pragma: no cover - defensive
        return None


def read_cmdline(pid: int) -> list[str]:
    """The process argv as a list (``/proc/<pid>/cmdline`` is NUL-separated). [] if unavailable."""
    raw = _read_proc(pid, "cmdline")
    if not raw:
        return []
    parts = raw.split("\x00")
    if parts and parts[-1] == "":
        parts.pop()
    return parts


def _stat_fields(pid: int) -> list[str] | None:
    """Split ``/proc/<pid>/stat`` past the comm field (which may itself contain spaces/parens).
    Returns the fields from ``state`` onward, so field N (N>=3) is ``fields[N-3]``."""
    raw = _read_proc(pid, "stat")
    if not raw:
        return None
    rparen = raw.rfind(")")
    if rparen == -1:
        return None
    return raw[rparen + 2 :].split()


def read_starttime(pid: int) -> int | None:
    """The process start time (stat field 22, clock ticks since boot). Combined with the pid this
    is a stable identity that survives nothing — it changes on PID reuse — so it is the guard
    against adopting/signalling a recycled pid."""
    fields = _stat_fields(pid)
    if fields is None or len(fields) < 20:
        return None
    try:
        return int(fields[19])  # field 22 == fields[22-3]
    except (ValueError, IndexError):  # pragma: no cover - defensive
        return None


def proc_pgid(pid: int) -> int | None:
    """The process group id (stat field 5). Used to (a) enumerate the serve's worker tree and
    (b) assert ``pgid == pid`` before a group-kill so we never signal an unrelated group."""
    fields = _stat_fields(pid)
    if fields is None or len(fields) < 3:
        # macOS and other non-/proc POSIX hosts still expose the process group
        # through getpgid(2). Serve children are created with
        # start_new_session=True, so pgid == pid remains the safety invariant.
        try:
            return os.getpgid(pid)
        except (ProcessLookupError, PermissionError, OSError):
            return None
    try:
        return int(fields[2])  # field 5 == fields[5-3]
    except (ValueError, IndexError):  # pragma: no cover - defensive
        return None


def _iter_all_pids() -> Iterable[int]:
    try:
        for name in os.listdir("/proc"):
            if name.isdigit():
                yield int(name)
    except OSError:  # pragma: no cover - defensive
        return


def tree_pids(root_pid: int) -> list[int]:
    """The serve and all its worker processes: every pid whose process group == ``root_pid``.
    The serve is spawned as a session/group leader (``start_new_session=True``), and its
    mp-spawned scheduler/tokenizer workers inherit that group, so pgid membership captures the
    whole tree without walking ppid chains. Falls back to ``[root_pid]`` off-/proc."""
    if not _HAS_PROC:
        return [root_pid] if pid_alive(root_pid) else []
    out = []
    for pid in _iter_all_pids():
        if proc_pgid(pid) == root_pid:
            out.append(pid)
    if root_pid not in out and pid_alive(root_pid):
        out.append(root_pid)
    return out


def signal_group(pid: int, sig: int) -> None:
    """Signal the serve's whole process group. Resolves the pgid and asserts ``pgid == pid``
    (the group-leader invariant of our own spawn) before ``killpg``; on any mismatch or lookup
    failure it falls back to signalling just the pid, so a re-adopted process with an unexpected
    group is never able to make us nuke an innocent group."""
    pgid = proc_pgid(pid)
    try:
        if pgid is not None and pgid == pid:
            os.killpg(pgid, sig)
        else:
            os.kill(pid, sig)
    except ProcessLookupError:
        pass
    except OSError:  # pragma: no cover - defensive
        pass


def set_oom_score_adj(pid: int, score: int) -> bool:
    """Best-effort write to ``/proc/<pid>/oom_score_adj``. Returns whether it stuck.
    Lowering the daemon's own score needs privilege we may not have under ``systemctl --user``;
    raising the serve's is the load-bearing half and usually succeeds. Never raises."""
    try:
        with open(f"/proc/{pid}/oom_score_adj", "w") as fh:
            fh.write(str(score))
        return True
    except OSError:
        return False


def read_pss_bytes(pid: int) -> int:
    """Proportional set size for one process from ``/proc/<pid>/smaps_rollup`` (shared pages
    counted fractionally — the honest per-process RAM number). 0 if unavailable."""
    raw = _read_proc(pid, "smaps_rollup")
    if not raw:
        return 0
    for line in raw.splitlines():
        if line.startswith("Pss:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024  # value is in kB
    return 0


def is_ft_serve_on_port(pid: int, port: int, *, starttime: int | None = None) -> bool:
    """Verify ``pid`` is (still) an ``ft serve`` bound to ``port`` — the re-adoption / liveness
    identity check. Requires: alive, unchanged start time (PID-reuse
    guard), an argv that looks like our serve invocation, and a matching ``--port`` (or the serve
    default when unspecified). Off-/proc it degrades to a bare liveness check."""
    if not pid_alive(pid):
        return False
    if starttime is not None and read_starttime(pid) != starttime:
        return False
    if not _HAS_PROC:
        return True
    argv = read_cmdline(pid)
    if not argv:
        return False
    joined = " ".join(argv)
    looks_like_serve = "serve" in argv and (
        "freetoken.cli" in joined or "freetoken" in joined or os.path.basename(argv[0]) in {"ft", "ft.exe"}
    )
    if not looks_like_serve:
        return False
    return _argv_port(argv) == port


def _argv_port(argv: list[str], default: int = 1919) -> int:
    """The integer following a standalone ``--port`` / ``-p`` in argv, else the serve default
    (``ServerArgs.server_port``). Also accepts ``--port=N``."""
    for i, tok in enumerate(argv):
        if tok in ("--port", "-p") and i + 1 < len(argv):
            nxt = argv[i + 1]
            if nxt.isdigit():
                return int(nxt)
        elif tok.startswith("--port="):
            val = tok.split("=", 1)[1]
            if val.isdigit():
                return int(val)
    return default


def now_monotonic() -> float:
    return time.monotonic()
