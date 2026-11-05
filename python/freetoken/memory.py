"""Conservative host-memory admission for bare-metal and cgrouped processes.

The kernel's host-wide ``MemAvailable`` figure is not an allocation budget for a
container.  This module combines it with the process's tightest finite cgroup
budget and deliberately keeps the probing code independent of torch and the
rest of the runtime so startup checks and diagnostic tools can share it.
"""

from __future__ import annotations

import re
from pathlib import Path


# Kernel pseudo-files are small, but bound every read and hierarchy walk.  A
# container may expose a synthetic or partially mounted cgroup tree; admission
# must not turn an unexpected file into an unbounded read or traversal.
_CGROUP_VALUE_MAX_BYTES = 128
_CGROUP_MEMBERSHIP_MAX_BYTES = 64 * 1024
_CGROUP_MAX_ANCESTORS = 64
_MEMINFO_MAX_BYTES = 256 * 1024
_UINT64_MAX = (1 << 64) - 1

# Cgroup v1 represents "unlimited" with a page-aligned value just below the
# signed 64-bit maximum (commonly 9223372036854771712).  No supported host can
# offer an exbibyte of usable RAM, so this threshold safely covers the kernel's
# sentinel variants without mistaking a practical finite limit for unlimited.
_CGROUP_V1_UNLIMITED_MIN = 1 << 60


def _read_bounded_ascii(path: Path, max_bytes: int) -> tuple[bool, str | None]:
    """Return ``(present, exact_text)`` for a small kernel pseudo-file.

    A missing path means that source/controller is unavailable.  Other I/O
    errors are authoritative probe failures and raise so the public resolver can
    fail closed instead of silently treating the process as unconstrained.
    ``None`` text means the present file was oversized or not ASCII.
    """
    path = Path(path)
    try:
        with path.open("rb") as stream:
            raw = stream.read(max_bytes + 1)
    except (FileNotFoundError, NotADirectoryError):
        return False, None
    except OSError as exc:
        raise ValueError(f"cannot read memory control file {path}: {exc}") from exc
    if len(raw) > max_bytes:
        return True, None
    try:
        return True, raw.decode("ascii")
    except UnicodeDecodeError:
        return True, None


def _parse_u64_counter(text: str | None) -> int | None:
    """Parse one non-negative decimal counter with at most one final newline."""
    match = re.fullmatch(r"(\d+)\n?", text or "")
    if match is None:
        return None
    value = int(match.group(1))
    return value if value <= _UINT64_MAX else None


def _host_memory_available(meminfo_path: Path) -> int | None:
    """Read host-wide ``MemAvailable`` bytes, or ``None`` when /proc is absent."""
    present, text = _read_bounded_ascii(meminfo_path, _MEMINFO_MAX_BYTES)
    if not present:
        return None
    if text is None:
        raise ValueError(f"malformed or oversized memory information: {meminfo_path}")
    match = re.search(
        r"^MemAvailable:[ \t]+(\d+)[ \t]+kB[ \t]*(?:\n|$)",
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"MemAvailable is unavailable or malformed: {meminfo_path}")
    kib = int(match.group(1))
    if kib > _UINT64_MAX // 1024:
        raise ValueError(f"MemAvailable overflows a byte counter: {meminfo_path}")
    return kib * 1024


def _cgroup_pair_remaining(
    directory: Path,
    limit_name: str,
    current_name: str,
    *,
    v1: bool = False,
) -> tuple[bool, int | None]:
    """Return ``(limit_present, finite_remaining_or_none)`` for one cgroup."""
    limit_path = Path(directory) / limit_name
    current_path = Path(directory) / current_name
    limit_present, limit_text = _read_bounded_ascii(limit_path, _CGROUP_VALUE_MAX_BYTES)
    if not limit_present:
        return False, None
    if not v1 and limit_text in ("max", "max\n"):
        return True, None

    limit = _parse_u64_counter(limit_text)
    if limit is None:
        raise ValueError(f"malformed cgroup memory limit: {limit_path}")
    if v1 and limit >= _CGROUP_V1_UNLIMITED_MIN:
        return True, None

    current_present, current_text = _read_bounded_ascii(
        current_path, _CGROUP_VALUE_MAX_BYTES
    )
    if not current_present:
        raise ValueError(f"cgroup memory usage is unavailable: {current_path}")
    current = _parse_u64_counter(current_text)
    if current is None:
        raise ValueError(f"malformed cgroup memory usage: {current_path}")
    return True, max(0, limit - current)


def _read_cgroup_memberships(proc_cgroup_path: Path) -> tuple[str | None, str | None]:
    """Return the v2 and v1-memory paths from bounded ``/proc/self/cgroup``.

    Hybrid hosts list leftover v1 hierarchies (``1:net_cls:/``) above the unified
    ``0::/...`` line; non-memory v1 controllers are simply skipped.
    """
    present, text = _read_bounded_ascii(proc_cgroup_path, _CGROUP_MEMBERSHIP_MAX_BYTES)
    if not present:
        return None, None
    if text is None:
        raise ValueError(
            f"malformed or oversized cgroup membership: {proc_cgroup_path}"
        )

    v2_path = None
    v1_path = None
    for line in text.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers, member_path = fields
        if not member_path.startswith("/"):
            continue
        # Real cgroup membership paths are absolute and contain no dot
        # components.  Reject lexical escapes in synthetic/malformed proc data.
        parts = Path(member_path).parts[1:]
        if any(part in ("", ".", "..") for part in parts):
            continue
        if hierarchy == "0" and not controllers:
            v2_path = member_path
        elif "memory" in controllers.split(","):
            v1_path = member_path
    return v2_path, v1_path


def _control_file_exists(path: Path) -> bool:
    """Existence probe that fails closed on errors other than a missing path."""
    path = Path(path)
    try:
        path.stat()
        return True
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as exc:
        raise ValueError(f"cannot inspect cgroup control file {path}: {exc}") from exc


def _cgroup_control_directory(
    root: Path, member_path: str | None, limit_name: str
) -> Path | None:
    """Map a proc membership path into a mounted, possibly namespaced tree."""
    root = Path(root)
    if member_path:
        parts = Path(member_path).parts[1:]
        if len(parts) >= _CGROUP_MAX_ANCESTORS:
            raise ValueError(
                f"cgroup membership nesting exceeds {_CGROUP_MAX_ANCESTORS - 1} levels"
            )
        candidate = root.joinpath(*parts)
        # A leaf can omit controller files when only an ancestor has the memory
        # controller enabled.  Find the nearest visible ancestor without a
        # filesystem scan or a walk outside the mounted hierarchy.
        for _ in range(_CGROUP_MAX_ANCESTORS):
            if _control_file_exists(candidate / limit_name):
                return candidate
            if candidate == root:
                break
            parent = candidate.parent
            if parent == candidate or (parent != root and root not in parent.parents):
                break
            candidate = parent

    # Container runtimes may mount the process's own subgroup as the hierarchy
    # root while /proc still reports a host-side membership path.
    if _control_file_exists(root / limit_name):
        return root
    return None


def _cgroup_hierarchy_remaining(
    root: Path,
    member_path: str | None,
    limit_name: str,
    current_name: str,
    *,
    v1: bool = False,
) -> tuple[bool, int | None]:
    """Return ``(hierarchy_seen, tightest finite ancestor headroom)``."""
    root = Path(root)
    current = _cgroup_control_directory(root, member_path, limit_name)
    if current is None:
        return False, None

    seen = False
    remaining = None
    for _ in range(_CGROUP_MAX_ANCESTORS):
        present, candidate = _cgroup_pair_remaining(
            current, limit_name, current_name, v1=v1
        )
        seen = seen or present
        if candidate is not None:
            remaining = candidate if remaining is None else min(remaining, candidate)
        if current == root:
            break
        parent = current.parent
        if parent == current or (parent != root and root not in parent.parents):
            break
        current = parent
    return seen, remaining


def _cgroup_memory_remaining(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_cgroup_path: Path = Path("/proc/self/cgroup"),
) -> int | None:
    """Return this process's tightest finite cgroup memory headroom.

    Cgroup v2 is authoritative whenever its memory controller is visible.
    ``max`` is a known-unlimited value.  Only when v2 is absent do we try the
    conventional, bounded v1 controller roots; no filesystem scan is used.
    Malformed, overflowing, or unreadable visible controls raise ``ValueError``.
    """
    root = Path(cgroup_root)
    v2_path, v1_path = _read_cgroup_memberships(proc_cgroup_path)
    v2_seen, remaining = _cgroup_hierarchy_remaining(
        root, v2_path, "memory.max", "memory.current"
    )
    if v2_seen:
        return remaining

    for v1_root in (root / "memory", root):
        v1_seen, remaining = _cgroup_hierarchy_remaining(
            v1_root,
            v1_path,
            "memory.limit_in_bytes",
            "memory.usage_in_bytes",
            v1=True,
        )
        if v1_seen:
            return remaining
    return None


def effective_memory_available(
    *,
    meminfo_path: Path = Path("/proc/meminfo"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_cgroup_path: Path = Path("/proc/self/cgroup"),
) -> int | None:
    """Return the effective available host-memory budget in bytes.

    The result is ``min(MemAvailable, tightest finite cgroup headroom)`` when
    both measurements exist, either known bound when only one exists, and
    ``None`` only when neither source is available -- no procfs, a non-Linux
    host, or a cgroup tree without a visible memory controller.  ``None`` means
    *unknown* and is never conflated with ``0``; callers keep their historical
    best-effort behaviour on it.  A known zero remains zero.

    A *present* but malformed, overflowing, or permission-denied control file
    is an authoritative signal that the budget cannot be trusted, so it fails
    closed to ``0`` rather than being mistaken for an unlimited controller.
    """
    try:
        host_available = _host_memory_available(Path(meminfo_path))
        cgroup_remaining = _cgroup_memory_remaining(
            Path(cgroup_root), Path(proc_cgroup_path)
        )
    except ValueError:
        return 0

    if host_available is None:
        return cgroup_remaining
    if cgroup_remaining is None:
        return host_available
    return min(host_available, cgroup_remaining)


__all__ = ["effective_memory_available"]
