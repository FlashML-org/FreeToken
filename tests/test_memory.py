from pathlib import Path

import pytest

from freetoken.memory import (
    _CGROUP_MAX_ANCESTORS,
    _CGROUP_V1_UNLIMITED_MIN,
    _UINT64_MAX,
    _cgroup_memory_remaining,
    effective_memory_available,
)


class FakeMemoryFiles:
    """Pure fake /proc and cgroup trees; never touches the runner's hierarchy."""

    def __init__(self, base: Path):
        self.root = base / "cgroup"
        self.root.mkdir()
        self.proc_cgroup = base / "proc-self-cgroup"
        self.meminfo = base / "meminfo"
        self.meminfo.write_text(
            "MemTotal: 32768 kB\nMemAvailable: 16384 kB\n", encoding="ascii"
        )

    def membership(self, *lines: str) -> None:
        self.proc_cgroup.write_text("\n".join(lines) + "\n", encoding="ascii")

    @staticmethod
    def pair(
        directory: Path,
        limit_name: str,
        current_name: str,
        limit: object,
        current: object,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / limit_name).write_text(f"{limit}\n", encoding="ascii")
        (directory / current_name).write_text(f"{current}\n", encoding="ascii")

    def v2(self, directory: Path, limit: object, current: object) -> None:
        self.pair(directory, "memory.max", "memory.current", limit, current)

    def v1(self, directory: Path, limit: object, current: object) -> None:
        self.pair(
            directory,
            "memory.limit_in_bytes",
            "memory.usage_in_bytes",
            limit,
            current,
        )

    def remaining(self) -> int | None:
        return _cgroup_memory_remaining(self.root, self.proc_cgroup)

    def effective(self) -> int | None:
        return effective_memory_available(
            meminfo_path=self.meminfo,
            cgroup_root=self.root,
            proc_cgroup_path=self.proc_cgroup,
        )


@pytest.fixture
def memory_files(tmp_path: Path) -> FakeMemoryFiles:
    return FakeMemoryFiles(tmp_path)


def test_v2_finite_budget_clamps_host_memavailable(memory_files: FakeMemoryFiles):
    memory_files.membership("0::/tenant/job")
    memory_files.v2(memory_files.root / "tenant" / "job", 12_000_000, 3_000_000)

    assert memory_files.remaining() == 9_000_000
    assert memory_files.effective() == 9_000_000


def test_host_memavailable_wins_when_lower(memory_files: FakeMemoryFiles):
    memory_files.membership("0::/tenant/job")
    memory_files.v2(memory_files.root / "tenant" / "job", 100_000_000, 1)

    assert memory_files.effective() == 16_384 * 1024


def test_nested_v2_uses_tightest_finite_ancestor(memory_files: FakeMemoryFiles):
    memory_files.membership("0::/tenant/job")
    memory_files.v2(memory_files.root, "max", "not-read-for-unlimited")
    memory_files.v2(memory_files.root / "tenant", 20_000_000, 14_000_000)
    memory_files.v2(memory_files.root / "tenant" / "job", 12_000_000, 3_000_000)

    assert memory_files.remaining() == 6_000_000


def test_unlimited_v2_leaf_still_honors_finite_parent(memory_files: FakeMemoryFiles):
    memory_files.membership("0::/tenant/job")
    memory_files.v2(memory_files.root / "tenant", 9_000_000, 4_000_000)
    memory_files.v2(
        memory_files.root / "tenant" / "job", "max", "not-read-for-unlimited"
    )

    assert memory_files.remaining() == 5_000_000


def test_namespaced_v2_mount_uses_root_controls(memory_files: FakeMemoryFiles):
    memory_files.membership("0::/host/tenant/job")
    memory_files.v2(memory_files.root, 10_000_000, 4_000_000)

    assert memory_files.remaining() == 6_000_000


def test_v2_max_is_known_unlimited(memory_files: FakeMemoryFiles):
    memory_files.membership("0::/")
    memory_files.v2(memory_files.root, "max", "malformed-but-irrelevant")

    assert memory_files.remaining() is None
    assert memory_files.effective() == 16_384 * 1024


def test_counter_accepts_no_trailing_newline(memory_files: FakeMemoryFiles):
    memory_files.membership("0::/")
    (memory_files.root / "memory.max").write_text("10000", encoding="ascii")
    (memory_files.root / "memory.current").write_text("4000", encoding="ascii")

    assert memory_files.remaining() == 6_000


@pytest.mark.parametrize("current", [10_000, 10_001])
def test_v2_current_at_or_above_limit_is_zero(
    memory_files: FakeMemoryFiles, current: int
):
    memory_files.membership("0::/")
    memory_files.v2(memory_files.root, 10_000, current)

    assert memory_files.remaining() == 0
    assert memory_files.effective() == 0


@pytest.mark.parametrize(
    "value",
    ["", "-1", "+1", "1.5", "garbage", " 1", "1 ", "\t1", "1\n", "1\r", "max "],
)
def test_malformed_v2_limit_fails_closed(memory_files: FakeMemoryFiles, value: str):
    memory_files.membership("0::/")
    memory_files.v2(memory_files.root, value, 1)

    with pytest.raises(ValueError, match="malformed cgroup memory limit"):
        memory_files.remaining()
    assert memory_files.effective() == 0


@pytest.mark.parametrize("value", ["-1", "+1", "1.5", "unknown", "1 "])
def test_malformed_v2_usage_fails_closed(memory_files: FakeMemoryFiles, value: str):
    memory_files.membership("0::/")
    memory_files.v2(memory_files.root, 10_000, value)

    with pytest.raises(ValueError, match="malformed cgroup memory usage"):
        memory_files.remaining()
    assert memory_files.effective() == 0


def test_overflow_and_oversized_controls_fail_closed(memory_files: FakeMemoryFiles):
    memory_files.membership("0::/")
    memory_files.v2(memory_files.root, _UINT64_MAX + 1, 1)
    assert memory_files.effective() == 0

    memory_files.v2(memory_files.root, "9" * 256, 1)
    with pytest.raises(ValueError, match="malformed cgroup memory limit"):
        memory_files.remaining()
    assert memory_files.effective() == 0


def test_finite_limit_without_usage_fails_closed(memory_files: FakeMemoryFiles):
    memory_files.membership("0::/")
    (memory_files.root / "memory.max").write_text("10000\n", encoding="ascii")

    with pytest.raises(ValueError, match="usage is unavailable"):
        memory_files.remaining()
    assert memory_files.effective() == 0


def test_permission_error_on_visible_counter_fails_closed(
    memory_files: FakeMemoryFiles, monkeypatch: pytest.MonkeyPatch
):
    memory_files.membership("0::/")
    memory_files.v2(memory_files.root, 10_000, 1_000)
    denied = memory_files.root / "memory.current"
    original_open = Path.open

    def deny_current(path: Path, *args, **kwargs):
        if path == denied:
            raise PermissionError("fixture denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_current)
    with pytest.raises(ValueError, match="cannot read memory control file"):
        memory_files.remaining()
    assert memory_files.effective() == 0


def test_excessive_membership_nesting_is_bounded(memory_files: FakeMemoryFiles):
    deep = "/" + "/".join(f"level-{i}" for i in range(_CGROUP_MAX_ANCESTORS))
    memory_files.membership(f"0::{deep}")
    memory_files.v2(memory_files.root, "max", 1)

    with pytest.raises(ValueError, match="membership nesting exceeds"):
        memory_files.remaining()
    assert memory_files.effective() == 0


def test_oversized_membership_read_fails_closed(memory_files: FakeMemoryFiles):
    memory_files.proc_cgroup.write_text("0::/" + "x" * (65 * 1024), encoding="ascii")

    with pytest.raises(ValueError, match="oversized cgroup membership"):
        memory_files.remaining()
    assert memory_files.effective() == 0


def test_malformed_leaf_is_not_masked_by_valid_parent(memory_files: FakeMemoryFiles):
    memory_files.membership("0::/tenant/job")
    memory_files.v2(memory_files.root / "tenant", 8_000_000, 3_000_000)
    memory_files.v2(memory_files.root / "tenant" / "job", "bad", 1)

    assert memory_files.effective() == 0


def test_v2_is_authoritative_over_stale_v1(memory_files: FakeMemoryFiles):
    memory_files.membership("0::/", "7:memory:/legacy")
    memory_files.v2(memory_files.root, "max", 1)
    memory_files.v1(memory_files.root / "memory" / "legacy", 4_000_000, 3_000_000)

    assert memory_files.remaining() is None
    assert memory_files.effective() == 16_384 * 1024


def test_missing_v2_controller_uses_bounded_v1(memory_files: FakeMemoryFiles):
    memory_files.membership("0::/unified", "7:memory:/legacy")
    memory_files.v1(memory_files.root / "memory" / "legacy", 8_000_000, 3_000_000)

    assert memory_files.remaining() == 5_000_000
    assert memory_files.effective() == 5_000_000


def test_v1_honors_tighter_parent(memory_files: FakeMemoryFiles):
    memory_files.membership("7:memory:/tenant/job")
    v1_root = memory_files.root / "memory"
    memory_files.v1(v1_root / "tenant", 10_000_000, 7_000_000)
    memory_files.v1(v1_root / "tenant" / "job", 8_000_000, 2_000_000)

    assert memory_files.remaining() == 3_000_000


def test_v1_unlimited_sentinel_is_known_unlimited(memory_files: FakeMemoryFiles):
    memory_files.membership("7:memory:/")
    memory_files.v1(
        memory_files.root / "memory",
        _CGROUP_V1_UNLIMITED_MIN,
        "malformed-but-irrelevant",
    )

    assert memory_files.remaining() is None
    assert memory_files.effective() == 16_384 * 1024


def test_v1_current_above_limit_is_zero(memory_files: FakeMemoryFiles):
    memory_files.membership("7:memory:/")
    memory_files.v1(memory_files.root / "memory", 10_000, 20_000)

    assert memory_files.remaining() == 0
    assert memory_files.effective() == 0


def test_malformed_membership_can_use_namespaced_root(memory_files: FakeMemoryFiles):
    memory_files.v2(memory_files.root, 9_000_000, 2_000_000)
    for proc_text in ("", "not:a:valid:line\n"):
        memory_files.proc_cgroup.write_text(proc_text, encoding="ascii")
        assert memory_files.remaining() == 7_000_000


def test_absent_cgroup_files_leave_host_measurement(memory_files: FakeMemoryFiles):
    memory_files.membership("0::/missing")

    assert memory_files.remaining() is None
    assert memory_files.effective() == 16_384 * 1024


def test_finite_cgroup_is_bound_when_meminfo_is_absent(memory_files: FakeMemoryFiles):
    memory_files.membership("0::/")
    memory_files.v2(memory_files.root, 7_000_000, 2_000_000)

    assert (
        effective_memory_available(
            meminfo_path=memory_files.meminfo.with_name("missing"),
            cgroup_root=memory_files.root,
            proc_cgroup_path=memory_files.proc_cgroup,
        )
        == 5_000_000
    )


def test_malformed_or_overflowing_meminfo_fails_closed(memory_files: FakeMemoryFiles):
    memory_files.membership("0::/missing")
    for available in ("not-a-number", str(_UINT64_MAX // 1024 + 1)):
        memory_files.meminfo.write_text(
            f"MemTotal: 32768 kB\nMemAvailable: {available} kB\n", encoding="ascii"
        )
        assert memory_files.effective() == 0


def test_permission_error_on_meminfo_fails_closed(
    memory_files: FakeMemoryFiles, monkeypatch: pytest.MonkeyPatch
):
    memory_files.membership("0::/missing")
    denied = memory_files.meminfo
    original_open = Path.open

    def deny_meminfo(path: Path, *args, **kwargs):
        if path == denied:
            raise PermissionError("fixture denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_meminfo)
    assert memory_files.effective() == 0


def test_completely_absent_probe_is_unknown(tmp_path: Path):
    assert (
        effective_memory_available(
            meminfo_path=tmp_path / "missing-meminfo",
            cgroup_root=tmp_path / "missing-cgroup",
            proc_cgroup_path=tmp_path / "missing-proc-cgroup",
        )
        is None
    )


def test_hybrid_v1_and_v2_membership_is_not_a_failure(memory_files: FakeMemoryFiles):
    """systemd hosts keep leftover v1 hierarchies (``1:net_cls:/``) above the unified
    ``0::/...`` line; those lines must be skipped, not treated as malformed or as a
    memory controller, and the real unified root carries no ``memory.max`` of its own."""
    memory_files.membership("2:cpu,cpuacct:/", "1:net_cls:/", "0::/tenant/job")
    memory_files.v2(memory_files.root / "tenant", 20_000_000, 14_000_000)
    memory_files.v2(
        memory_files.root / "tenant" / "job", "max", "not-read-for-unlimited"
    )

    assert memory_files.remaining() == 6_000_000
    assert memory_files.effective() == 6_000_000
