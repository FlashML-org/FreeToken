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


@pytest.fixture(params=[False, True], ids=["v2", "v1"])
def file_cache_cgroup(
    memory_files: FakeMemoryFiles, request: pytest.FixtureRequest
) -> tuple[Path, str]:
    v1 = request.param
    memory_files.membership("7:memory:/" if v1 else "0::/")
    directory = memory_files.root / "memory" if v1 else memory_files.root
    write = memory_files.v1 if v1 else memory_files.v2
    write(directory, 24 << 30, 20 << 30)
    memory_files.meminfo.write_text("MemAvailable: 33554432 kB\n", encoding="ascii")
    return directory, "total_" if v1 else ""


@pytest.mark.parametrize(
    "current,inactive,active,remaining",
    [
        (20, 8, 6, 18),
        (20, 14, 0, 18),
        (20, 0, 14, 18),
        (20, 10, 11, 24),
        (26, 1, 0, 0),
        (26, 4, 0, 2),
        (24, 0, 0, 0),
    ],
)
def test_file_cache_counts_as_room(
    memory_files: FakeMemoryFiles,
    file_cache_cgroup,
    current: int,
    inactive: int,
    active: int,
    remaining: int,
):
    directory, prefix = file_cache_cgroup
    current_name = "memory.usage_in_bytes" if prefix else "memory.current"
    (directory / current_name).write_text(f"{current << 30}\n", encoding="ascii")
    (directory / "memory.stat").write_text(
        f"{prefix}inactive_file {inactive << 30}\n"
        f"{prefix}active_file {active << 30}\n",
        encoding="ascii",
    )

    assert memory_files.remaining() == remaining << 30
    assert memory_files.effective() == remaining << 30


@pytest.mark.parametrize("key", ["inactive_file", "active_file"])
def test_file_cache_missing_one_key_uses_the_other(
    memory_files: FakeMemoryFiles, file_cache_cgroup, key: str
):
    directory, prefix = file_cache_cgroup
    (directory / "memory.stat").write_text(
        f"{prefix}{key} {14 << 30}\n", encoding="ascii"
    )

    assert memory_files.remaining() == 18 << 30


def test_file_cache_uses_only_matching_file_lists(
    memory_files: FakeMemoryFiles, file_cache_cgroup
):
    directory, prefix = file_cache_cgroup
    other_prefix = "" if prefix else "total_"
    (directory / "memory.stat").write_text(
        f"file {20 << 30}\nshmem {10 << 30}\ncache {20 << 30}\n"
        f"{other_prefix}inactive_file {8 << 30}\n"
        f"{other_prefix}active_file {6 << 30}\n",
        encoding="ascii",
    )

    assert memory_files.remaining() == 4 << 30


@pytest.mark.parametrize(
    "dirty,writeback,remaining",
    [(2, 1, 15), (2, 0, 16), (0, 1, 17), (16, 0, 4), (8, 8, 4)],
)
def test_dirty_and_writeback_pages_are_not_room(
    memory_files: FakeMemoryFiles,
    file_cache_cgroup,
    dirty: int,
    writeback: int,
    remaining: int,
):
    directory, prefix = file_cache_cgroup
    dirty_key, writeback_key = (
        ("total_dirty", "total_writeback")
        if prefix
        else ("file_dirty", "file_writeback")
    )
    (directory / "memory.stat").write_text(
        f"{prefix}inactive_file {8 << 30}\n{prefix}active_file {6 << 30}\n"
        f"{dirty_key} {dirty << 30}\n{writeback_key} {writeback << 30}\n",
        encoding="ascii",
    )

    assert memory_files.remaining() == remaining << 30
    assert memory_files.effective() == remaining << 30


def test_other_hierarchy_dirty_counters_are_ignored(
    memory_files: FakeMemoryFiles, file_cache_cgroup
):
    directory, prefix = file_cache_cgroup
    other_dirty, other_writeback = (
        ("file_dirty", "file_writeback")
        if prefix
        else ("total_dirty", "total_writeback")
    )
    (directory / "memory.stat").write_text(
        f"{prefix}inactive_file {8 << 30}\n{prefix}active_file {6 << 30}\n"
        f"{other_dirty} {2 << 30}\n{other_writeback} {1 << 30}\n",
        encoding="ascii",
    )

    assert memory_files.remaining() == 18 << 30


@pytest.mark.parametrize(
    "stat_text",
    [
        None,
        "",
        "anon 1\n",
        "{prefix}active_file x\n",
        "{prefix}active_file -1\n",
        "{prefix}active_file +1\n",
        "{prefix}active_file 1.5\n",
        "{prefix}active_file 1 2\n",
        "{prefix}active_file 1 \n",
        "{prefix}active_file 1\r\n",
        "{prefix}active_file {overflow}\n",
        "{prefix}inactive_file 1\n{prefix}inactive_file 2\n",
        "{prefix}active_file 1\n{prefix}active_file 2\n",
        "{prefix}inactive_file 8589934592\n{prefix}active_file x\n",
        "{prefix}inactive_file 8589934592\nbad line here\n",
        "{prefix}inactive_file 8589934592\nanon {overflow}\n",
        "{prefix}inactive_file 8589934592\nanon 1\nanon 2\n",
        "{prefix}inactive_file 8589934592\n{prefix}active_file " + "9" * 5000 + "\n",
        "{prefix}inactive_file 8589934592\n\n",
    ],
    ids=[
        "absent",
        "empty",
        "missing-keys",
        "non-numeric",
        "negative",
        "signed",
        "fractional",
        "extra-field",
        "trailing-space",
        "carriage-return",
        "overflow",
        "duplicate-inactive",
        "duplicate-active",
        "partial-cache",
        "malformed-line",
        "unrelated-overflow",
        "unrelated-duplicate",
        "oversized-counter",
        "blank-line",
    ],
)
def test_unusable_file_cache_subtracts_nothing(
    memory_files: FakeMemoryFiles, file_cache_cgroup, stat_text: str | None
):
    directory, prefix = file_cache_cgroup
    if stat_text is not None:
        (directory / "memory.stat").write_text(
            stat_text.format(prefix=prefix, overflow=_UINT64_MAX + 1), encoding="ascii"
        )

    assert memory_files.remaining() == 4 << 30
    assert memory_files.effective() == 4 << 30


# The oversized case is well-formed line by line (70,000 bytes of distinct
# counters), so only the read bound can reject it.
@pytest.mark.parametrize(
    "suffix",
    [b"\xff", b"".join(b"pad%04d 0\n" % n for n in range(7000))],
    ids=["non-ascii", "oversized"],
)
def test_non_ascii_or_oversized_file_cache_subtracts_nothing(
    memory_files: FakeMemoryFiles, file_cache_cgroup, suffix: bytes
):
    directory, prefix = file_cache_cgroup
    (directory / "memory.stat").write_bytes(
        f"{prefix}inactive_file {14 << 30}\n".encode("ascii") + suffix
    )

    assert memory_files.remaining() == 4 << 30


def test_unreadable_file_cache_subtracts_nothing(
    memory_files: FakeMemoryFiles, file_cache_cgroup, monkeypatch: pytest.MonkeyPatch
):
    directory, _ = file_cache_cgroup
    denied = directory / "memory.stat"
    original_open = Path.open

    def deny_stat(path: Path, *args, **kwargs):
        if path == denied:
            raise PermissionError("fixture denied memory.stat")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_stat)
    assert memory_files.remaining() == 4 << 30
    assert memory_files.effective() == 4 << 30


@pytest.mark.parametrize("leaf_current,remaining", [(20, 6), (4, 18)])
def test_file_cache_is_adjusted_at_each_ancestor(
    memory_files: FakeMemoryFiles, file_cache_cgroup, leaf_current: int, remaining: int
):
    directory, prefix = file_cache_cgroup
    memory_files.membership("7:memory:/job" if prefix else "0::/job")
    write = memory_files.v1 if prefix else memory_files.v2
    write(directory / "job", 24 << 30, leaf_current << 30)
    for level, inactive, active in [(directory, 8, 6), (directory / "job", 1, 1)]:
        (level / "memory.stat").write_text(
            f"{prefix}inactive_file {inactive << 30}\n"
            f"{prefix}active_file {active << 30}\n",
            encoding="ascii",
        )

    assert memory_files.remaining() == remaining << 30


def test_file_cache_room_is_still_clamped_by_host(
    memory_files: FakeMemoryFiles, file_cache_cgroup
):
    directory, prefix = file_cache_cgroup
    memory_files.meminfo.write_text("MemAvailable: 8388608 kB\n", encoding="ascii")
    (directory / "memory.stat").write_text(
        f"{prefix}inactive_file {14 << 30}\n", encoding="ascii"
    )

    assert memory_files.remaining() == 18 << 30
    assert memory_files.effective() == 8 << 30


@pytest.mark.parametrize("control", ["limit", "usage"])
def test_file_cache_does_not_mask_malformed_controls(
    memory_files: FakeMemoryFiles, file_cache_cgroup, control: str
):
    directory, prefix = file_cache_cgroup
    limit_name, current_name = (
        ("memory.limit_in_bytes", "memory.usage_in_bytes")
        if prefix
        else ("memory.max", "memory.current")
    )
    name = current_name if control == "usage" else limit_name
    (directory / name).write_text("bad\n", encoding="ascii")
    (directory / "memory.stat").write_text(
        f"{prefix}inactive_file {14 << 30}\n", encoding="ascii"
    )

    with pytest.raises(ValueError, match=f"malformed cgroup memory {control}"):
        memory_files.remaining()
    assert memory_files.effective() == 0


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
