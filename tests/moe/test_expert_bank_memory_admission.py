import inspect
from types import SimpleNamespace

import pytest
import torch

from freetoken.layers.quantization import QuantKind
from freetoken.memory import effective_memory_available


def _record_read_mode(monkeypatch: pytest.MonkeyPatch, expert_banks) -> list:
    """Stand in for the raw-checkpoint bank builder and record the read mode it is handed.

    The mode is looked up by parameter name on the real builder's signature, so a
    reordered signature cannot make this read a different argument.
    """
    signature = inspect.signature(expert_banks._method_expert_banks)
    chosen = []

    def build(*args, **kwargs):
        chosen.append(signature.bind(*args, **kwargs).arguments["parallel"])
        return expert_banks.ExpertBanks("bf16", {})

    monkeypatch.setattr(expert_banks, "_method_expert_banks", build)
    return chosen


@pytest.mark.parametrize(
    ("cgroup_current", "parallel"),
    [
        pytest.param(701, False, id="299-bytes-left"),
        pytest.param(699, True, id="301-bytes-left"),
    ],
)
def test_loader_auto_admission_consumes_effective_memory(
    tmp_path, monkeypatch: pytest.MonkeyPatch, cgroup_current: int, parallel: bool
):
    """Regression: production auto-load must use the shared effective budget."""
    import freetoken.checkpoint.ftw as ftw
    import freetoken.models.weight as model_weight
    import freetoken.moe.expert_banks as expert_banks

    # The bank estimate is 200 bytes plus a 100-byte parallel-reader transient.
    # Host MemAvailable is large; the fake cgroup has 299 bytes left (one short,
    # so the serial build) or 301 (enough, so the parallel reader is kept).
    (tmp_path / "a.safetensors").write_bytes(b"a" * 100)
    (tmp_path / "b.safetensors").write_bytes(b"b" * 100)
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable: 1024 kB\n", encoding="ascii")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::/\n", encoding="ascii")
    cgroup_root = tmp_path / "cgroup"
    cgroup_root.mkdir()
    (cgroup_root / "memory.max").write_text("1000\n", encoding="ascii")
    (cgroup_root / "memory.current").write_text(f"{cgroup_current}\n", encoding="ascii")

    def effective_fixture():
        return effective_memory_available(
            meminfo_path=meminfo,
            cgroup_root=cgroup_root,
            proc_cgroup_path=proc_cgroup,
        )

    monkeypatch.setattr(expert_banks, "_PARALLEL_READER_SUPPORTED", True)
    monkeypatch.setattr(expert_banks, "effective_memory_available", effective_fixture)
    monkeypatch.setattr(model_weight, "experts_scattered", lambda _path: True)
    monkeypatch.setattr(ftw, "is_ftw_checkpoint", lambda _path: False)
    chosen = _record_read_mode(monkeypatch, expert_banks)
    config = SimpleNamespace(num_moe_layers=1, expert_quant="none")

    expert_banks.load_expert_banks(
        str(tmp_path),
        config,
        method=SimpleNamespace(kind=QuantKind.NONE),
        device=torch.device("cpu"),
    )

    assert chosen == [parallel]


def test_explicit_parallel_loader_override_bypasses_auto_admission(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    import freetoken.checkpoint.ftw as ftw
    import freetoken.moe.expert_banks as expert_banks

    monkeypatch.setattr(expert_banks, "_PARALLEL_READER_SUPPORTED", True)
    monkeypatch.setattr(ftw, "is_ftw_checkpoint", lambda _path: False)
    monkeypatch.setattr(
        expert_banks,
        "effective_memory_available",
        lambda: (_ for _ in ()).throw(AssertionError("auto admission should not run")),
    )
    chosen = _record_read_mode(monkeypatch, expert_banks)
    config = SimpleNamespace(num_moe_layers=1, expert_quant="none")

    expert_banks.load_expert_banks(
        str(tmp_path),
        config,
        method=SimpleNamespace(kind=QuantKind.NONE),
        device=torch.device("cpu"),
        parallel=True,
    )

    assert chosen == [True]


def test_benchmark_ram_estimate_uses_effective_headroom(monkeypatch):
    import freetoken.moe.benchbw as benchbw

    # A finite cgroup budget (here 3 GiB) must reach the benchmark unchanged
    # instead of the host-wide figure the old sysconf probe reported.
    monkeypatch.setattr(benchbw, "effective_memory_available", lambda: 3 << 30)

    assert benchbw._available_ram_bytes() == 3 << 30


def test_benchmark_legacy_fallback_is_only_for_unknown_probe(monkeypatch):
    import freetoken.moe.benchbw as benchbw

    monkeypatch.setattr(benchbw, "effective_memory_available", lambda: None)

    assert benchbw._available_ram_bytes() == 8 << 30
