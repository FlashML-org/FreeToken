"""File-backed shared HostBank (TP ranks must not duplicate the DSV4 pool)."""

from __future__ import annotations

import torch

from freetoken.distributed import reset_tp_info, set_tp_info
from freetoken.moe.host_banks import (
    HostBank,
    alloc_layer_banks,
    prepare_shared_banks,
    resolve_bank_share_dir,
)


def test_shared_bank_second_map_sees_writes(tmp_path):
    path = tmp_path / "w.bin"
    a = HostBank((8,), torch.float32, share_path=path, create=True)
    a.tensor[:] = 3.5
    a.flush()
    b = HostBank((8,), torch.float32, share_path=path, create=False)
    assert float(b.tensor[0]) == 3.5
    assert a._share_path == str(path)


def test_alloc_layer_banks_uses_share_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FREETOKEN_BANK_SHARE_DIR", str(tmp_path))
    monkeypatch.delenv("FREETOKEN_BANK_SHARE", raising=False)
    reset_tp_info()
    specs = {"gate_up_packed": ((4, 8), torch.uint8)}
    hb = alloc_layer_banks(specs, 2)
    assert (tmp_path / "gate_up_packed.L0.bin").is_file()
    assert (tmp_path / "gate_up_packed.L1.bin").is_file()
    hb["gate_up_packed"][0].tensor.fill_(7)
    hb["gate_up_packed"][0].flush()
    other = HostBank((4, 8), torch.uint8, share_path=tmp_path / "gate_up_packed.L0.bin", create=False)
    assert int(other.tensor[0, 0]) == 7


def test_prepare_shared_banks_tp2(tmp_path, monkeypatch):
    reset_tp_info()
    set_tp_info(0, 2)
    try:
        monkeypatch.delenv("FREETOKEN_BANK_SHARE_DIR", raising=False)
        monkeypatch.delenv("FREETOKEN_BANK_SHARE", raising=False)
        d = prepare_shared_banks(port=1919)
        assert d == "/tmp/freetoken-banks-1919"
        monkeypatch.setenv("FREETOKEN_BANK_SHARE_DIR", str(tmp_path))
        assert prepare_shared_banks(port=1919) == str(tmp_path)
        monkeypatch.setenv("FREETOKEN_BANK_SHARE", "0")
        assert prepare_shared_banks(port=1919) is None
        assert resolve_bank_share_dir() is None
    finally:
        reset_tp_info()


def test_anonymous_mmap_when_tp1(monkeypatch):
    reset_tp_info()
    monkeypatch.delenv("FREETOKEN_BANK_SHARE_DIR", raising=False)
    monkeypatch.delenv("FREETOKEN_BANK_SHARE", raising=False)
    assert resolve_bank_share_dir() is None
    hb = alloc_layer_banks({"w": ((2,), torch.uint8)}, 1)
    assert hb["w"][0]._share_path is None
