"""File-backed shared HostBank (TP ranks must not duplicate the DSV4 pool)."""

from __future__ import annotations

import pytest
import torch

from freetoken.distributed import reset_tp_info, set_tp_info
from freetoken.moe import host_banks as hb
from freetoken.moe.host_banks import (
    DEFAULT_BANK_SHARE_NEED_BYTES,
    HostBank,
    alloc_layer_banks,
    default_bank_share_dir,
    normalize_bank_share_dir,
    prepare_shared_banks,
    require_share_dir_capacity,
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
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        d = prepare_shared_banks(port=1919, need_bytes=0)
        assert d == str(tmp_path / "cache" / "freetoken" / "banks-1919")
        assert not d.startswith("/tmp/freetoken-banks")
        monkeypatch.setenv("FREETOKEN_BANK_SHARE_DIR", str(tmp_path / "disk"))
        assert prepare_shared_banks(port=1919, need_bytes=0) == str(tmp_path / "disk")
        monkeypatch.setenv("FREETOKEN_BANK_SHARE", "0")
        assert prepare_shared_banks(port=1919, need_bytes=0) is None
        assert resolve_bank_share_dir() is None
    finally:
        reset_tp_info()


def test_default_bank_share_dir_is_not_tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    d = default_bank_share_dir(1919)
    assert d == str(tmp_path / "xdg" / "freetoken" / "banks-1919")
    assert DEFAULT_BANK_SHARE_NEED_BYTES == 43 * 256 * 13_369_344


def test_normalize_rejects_dotdot():
    with pytest.raises(RuntimeError, match=r"\.\."):
        normalize_bank_share_dir("/var/tmp/../etc/freetoken")


def test_require_share_dir_rejects_tmpfs(tmp_path, monkeypatch):
    monkeypatch.setattr(hb, "_fs_type", lambda path: "tmpfs")
    monkeypatch.setattr(hb, "_fs_free_bytes", lambda path: 200 * (1 << 30))
    with pytest.raises(RuntimeError, match="tmpfs"):
        require_share_dir_capacity(str(tmp_path), need_bytes=DEFAULT_BANK_SHARE_NEED_BYTES)


def test_require_share_dir_rejects_small_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(hb, "_fs_type", lambda path: "ext4")
    monkeypatch.setattr(hb, "_fs_free_bytes", lambda path: 10 * (1 << 30))
    with pytest.raises(RuntimeError, match="too small|has 10"):
        require_share_dir_capacity(str(tmp_path), need_bytes=DEFAULT_BANK_SHARE_NEED_BYTES)


def test_anonymous_mmap_when_tp1(monkeypatch):
    reset_tp_info()
    monkeypatch.delenv("FREETOKEN_BANK_SHARE_DIR", raising=False)
    monkeypatch.delenv("FREETOKEN_BANK_SHARE", raising=False)
    assert resolve_bank_share_dir() is None
    hb = alloc_layer_banks({"w": ((2,), torch.uint8)}, 1)
    assert hb["w"][0]._share_path is None
