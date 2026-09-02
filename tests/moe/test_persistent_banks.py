"""Persistent tmpfs expert banks: a second process maps the same files and skips the fill.

CPU-only, single rank (no torch.distributed). The tmpfs check is satisfied by
pointing the pool at /dev/shm when present; elsewhere the check itself is the
subject of the first test.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from freetoken.moe import host_banks as hb
from freetoken.moe.host_banks import (
    PersistentPool,
    alloc_persistent_layer_banks,
    persist_present_bytes,
    persistent_fill_done,
)

SPECS = {"a": ((4, 8), torch.uint8), "b": ((4, 2), torch.uint8)}
L = 3


@pytest.fixture(autouse=True)
def _no_pin(monkeypatch):
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")


@pytest.fixture
def shm_root(tmp_path, monkeypatch):
    if not os.path.isdir("/dev/shm"):
        pytest.skip("no /dev/shm on this platform")
    root = os.path.join("/dev/shm", f"ft-persist-test-{os.getpid()}")
    monkeypatch.setenv(hb.PERSIST_DIR_ENV, root)
    yield root
    import shutil

    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def ckpt(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"x" * 64)
    (tmp_path / "config.json").write_text("{}")
    return str(tmp_path)


def test_persist_refuses_non_tmpfs(tmp_path, monkeypatch, ckpt):
    monkeypatch.setenv(hb.PERSIST_DIR_ENV, str(tmp_path / "pool"))
    if hb._fs_type(str(tmp_path)) in hb._TMPFS_TYPES:
        pytest.skip("tmp_path is itself on tmpfs here")
    with pytest.raises(RuntimeError, match="not tmpfs"):
        PersistentPool.for_checkpoint(ckpt, SPECS, L, tag="t")


def test_unset_env_means_no_pool(monkeypatch, ckpt):
    monkeypatch.delenv(hb.PERSIST_DIR_ENV, raising=False)
    assert PersistentPool.for_checkpoint(ckpt, SPECS, L) is None
    assert persist_present_bytes() == 0


def test_fill_then_reuse_across_pools(shm_root, ckpt):
    pool = PersistentPool.for_checkpoint(ckpt, SPECS, L, tag="t")
    banks, filled = alloc_persistent_layer_banks(SPECS, L, pool)
    assert filled is False
    assert not pool.is_filled()
    # "fill": distinct bytes per (bank, layer)
    for name in SPECS:
        for layer in range(L):
            banks[name][layer].tensor.fill_(10 * (1 + list(SPECS).index(name)) + layer)
    persistent_fill_done(pool, filled_by_me=True)
    assert pool.is_filled()
    assert persist_present_bytes() == sum(
        ((torch.Size(shape).numel() + hb._BLK - 1) // hb._BLK) * hb._BLK * L for shape, _ in SPECS.values()
    )

    # a "second process": a fresh pool object over the same root sees the fill
    pool2 = PersistentPool.for_checkpoint(ckpt, SPECS, L, tag="t")
    assert pool2.key == pool.key and pool2.dir == pool.dir
    banks2, filled2 = alloc_persistent_layer_banks(SPECS, L, pool2)
    assert filled2 is True
    for name in SPECS:
        for layer in range(L):
            want = 10 * (1 + list(SPECS).index(name)) + layer
            assert int(banks2[name][layer].tensor.min()) == want == int(banks2[name][layer].tensor.max())


def test_changed_checkpoint_invalidates(shm_root, ckpt):
    pool = PersistentPool.for_checkpoint(ckpt, SPECS, L, tag="t")
    alloc_persistent_layer_banks(SPECS, L, pool)
    persistent_fill_done(pool, filled_by_me=True)
    assert pool.is_filled()
    # RED control: touch a shard -> identity changes -> a new key, the old pool is evicted
    with open(os.path.join(ckpt, "model.safetensors"), "ab") as fh:
        fh.write(b"y")
    pool2 = PersistentPool.for_checkpoint(ckpt, SPECS, L, tag="t")
    assert pool2.key != pool.key
    _banks, filled = alloc_persistent_layer_banks(SPECS, L, pool2)
    assert filled is False
    assert not os.path.isdir(pool.dir), "stale pool must be evicted, the box holds one"


def test_size_change_invalidates_in_place(shm_root, ckpt):
    pool = PersistentPool.for_checkpoint(ckpt, SPECS, L, tag="t")
    alloc_persistent_layer_banks(SPECS, L, pool)
    persistent_fill_done(pool, filled_by_me=True)
    # same identity written by hand but a bank file of the wrong size -> not filled
    os.truncate(pool.bank_path("a", 0), 1)
    _banks, filled = alloc_persistent_layer_banks(SPECS, L, pool)
    assert filled is False
    assert os.stat(pool.bank_path("a", 0)).st_size == hb._BLK


def test_manifest_is_the_identity(shm_root, ckpt):
    pool = PersistentPool.for_checkpoint(ckpt, SPECS, L, tag="t")
    alloc_persistent_layer_banks(SPECS, L, pool)
    persistent_fill_done(pool, filled_by_me=True)
    with open(pool.manifest_path, encoding="utf-8") as fh:
        m = json.load(fh)
    assert m["num_layers"] == L and set(m["specs"]) == set(SPECS)
    assert any(s[0] == "model.safetensors" for s in m["shards"])
