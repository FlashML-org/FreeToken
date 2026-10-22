"""The hash-agnostic ``RowStore`` (``kernel/csrc/row_store``): byte-exact staged rows across extents and files,
dedup of repeated ids, a destination stride, and bounds checks before any I/O. The PLE hash wrapper on top of
it is covered by ``tests/models/qwen4_exp/test_ple_disk.py``."""

from __future__ import annotations

import pytest
import torch

_row_store = pytest.importorskip("freetoken.kernel._row_store")

ROW_BYTES = 264  # deliberately unaligned rows exercise reads across page boundaries


def _table(rows: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (rows, ROW_BYTES), dtype=torch.uint8, generator=gen)


def _stage(store, ids: torch.Tensor, dst_stride: int = 0) -> torch.Tensor:
    stride = dst_stride or store.row_bytes
    staging = torch.zeros(ids.numel() * stride, dtype=torch.uint8)
    store.stage_rows(ids.data_ptr(), ids.numel(), staging.data_ptr(), dst_stride)
    store.flush(0)
    return staging.view(ids.numel(), stride)


@pytest.mark.parametrize("use_io_uring", [True, False])
@pytest.mark.parametrize("cache_rows", [0, 64])
def test_rows_come_back_bitwise_across_extents_and_files(tmp_path, use_io_uring, cache_rows):
    # two files; file 0 carries extents 0 and 2 behind an unrelated header, file 1 carries extent 1
    per_extent = 1000
    table = _table(3 * per_extent, seed=1)
    header = b"safetensors-like header, not a row" + bytes(4096 - 34)
    f0 = tmp_path / "shard0.bin"
    f1 = tmp_path / "shard1.bin"
    f0.write_bytes(header + table[:per_extent].numpy().tobytes() + table[2 * per_extent :].numpy().tobytes())
    f1.write_bytes(table[per_extent : 2 * per_extent].numpy().tobytes())
    store = _row_store.RowStore(
        paths=[str(f0), str(f1)],
        extent_file=[0, 1, 0],
        extent_base=[len(header), 0, len(header) + per_extent * ROW_BYTES],
        rows_per_extent=per_extent,
        row_bytes=ROW_BYTES,
        row_stride=ROW_BYTES,
        use_io_uring=use_io_uring,
    )
    assert store.total_rows == 3 * per_extent and store.row_bytes == ROW_BYTES
    store.set_row_cache(cache_rows)

    ids = torch.tensor([0, 999, 1000, 1999, 2000, 2999, 1234, 1234, 42, 0], dtype=torch.int64)
    got = _stage(store, ids)
    assert torch.equal(got, table[ids])

    # a wider destination stride leaves the gap untouched
    got = _stage(store, ids, dst_stride=ROW_BYTES + 40)
    assert torch.equal(got[:, :ROW_BYTES], table[ids]) and int(got[:, ROW_BYTES:].sum()) == 0

    # a fill larger than the reader's in-flight window pipelines through
    many = torch.randint(0, 3 * per_extent, (5000,), dtype=torch.int64, generator=torch.Generator().manual_seed(2))
    assert torch.equal(_stage(store, many), table[many])

    # flushing nothing still signals the flag
    flag = torch.zeros(1, dtype=torch.int64)
    store.flush(flag.data_ptr())
    assert int(flag[0]) == 1


def test_geometry_and_bounds_are_checked_up_front(tmp_path):
    table = _table(64, seed=3)
    path = tmp_path / "t.bin"
    path.write_bytes(table.numpy().tobytes())
    make = lambda **kw: _row_store.RowStore(
        paths=[str(path)], extent_file=[0], extent_base=[0], rows_per_extent=64,
        row_bytes=ROW_BYTES, row_stride=ROW_BYTES, use_io_uring=False, **kw,
    )
    store = make()
    ids = torch.tensor([3, 64], dtype=torch.int64)
    staging = torch.zeros(2 * ROW_BYTES, dtype=torch.uint8)
    with pytest.raises(IndexError):
        store.stage_rows(ids.data_ptr(), 2, staging.data_ptr(), 0)
    with pytest.raises(RuntimeError, match="overlap"):
        store.stage_rows(ids[:1].data_ptr(), 1, staging.data_ptr(), ROW_BYTES - 1)
    with pytest.raises(RuntimeError, match="extent needs"):
        _row_store.RowStore(paths=[str(path)], extent_file=[0], extent_base=[0], rows_per_extent=65,
                            row_bytes=ROW_BYTES, row_stride=ROW_BYTES, use_io_uring=False)
    with pytest.raises(RuntimeError, match="exceeds a page"):
        _row_store.RowStore(paths=[str(path)], extent_file=[0], extent_base=[0], rows_per_extent=1,
                            row_bytes=4097, row_stride=4097, use_io_uring=False)


@pytest.mark.parametrize("row_bytes", [90, 160, 264, 4096])
@pytest.mark.parametrize("use_io_uring", [True, False])
def test_row_cache_hits_eviction_and_disable(tmp_path, row_bytes, use_io_uring):
    table = torch.randint(0, 256, (65, row_bytes), generator=torch.Generator().manual_seed(13), dtype=torch.uint8)
    path = tmp_path / "cache.bin"
    path.write_bytes(table.numpy().tobytes())
    store = _row_store.RowStore([str(path)], [0], [0], 65, row_bytes, row_bytes, use_io_uring)
    assert store.io_stats()["row_cache_bytes"] == 0
    store.set_row_cache(8)  # one set forces eviction after eight distinct rows
    ids = torch.tensor([0, 1, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64)
    stride = row_bytes + 11
    assert torch.equal(_stage(store, ids, stride)[:, :row_bytes], table[ids])
    before = store.io_stats()
    assert before["read_requests_completed"] == 8
    got = _stage(store, ids, stride)
    assert torch.equal(got[:, :row_bytes], table[ids])
    assert not got[:, row_bytes:].any()
    after = store.io_stats()
    assert after["read_requests_completed"] == before["read_requests_completed"]
    assert after["row_cache_hits"] - before["row_cache_hits"] == len(ids)
    assert after["row_cache_bytes"] == 8 * (row_bytes + 8) + 1

    for start in range(0, 65, 7):
        evict = torch.arange(start, min(start + 19, 65))
        assert torch.equal(_stage(store, evict), table[evict])
    assert torch.equal(_stage(store, ids), table[ids])
    for capacity in (16, 0, 8):
        store.set_row_cache(capacity)
        assert store.io_stats()["row_cache_hits"] == 0
        assert store.io_stats()["row_cache_capacity"] == capacity
        _stage(store, ids)
        before = store.io_stats()
        assert torch.equal(_stage(store, ids), table[ids])
        delta = store.io_stats()["read_requests_completed"] - before["read_requests_completed"]
        assert delta == (8 if capacity == 0 else 0)


def test_row_cache_resize_preserves_pending_reads_and_bounds_checks(tmp_path):
    table = _table(64, seed=17)
    path = tmp_path / "cache.bin"
    path.write_bytes(table.numpy().tobytes())
    store = _row_store.RowStore([str(path)], [0], [0], 64, ROW_BYTES, ROW_BYTES, False)
    store.set_row_cache(16)
    ids = torch.tensor([3], dtype=torch.int64)
    output = torch.full((1, ROW_BYTES), 173, dtype=torch.uint8)
    store.stage_rows(ids.data_ptr(), 1, output.data_ptr())
    with pytest.raises(RuntimeError, match="unflushed"):
        store.set_row_cache(0)
    store.flush()
    assert torch.equal(output, table[ids])
    for capacity in (-8, 1, 7, 9, (1 << 24) + 8):
        with pytest.raises(ValueError, match="capacity"):
            store.set_row_cache(capacity)
    assert store.io_stats()["row_cache_capacity"] == 16
    output = torch.full((2, ROW_BYTES), 173, dtype=torch.uint8)
    invalid = torch.tensor([3, 64], dtype=torch.int64)
    with pytest.raises(IndexError):
        store.stage_rows(invalid.data_ptr(), 2, output.data_ptr())
    assert (output == 173).all()
    assert torch.equal(_stage(store, ids), table[ids])


def test_row_caches_do_not_share_checkpoint_rows(tmp_path):
    stores, tables = [], []
    for seed in (19, 23):
        table = _table(64, seed=seed)
        path = tmp_path / f"table-{seed}.bin"
        path.write_bytes(table.numpy().tobytes())
        store = _row_store.RowStore([str(path)], [0], [0], 64, ROW_BYTES, ROW_BYTES, False)
        store.set_row_cache(8)
        stores.append(store)
        tables.append(table)
    ids = torch.tensor([1, 3, 3, 5], dtype=torch.int64)
    for _ in range(3):
        for store, table in zip(stores, tables):
            assert torch.equal(_stage(store, ids), table[ids])


def test_failed_read_does_not_cache_incomplete_payload(tmp_path):
    table = _table(64, seed=29)
    path = tmp_path / "table.bin"
    payload = table.numpy().tobytes()
    path.write_bytes(payload)
    store = _row_store.RowStore([str(path)], [0], [0], 64, ROW_BYTES, ROW_BYTES, False)
    store.set_row_cache(8)
    ids = torch.tensor([63], dtype=torch.int64)
    with path.open("r+b") as file:
        file.truncate(len(payload) - 1)
    with pytest.raises(RuntimeError, match="short read"):
        _stage(store, ids)
    assert store.io_stats()["row_cache_hits"] == 0
    with path.open("r+b") as file:
        file.write(payload)
    assert torch.equal(_stage(store, ids), table[ids])
    before = store.io_stats()
    assert torch.equal(_stage(store, ids), table[ids])
    assert store.io_stats()["read_requests_completed"] == before["read_requests_completed"]
