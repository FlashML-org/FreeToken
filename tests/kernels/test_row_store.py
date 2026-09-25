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
def test_rows_come_back_bitwise_across_extents_and_files(tmp_path, use_io_uring):
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
