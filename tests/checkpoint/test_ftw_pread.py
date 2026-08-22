import json
import os

import pytest

from freetoken.checkpoint.ftw import (
    ALIGN,
    FORMAT_TAG,
    FORMAT_VERSION,
    INDEX_NAME,
    _SHARD_FMT,
    _pread_into,
    FTWReader,
)


def _make_ftw_checkpoint(tmp_path, shard_bytes_on_disk: int, shard_bytes_in_index: int,
                         tensor_nbytes: int):
    """Build a minimal FTW checkpoint. The shard file may be truncated on disk
    while the index still claims the full size — simulating an interrupted write."""
    shard_name = _SHARD_FMT.format(0)
    (tmp_path / shard_name).write_bytes(b"A" * shard_bytes_on_disk)
    index = {
        "format": FORMAT_TAG,
        "version": FORMAT_VERSION,
        "align": ALIGN,
        "shard_limit": 8 << 30,
        "total_bytes": shard_bytes_in_index,
        "tensors": [
            {"name": "w", "kind": "weight", "dtype": "uint8",
             "shape": [tensor_nbytes], "global_off": 0, "nbytes": tensor_nbytes}
        ],
        "shards": [
            {"file": shard_name, "global_off": 0, "nbytes": shard_bytes_in_index}
        ],
    }
    (tmp_path / INDEX_NAME).write_text(json.dumps(index))
    return str(tmp_path)


def test_pread_into_truncated_raises(tmp_path):
    """O_DIRECT path: truncated shard must raise OSError, not silently load garbage."""
    p = tmp_path / "shard.ftw"
    p.write_bytes(b"A" * 4096)
    fd = os.open(str(p), os.O_RDONLY)
    try:
        buf = bytearray(b"X" * 8192)
        with pytest.raises(OSError, match="unexpected EOF"):
            _pread_into(fd, memoryview(buf), 0)
    finally:
        os.close(fd)


def test_mmap_read_into_truncated_raises(tmp_path):
    """mmap path: truncated shard must raise OSError with shard name."""
    path = _make_ftw_checkpoint(
        tmp_path,
        shard_bytes_on_disk=4096,
        shard_bytes_in_index=8192,
        tensor_nbytes=8192,
    )
    reader = FTWReader(path)
    reader._direct = 0
    reader._probed = True
    dest = bytearray(b"X" * 8192)
    with pytest.raises(OSError, match="unexpected EOF"):
        reader.read_into(memoryview(dest), reader.tensors["w"], workers=1)


def test_mmap_read_into_complete(tmp_path):
    """mmap path: valid shard must pass the length check and read correctly."""
    path = _make_ftw_checkpoint(
        tmp_path,
        shard_bytes_on_disk=8192,
        shard_bytes_in_index=8192,
        tensor_nbytes=8192,
    )
    reader = FTWReader(path)
    reader._direct = 0
    reader._probed = True
    dest = bytearray(b"X" * 8192)
    reader.read_into(memoryview(dest), reader.tensors["w"], workers=1)
    assert bytes(dest) == b"A" * 8192
