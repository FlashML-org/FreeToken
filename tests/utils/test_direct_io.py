"""DirectFile: positional reads that bypass the page cache, in aligned pieces, short only at the end of the file."""

from __future__ import annotations

import mmap
import random
from concurrent.futures import ThreadPoolExecutor

import pytest

from freetoken.utils.direct_io import DIRECT_ALIGN, DIRECT_READS, DirectFile

pytestmark = pytest.mark.skipif(not DIRECT_READS, reason="the platform has no direct reads")

SIZE = 9 * DIRECT_ALIGN + 1234


@pytest.fixture(scope="module")
def data_file(tmp_path_factory) -> tuple[str, bytes]:
    data = random.Random(11).randbytes(SIZE)
    path = tmp_path_factory.mktemp("direct") / "data.bin"
    path.write_bytes(data)
    return str(path), data


def _aligned(nbytes: int) -> memoryview:
    return memoryview(mmap.mmap(-1, nbytes))


def _open(path: str) -> DirectFile:
    try:
        return DirectFile(path)
    except OSError as error:
        pytest.skip(f"the test's file system refuses direct reads: {error}")


@pytest.mark.parametrize("offset, nbytes", [(0, DIRECT_ALIGN), (DIRECT_ALIGN, 4 * DIRECT_ALIGN), (8 * DIRECT_ALIGN, DIRECT_ALIGN)])
def test_an_aligned_read_returns_the_file_bytes(data_file, offset, nbytes):
    path, data = data_file
    view = _aligned(nbytes)
    with _open(path) as file:
        assert file.pread_into(view, offset) == nbytes
    assert bytes(view) == data[offset:offset + nbytes]


def test_a_read_across_the_end_returns_the_bytes_up_to_it(data_file):
    path, data = data_file
    offset = 9 * DIRECT_ALIGN
    view = _aligned(DIRECT_ALIGN)
    with _open(path) as file:
        assert file.pread_into(view, offset) == SIZE - offset
    assert bytes(view[:SIZE - offset]) == data[offset:]


def test_a_read_past_the_end_returns_nothing(data_file):
    path, _data = data_file
    with _open(path) as file:
        assert file.pread_into(_aligned(DIRECT_ALIGN), 12 * DIRECT_ALIGN) == 0


def test_threads_read_one_file_concurrently(data_file):
    path, data = data_file
    view = _aligned(10 * DIRECT_ALIGN)

    def read(block: int) -> int:
        return file.pread_into(view[block * DIRECT_ALIGN:(block + 1) * DIRECT_ALIGN], block * DIRECT_ALIGN)

    with _open(path) as file, ThreadPoolExecutor(8) as pool:
        got = list(pool.map(read, range(10)))
    assert got == [DIRECT_ALIGN] * 9 + [1234]
    assert bytes(view[:SIZE]) == data
