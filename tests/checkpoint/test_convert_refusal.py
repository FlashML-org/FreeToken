"""convert_checkpoint refuses an output directory that already holds FTW output, before it writes anything.

No GPU: the refusal sits ahead of the TP/CUDA setup, so a CPU box reaches it with an empty source dir."""

import os

import pytest
import torch

from freetoken.checkpoint.convert import _refuse_occupied_output, convert_checkpoint
from freetoken.checkpoint.ftw import INDEX_NAME, FTWWriter


def _listing(d):
    return sorted((name, os.stat(os.path.join(d, name)).st_size, os.stat(os.path.join(d, name)).st_mtime_ns)
                  for name in os.listdir(d))


def _refused(src, out):
    before = _listing(out)
    with pytest.raises(SystemExit, match="already holds an FTW checkpoint") as ei:
        convert_checkpoint(str(src), str(out))
    assert _listing(out) == before  # nothing written, nothing touched
    return str(ei.value)


def test_finished_ftw_in_the_output_dir_is_refused_by_name(tmp_path):
    out = tmp_path / "out"
    writer = FTWWriter(str(out))
    writer.add_tensor("model.a.weight", torch.zeros(4, dtype=torch.bfloat16))
    writer.finalize({})
    src = tmp_path / "src"
    src.mkdir()
    msg = _refused(src, out)
    assert str(out) in msg and INDEX_NAME in msg


def test_shards_from_an_interrupted_conversion_are_refused_by_name(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "freetoken-00001.ftw").write_bytes(b"\0" * 64)
    (out / "freetoken-00000.ftw").write_bytes(b"\0" * 64)
    src = tmp_path / "src"
    src.mkdir()
    msg = _refused(src, out)
    assert str(out) in msg and "freetoken-00000.ftw" in msg and INDEX_NAME not in msg


def test_empty_absent_or_unrelated_output_dirs_pass(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _refuse_occupied_output(str(empty)) is None
    absent = tmp_path / "absent"
    assert _refuse_occupied_output(str(absent)) is None
    assert not absent.exists()  # the check creates nothing
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text("{}")  # copied metadata alone is not FTW output
    (other / "notes.ftw.bak").write_bytes(b"")
    assert _refuse_occupied_output(str(other)) is None
