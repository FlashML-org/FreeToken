"""FTW replay: dropping entries by name before their bytes are read, the mmap path, and the vision-tower presence check."""

import math

import torch

from freetoken.checkpoint.ftw import ALIGN, FTWReader, FTWWriter, ftw_tensor_names, iter_ftw_weights
from freetoken.models.weight import ftw_lacks_vision, load_weight


def _write_ftw(out_dir, names):
    writer = FTWWriter(str(out_dir))
    tensors = {name: torch.full((4, 8), float(i), dtype=torch.bfloat16) for i, name in enumerate(names)}
    for name, tensor in tensors.items():
        writer.add_tensor(name, tensor)
    writer.finalize({})
    return tensors


def test_keep_drops_entries_before_their_bytes_are_read(tmp_path, monkeypatch):
    tensors = _write_ftw(tmp_path, ["model.a.weight", "visual.b.weight", "model.c.weight"])
    read = []
    original = FTWReader.read_into

    def spy(self, dest, entry, **kwargs):
        read.append(entry["name"])
        return original(self, dest, entry, **kwargs)

    monkeypatch.setattr(FTWReader, "read_into", spy)
    got = dict(iter_ftw_weights(str(tmp_path), keep=lambda name: not name.startswith("visual.")))
    assert list(got) == ["model.a.weight", "model.c.weight"]
    assert read == ["model.a.weight", "model.c.weight"]
    for name, tensor in got.items():
        assert torch.equal(tensor, tensors[name])


def test_the_mmap_path_reads_what_was_written(tmp_path, monkeypatch):
    # the reader maps the shards where there is no O_DIRECT (Windows) or the filesystem refuses it
    tensors = _write_ftw(tmp_path, ["model.a.weight", "model.b.weight"])
    reader = FTWReader(str(tmp_path))
    monkeypatch.setattr(reader, "_direct", 0)
    for entry in reader.entries("weight"):
        dest = bytearray(math.ceil(entry["nbytes"] / ALIGN) * ALIGN)
        reader.read_into(memoryview(dest), entry)
        assert bytes(dest[:entry["nbytes"]]) == tensors[entry["name"]].view(torch.uint8).numpy().tobytes()
    reader.close()


def test_no_keep_replays_every_entry(tmp_path):
    tensors = _write_ftw(tmp_path, ["model.a.weight", "visual.b.weight"])
    got = dict(iter_ftw_weights(str(tmp_path)))
    assert list(got) == list(tensors)
    assert torch.equal(got["visual.b.weight"], tensors["visual.b.weight"])


def test_load_weight_text_only_skips_the_tower(tmp_path):
    names = ["model.a.weight", "visual.b.weight", "vision_tower.c.weight"]
    _write_ftw(tmp_path, names)
    cpu = torch.device("cpu")
    assert [n for n, _ in load_weight(str(tmp_path), cpu, include_vision=False)] == ["model.a.weight"]
    assert [n for n, _ in load_weight(str(tmp_path), cpu)] == names


def test_ftw_lacks_vision(tmp_path):
    _write_ftw(tmp_path / "text", ["model.a.weight"])
    _write_ftw(tmp_path / "vl", ["model.a.weight", "visual.b.weight"])
    assert ftw_lacks_vision(str(tmp_path / "text"))
    assert not ftw_lacks_vision(str(tmp_path / "vl"))
    assert not ftw_lacks_vision(str(tmp_path))
    assert ftw_tensor_names(str(tmp_path / "vl"), "weight") == ["model.a.weight", "visual.b.weight"]
