from __future__ import annotations

import json
from pathlib import Path

import torch

from freetoken.checkpoint.convert import _copy_metadata
from freetoken.checkpoint.ftw import FTWReader, FTWWriter, iter_ftw_weights


def test_copy_metadata_keeps_only_qwen4_host_mapped_shards(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    (source / "model-plefp8-00000.safetensors").write_bytes(b"ple")
    (source / "model-00001.safetensors").write_bytes(b"dense")
    ple_name = (
        "model.language_model.layers.0.ple.ple_embedding."
        "ngram_embedding.shard_0.weight"
    )
    index = {
        "metadata": {"total_size": 8},
        "weight_map": {
            ple_name: "model-plefp8-00000.safetensors",
            "model.embed_tokens.weight": "model-00001.safetensors",
        },
    }
    (source / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )

    copied = _copy_metadata(str(source), str(output))

    assert (output / "config.json").is_file()
    assert (output / "model-plefp8-00000.safetensors").read_bytes() == b"ple"
    assert not (output / "model-00001.safetensors").exists()
    slim = json.loads((output / "model.safetensors.index.json").read_text())
    assert slim["weight_map"] == {ple_name: "model-plefp8-00000.safetensors"}
    assert slim["metadata"]["freetoken_host_mapped_only"] is True
    assert sorted(copied) == [
        "config.json",
        "model-plefp8-00000.safetensors",
        "model.safetensors.index.json",
    ]


def test_ftw_buffered_reader_works_on_the_current_platform(tmp_path: Path) -> None:
    output = tmp_path / "ftw"
    writer = FTWWriter(str(output), shard_limit=4096)
    expected = torch.arange(64, dtype=torch.int32).reshape(8, 8)
    writer.add_tensor("weight", expected, kind="weight")
    writer.finalize({})

    loaded = list(iter_ftw_weights(str(output), workers=2))

    assert len(loaded) == 1
    assert loaded[0][0] == "weight"
    assert torch.equal(loaded[0][1], expected)


def test_ftw_reader_can_drop_and_reopen_source_maps(tmp_path: Path) -> None:
    output = tmp_path / "ftw"
    writer = FTWWriter(str(output), shard_limit=4096)
    expected = torch.arange(64, dtype=torch.int32).reshape(8, 8)
    writer.add_tensor("weight", expected, kind="weight")
    writer.finalize({})

    reader = FTWReader(str(output))
    reader._direct = 0
    reader._probed = True
    entry = reader.entries("weight")[0]
    destination = bytearray(4096)
    reader.read_into(memoryview(destination), entry, workers=2)
    assert reader._maps

    reader.drop_maps()
    assert not reader._maps
    destination[:] = b"\0" * len(destination)
    reader.read_into(memoryview(destination), entry, workers=2)
    actual = torch.frombuffer(destination, dtype=torch.int32, count=64).reshape(8, 8)
    assert torch.equal(actual, expected)
    reader.close()
