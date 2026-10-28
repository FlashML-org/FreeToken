from __future__ import annotations

import json
import struct

from freetoken.moe.disk_tier import _header_offsets


def test_safetensors_offsets_are_absolute(tmp_path):
    header = {
        "weight": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
        "__metadata__": {"format": "pt"},
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    path = tmp_path / "part.safetensors"
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"data")

    offsets = _header_offsets(str(path))

    assert offsets["weight"] == (8 + len(header_bytes), 12 + len(header_bytes))
