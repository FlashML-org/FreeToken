from types import SimpleNamespace

import numpy as np
import torch

from freetoken.models.gguf import reader
from freetoken.models.gguf.dequant import GGML_Q4_0


def test_gguf_tensor_iterator_exposes_packed_rows(monkeypatch):
    raw = np.arange(36, dtype=np.uint8)
    monkeypatch.setattr(reader, "_reader", lambda _: SimpleNamespace(data=raw))
    info = reader.GgufTensorInfo("tensor", "model.gguf", 0, raw.size, GGML_Q4_0, (32, 2))
    monkeypatch.setattr(reader, "gguf_tensor_index", lambda _: {info.name: info})

    tensor = next(reader.iter_gguf_tensors("model.gguf"))

    assert tensor is info
    assert tensor.shape == (2, 32)
    assert tensor.rows == 2
    assert tensor.row_bytes == 18
    torch.testing.assert_close(tensor.packed(), torch.from_numpy(raw.reshape(2, 18)))
