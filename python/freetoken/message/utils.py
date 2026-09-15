from __future__ import annotations

from typing import Any, Dict, Type

import numpy as np
import torch


_TYPE_KEY = "__type__"
# Message payloads carry free-form client dicts (tool JSON Schemas, chat_template_kwargs) that
# may legitimately use our tag key as a field name. Wrapping such a dict keeps the decoder from
# reading it as a serialized class -- without this, a request could crash the tokenizer worker.
_RAW_DICT_KEY = "__raw_dict__"

def _serialize_any(value: Any) -> Any:
    if isinstance(value, dict):
        encoded = {k: _serialize_any(v) for k, v in value.items()}
        if _TYPE_KEY in encoded or _RAW_DICT_KEY in encoded:
            return {_RAW_DICT_KEY: encoded}
        return encoded
    elif isinstance(value, (list, tuple)):
        return type(value)(_serialize_any(v) for v in value)
    elif isinstance(value, (int, float, str, type(None), bool, bytes)):
        return value
    else:
        return serialize_type(value)


def serialize_type(self) -> Dict:
    # find all member variables
    serialized = {}

    if isinstance(self, torch.Tensor):
        assert not self.is_cuda, "wire tensors must live on CPU"
        t = self.contiguous()
        serialized["__type__"] = "Tensor"
        serialized["dtype"] = str(t.dtype)
        # 1-D tensors omit the shape so the payload matches the legacy wire format.
        if t.dim() != 1:
            serialized["shape"] = list(t.shape)
        if t.dtype == torch.bfloat16:
            t = t.view(torch.uint16)  # numpy has no bf16; ship the raw bytes
        serialized["buffer"] = t.numpy().tobytes()
        return serialized

    # normal type
    serialized["__type__"] = self.__class__.__name__
    for k, v in self.__dict__.items():
        serialized[k] = _serialize_any(v)
    return serialized


def _deserialize_any(cls_map: Dict[str, Type], data: Any) -> Any:
    if isinstance(data, dict):
        if len(data) == 1 and _RAW_DICT_KEY in data:
            inner = data[_RAW_DICT_KEY]
            return {k: _deserialize_any(cls_map, v) for k, v in inner.items()}
        if _TYPE_KEY in data:
            return deserialize_type(cls_map, data)
        else:
            return {k: _deserialize_any(cls_map, v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(_deserialize_any(cls_map, d) for d in data)
    elif isinstance(data, (int, float, str, type(None), bool, bytes)):
        return data
    else:
        raise ValueError(f"Cannot deserialize type {type(data)}")


def deserialize_type(cls_map: Dict[str, Type], data: Dict) -> Any:
    type_name = data["__type__"]
    if type_name == "Tensor":
        buffer = data["buffer"]
        dtype_str = data["dtype"].replace("torch.", "")
        assert isinstance(buffer, bytes)
        is_bf16 = dtype_str == "bfloat16"
        np_tensor = np.frombuffer(buffer, dtype=getattr(np, "uint16" if is_bf16 else dtype_str))
        tensor = torch.from_numpy(np_tensor.copy())
        if is_bf16:
            tensor = tensor.view(torch.bfloat16)
        shape = data.get("shape")
        return tensor if shape is None else tensor.view(shape)

    cls = cls_map.get(type_name)
    if cls is None:
        raise ValueError(f"Unknown serialized message type {type_name!r}")
    kwargs = {}
    for k, v in data.items():
        if k == _TYPE_KEY:
            continue
        kwargs[k] = _deserialize_any(cls_map, v)
    return cls(**kwargs)
