from __future__ import annotations

from typing import Any, Dict, Type
import math

import torch


_TYPE_KEY = "__type__"
# Message payloads carry free-form client dicts (tool JSON Schemas, chat_template_kwargs) that
# may legitimately use our tag key as a field name. Wrapping such a dict keeps the decoder from
# reading it as a serialized class -- without this, a request could crash the tokenizer worker.
_RAW_DICT_KEY = "__raw_dict__"
_TENSOR_DTYPES = {str(dtype): dtype for dtype in (
    torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    torch.float16, torch.bfloat16, torch.float32, torch.float64,
)}

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
        if self.device.type != "cpu" or str(self.dtype) not in _TENSOR_DTYPES:
            raise ValueError("message tensors must use a supported CPU dtype")
        serialized["__type__"] = "Tensor"
        serialized["buffer"] = self.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        serialized["dtype"] = str(self.dtype)
        # Keep the original 1-D wire format readable by older workers.
        if self.dim() != 1:
            serialized["shape"] = list(self.shape)
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
        dtype = _TENSOR_DTYPES.get(data["dtype"])
        if dtype is None or not isinstance(buffer, bytes):
            raise ValueError("invalid serialized tensor dtype or data")
        itemsize = torch.empty((), dtype=dtype).element_size()
        shape = data.get("shape", [len(buffer) // itemsize])
        if (not isinstance(shape, (list, tuple)) or len(shape) > 8
                or any(type(n) is not int or n < 0 for n in shape)
                or math.prod(shape) * itemsize != len(buffer)):
            raise ValueError("serialized tensor shape does not match its data")
        if not buffer:
            return torch.empty(shape, dtype=dtype)
        return torch.frombuffer(bytearray(buffer), dtype=dtype).reshape(shape)

    cls = cls_map.get(type_name)
    if cls is None:
        raise ValueError(f"Unknown serialized message type {type_name!r}")
    kwargs = {}
    for k, v in data.items():
        if k == _TYPE_KEY:
            continue
        kwargs[k] = _deserialize_any(cls_map, v)
    return cls(**kwargs)
