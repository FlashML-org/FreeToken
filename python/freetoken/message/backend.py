from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
from freetoken.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    pass


@dataclass
class MMItem:
    """One multimodal input of a request (an image, video or audio clip).

    Family extras from the processor live in model_specific_data and read back as attributes (item.grid_thw).
    offsets are half-open [start, end) spans in input_ids; an item may occupy several.
    Exactly one of feature / precomputed_embeddings is set.
    """

    modality: str
    hash: int
    pad_value: int
    offsets: List[List[int]]
    feature: torch.Tensor | None = None  # raw processor output (CPU), encoded by the engine
    precomputed_embeddings: torch.Tensor | None = None  # final encoder output, skips encoding
    model_specific_data: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        data = self.__dict__.get("model_specific_data")
        if data is not None and name in data:
            return data[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def num_tokens(self) -> int:
        return sum(end - start for start, end in self.offsets)

    def is_image(self) -> bool:
        return self.modality == "image"

    def validate(self) -> None:
        if not self.offsets or any(end <= start for start, end in self.offsets):
            raise ValueError(f"MMItem offsets must be non-empty half-open spans: {self.offsets}")
        if (self.feature is None) == (self.precomputed_embeddings is None):
            raise ValueError("MMItem needs exactly one of feature / precomputed_embeddings")

    def encoder(self) -> Dict:
        return serialize_type(self)


@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams
    # per-image processor outputs, in prompt order
    mm_items: List[MMItem] | None = None
    # precomputed [3, len(input_ids)] mrope positions and decode delta; None for text-only requests and 1-D rope models
    mrope_positions: torch.Tensor | None = None
    mrope_delta: int = 0


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int


@dataclass
class CacheRebuildBackendMsg(BaseBackendMsg):
    # tokenizer worker -> scheduler: request a runtime KV/MoE/GDN cache resize.
    request_id: str
    moe_cache_size: int | None = None
    num_pages: int | None = None
    num_mamba_slots: int | None = None
    num_swa_pages: int | None = None
    mode: str = "if_idle"  # only "if_idle" is supported; "drain" is deferred (rejected)
