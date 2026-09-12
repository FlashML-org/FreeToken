"""Runtime knobs of the multimodal path. The architecture side (vision_config, mrope) lives in ModelConfig."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# encoder tower kinds a family can register; --mm-disable <kind> leaves that tower unbuilt and refuses its inputs
ENCODER_KINDS = ("vision", "audio")
# checkpoint config sections that describe encoder towers; the parser never sees the section of a tower this process does not build
ENCODER_SECTIONS = ("vision_config", "audio_config")


@dataclass(frozen=True)
class MultimodalConfig:
    # encoder kinds (ENCODER_KINDS) whose tower this process does not build; --text-model-only names them all
    disabled_encoders: frozenset[str] = frozenset()
    # Where encoded items wait between prefill chunks. "cpu": pinned host memory, "cuda": the device.
    embed_cache_device: Literal["cpu", "cuda"] = "cpu"
    # Encoder tower block weights. "host": pinned host banks streamed two blocks at a time behind the compute, "gpu": resident.
    encoder_weights: Literal["gpu", "host"] = "host"
    # Pixel budget the processor resizes each image into. None: the checkpoint's own default (Qwen VL: 16,777,216 = 4096x4096).
    max_pixels: int | None = None

    @property
    def text_model_only(self) -> bool:
        return set(ENCODER_KINDS) <= self.disabled_encoders


__all__ = ["ENCODER_KINDS", "ENCODER_SECTIONS", "MultimodalConfig"]
