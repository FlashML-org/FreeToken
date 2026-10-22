"""llama.cpp vision towers (``mmproj-*.gguf``): find the file next to a text GGUF, read it, and dequantize its tensors.

llama.cpp keeps the tower in its own file (``general.architecture = clip``). FreeToken loads it from ``--mmproj`` (passed to every process as ``FREETOKEN_MMPROJ``) or finds a ``*mmproj*.gguf`` that fits the text model in the model's own folder, as llama-server does for a ``--models-dir`` entry. Every tensor loads as a float tensor: the towers run in bf16 whatever the file stores.
"""

from __future__ import annotations

import functools
import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import torch

from freetoken.layers.quantization.ggml_types import is_quantized
from freetoken.utils.logger import init_logger

from .reader import FTW_METADATA_GGUF, GgufTensorInfo, gguf_tensor_index, is_gguf_path, load_gguf_metadata
from .tensors import float_tensor

logger = init_logger(__name__)

MMPROJ_ENV = "FREETOKEN_MMPROJ"

# BF16 and F32 convert to bf16 exactly, F16 loses only its subnormals, the rest is lossy in the file
_TYPE_RANK = {"BF16": 0, "F32": 1, "F16": 2}


@dataclass(frozen=True)
class Mmproj:
    path: str
    source: str  # "--mmproj" or "auto-discovered"
    projector: str
    weight_type: str  # ggml type of the block weights (largest matrix without blocks), e.g. "BF16", "Q8_0"
    skipped: tuple[str, ...] = ()


def _header(path: str):
    from freetoken.third_party import gguf

    # uncached: candidates that are not picked must not stay mapped in this process
    return gguf.GGUFReader(path)


def _field(reader, name: str):
    field = reader.fields.get(name)
    return None if field is None else field.contents()


def _model_key(name: str | None) -> str | None:
    """A model name compared across converters: ``unsloth/Qwen3.6-35B-A3B`` and ``Qwen3.6-35B-A3B`` are one model."""
    if name is None:
        return None
    return re.sub(r"[^0-9a-z]", "", str(name).rsplit("/", 1)[-1].lower())


def _describe(path: str) -> tuple[str, str, int, str | None] | None:
    """(projector type, block weight type, output width, model name) of a vision mmproj, or None for any other GGUF."""
    from freetoken.third_party.gguf import GGMLQuantizationType

    reader = _header(path)
    if _field(reader, "general.architecture") != "clip" or not _field(reader, "clip.has_vision_encoder"):
        return None
    projector = _field(reader, "clip.vision.projector_type") or _field(reader, "clip.projector_type")
    if projector is None:
        return None
    types = {t.name: t.tensor_type for t in reader.tensors}
    weight = types.get("v.blk.0.attn_qkv.weight", types.get("v.blk.0.attn_q.weight"))
    if weight is None and reader.tensors:
        # no blocks: the largest matrix; converters keep patch embeddings F32 in every file type, so F32 ranks last
        weight = max(reader.tensors, key=lambda t: (t.tensor_type != GGMLQuantizationType.F32, len(t.shape) >= 2, int(t.n_elements))).tensor_type
    width = int(_field(reader, "clip.vision.projection_dim") or 0)
    name = _model_key(_field(reader, "general.basename") or _field(reader, "general.name"))
    return str(projector), "unknown" if weight is None else GGMLQuantizationType(weight).name, width, name


@functools.cache
def _find(model_path: str, override: str) -> Mmproj | None:
    if override:
        described = _describe(override) if is_gguf_path(override) else None
        if described is None:
            raise ValueError(f"--mmproj {override!r} is not a llama.cpp vision mmproj GGUF")
        return Mmproj(override, "--mmproj", *described[:2])
    md = load_gguf_metadata(model_path)
    width = md.get(f"{md.get('general.architecture')}.embedding_length")
    folder = os.path.dirname(os.path.abspath(model_path))
    found = []
    for path in sorted(glob.glob(os.path.join(glob.escape(folder), "*mmproj*.gguf"))):
        try:
            described = _describe(path)
        except Exception as e:  # noqa: BLE001 -- a broken download next to the model must not stop it from serving text
            logger.warning("vision: skipping unreadable %s: %s", path, e)
            continue
        if described is not None and (width is None or described[2] == int(width)):
            found.append((_TYPE_RANK.get(described[1], len(_TYPE_RANK)), path, described))
    # same-width towers of several models in one folder: take the one named like the text model, else do not guess
    names = {f[2][3] for f in found}
    if len(names) > 1:
        text_name = _model_key(md.get("general.basename") or md.get("general.name"))
        found = [f for f in found if text_name is not None and f[2][3] == text_name]
        if not found:
            logger.warning("vision: %s holds mmproj files of several models (%s); pick one with --mmproj", folder, ", ".join(sorted(map(str, names))))
            return None
    from .config import mm_fields

    found.sort(key=lambda f: f[:2])
    for i, (_, path, described) in enumerate(found):
        mmproj = Mmproj(path, "auto-discovered", *described[:2], skipped=tuple(f[1] for f in found[i + 1 :]))
        try:
            # the family's own checks (tower type, image tokens); a file that fails them leaves the model text only
            mm_fields(model_path, mmproj)
        except (KeyError, ValueError) as e:
            logger.warning("vision: skipping %s: %s", path, e)
            continue
        return mmproj
    return None


def find_mmproj(model_path: str) -> Mmproj | None:
    """The vision mmproj that goes with the text GGUF ``model_path``, or None when it has none."""
    # an FTW converted from a GGUF keeps only the text metadata, and no tower
    if not is_gguf_path(model_path) or os.path.basename(model_path) == FTW_METADATA_GGUF:
        return None
    return _find(model_path, os.environ.get(MMPROJ_ENV, ""))


@functools.cache
def _warn_inline_vision(model_path: str) -> None:
    logger.warning("vision: %s carries vision tensors itself (Ollama layout), which are not read; use a llama.cpp mmproj file", model_path)


def warn_inline_vision(model_path: str, tensor_names: set[str]) -> None:
    """Old Ollama GGUFs carry the tower inside the text file under another layout; FreeToken reads only llama.cpp mmproj files."""
    if any(n.startswith(("v.", "mm.")) for n in tensor_names):
        _warn_inline_vision(model_path)


def mmproj_tensor(info: GgufTensorInfo) -> torch.Tensor:
    """A tensor in torch order: float types as stored, quantized types dequantized to fp32."""
    if not is_quantized(info.ggml_type):
        return float_tensor(info)
    from freetoken.third_party.gguf import GGMLQuantizationType, quants

    flat = quants.dequantize(np.array(info.data()), GGMLQuantizationType(info.ggml_type))
    return torch.from_numpy(np.ascontiguousarray(flat)).reshape(*reversed(info.ne))


@functools.cache
def log_mmproj(model_path: str, enabled: bool) -> None:
    """One line per process on the vision tower the text GGUF ``model_path`` loads, or why it loads none."""
    mmproj = find_mmproj(model_path)
    if mmproj is None:
        logger.info("vision: no mmproj found for %s, vision disabled", model_path)
        return
    if not enabled:
        logger.info("vision: mmproj %s ignored (--mm-disable)", mmproj.path)
        return
    tensors = gguf_tensor_index(mmproj.path)
    gib = sum(i.nbytes for i in tensors.values()) / (1 << 30)
    logger.info(
        "vision: mmproj=%s (%s) projector=%s type=%s -> bf16, %d tensors, %.2f GiB",
        mmproj.path, mmproj.source, mmproj.projector, mmproj.weight_type, len(tensors), gib,
    )
    if mmproj.skipped:
        logger.info("vision: other mmproj files not used: %s (pick one with --mmproj)", ", ".join(mmproj.skipped))
    if mmproj.weight_type not in _TYPE_RANK:
        logger.warning("vision: mmproj %s is %s; it runs in bf16 but keeps the quantization error of the file", mmproj.path, mmproj.weight_type)


__all__ = [
    "MMPROJ_ENV",
    "Mmproj",
    "find_mmproj",
    "log_mmproj",
    "mmproj_tensor",
    "warn_inline_vision",
]
