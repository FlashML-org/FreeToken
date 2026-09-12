"""Per-family multimodal processors and the architecture registry; MMProcessor.apply is the image half of tokenization."""

from __future__ import annotations

import hashlib
import importlib
import io
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

from freetoken.message import MMItem
from freetoken.mm import check_mm_pad_shift


@dataclass
class PromptReplacement:
    """The token sequence that replaces one placeholder; is_embed marks the positions that take image embeddings (None = all of them)."""

    full: list[int]
    is_embed: list[bool] | None = None

    @classmethod
    def select_token_id(cls, full: list[int], embed_token_id: int) -> PromptReplacement:
        return cls(full, [t == embed_token_id for t in full])

    def embed_spans(self) -> list[list[int]]:
        """Half-open [lo, hi) runs of embedding positions, relative to the start of full."""
        mask = self.is_embed if self.is_embed is not None else [True] * len(self.full)
        spans: list[list[int]] = []
        lo: int | None = None
        for i, flag in enumerate([*mask, False]):
            if flag and lo is None:
                lo = i
            elif not flag and lo is not None:
                spans.append([lo, i])
                lo = None
        return spans


@dataclass
class MMResult:
    input_ids: torch.Tensor
    mm_items: list[MMItem]
    mrope_positions: torch.Tensor | None
    mrope_delta: int


def _find_all(ids: list[int], target: list[int]) -> list[int]:
    """Start index of every non-overlapping occurrence of target."""
    n = len(target)
    found: list[int] = []
    i = 0
    while i <= len(ids) - n:
        if ids[i : i + n] == target:
            found.append(i)
            i += n
        else:
            i += 1
    return found


class MMProcessor:
    """How a checkpoint's images become items, tokens and positions; apply is the family-agnostic driver, subclasses fill the hooks.

    Built from the checkpoint config alone and import-light: tokenizer workers call apply without the model layer, the engine calls dummy_items.
    Image preprocessing must stay on the CPU with one fixed implementation: radix keys and the embedding cache are content hashes of pixel_values, and resize results differ across backends/devices.
    """

    # the token sequence the chat template renders once per image
    placeholder: list[int]

    def process(self, images: list[Any], max_pixels: int | None) -> list[MMItem]:
        """One MMItem per image with feature and hash; offsets are assigned by apply."""
        raise NotImplementedError

    def prompt_replacement(self, item: MMItem) -> PromptReplacement:
        """The token sequence that stands in for this item's placeholder in the prompt."""
        raise NotImplementedError

    def positions(self, length: int, items: list[MMItem]) -> tuple[torch.Tensor, int] | None:
        """[3, length] t/h/w rope positions and the decode delta; None for 1-D rope families."""
        return None

    def dummy_items(self, dtype: torch.dtype, device: torch.device) -> list[MMItem]:
        """The smallest encodable item of each modality the family serves, for warming the encoders."""
        raise NotImplementedError

    def apply(
        self, input_ids: torch.Tensor, images: list[bytes], max_pixels: int | None
    ) -> MMResult:
        """Swap each image placeholder for its replacement sequence and precompute rope positions."""
        from PIL import Image

        ids = input_ids.tolist()
        target = self.placeholder
        slots = _find_all(ids, target)
        if len(slots) != len(images):
            raise ValueError(
                f"prompt renders {len(slots)} image placeholders but the request "
                f"carries {len(images)} images"
            )

        pils = [Image.open(io.BytesIO(raw)).convert("RGB") for raw in images]
        items = self.process(pils, max_pixels)
        out: list[int] = []
        cursor = 0
        for slot, item in zip(slots, items):
            repl = self.prompt_replacement(item)
            out.extend(ids[cursor:slot])
            base = len(out)
            full = list(repl.full)
            spans = repl.embed_spans()
            # embedding slots carry the content pad id so radix keys and the model's scatter mask see the image
            for lo, hi in spans:
                full[lo:hi] = [item.pad_value] * (hi - lo)
            out.extend(full)
            item.offsets = [[base + lo, base + hi] for lo, hi in spans]
            item.validate()
            cursor = slot + len(target)
        out.extend(ids[cursor:])
        new_ids = torch.tensor(out, dtype=input_ids.dtype)

        positions, delta = self.positions(len(out), items) or (None, 0)
        return MMResult(new_ids, items, positions, delta)


def image_positions(
    length: int, blocks: list[tuple[int, int, int, int]]
) -> tuple[torch.Tensor, int]:
    """t/h/w positions: text advances all three rows, an image freezes t and spreads h/w over its grid, then the running position jumps by max(h, w).

    blocks are (offset, n_tokens, grid_h, grid_w) in prompt order; delta shifts decode positions (sequence index + delta)."""
    pos = torch.empty((3, length), dtype=torch.int32)
    st = 0
    cursor = 0
    for offset, n_tokens, h_m, w_m in blocks:
        run = offset - cursor
        if run:
            text = torch.arange(st, st + run, dtype=torch.int32)
            pos[:, cursor:offset] = text
            st += run
            cursor = offset
        idx = torch.arange(n_tokens, dtype=torch.int32)
        pos[0, cursor : cursor + n_tokens] = st
        pos[1, cursor : cursor + n_tokens] = st + idx // w_m
        pos[2, cursor : cursor + n_tokens] = st + idx % w_m
        st += max(h_m, w_m)
        cursor += n_tokens
    run = length - cursor
    if run:
        pos[:, cursor:] = torch.arange(st, st + run, dtype=torch.int32)
        st += run
    delta = int(pos.max().item()) + 1 - length
    return pos, delta


def content_hash(feature: torch.Tensor, extra: bytes = b"") -> int:
    """62-bit hash of the wire bytes plus family extras (e.g. the grid, so transposed images differ)."""
    buf = feature.contiguous()
    if buf.dtype == torch.bfloat16:
        buf = buf.view(torch.uint16)
    hasher = hashlib.sha256(memoryview(buf.numpy()))
    hasher.update(extra)
    return int.from_bytes(hasher.digest()[:8], "little") % (1 << 62)


@lru_cache(maxsize=4)
def get_mm_processor(model_path: str) -> MMProcessor | None:
    """The family's processor, or None when the family takes no multimodal input or the checkpoint ships none of its encoders."""
    from freetoken.models.register import get_model_spec
    from freetoken.utils import cached_load_hf_config

    try:
        config = cached_load_hf_config(model_path)
        spec = get_model_spec((config.architectures or [None])[0])
    except Exception:  # noqa: BLE001 -- missing/foreign config or unknown architecture: no multimodal input
        return None
    if spec.mm_processor is None or all(getattr(config, e.config_key, None) is None for e in spec.encoders):
        return None
    check_mm_pad_shift(config.text_config.vocab_size)
    module, _, cls = spec.mm_processor.partition(":")
    return getattr(importlib.import_module(module), cls)(config, model_path)


__all__ = [
    "MMProcessor",
    "MMResult",
    "PromptReplacement",
    "content_hash",
    "get_mm_processor",
    "image_positions",
]
