"""Multimodal input handling: pad-id scheme, per-family processors, media fetch, embedding cache."""

from __future__ import annotations

import torch

# image spans become content pseudo ids above every real token id: the radix cache keys them by content and the model finds image rows by id >= MM_PAD_SHIFT_VALUE
# they are clamped back into the vocab before the embedding lookup; those rows are overwritten anyway
MM_PAD_SHIFT_VALUE = 1_000_000


def restore_placeholder(input_ids: torch.Tensor, token_id: int) -> torch.Tensor:
    """input_ids with every content pad id put back to the family's placeholder token."""
    return input_ids.masked_fill(input_ids >= MM_PAD_SHIFT_VALUE, token_id)


def mm_pad_value(item_hash: int) -> int:
    return MM_PAD_SHIFT_VALUE + item_hash % (1 << 30)


def check_mm_pad_shift(vocab_size: int) -> None:
    if vocab_size > MM_PAD_SHIFT_VALUE:
        raise ValueError(
            f"vocab_size ({vocab_size}) exceeds MM_PAD_SHIFT_VALUE ({MM_PAD_SHIFT_VALUE}); "
            "image pseudo-ids would alias real token ids"
        )


__all__ = [
    "MM_PAD_SHIFT_VALUE",
    "check_mm_pad_shift",
    "mm_pad_value",
]
