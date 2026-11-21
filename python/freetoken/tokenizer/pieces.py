from __future__ import annotations

import json
import re

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

# The split patterns of the Qwen2 and Qwen3.6 byte-level BPE tokenizers. Under either, a match never runs from a
# newline into the character after it, nor from a letter into a space that a letter follows, so the text can be cut
# there and its pieces encoded on their own (``piece_cuts``).
_PIECE_PATTERNS = frozenset({
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
})
# after a newline, before printable ASCII; or at a space between two ASCII letters
_PIECE_CUT = re.compile(r"(?<=\n)(?=[!-~])|(?<=[A-Za-z])(?= [A-Za-z])")
# a text is encoded in pieces of about this many characters, at most _MAX_PIECES of them, in parallel
_PIECE_CHARS = 2048
_MAX_PIECES = 32


def piece_encoder(tokenizer: PreTrainedTokenizerBase):
    """The tokenizer's Rust backend when a text cut at ``_PIECE_CUT`` encodes to the ids of the whole, else None.

    Pieces are encoded alone, so no rule may look across a cut: a split pattern of ``_PIECE_PATTERNS`` then byte-level
    without its own regex, a normalizer that is NFC or none, a BPE model, and added tokens that hold no whitespace and
    strip none (none can then span a cut).
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or getattr(tokenizer, "split_special_tokens", False):
        return None
    try:
        spec = json.loads(backend.to_str())
    except Exception:  # noqa: BLE001 -- a backend that does not serialize keeps the whole-text encode
        return None
    pre = spec.get("pre_tokenizer") or {}
    steps = pre.get("pretokenizers") if pre.get("type") == "Sequence" else None
    if not steps or len(steps) != 2:
        return None
    split, byte_level = steps
    if not (
        split.get("type") == "Split"
        and (split.get("pattern") or {}).get("Regex") in _PIECE_PATTERNS
        and split.get("behavior") == "Isolated"
        and not split.get("invert")
        and byte_level.get("type") == "ByteLevel"
        and not byte_level.get("use_regex")
        and not byte_level.get("add_prefix_space")
        and spec.get("normalizer") in (None, {"type": "NFC"})
        and (spec.get("model") or {}).get("type") == "BPE"
    ):
        return None
    for t in spec.get("added_tokens", []):
        if t.get("lstrip") or t.get("rstrip") or t.get("single_word") or any(c.isspace() for c in t["content"]):
            return None
    return backend


def piece_cuts(text: str, pieces: int) -> list[int]:
    """Up to ``pieces - 1`` cut positions, each the first ``_PIECE_CUT`` from an even share of the text on."""
    cuts: list[int] = []
    for i in range(1, pieces):
        at = max(len(text) * i // pieces, cuts[-1] + 1 if cuts else 1)
        m = _PIECE_CUT.search(text, at)
        if m is None:
            break
        cuts.append(m.start())
    return cuts


def warm(backend) -> None:
    """Start the backend's thread pool, so that the first long prompt does not."""
    backend.encode_batch(["warm", "up"], add_special_tokens=False)


def encode_pieces(backend, text: str) -> torch.Tensor | None:
    """``tokenizer.encode(text, add_special_tokens=False)`` as an int32 tensor, from ``text`` cut at ``piece_cuts``
    and the pieces encoded in parallel by ``backend`` (a ``piece_encoder``). None when the text is too short to cut,
    so the caller keeps its whole-text encode."""
    bounds = [0, *piece_cuts(text, min(_MAX_PIECES, len(text) // _PIECE_CHARS)), len(text)]
    if len(bounds) == 2:
        return None
    encodings = backend.encode_batch([text[a:b] for a, b in zip(bounds, bounds[1:])], add_special_tokens=False)
    return torch.from_numpy(np.concatenate([np.asarray(enc.ids, dtype=np.int32) for enc in encodings]))
