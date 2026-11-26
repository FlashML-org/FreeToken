"""Opt-in switch for the decode expert gather. Default off."""

from __future__ import annotations

import os

DECODE_GATHER_ENV = "QWEN36_DECODE_GATHER"
# 8 CTAs x 3 stages x 16 KiB keeps 384 KiB of reads in flight.
TUNED_DECODE_GATHER = ("tma", 8, 3, 16384)
_DEFAULT_VALUES = {"", "0", "off", "default"}


def decode_gather_config() -> tuple | None:
    """The tuned gather config, or ``None`` to keep ``fast_index_copy_multi``.

    Any other value raises, so a typo cannot silently select the default kernel.
    """
    raw = os.getenv(DECODE_GATHER_ENV, "")
    value = raw.strip().lower()
    if value in _DEFAULT_VALUES:
        return None
    if value == "tuned":
        return TUNED_DECODE_GATHER
    raise ValueError(
        f"{DECODE_GATHER_ENV}={raw!r}: expected 'tuned' (or unset for the default gather)"
    )
