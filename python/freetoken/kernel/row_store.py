"""Python face of the disk row store extension (``kernel/csrc/row_store``) and its CUDA-graph sync.

``RowStore`` reads fixed-width rows straight out of checkpoint shard files (io_uring or a pread pool,
O_DIRECT) into pinned staging by row id; ``PleStore`` adds PLE's n-gram hashing.

Graph sync: a captured graph cannot wait for host I/O, so a captured ``lookup`` WAITs on a pinned
flag through a stream memop (``wait_reset``) that the host ``signal``s once the fill landed; where
the driver rejects memops in capture, the caller falls back to launch-gating (fill before replay).
``probe_wait_sync`` decides which, once, at startup.
"""

from __future__ import annotations

import torch

from freetoken.kernel import _row_store
from freetoken.kernel.pinned import alloc_pinned_tensor

RowStore = _row_store.RowStore
PleStore = _row_store.PleStore


def probe_wait_sync(mode: str, device: torch.device) -> bool:
    """``mode``: ``auto`` (probe), ``wait`` (require memops) or ``gate`` (never use them)."""
    if mode not in ("auto", "wait", "gate"):
        raise ValueError(f"unknown row-store sync mode {mode!r}; expected auto, wait or gate")
    if mode == "gate":
        return False
    scratch = alloc_pinned_tensor(1, dtype=torch.int64)
    scratch.zero_()
    stream = torch.cuda.current_stream(device)
    ok = (
        _row_store.memop_write(stream.cuda_stream, scratch.data_ptr(), 7) == 0
        and _row_store.memop_wait_geq(stream.cuda_stream, scratch.data_ptr(), 7) == 0
    )
    if ok:
        stream.synchronize()
        ok = int(scratch[0]) == 7
    if mode == "wait" and not ok:
        raise RuntimeError("wait-sync requested but stream memops are unavailable")
    return ok


def wait_reset(stream: torch.cuda.Stream, flag: torch.Tensor) -> None:
    """Enqueue WAIT(flag >= 1) then flag := 0 on ``stream`` (capturable)."""
    _row_store.memop_wait_reset(stream.cuda_stream, flag.data_ptr())


def signal(flag: torch.Tensor) -> None:
    """Host side: release a pending WAIT (a release-store of 1)."""
    _row_store.signal_flag(flag.data_ptr())


__all__ = ["RowStore", "PleStore", "probe_wait_sync", "wait_reset", "signal"]
