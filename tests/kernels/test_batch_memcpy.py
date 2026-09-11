"""The cudaMemcpyBatchAsync load probe must be ordered against the current stream.

_probe allocates its destination with torch.zeros (a fill enqueued on the CURRENT
stream) but enqueues the copy on a fresh probe stream. With nothing joining the two,
the fill can land *after* the copy and clobber it, so the probe reads back zeros and
load_batch_memcpy raises "probe copied wrong bytes" on a GPU that supports the API
perfectly well. It only reproduces once the caching allocator and the stream pool are
warm -- in a cold process torch.zeros and torch.cuda.Stream() each synchronize the
device and hide the race, which is why the probe passes standalone but fails inside a
server that is mid-warmup.
"""

from __future__ import annotations

import os

import pytest
import torch

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
JIT = pytest.mark.skipif(
    os.getenv("FREETOKEN_DISABLE_JIT", "").strip().lower() in {"1", "true", "yes", "on"},
    reason="batch_memcpy has no AOT prebuild; needs runtime JIT",
)


def _cuda_at_least(major: int, minor: int) -> bool:
    cuda = torch.version.cuda
    if cuda is None:
        return False
    return tuple(int(x) for x in cuda.split(".")[:2]) >= (major, minor)


BATCH_API = pytest.mark.skipif(
    not _cuda_at_least(13, 0), reason="the cudaMemcpyBatchAsync binding needs CUDA >= 13.0"
)

# ~1s of spin on the current stream. The probe's own copy is microseconds, so any
# ordering bug shows up as the whole backlog landing on top of the copied bytes.
_BACKLOG_CYCLES = 1_500_000_000


def _warm_device_state() -> None:
    """Reach the state a server is in at prefill warmup: the caching allocator holds a
    free block of the probe's 16-byte size and the per-device stream pool is populated,
    so neither torch.zeros nor torch.cuda.Stream() synchronizes on the probe's behalf."""
    for _ in range(4):
        block = torch.zeros(16, dtype=torch.uint8, device="cuda")
        del block
        pinned = torch.arange(16, dtype=torch.uint8).pin_memory()
        del pinned
    pool = [torch.cuda.Stream() for _ in range(40)]
    torch.cuda.synchronize()
    del pool


@CUDA
@JIT
@BATCH_API
def test_probe_survives_a_busy_current_stream():
    from freetoken.kernel.batch_memcpy import load_batch_memcpy

    _warm_device_state()
    try:
        torch.cuda._sleep(_BACKLOG_CYCLES)
        # Guard against a vacuous test: if something drained the queue the race is
        # not being exercised at all and a pass would mean nothing.
        assert not torch.cuda.current_stream().query(), "backlog drained before the probe ran"
        load_batch_memcpy()
    finally:
        torch.cuda.synchronize()
