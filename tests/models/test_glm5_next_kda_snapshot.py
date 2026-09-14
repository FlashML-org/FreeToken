"""KDA hybrid-radix track-snapshot contract (prefix caching for glm5_next).

The scheduler snapshots each request's linear state at the deepest chunk-aligned
(x64) boundary of a prefill into a donatable pool slot (FLAMetadata.track_*; the
op writes it from the chunk kernel's per-chunk h + the raw conv window). A later
request restores by copying that slot and continuing with
``has_initial_state=True``. Checks, at the KDA-op level:

* the snapshot equals the TRUE state after exactly 64 tokens (independent run)
* restore + continuation reproduces the uninterrupted run's outputs
* the 64-boundary is kpool-aligned by construction (64 % 4 == 0)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.models.test_glm5_next_kda_op import (  # reuse the op harness
    _make_op,
    _make_pool,
    _patch_ctx,
    _fla,
    _assert_close,
    HIDDEN,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

CHUNK = 64


@pytest.fixture(autouse=True)
def _single_rank_tp():
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _prefill(op, pool, monkeypatch, x, slot, t0=0, has_init=False, track=None):
    t = x.shape[0]
    fla = _fla(
        torch.tensor([0, t], dtype=torch.int32, device="cuda"),
        torch.tensor([slot], dtype=torch.int32, device="cuda"),
        torch.tensor([has_init], device="cuda"),
        None if has_init else torch.tensor([slot], dtype=torch.int64, device="cuda"),
    )
    if track is not None:
        fla.track_dst, fla.track_h_row, fla.track_conv_src = track
    batch = SimpleNamespace(is_decode=False, fla_metadata=fla)
    _patch_ctx(monkeypatch, pool, batch)
    return op.forward(x)


def test_snapshot_restore_roundtrip(monkeypatch):
    from freetoken.kernel.fla.index import prepare_chunk_offsets

    op = _make_op(seed=5)
    pool = _make_pool(num_slots=6)
    total = 100  # crosses one x64 boundary; tail 36 tokens
    torch.manual_seed(20)
    x = torch.randn(total, HIDDEN, device="cuda", dtype=torch.bfloat16)

    # --- ground truth: state after exactly CHUNK tokens (independent run, slot 3)
    _prefill(op, pool, monkeypatch, x[:CHUNK], slot=3)
    true_rec = pool.recurrent_states[0, 3].clone()
    true_conv = pool.conv_states[0, 3].clone()

    # --- tracked run (slot 1, snapshot into slot 2), as _build_track_metadata would
    km1 = pool.conv_states.shape[-1]
    cu_host = torch.tensor([0, total], dtype=torch.int64)
    boh = prepare_chunk_offsets(cu_host, CHUNK).tolist()
    c = (total - 1) // CHUNK  # deepest mid-chunk boundary: 1 -> position 64
    track = (
        torch.tensor([2], dtype=torch.int64, device="cuda"),
        torch.tensor([boh[0] + c], dtype=torch.int64, device="cuda"),
        torch.tensor([[c * CHUNK - km1 + j for j in range(km1)]], dtype=torch.int64, device="cuda"),
    )
    out_full = _prefill(op, pool, monkeypatch, x, slot=1, track=track)

    _assert_close(pool.recurrent_states[0, 2], true_rec, "snapshot recurrent state")
    _assert_close(pool.conv_states[0, 2], true_conv, "snapshot conv state")

    # --- restore: copy snapshot -> fresh slot 4, continue [64, 100)
    pool.copy_from(2, 4)
    out_cont = _prefill(
        op, pool, monkeypatch, x[CHUNK:], slot=4, t0=CHUNK, has_init=True
    )
    _assert_close(out_cont, out_full[CHUNK:], "restored continuation outputs")
    _assert_close(
        pool.recurrent_states[0, 4], pool.recurrent_states[0, 1], "final states agree"
    )

    # kpool alignment is subsumed by the x64 snapshot boundary.
    assert CHUNK % 4 == 0