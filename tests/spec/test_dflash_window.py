"""DFlashWindow on a CPU LinearStatePool: the rows past the window keep a verify's unaccepted rows and a prefill's rows past its tracked boundary out of every window read, also after the copy with the GDN slot."""

from __future__ import annotations

import torch

from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.spec.dflash.window import DFlashWindow

W, H, BLOCK, LIVE, TRACK = 16, 8, 8, 3, 5
R = DFlashWindow.length(W, BLOCK)


def _group():
    return LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=1, num_value_heads=2,
        key_head_dim=4, value_head_dim=4, conv_kernel_dim=2, output_gate="silu",
    )


def _window():
    specs = DFlashWindow.slot_specs(R, H)
    pool = LinearStatePool(group=_group(), num_slots=6, dtype=torch.bfloat16, device=torch.device("cpu"), tp_size=1, slot_states=specs)
    return pool, DFlashWindow({s.name: pool.slot_state(s.name) for s in specs}, R, H)


def _rows(n, seed):
    return torch.randn(n, H, generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)


def _store(window, rows, start, slot=LIVE):
    """As the proposer stores a chunk of positions ``start ..``: its last R rows at most, in one call."""
    kept, end = rows[-R:], start + rows.shape[0]
    window.write(kept, torch.arange(end - kept.shape[0], end), torch.full((kept.shape[0],), slot))


def test_rows_past_the_window_keep_stale_verify_and_prefill_rows_out_of_every_read():
    """A verify writes k+1 rows at C..C+k and keeps n <= k; a prefill chunk ending at e is snapshot at the tracked boundary b with e-b <= 64: the window of every committed length reads back intact."""
    pool, window = _window()
    C, k = 300, BLOCK - 1
    history = _rows(C, 1)
    for start in range(0, C, R):
        _store(window, history[start : start + R], start)
    verify = _rows(k + 1, 2)
    _store(window, verify, C)
    for n in range(k + 1):
        want = torch.cat([history[C + n - W : C], verify[:n]])
        assert torch.equal(window.read(LIVE, C + n - W, W), want), f"committed {n} of {k} drafts"

    s, e = C, C + 100
    chunk = _rows(e - s, 3)
    _store(window, chunk, s)
    b = s + 64 * ((e - s - 1) // 64)
    pool.copy_from(LIVE, TRACK)
    assert torch.equal(window.read(TRACK, b - W, W), chunk[b - W - s : b - s])
    # and the prefill left the live window of e intact as well
    assert torch.equal(window.read(LIVE, e - W, W), chunk[-W:])

