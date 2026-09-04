"""mRoPE plumbing through the QSA layer (GPU): the per-token table path (what an image-carrying
prefill batch uses) must reproduce the position-indexed cache path exactly for text, including
a chunked continuation sharing the batch (the indexer ropes a straddling group at its first
token, which lives in the table's lead rows) and the decode `positions + delta` path.
"""

from __future__ import annotations

import torch
from freetoken.models.qwen4_exp.mrope import mrope_table

from .common import Fixture, parsed_config, requires_cuda

QSA_LAYER = 3
SECTION = (11, 11, 10)


def _inv_freq(config, device):
    rc = config.rotary_config
    return 1.0 / (
        rc.base
        ** (
            torch.arange(0, rc.rotary_dim, 2, dtype=torch.float, device=device)
            / rc.rotary_dim
        )
    )


def _run(config, table_path: bool, seed: int = 5):
    """Prefill B (chunk 1) -> prefill [A, B chunk 2] -> one decode step; return the two outputs."""
    fixture = Fixture(config, num_pages=64)
    attn = fixture.layer(QSA_LAYER)
    ratio = config.qwen4_args.index_ratio
    gen = torch.Generator(device=fixture.device).manual_seed(seed)
    hidden = config.hidden_size
    x_b1 = torch.randn(
        41, hidden, device=fixture.device, dtype=fixture.dtype, generator=gen
    )
    x_ab = torch.randn(
        37 + 30, hidden, device=fixture.device, dtype=fixture.dtype, generator=gen
    )
    x_dec = torch.randn(
        2, hidden, device=fixture.device, dtype=fixture.dtype, generator=gen
    )
    assert 41 % ratio, (
        "chunk 1 must end mid-group so a group straddles the chunk boundary"
    )

    b = fixture.req(1, 0, 41)
    # chunk 1 always takes the cache path
    attn.forward(x_b1, fixture.batch([b], "prefill"))
    b.cached_len, b.device_len, b.extend_len = 41, 71, 30
    fixture.allocate(1, 41, 71)
    a = fixture.req(0, 0, 37)
    batch = fixture.batch([a, b], "prefill")
    if table_path:
        batch.rope_positions, batch.mrope_cos_sin = mrope_table(
            [a, b], ratio, _inv_freq(config, fixture.device), SECTION, fixture.device
        )
        assert batch.mrope_cos_sin.shape[0] == 37 + 30 + (ratio - 1)
    out_prefill = attn.forward(x_ab, batch).clone()

    fixture.step(a)
    fixture.step(b)
    batch = fixture.batch([a, b], "decode")
    if table_path:
        # decode never has a table; exercise the explicit rope_positions path (delta 0)
        batch.rope_positions = batch.positions + 0
    out_decode = attn.forward(x_dec, batch).clone()
    return out_prefill, out_decode


@requires_cuda
def test_table_path_matches_cache_path():
    config = parsed_config()
    ref_prefill, ref_decode = _run(config, table_path=False)
    tab_prefill, tab_decode = _run(config, table_path=True)
    assert torch.equal(ref_prefill, tab_prefill)
    assert torch.equal(ref_decode, tab_decode)
