"""qwen35moe GGUF GDN value-head de-interleaving.

llama.cpp stores the GDN *value* projections with the ``mrope_interleaved`` head order
(even heads 0..nv/2-1 first, then odd heads nv/2..nv-1). FreeToken uses the HF contiguous
head order, so the loader de-interleaves the value projections on load. These tests pin the
permutation and the packed/dense de-interleave helpers (no model weights required).
"""

import torch

from freetoken.models.qwen3_5_moe.gguf import (
    _gdn_head_perm,
    _deint_dense_rows,
    _deint_q8_cols,
    _deint_q8_rows,
)


def _interleave(x: torch.Tensor, nv: int, rows_per_head: int = 1) -> torch.Tensor:
    """Build the GGUF head-interleaved layout from an HF-contiguous tensor.

    GGUF position ``perm[h]`` holds HF head ``h`` (so interleaved[perm[h]] = x[h]).
    """
    m = x.reshape(nv, rows_per_head, -1)
    perm = _gdn_head_perm(nv)
    out = m.clone()
    for h in range(nv):
        out[perm[h]] = m[h]
    return out.reshape(x.shape)


def test_head_permutation_is_a_bijection():
    nv = 32
    perm = _gdn_head_perm(nv)
    assert sorted(perm) == list(range(nv))  # valid permutation
    # GGUF layout: even HF heads occupy positions 0..15, odd heads 16..31.
    even = sorted(perm[h] for h in range(0, nv, 2))
    odd = sorted(perm[h] for h in range(1, nv, 2))
    assert even == list(range(nv // 2))
    assert odd == list(range(nv // 2, nv))


def test_deint_dense_rows_recovers_contiguous():
    nv, hd = 32, 128
    x = torch.randn(nv, hd)
    inter = _interleave(x, nv, rows_per_head=1)
    rec = _deint_dense_rows(inter, nv)
    assert torch.allclose(rec, x, atol=1e-6)


def test_deint_q8_rows_recovers_contiguous():
    nv, hd = 32, 128
    rph = 64  # rows per value head in the projection output dim
    x = torch.randn(nv * rph, 4)  # packed rows x (row_bytes mocked)
    inter = _interleave(x, nv, rows_per_head=rph)
    rec = _deint_q8_rows(inter, nv, rph)
    assert torch.allclose(rec, x, atol=1e-6)


def test_deint_q8_cols_recovers_contiguous():
    nv = 32
    rows, blocks_per_head, bb = 16, 4, 34
    x = torch.randn(rows, nv * blocks_per_head * bb)
    # interleave column head-groups: inter[:, perm[h]*bbp:(perm[h]+1)*bbp] = x[:, h*bbp:(h+1)*bbp]
    bbp = blocks_per_head * bb
    inter = torch.empty_like(x)
    perm = _gdn_head_perm(nv)
    for h in range(nv):
        inter[:, perm[h] * bbp:(perm[h] + 1) * bbp] = x[:, h * bbp:(h + 1) * bbp]
    rec = _deint_q8_cols(inter, nv, blocks_per_head)
    assert torch.allclose(rec, x, atol=1e-6)
