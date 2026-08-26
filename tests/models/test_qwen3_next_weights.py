"""Qwen3-Next GDN in_proj de-interleaving and sigmoid routing.

Qwen3-Next modelopt checkpoints ship the pre-fused GDN input projections in HF's
interleaved layout (``Qwen3NextGatedDeltaNet.fix_query_key_value_ordering``):
per k-head group the raw rows are ``[q_g | k_g | v_pair | z_pair]`` (qkvz) and
``[b_pair | a_pair]`` (ba). The engine's GDN instead splits contiguous
``[q|k|v|z]`` / ``[b|a]`` blocks, so ``_gdn_split_reorder`` must de-interleave
at load time. Loading the interleaved layout as-is scrambles q/k/v/z silently --
output is fluent garbage and decode speed looks normal -- so the permutation is
verified by round-trip here: build the interleaved layout from a known
contiguous reference and require the loader to recover it exactly, for both the
bf16 weights and the per-128-row fp8 scale blocks that alias head_dim.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.models.qwen3_5_moe.weight import _gdn_split_reorder
from freetoken.moe.fused import _torch_fused_topk

# Small but non-degenerate GDN geometry: 4 k-head groups, 2 v-heads per group.
NK, NV, HD, HIDDEN = 4, 8, 8, 20
RATIO = NV // NK


def _g():
    return SimpleNamespace(num_key_heads=NK, num_value_heads=NV, key_head_dim=HD)


def _interleave_qkvz(contig: torch.Tensor) -> torch.Tensor:
    """Inverse of _gdn_split_reorder: HF's fix_query_key_value_ordering layout."""
    q, k, v, z = torch.split(contig, [NK * HD, NK * HD, NV * HD, NV * HD], dim=0)

    def head(t, h):
        return t[h * HD : (h + 1) * HD]

    parts = []
    for g in range(NK):
        parts += [head(q, g), head(k, g)]
        parts += [head(v, h) for h in range(g * RATIO, (g + 1) * RATIO)]
        parts += [head(z, h) for h in range(g * RATIO, (g + 1) * RATIO)]
    return torch.cat(parts, dim=0)


def _interleave_ba(contig: torch.Tensor) -> torch.Tensor:
    b, a = torch.split(contig, [NV, NV], dim=0)
    parts = []
    for g in range(NK):
        parts += [b[h : h + 1] for h in range(g * RATIO, (g + 1) * RATIO)]
        parts += [a[h : h + 1] for h in range(g * RATIO, (g + 1) * RATIO)]
    return torch.cat(parts, dim=0)


def test_in_proj_qkvz_round_trip():
    contig = torch.arange(NK * (2 + 2 * RATIO) * HD * HIDDEN, dtype=torch.float32)
    contig = contig.view(-1, HIDDEN)
    interleaved = _interleave_qkvz(contig)
    name = "model.layers.0.linear_attn.in_proj_qkvz.weight"
    out = _gdn_split_reorder(name, interleaved, _g())
    assert torch.equal(out, contig)


def test_in_proj_ba_round_trip():
    contig = torch.arange(2 * NV * HIDDEN, dtype=torch.float32).view(-1, HIDDEN)
    interleaved = _interleave_ba(contig)
    name = "model.layers.0.linear_attn.in_proj_ba.weight"
    out = _gdn_split_reorder(name, interleaved, _g())
    assert torch.equal(out, contig)


def test_in_proj_qkvz_scale_blocks_round_trip():
    """The fp8 per-128-row ``weight_scale_inv`` aliases head_dim: each scale row
    covers exactly one head_dim-sized chunk of the interleaved layout."""
    n_blocks = NK * (2 + 2 * RATIO)  # one scale row per (group, slot) block
    contig_scale = torch.arange(n_blocks, dtype=torch.float32).view(-1, 1)
    q, k, v, z = torch.split(contig_scale, [NK, NK, NV, NV], dim=0)
    parts = []
    for g in range(NK):
        parts += [q[g : g + 1], k[g : g + 1]]
        parts += [v[h : h + 1] for h in range(g * RATIO, (g + 1) * RATIO)]
        parts += [z[h : h + 1] for h in range(g * RATIO, (g + 1) * RATIO)]
    interleaved_scale = torch.cat(parts, dim=0)
    name = "model.layers.0.linear_attn.in_proj_qkvz.weight_scale_inv"
    out = _gdn_split_reorder(name, interleaved_scale, _g())
    assert torch.equal(out, contig_scale)


def test_other_names_pass_through():
    t = torch.arange(2 * NV * HIDDEN, dtype=torch.float32).view(-1, HIDDEN)
    name = "model.layers.0.linear_attn.out_proj.weight"
    out = _gdn_split_reorder(name, t, _g())
    assert out is t


def test_ambiguous_qkvz_rows_rejected():
    bad = torch.zeros(NK * (2 + 2 * RATIO) * HD + 1, HIDDEN)
    name = "model.layers.0.linear_attn.in_proj_qkvz.weight"
    with pytest.raises(ValueError, match="cannot de-interleave"):
        _gdn_split_reorder(name, bad, _g())


def test_sigmoid_router_matches_reference():
    """Qwen3-Next scores experts with sigmoid + top-10: verify the torch router
    against a hand-rolled reference, including renormalization."""
    torch.manual_seed(0)
    logits = torch.randn(5, 32)
    weights, ids = _torch_fused_topk(logits, 10, True, None, "sigmoid")
    probs = torch.sigmoid(logits.float())
    ref_weights, ref_ids = torch.topk(probs, 10, dim=-1)
    ref_weights = ref_weights / ref_weights.sum(dim=-1, keepdim=True)
    assert torch.equal(ids, ref_ids)
    assert torch.allclose(weights, ref_weights)
