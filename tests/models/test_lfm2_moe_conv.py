"""LFM2 short-conv op: HF reference math, state contract, and the hybrid-radix track snapshot.

The op's own arithmetic (the B/C/x split, the depthwise conv over ``B * x``, the C gate) is
checked against a transcription of HF ``Lfm2MoeShortConv.slow_forward``; the state contract
(prefill == per-token decode == chunked prefill, fresh slots zeroed, state == the last K-1 conv
inputs) and the track snapshot (the state at a x64 boundary, restorable) are checked against
independent runs.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = "cuda"
CONV_DIM = 64
KERNEL = 3
CHUNK = 64  # the hybrid-radix snapshot granularity (kernel.fla.chunk CHUNK_SIZE)


@pytest.fixture(autouse=True)
def _single_rank_tp():
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _config(conv_bias=False):
    from freetoken.models.config import ModelConfig, RotaryConfig, ShortConvGroupConfig

    group = ShortConvGroupConfig(
        name="conv", layer_ids=(0,), conv_dim=CONV_DIM, conv_kernel_dim=KERNEL, conv_bias=conv_bias
    )
    return ModelConfig(
        num_layers=1, num_qo_heads=1, num_kv_heads=1, head_dim=64, hidden_size=CONV_DIM,
        vocab_size=8, intermediate_size=8, rms_norm_eps=1e-5, hidden_act="silu",
        tie_word_embeddings=True, num_experts=0, num_experts_per_tok=1, moe_intermediate_size=8,
        norm_topk_prob=True, model_type="lfm2_moe", architectures=["Lfm2MoeForCausalLM"],
        rotary_config=RotaryConfig(head_dim=64, rotary_dim=64, max_position=64, base=1e4, scaling=None),
        attention_groups=(group,),
    )


def _make_op(seed=0, conv_bias=False):
    from freetoken.models.lfm2_moe.conv import Lfm2ShortConv

    op = Lfm2ShortConv(_config(conv_bias), layer_id=0, prefix="layers.0.conv")
    torch.manual_seed(seed)
    op.in_proj.weight = torch.randn(3 * CONV_DIM, CONV_DIM, device=DEV, dtype=torch.bfloat16) * 0.2
    op.conv.weight = torch.randn(CONV_DIM, 1, KERNEL, device=DEV, dtype=torch.bfloat16) * 0.5
    op.out_proj.weight = torch.randn(CONV_DIM, CONV_DIM, device=DEV, dtype=torch.bfloat16) * 0.2
    if conv_bias:
        op.bias = torch.randn(CONV_DIM, device=DEV, dtype=torch.bfloat16) * 0.5
    return op


def _make_pool(num_slots=8):
    from freetoken.kvcache.linear_state_pool import LinearStatePool

    return LinearStatePool(
        _config().short_conv_group(), num_slots, dtype=torch.bfloat16,
        device=torch.device(DEV), tp_size=1,
    )


def _fla(cu, indices, has_init=None, fresh=None, track=None):
    from freetoken.attention.linear import FLAMetadata

    fla = FLAMetadata(
        cu_seqlens=cu, cache_indices=indices, has_initial_state=has_init,
        fresh_state_indices=fresh,
    )
    if track is not None:
        fla.track_dst, fla.track_h_row, fla.track_conv_src = track
    return fla


def _forward(op, pool, monkeypatch, x, fla, is_decode=False):
    """One op forward under a fresh batch (as the scheduler builds per step) + context."""
    from freetoken.core import Context

    batch = SimpleNamespace(is_decode=is_decode, fla_metadata=fla)
    ctx = Context(page_size=1)
    ctx.linear_state_pool = pool
    monkeypatch.setattr("freetoken.core._GLOBAL_CTX", ctx)
    with ctx.forward_batch(batch):
        return op.forward(x)


def _prefill(op, pool, monkeypatch, x, slot, t0=0, has_init=False, track=None):
    t = x.shape[0]
    device = x.device
    fla = _fla(
        torch.tensor([0, t], dtype=torch.int32, device=device),
        torch.tensor([slot], dtype=torch.int32, device=device),
        torch.tensor([has_init], device=device),
        None if has_init or t0 > 0 else torch.tensor([slot], dtype=torch.int64, device=device),
        track,
    )
    return _forward(op, pool, monkeypatch, x, fla)


def _reference(op, x, left=None):
    """HF ``Lfm2MoeShortConv.slow_forward`` for one sequence; returns (out, next_state, bx)."""
    proj = x.float() @ op.in_proj.weight.float().T
    b, c, xv = proj.split(op.conv_dim, dim=-1)
    bx = b * xv
    left = torch.zeros(op.conv_dim, KERNEL - 1, device=x.device) if left is None else left
    padded = torch.cat([left.float(), bx.T], dim=1)  # [conv_dim, K-1 + T]
    weight = op.conv.weight.squeeze(1).float()
    total = x.shape[0]
    conv = torch.stack([(padded[:, t : t + KERNEL] * weight).sum(-1) for t in range(total)], dim=1)
    if op.bias is not None:
        conv = conv + op.bias.float()[:, None]
    out = (c.transpose(0, 1) * conv).T.float() @ op.out_proj.weight.float().T
    return out.to(x.dtype), padded[:, -(KERNEL - 1) :], bx


def _assert_close(ours, ref, tag, atol=3e-2):
    err = (ours.float() - ref.float()).abs().max().item()
    scale = ref.float().abs().max().item() + 1e-8
    assert err / scale < atol, f"{tag}: max abs err {err:.5f} (ref scale {scale:.3f})"


def test_prefill_matches_the_hf_reference(monkeypatch):
    op = _make_op(seed=3)
    pool = _make_pool()
    torch.manual_seed(4)
    x = torch.randn(23, CONV_DIM, device=DEV, dtype=torch.bfloat16)

    out = _prefill(op, pool, monkeypatch, x, slot=1)

    ref, state, _bx = _reference(op, x)
    _assert_close(out, ref, "prefill out")
    _assert_close(pool.conv_states[0, 1], state, "conv state")


def test_prefill_decode_and_chunked_agree(monkeypatch):
    op = _make_op(seed=5)
    torch.manual_seed(6)
    total = 20
    x = torch.randn(total, CONV_DIM, device=DEV, dtype=torch.bfloat16)

    ref, _state, _bx = _reference(op, x)

    # one-shot prefill
    pool_full = _make_pool()
    out_full = _prefill(op, pool_full, monkeypatch, x, slot=1)
    _assert_close(out_full, ref, "one-shot prefill")

    # prefill one token, then decode the rest one at a time (the state rolls per step)
    pool_dec = _make_pool()
    out_dec = [_prefill(op, pool_dec, monkeypatch, x[:1], slot=1)]
    slot = torch.tensor([1], dtype=torch.int32, device=DEV)
    for t in range(1, total):
        fla = _fla(torch.arange(2, dtype=torch.int32, device=DEV), slot)
        out_dec.append(_forward(op, pool_dec, monkeypatch, x[t : t + 1], fla, is_decode=True))
    for t, row in enumerate(out_dec):
        _assert_close(row, out_full[t : t + 1], f"decode token {t}")
    _assert_close(pool_dec.conv_states[0, 1], pool_full.conv_states[0, 1], "final state")

    # chunked prefill: [0, 7) then [7, total) with the live state as left context
    pool_chunk = _make_pool()
    out_a = _prefill(op, pool_chunk, monkeypatch, x[:7], slot=1, t0=0)
    out_b = _prefill(op, pool_chunk, monkeypatch, x[7:], slot=1, t0=7, has_init=True)
    _assert_close(torch.cat([out_a, out_b], dim=0), out_full, "chunked prefill")


def test_packed_prefill_matches_per_sequence_runs(monkeypatch):
    """Two ragged requests in one packed prefill keep their own state blocks and outputs."""
    op = _make_op(seed=7)
    torch.manual_seed(8)
    lens = [33, 70]
    x = torch.randn(sum(lens), CONV_DIM, device=DEV, dtype=torch.bfloat16)

    pool = _make_pool()
    t0 = 0
    per_seq = []
    for i, n in enumerate(lens):
        per_seq.append(_prefill(op, _make_pool(), monkeypatch, x[t0 : t0 + n], slot=i + 1))
        t0 += n

    cu = torch.tensor([0, *itertools.accumulate(lens)], dtype=torch.int32, device=DEV)
    fla = _fla(cu, torch.tensor([1, 2], dtype=torch.int32, device=DEV),
               fresh=torch.tensor([1, 2], dtype=torch.int64, device=DEV))
    packed = _forward(op, pool, monkeypatch, x, fla)

    t0 = 0
    for i, n in enumerate(lens):
        _assert_close(packed[t0 : t0 + n], per_seq[i], f"packed seq {i}")
        t0 += n


def test_state_is_the_last_k_minus_1_conv_inputs(monkeypatch):
    op = _make_op(seed=9)
    pool = _make_pool()
    torch.manual_seed(10)
    x = torch.randn(11, CONV_DIM, device=DEV, dtype=torch.bfloat16)

    _prefill(op, pool, monkeypatch, x, slot=1)

    # the conv input stream, through the op's own projection (so the dtypes match exactly)
    b, _c, xv = op.in_proj.forward(x).split(CONV_DIM, dim=-1)
    bx = b * xv
    assert torch.equal(pool.conv_states[0, 1], bx[-(KERNEL - 1) :].T.to(torch.bfloat16))


def test_fresh_slots_start_from_zero(monkeypatch):
    """A recycled slot must not leak its previous sequence's history into a fresh request."""
    op = _make_op(seed=11)
    pool = _make_pool()
    torch.manual_seed(12)
    x = torch.randn(9, CONV_DIM, device=DEV, dtype=torch.bfloat16)

    clean = _prefill(op, _make_pool(), monkeypatch, x, slot=1)
    pool.conv_states[0, 2] = 7.5  # stale history in the slot the fresh request takes
    reused = _prefill(op, pool, monkeypatch, x, slot=2)
    assert torch.equal(reused, clean)


def test_track_snapshot_is_the_state_at_the_aligned_boundary(monkeypatch):
    """The hybrid-radix snapshot at the deepest x64 boundary equals an independent 64-token run,
    and restoring it (COW + ``has_initial_state``) reproduces the uninterrupted continuation."""
    from freetoken.kernel.fla.index import prepare_chunk_offsets

    op = _make_op(seed=13)
    pool = _make_pool(num_slots=6)
    total = 100
    torch.manual_seed(14)
    x = torch.randn(total, CONV_DIM, device=DEV, dtype=torch.bfloat16)

    # ground truth: the state after exactly CHUNK tokens
    ref_pool = _make_pool()
    _prefill(op, ref_pool, monkeypatch, x[:CHUNK], slot=3)
    true_state = ref_pool.conv_states[0, 3].clone()

    km1 = KERNEL - 1
    cu_host = torch.tensor([0, total], dtype=torch.int64)
    boh = prepare_chunk_offsets(cu_host, CHUNK).tolist()
    c = (total - 1) // CHUNK  # deepest mid-chunk boundary -> position 64
    assert c * CHUNK == CHUNK
    track = (
        torch.tensor([2], dtype=torch.int64, device=DEV),
        torch.tensor([boh[0] + c], dtype=torch.int64, device=DEV),
        torch.tensor([[c * CHUNK - km1 + j for j in range(km1)]], dtype=torch.int64, device=DEV),
    )
    out_full = _prefill(op, pool, monkeypatch, x, slot=1, track=track)

    _assert_close(pool.conv_states[0, 2], true_state, "track snapshot")
    _assert_close(out_full, _reference(op, x)[0], "tracked prefill out")

    # restore: snapshot -> fresh live slot, continue [64, 100) as a continuation
    pool.copy_from(2, 4)
    out_cont = _prefill(op, pool, monkeypatch, x[CHUNK:], slot=4, t0=CHUNK, has_init=True)
    _assert_close(out_cont, out_full[CHUNK:], "restored continuation")
    _assert_close(pool.conv_states[0, 4], pool.conv_states[0, 1], "final state")
