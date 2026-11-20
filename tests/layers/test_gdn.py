"""The GatedDeltaNet layer against the pure-torch HF reference math (``layers/linear_attn/gdn_reference.py``): a ragged bs=3 prefill (up to 1000 tokens), decode steps and a speculative verify off the carried state, for both GQA head ratios and both output gates."""

from __future__ import annotations

import pytest
import torch

from freetoken.core import Batch, Context, Req, SamplingParams
from freetoken.layers.linear_attn import GatedDeltaNet
from freetoken.layers.linear_attn.gdn_reference import GatedDeltaNetReference
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.utils import torch_dtype

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = torch.device("cuda")
HIDDEN, HEAD_DIM, CONV_K, EPS = 256, 128, 4, 1e-6
RTOL = ATOL = 2e-2
# (num_k_heads, num_v_heads) per value:key head ratio; 3:1 is the Qwen3.8-Flash-Next shape.
HEADS = {2: (8, 16), 3: (16, 48)}
K = 3  # drafts per verify step


def _bf(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(DEV, torch.bfloat16)


def _state_dict(ref) -> dict[str, torch.Tensor]:
    """HF's four in_proj matrices fused into the op's single qkv|z|b|a GEMM. A_log / dt_bias
    stay fp32, as the weight loader keeps them."""
    return {
        "in_proj.weight": _bf(torch.cat([ref.in_proj_qkv.weight, ref.in_proj_z.weight,
                                         ref.in_proj_b.weight, ref.in_proj_a.weight], dim=0)),
        "conv1d.weight": _bf(ref.conv1d.weight),
        "dt_bias": ref.dt_bias.detach().to(DEV, torch.float32),
        "A_log": ref.A_log.detach().to(DEV, torch.float32),
        "norm.weight": _bf(ref.norm.weight),
        "out_proj.weight": _bf(ref.out_proj.weight),
    }


def _make_layer(ratio: int, output_gate: str, seed: int = 0):
    """fp32 reference + bf16 kernel op over one set of weights. The op is built on meta under
    the serving dtype, the way the engine builds a model, so load_state_dict's dtype check bites."""
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    num_k, num_v = HEADS[ratio]
    torch.manual_seed(seed)
    ref = GatedDeltaNetReference(
        hidden_size=HIDDEN, num_k_heads=num_k, num_v_heads=num_v, head_k_dim=HEAD_DIM,
        head_v_dim=HEAD_DIM, conv_kernel_size=CONV_K, rms_norm_eps=EPS, output_gate=output_gate,
    ).to(DEV).float().eval()
    with torch.no_grad():
        # HF inits A_log = log(U(0.01, 16)); a zero dt_bias or a unit gate norm would hide sign errors.
        ref.A_log.uniform_(0.01, 16.0).log_()
        ref.dt_bias.uniform_(-1.0, 1.0)
        ref.norm.weight.normal_(1.0, 0.1)
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        op = GatedDeltaNet(
            hidden_size=HIDDEN, num_k_heads=num_k, num_v_heads=num_v, head_k_dim=HEAD_DIM,
            head_v_dim=HEAD_DIM, conv_kernel_size=CONV_K, rms_norm_eps=EPS, layer_id=0,
            output_gate=output_gate,
        )
    op.load_state_dict(_state_dict(ref))
    return op, ref


def _ctx(monkeypatch, ratio: int, num_slots: int = 8) -> Context:
    import freetoken.core as core
    from freetoken.kvcache.linear_state_pool import LinearStatePool

    num_k, num_v = HEADS[ratio]
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=num_k, num_value_heads=num_v,
        key_head_dim=HEAD_DIM, value_head_dim=HEAD_DIM, conv_kernel_dim=CONV_K,
        output_gate="sigmoid",
    )
    ctx = Context(page_size=64)
    ctx.linear_state_pool = LinearStatePool(group, num_slots, torch.bfloat16, DEV, tp_size=1)
    ctx.linear_state_pool.attach_verify_scratch(num_slots, K + 1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    return ctx


def _prefill(op, ctx: Context, lengths: list[int], seed: int):
    """One ragged prefill batch, one state slot per request. Returns the per-request hidden
    states, the reqs (for a follow-up decode) and the packed output."""
    torch.manual_seed(seed)
    hidden = [torch.randn(n, HIDDEN, device=DEV, dtype=torch.bfloat16) for n in lengths]
    reqs = [
        Req(input_ids=torch.zeros(n, dtype=torch.int32), table_idx=i + 1, cached_len=0,
            output_len=1, uid=i, sampling_params=SamplingParams(), cache_handle=None)
        for i, n in enumerate(lengths)
    ]
    batch = Batch(reqs=reqs, phase="prefill")
    batch.padded_reqs = reqs
    with ctx.forward_batch(batch):
        out = op.forward(torch.cat(hidden, dim=0))
    return hidden, reqs, out


def _step(op, ctx: Context, reqs, hidden: torch.Tensor, phase: str = "decode", rows: int = 1) -> torch.Tensor:
    batch = Batch(reqs=reqs, phase=phase, rows_per_req=rows)
    batch.padded_reqs = reqs
    with ctx.forward_batch(batch):
        return op.forward(hidden)


@torch.no_grad()
def _ref_out(ref, hidden: torch.Tensor) -> torch.Tensor:
    return ref(hidden.float().unsqueeze(0))[0]


# 3:1 with sigmoid is Qwen3.8-Flash-Next; the sigmoid case also catches a layer that ignores output_gate (the default is silu)
@pytest.mark.parametrize("ratio,output_gate", [(2, "silu"), (3, "sigmoid")])
def test_prefill_decode_and_verify_match_reference(monkeypatch, ratio, output_gate):
    """bs=3 ragged prefill, a decode step, a verify of K+1 rows committed at a different accept count per request, then another decode step.
    Each step is checked against the reference over the request's whole sequence so far, so a state that did not survive a step, or a verify that wrote the live state, shows up in the next one."""
    op, ref = _make_layer(ratio, output_gate, seed=ratio)
    ctx = _ctx(monkeypatch, ratio)
    lengths = [128, 1000, 37]
    seqs, reqs, out = _prefill(op, ctx, lengths, seed=13)

    off = 0
    for h, n in zip(seqs, lengths):
        torch.testing.assert_close(
            out[off:off + n].float(), _ref_out(ref, h), rtol=RTOL, atol=ATOL
        )
        off += n

    nxt = torch.randn(len(lengths), HIDDEN, device=DEV, dtype=torch.bfloat16)
    dec = _step(op, ctx, reqs, nxt)
    seqs = [torch.cat([h, nxt[i:i + 1]]) for i, h in enumerate(seqs)]
    for i, h in enumerate(seqs):
        torch.testing.assert_close(dec[i].float(), _ref_out(ref, h)[-1], rtol=RTOL, atol=ATOL)

    rows = torch.randn(len(lengths), K + 1, HIDDEN, device=DEV, dtype=torch.bfloat16)
    ver = _step(op, ctx, reqs, rows.flatten(0, 1), "verify", K + 1).view(len(lengths), K + 1, -1)
    for i, h in enumerate(seqs):
        want = _ref_out(ref, torch.cat([h, rows[i]]))[-(K + 1):]
        torch.testing.assert_close(ver[i].float(), want, rtol=RTOL, atol=ATOL)
    accepted = [1, 2, K + 1]
    ctx.linear_state_pool.commit_verify([r.table_idx for r in reqs], accepted)
    seqs = [torch.cat([h, rows[i, :n]]) for i, (h, n) in enumerate(zip(seqs, accepted))]

    last = torch.randn(len(lengths), HIDDEN, device=DEV, dtype=torch.bfloat16)
    dec = _step(op, ctx, reqs, last)
    for i, h in enumerate(seqs):
        want = _ref_out(ref, torch.cat([h, last[i:i + 1]]))[-1]
        torch.testing.assert_close(dec[i].float(), want, rtol=RTOL, atol=ATOL)


def _split_ba(op, seed: int) -> None:
    """The op's input GEMM split the way a checkpoint with qkv|z quantized and b|a not loads it: a qkv|z linear and a
    bf16 b|a linear (torch bf16 both, here). The b|a weight is k * 2^-6 for integers |k| <= 8, so with hidden states
    of k * 2^-4 every partial sum of a b|a dot is exact in fp32 and any summation order gives the same bits."""
    from freetoken.layers import LinearColParallelMerged

    w, hv, cut = op.in_proj.weight, op.num_v_heads, op.conv_dim + op.value_dim
    with torch.device(DEV), torch_dtype(torch.bfloat16):
        op.in_proj_qkvz = LinearColParallelMerged(HIDDEN, [op.conv_dim, op.value_dim], has_bias=False)
        op.in_proj_ba = LinearColParallelMerged(HIDDEN, [hv, hv], has_bias=False)
    op.in_proj_qkvz.weight = w[:cut].contiguous()
    gen = torch.Generator(device=DEV).manual_seed(seed)
    op.in_proj_ba.weight = (torch.randint(-8, 9, (2 * hv, HIDDEN), device=DEV, generator=gen) * 2.0**-6).to(torch.bfloat16)
    del op.in_proj
    op._split_in_proj = True
    op._ba_weight = False


def _exact_rows(rows: int, gen: torch.Generator) -> torch.Tensor:
    return (torch.randint(-8, 9, (rows, HIDDEN), device=DEV, generator=gen) * 2.0**-4).to(torch.bfloat16)


@pytest.fixture
def cublas_bf16_fp32_partials():
    """cuBLAS keeps bf16 GEMM partials in fp32 for the test; afterwards the setting comes back whole (its split-K part
    too). monkeypatch cannot undo this flag: cuBLASModule serves it through __getattr__ and __setattr__, so there is no
    instance attribute to restore and undo() deletes one that is not there."""
    matmul = torch.backends.cuda.matmul
    was = torch._C._get_cublas_allow_bf16_reduced_precision_reduction()
    matmul.allow_bf16_reduced_precision_reduction = False
    try:
        yield
    finally:
        matmul.allow_bf16_reduced_precision_reduction = tuple(was)


@pytest.mark.usefixtures("cublas_bf16_fp32_partials")
def test_split_ba_in_the_recurrence_kernel_is_the_launched_projection_bit_for_bit_on_exact_dots(monkeypatch):
    """With b|a its own bf16 linear, a decode or verify step takes b and a inside the recurrence kernel and does not
    launch the linear; a prefill still launches it. On b|a dots every summation order gets exactly (cuBLAS kept to fp32
    partials), a prefill, a decode step, a verify committed at a different accept count per request and another
    decode step give the outputs and states of the launched linear, bit for bit."""
    import freetoken.layers.linear_attn.gdn as gdn

    assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    lengths, accepted = [128, 37, 5], [1, 2, K + 1]
    runs = []
    for fold in (True, False):
        monkeypatch.setattr(gdn, "_BA_IN_KERNEL", fold)
        op, _ = _make_layer(2, "silu", seed=4)
        _split_ba(op, seed=5)
        calls, launch = [], op.in_proj_ba.forward
        monkeypatch.setattr(op.in_proj_ba, "forward", lambda x: calls.append(x.shape[0]) or launch(x))
        ctx = _ctx(monkeypatch, 2)
        _, reqs, pre = _prefill(op, ctx, lengths, seed=13)
        gen = torch.Generator(device=DEV).manual_seed(14)
        x = _exact_rows(len(lengths), gen)
        if fold:
            exact = (x.double() @ op.in_proj_ba.weight.double().t()).float().to(torch.bfloat16)
            assert torch.equal(launch(x), exact), "F.linear does not give the exact dot here: the arms cannot be bit-equal"
        outs = [pre, _step(op, ctx, reqs, x)]
        outs.append(_step(op, ctx, reqs, _exact_rows(len(lengths) * (K + 1), gen), "verify", K + 1))
        ctx.linear_state_pool.commit_verify([r.table_idx for r in reqs], accepted)
        outs.append(_step(op, ctx, reqs, _exact_rows(len(lengths), gen)))
        pool = ctx.linear_state_pool
        runs.append((outs, pool.recurrent_states.clone(), pool.conv_states.clone(), calls, op._ba_weight, op._ba_off))
    (folded, state1, conv1, calls1, w1, off1), (launched, state0, conv0, calls0, w0, off0) = runs
    assert isinstance(w1, torch.Tensor) and off1 is None, off1
    assert w0 is None and off0 == "FREETOKEN_GDN_BA_IN_KERNEL=0", off0
    assert calls1 == [sum(lengths)], calls1
    assert calls0 == [sum(lengths), len(lengths), len(lengths) * (K + 1), len(lengths)], calls0
    for what, a, b in zip(("prefill", "decode", "verify", "decode after the commit"), folded, launched):
        assert torch.equal(a, b), what
    assert float(state1.float().norm()) > 0
    assert torch.equal(state1, state0) and torch.equal(conv1, conv0)
