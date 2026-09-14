"""qwen4_exp GatedDeltaNet op vs the pure-torch HF reference math.

The oracle is ``models/qwen4_exp/gdn_reference.py``, whose two delta rules and forward are
transcribed from the ``modeling_qwen4_exp.py`` snapshot, so no transformers build carrying
qwen4_exp is needed here. Covered: prefill at 128 and 1000 tokens, a decode step continuing
from the prefill state, ragged bs=3, both GQA head ratios, and the sigmoid output gate.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.core import Batch, Context, Req, SamplingParams
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.models.qwen4_exp.gdn import Qwen4ExpGatedDeltaNet
from freetoken.models.qwen4_exp.gdn_reference import Qwen4ExpGatedDeltaNetReference
from freetoken.utils import torch_dtype

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = torch.device("cuda")
HIDDEN, HEAD_DIM, CONV_K, EPS = 256, 128, 4, 1e-6
RTOL = ATOL = 2e-2
# (num_k_heads, num_v_heads) per value:key head ratio; 3:1 is the Qwen3.8-Flash-Next shape.
HEADS = {2: (8, 16), 3: (16, 48)}


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


def _make_layer(ratio: int, output_gate: str = "sigmoid", seed: int = 0):
    """fp32 reference + bf16 kernel op over one set of weights. The op is built on meta under
    the serving dtype, the way the engine builds a model, so load_state_dict's dtype check bites."""
    num_k, num_v = HEADS[ratio]
    torch.manual_seed(seed)
    ref = Qwen4ExpGatedDeltaNetReference(
        hidden_size=HIDDEN, num_k_heads=num_k, num_v_heads=num_v, head_k_dim=HEAD_DIM,
        head_v_dim=HEAD_DIM, conv_kernel_size=CONV_K, rms_norm_eps=EPS, output_gate=output_gate,
    ).to(DEV).float().eval()
    with torch.no_grad():
        # HF inits A_log = log(U(0.01, 16)); a zero dt_bias or a unit gate norm would hide sign errors.
        ref.A_log.uniform_(0.01, 16.0).log_()
        ref.dt_bias.uniform_(-1.0, 1.0)
        ref.norm.weight.normal_(1.0, 0.1)
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        op = Qwen4ExpGatedDeltaNet(
            hidden_size=HIDDEN, num_k_heads=num_k, num_v_heads=num_v, head_k_dim=HEAD_DIM,
            head_v_dim=HEAD_DIM, conv_kernel_size=CONV_K, rms_norm_eps=EPS, layer_id=0,
            output_gate=output_gate,
        )
    op.load_state_dict(_state_dict(ref))
    return op, ref


def _ctx(ratio: int, num_slots: int = 8) -> Context:
    import freetoken.core as core
    from freetoken.kvcache.linear_state_pool import LinearStatePool

    num_k, num_v = HEADS[ratio]
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=num_k, num_value_heads=num_v,
        key_head_dim=HEAD_DIM, value_head_dim=HEAD_DIM, conv_kernel_dim=CONV_K,
        output_gate="sigmoid",
    )
    core._GLOBAL_CTX = None
    ctx = Context(page_size=64)
    ctx.linear_state_pool = LinearStatePool(group, num_slots, torch.bfloat16, DEV, tp_size=1)
    core.set_global_ctx(ctx)
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


def _decode(op, ctx: Context, reqs, hidden: torch.Tensor) -> torch.Tensor:
    batch = Batch(reqs=reqs, phase="decode")
    batch.padded_reqs = reqs
    batch.linear_table_idx = torch.tensor(
        [r.table_idx for r in reqs], dtype=torch.int32, device=DEV
    )
    with ctx.forward_batch(batch):
        return op.forward(hidden)


@torch.no_grad()
def _ref_out(ref, hidden: torch.Tensor, use_chunk_rule: bool = False) -> torch.Tensor:
    return ref(hidden.float().unsqueeze(0), use_chunk_rule=use_chunk_rule)[0]


@pytest.mark.parametrize("length", (1000,))
@pytest.mark.parametrize("ratio", (2, 3))
def test_prefill_matches_reference(ratio, length):
    op, ref = _make_layer(ratio, seed=ratio)
    hidden, _, out = _prefill(op, _ctx(ratio), [length], seed=11)
    torch.testing.assert_close(out.float(), _ref_out(ref, hidden[0]), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("ratio", (2, 3))
def test_ragged_prefill_then_decode(ratio):
    """bs=3 ragged prefill, then one decode step per request off the carried conv + recurrent
    state. The decode oracle is the whole (prefill + 1) sequence in one reference pass, so a
    state that did not survive the prefill shows up immediately."""
    op, ref = _make_layer(ratio, seed=ratio)
    ctx = _ctx(ratio)
    lengths = [128, 1000, 37]
    hidden, reqs, out = _prefill(op, ctx, lengths, seed=13)

    off = 0
    for h, n in zip(hidden, lengths):
        torch.testing.assert_close(
            out[off:off + n].float(), _ref_out(ref, h), rtol=RTOL, atol=ATOL
        )
        off += n

    nxt = torch.randn(len(lengths), HIDDEN, device=DEV, dtype=torch.bfloat16)
    dec = _decode(op, ctx, reqs, nxt)
    for i, h in enumerate(hidden):
        full = _ref_out(ref, torch.cat([h, nxt[i:i + 1]], dim=0))
        torch.testing.assert_close(dec[i].float(), full[-1], rtol=RTOL, atol=ATOL)


def test_chunk_and_recurrent_rules_agree():
    """The chunked form (what the fla prefill kernel implements) against the sequential
    definition, both fp32: the chunk oracle is only worth anything if it reproduces the
    recurrence to fp32 precision."""
    _, ref = _make_layer(3, seed=1)
    torch.manual_seed(17)
    hidden = torch.randn(1000, HIDDEN, device=DEV, dtype=torch.bfloat16)
    torch.testing.assert_close(
        _ref_out(ref, hidden, use_chunk_rule=True), _ref_out(ref, hidden), rtol=1e-4, atol=1e-4
    )


def test_output_gate_comes_from_the_config():
    """The gate activation is the group config's string, not a hardcoded silu. Both gates track
    their own reference, and the two are far apart -- so a stuck activation cannot pass."""
    op_silu, ref_silu = _make_layer(3, output_gate="silu", seed=2)
    hidden, _, out_silu = _prefill(op_silu, _ctx(3), [128], seed=19)
    torch.testing.assert_close(out_silu.float(), _ref_out(ref_silu, hidden[0]), rtol=RTOL, atol=ATOL)

    op_sig, ref_sig = _make_layer(3, output_gate="sigmoid", seed=2)
    _, _, out_sig = _prefill(op_sig, _ctx(3), [128], seed=19)
    torch.testing.assert_close(out_sig.float(), _ref_out(ref_sig, hidden[0]), rtol=RTOL, atol=ATOL)

    assert (out_sig.float() - out_silu.float()).abs().max().item() > 10 * ATOL
