"""The GatedDeltaNet layer against the pure-torch HF reference math (``layers/linear_attn/gdn_reference.py``): a ragged bs=3 prefill (up to 1000 tokens) and one decode step off the carried state, for both GQA head ratios and both output gates."""

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


def _decode(op, ctx: Context, reqs, hidden: torch.Tensor) -> torch.Tensor:
    batch = Batch(reqs=reqs, phase="decode")
    batch.padded_reqs = reqs
    with ctx.forward_batch(batch):
        return op.forward(hidden)


@torch.no_grad()
def _ref_out(ref, hidden: torch.Tensor) -> torch.Tensor:
    return ref(hidden.float().unsqueeze(0))[0]


# 3:1 with sigmoid is Qwen3.8-Flash-Next; the sigmoid case also catches a layer that ignores output_gate (the default is silu)
@pytest.mark.parametrize("ratio,output_gate", [(2, "silu"), (3, "sigmoid")])
def test_ragged_prefill_then_decode(monkeypatch, ratio, output_gate):
    """bs=3 ragged prefill, then one decode step per request off the carried conv + recurrent
    state. The decode oracle is the whole (prefill + 1) sequence in one reference pass, so a
    state that did not survive the prefill shows up immediately."""
    op, ref = _make_layer(ratio, output_gate, seed=ratio)
    ctx = _ctx(monkeypatch, ratio)
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
