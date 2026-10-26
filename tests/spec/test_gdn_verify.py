"""GDN target-verify numerical equivalence.

The reference is the production DECODE path run token by token (same op, same
pool); the verify path must (a) produce the same per-token outputs, (b) leave the
live recurrent state bitwise untouched until commit (disable_state_update), and
(c) after LinearVerifyScratch.commit leave live SSM+conv state matching the
reference after exactly n tokens -- for every possible accept count.
"""

from types import SimpleNamespace

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

BS, K_DRAFT = 3, 4          # verify width = K_DRAFT + 1
HIDDEN, NK, NV, HD, CONV_K = 64, 2, 4, 32, 4
STEPS = K_DRAFT + 1


def _make_pool(device):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.config import LinearGatedDeltaGroupConfig

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,),
        num_key_heads=NK, num_value_heads=NV,
        key_head_dim=HD, value_head_dim=HD,
        conv_kernel_dim=CONV_K, output_gate=True,
    )
    return LinearStatePool(group, num_slots=BS + 1, dtype=torch.bfloat16,
                           device=device, tp_size=1)


def _make_op(device):
    from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet
    from freetoken.utils import torch_dtype

    with torch.device(device), torch_dtype(torch.bfloat16):
        op = Qwen3_5GatedDeltaNet(
            hidden_size=HIDDEN, num_k_heads=NK, num_v_heads=NV,
            head_k_dim=HD, head_v_dim=HD, conv_kernel_size=CONV_K,
            rms_norm_eps=1e-6, layer_id=0,
        )
    g = torch.Generator(device="cpu").manual_seed(11)
    sd = op.state_dict()
    for name, p in sd.items():
        fresh = torch.randn(p.shape, generator=g, dtype=torch.float32) * 0.05
        sd[name] = fresh.to(dtype=p.dtype, device=device)
    op.load_state_dict(sd)
    return op


def _ctx(monkeypatch, pool):
    from freetoken import core

    ctx = SimpleNamespace(linear_state_pool=pool, batch=None)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    return ctx


def _decode_batch(device):
    from freetoken.attention.linear import FLAMetadata

    slots = torch.arange(BS, dtype=torch.int32, device=device)
    return SimpleNamespace(
        phase="decode", is_decode=True, is_prefill=False,
        fla_metadata=FLAMetadata(
            cu_seqlens=torch.arange(BS + 1, dtype=torch.int32, device=device),
            cache_indices=slots,
        ),
        verify=None,
    )


def _verify_batch(device, gdn_scratch):
    from freetoken.attention.linear import FLAMetadata

    slots = torch.arange(BS, dtype=torch.int32, device=device)
    return SimpleNamespace(
        phase="verify", is_decode=False, is_prefill=False,
        fla_metadata=FLAMetadata(
            cu_seqlens=torch.arange(BS + 1, dtype=torch.int32, device=device) * STEPS,
            cache_indices=slots,
            has_initial_state=torch.ones(BS, dtype=torch.bool, device=device),
            intermediate_state_indices=torch.arange(BS, dtype=torch.int32, device=device),
        ),
        verify=SimpleNamespace(gdn=gdn_scratch),
    )


@cuda
@pytest.mark.parametrize("num_emitted", [1, 2, STEPS])
def test_verify_matches_stepwise_decode(monkeypatch, num_emitted):

    device = torch.device("cuda")
    torch.manual_seed(3)
    pool = _make_pool(device)
    op = _make_op(device)
    ctx = _ctx(monkeypatch, pool)

    # Non-trivial initial state (mid-generation), identical for both runs.
    init_rec = torch.randn_like(pool.recurrent_states) * 0.1
    init_conv = (torch.randn(pool.conv_states.shape, dtype=torch.float32,
                             device=device) * 0.1).to(pool.conv_states.dtype)
    # Same hidden rows for both paths: req i, token t -> hidden[i, t].
    hidden = torch.randn(BS, STEPS, HIDDEN, dtype=torch.bfloat16, device=device) * 0.5

    # ---- reference: production decode path, one token per step ----
    pool.recurrent_states.copy_(init_rec)
    pool.conv_states.copy_(init_conv)
    ctx.batch = _decode_batch(device)
    ref_out = []
    ref_rec, ref_conv = [], []
    for t in range(STEPS):
        ref_out.append(op.forward(hidden[:, t]))
        ref_rec.append(pool.recurrent_states[:, :BS].clone())
        ref_conv.append(pool.conv_states[:, :BS].clone())
    ref_out = torch.stack(ref_out, dim=1)  # [BS, STEPS, HIDDEN]

    # ---- verify path: one forward over the packed k+1 rows ----
    pool.recurrent_states.copy_(init_rec)
    pool.conv_states.copy_(init_conv)
    scratch = pool.verify_scratch(BS, STEPS)
    ctx.batch = _verify_batch(device, scratch)
    packed = hidden.reshape(BS * STEPS, HIDDEN)
    ver_out = op.forward(packed).reshape(BS, STEPS, HIDDEN)

    torch.testing.assert_close(ver_out, ref_out, rtol=2e-2, atol=2e-2)

    # Live SSM state must be bitwise untouched before commit (disable_state_update).
    assert torch.equal(pool.recurrent_states, init_rec)

    # ---- commit at num_emitted: live state == reference after num_emitted tokens ----
    # Loose tolerance by design: the two paths run the in_proj GEMM at different M
    # (batch-shape-dependent bf16 tiling), so states drift at bf16 noise level. The
    # bitwise semantics (mode equivalence, [V,K] raw layout, step indexing) are pinned
    # by test_kernel_verify_stash_replay_matches_decode_bitwise below.
    pool.commit_verify(scratch, list(range(BS)), [num_emitted] * BS, list(range(BS)))
    torch.testing.assert_close(
        pool.recurrent_states[:, :BS], ref_rec[num_emitted - 1],
        rtol=5e-2, atol=5e-2,
    )
    torch.testing.assert_close(
        pool.conv_states[:, :BS].float(), ref_conv[num_emitted - 1].float(),
        rtol=5e-2, atol=5e-2,
    )


@cuda
def test_kernel_verify_stash_replay_matches_decode_bitwise():
    """The fused kernel's verify mode (disable_state_update + input stash) followed by
    gdn_state_replay over the first n tokens must reproduce decode mode BITWISE on
    identical per-token inputs, for every n, across layers and with verify rows that
    differ from the live slots; the verify itself must leave the live slab untouched."""
    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update as fused
    from freetoken.kernel.fla import gdn_state_replay

    torch.manual_seed(0)
    dev = torch.device("cuda")
    layers, bs, steps, h, hv, kd, vd = 2, 3, STEPS, NK, NV, HD, HD
    slots = [2, 0, 1]  # verify row i commits into live slot slots[i]

    def r(*s, dt=torch.bfloat16, sc=0.5):
        return (torch.randn(*s, device=dev) * sc).to(dt)

    q, k, v = r(layers, 1, bs * steps, h, kd), r(layers, 1, bs * steps, h, kd), r(layers, 1, bs * steps, hv, vd)
    a, b = r(layers, bs * steps, hv), r(layers, bs * steps, hv)
    a_log, dt_bias = r(layers, hv, dt=torch.float32, sc=0.1), r(layers, hv, dt=torch.float32, sc=0.1)
    state0 = torch.randn(layers, bs + 1, hv, kd, vd, device=dev) * 0.1
    slot_t = torch.tensor(slots, dtype=torch.int32, device=dev)
    common = dict(softplus_beta=1.0, softplus_threshold=20.0, scale=kd ** -0.5,
                  use_qk_l2norm_in_kernel=True, initial_state_indices=slot_t)

    per_step = []  # per_step[t][l] = live slab after tokens 0..t
    st_decode = state0.clone()
    for t in range(steps):
        rows = torch.arange(bs, device=dev) * steps + t
        for l in range(layers):
            fused(A_log=a_log[l], dt_bias=dt_bias[l], a=a[l][rows], q=q[l][:, rows], k=k[l][:, rows],
                  v=v[l][:, rows], b=b[l][rows], initial_state_source=st_decode[l],
                  cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=dev), **common)
        per_step.append(st_decode.clone())

    stash = tuple(torch.zeros(layers, bs, steps, *shape, device=dev)
                  for shape in ((h, kd), (hv, vd), (hv,), (hv,)))
    st_verify = state0.clone()
    for l in range(layers):
        fused(A_log=a_log[l], dt_bias=dt_bias[l], a=a[l], q=q[l], k=k[l], v=v[l], b=b[l],
              initial_state_source=st_verify[l],
              cu_seqlens=torch.arange(bs + 1, dtype=torch.int64, device=dev) * steps,
              disable_state_update=True, input_stash=tuple(t[l] for t in stash),
              intermediate_state_indices=torch.arange(bs, dtype=torch.int32, device=dev), **common)
    assert torch.equal(st_verify, state0)  # live slab untouched

    for n in range(1, steps + 1):
        st = state0.clone()
        meta = torch.tensor([list(range(bs)), slots, [n] * bs], dtype=torch.int32, device=dev)
        gdn_state_replay(st, stash, meta)
        assert torch.equal(st[:, slots], per_step[n - 1][:, slots]), f"n={n}"
        assert torch.equal(st[:, bs], state0[:, bs])  # the slot no row commits to
