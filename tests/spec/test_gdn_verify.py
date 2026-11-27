"""The GDN verify kernels: the fused recurrence's verify mode stashes each row's inputs, and gdn_state_replay over the first n rows reproduces decode mode bitwise for every n.
Also the recurrence kernel with the b|a projection in it, against the kernel handed that projection's output.

The layer-level verify (outputs, live state, commit) is checked against the reference in tests/layers/test_gdn.py.
"""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NK, NV, HD = 2, 4, 32
STEPS = 5  # verify width: k drafts + 1


@cuda
# the second geometry launches more layer x row x head programs than one grid axis holds (48 * 29 * 48 > 65535)
@pytest.mark.parametrize("layers,bs,h,hv,dim", [(2, 3, NK, NV, HD), (48, 29, 16, 48, 16)], ids=["small", "past-grid-axis-limit"])
def test_kernel_verify_stash_replay_matches_decode_bitwise(layers, bs, h, hv, dim):
    """The fused kernel's verify mode (disable_state_update + input stash) followed by
    gdn_state_replay over the first n tokens must reproduce decode mode BITWISE on
    identical per-token inputs, for every n, across layers and with verify rows that
    differ from the live slots; the verify itself must leave the live slab untouched."""
    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update as fused
    from freetoken.kernel.fla import gdn_state_replay

    torch.manual_seed(0)
    dev = torch.device("cuda")
    steps, kd, vd = STEPS, dim, dim
    slots = [(i + 2) % bs for i in range(bs)]  # verify row i commits into live slot slots[i]

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
              disable_state_update=True, input_stash=tuple(t[l] for t in stash), **common)
    assert torch.equal(st_verify, state0)  # live slab untouched

    for n in range(1, steps + 1):
        st = state0.clone()
        meta = torch.tensor([list(range(bs)), slots, [n] * bs], dtype=torch.int32, device=dev)
        gdn_state_replay(st, stash, meta)
        assert torch.equal(st[:, slots], per_step[n - 1][:, slots]), f"n={n}"
        assert torch.equal(st[:, bs], state0[:, bs])  # the slot no row commits to


def _assert_same_bits(out, ref, what):
    """Equal bits of a bf16 or fp32 tensor; a NaN matches a NaN of any payload."""
    assert out.shape == ref.shape and out.dtype == ref.dtype, f"{what}: {out.shape} {out.dtype} against {ref.shape} {ref.dtype}"
    as_int = torch.int16 if out.dtype == torch.bfloat16 else torch.int32
    differ = (out.contiguous().view(as_int) != ref.contiguous().view(as_int)) & ~(out.isnan() & ref.isnan())
    if bool(differ.any()):
        first = tuple(int(i) for i in differ.nonzero()[0])
        raise AssertionError(
            f"{what}: {int(differ.sum())} of {out.numel()} elements differ, first at {first}: "
            f"fused {float(out[first])!r}, unfused {float(ref[first])!r}"
        )


def _exact_ba(tokens, hv, hidden, gen, dev, pad=40):
    """Hidden states of k * 2^-4 and a b|a weight of k * 2^-6 for integers |k| <= 8: every product is a multiple of
    2^-10 and every partial sum of a dot is exact in fp32, so any summation order gives the exact dot. The hidden
    states are rows of a wider buffer, as a layer's input may be. Returns (x, weight, the exact b|a rounded to bf16)."""
    x = (torch.randint(-8, 9, (tokens, pad + hidden), device=dev, generator=gen) * 2.0**-4).to(torch.bfloat16)[:, pad:]
    w = (torch.randint(-8, 9, (2 * hv, hidden), device=dev, generator=gen) * 2.0**-6).to(torch.bfloat16)
    exact = x.double() @ w.double().t()
    assert bool(((x.double().abs() @ w.double().abs().t()) * 2**10 < 2**24).all()), "a partial sum is not exact in fp32"
    ba = exact.float().to(torch.bfloat16)  # exact in fp32, so this is the one rounding
    assert bool((ba.double() != exact).any()), "no dot needs rounding to bf16"
    return x, w, ba


def _ba_common(h, hv, dim, bs, steps, gen, dev):
    def r(*shape, sc=0.5, dt=torch.bfloat16):
        return (torch.randn(*shape, device=dev, generator=gen) * sc).to(dt)

    tokens = bs * steps
    common = dict(
        A_log=r(hv, sc=0.1, dt=torch.float32), dt_bias=r(hv, sc=0.1, dt=torch.float32), softplus_beta=1.0,
        softplus_threshold=20.0, q=r(1, tokens, h, dim), k=r(1, tokens, h, dim), v=r(1, tokens, hv, dim),
        initial_state_indices=torch.tensor([(i + 1) % bs for i in range(bs)], dtype=torch.int32, device=dev),
        scale=dim ** -0.5, use_qk_l2norm_in_kernel=True,
        cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=dev) * steps, disable_state_update=steps > 1,
    )
    state0 = torch.randn(bs + 1, hv, dim, dim, device=dev, generator=gen) * 0.1
    return common, state0


def _stash(bs, steps, h, hv, dim, dev):
    return tuple(torch.zeros(bs, steps, *shape, device=dev) for shape in ((h, dim), (hv, dim), (hv,), (hv,)))


@cuda
@pytest.mark.parametrize("hidden", [2048, 200])
@pytest.mark.parametrize("bs,steps", [(1, 1), (4, 1), (2, 4), (1, 6)])
@pytest.mark.parametrize("h,hv,dim", [(16, 32, 128), (1, 5, 32)], ids=["qwen3.6-35b", "one-program-a-head"])
def test_kernel_ba_fold_is_the_projection_bitwise_on_exact_dots(h, hv, dim, bs, steps, hidden):
    """The recurrence kernel taking b|a from the hidden states and the b|a weight, against the kernel handed the
    projection's output, on dots every summation order gets exactly: the b|a it used, its output, the state (decode)
    and the stash (verify, several tokens a sequence) are the same bits. Twice, with the b|a it used written out and
    without."""
    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update as fused

    dev = torch.device("cuda")
    tokens = bs * steps
    gen = torch.Generator(device="cuda").manual_seed(hidden + 131 * tokens + hv)
    common, state0 = _ba_common(h, hv, dim, bs, steps, gen, dev)
    x, w, ba = _exact_ba(tokens, hv, hidden, gen, dev)
    verify = steps > 1

    state_ref, stash_ref = state0.clone(), _stash(bs, steps, h, hv, dim, dev)
    # b and a as the layer splits them: strided column views of the projection's output
    ref = fused(a=ba[:, hv:], b=ba[:, :hv], initial_state_source=state_ref, input_stash=stash_ref if verify else None, **common)
    for store in (True, False):
        state, stash = state0.clone(), _stash(bs, steps, h, hv, dim, dev)
        used = torch.full((tokens, 2 * hv), float("nan"), dtype=torch.bfloat16, device=dev)
        out = fused(
            a=None, b=None, initial_state_source=state, input_stash=stash if verify else None,
            ba_fold=(x, w, used) if store else (x, w), **common,
        )
        what = f"{bs} sequences of {steps}, {hv} heads of {dim}, hidden {hidden}, b|a written {store}"
        if store:
            _assert_same_bits(used, ba, f"the b|a the kernel used, {what}")
        _assert_same_bits(out, ref, f"the output, {what}")
        _assert_same_bits(state.reshape(-1, dim), state_ref.reshape(-1, dim), f"the recurrent state, {what}")
        if verify:
            for name, s, s_ref in zip(("k", "v", "g", "beta"), stash, stash_ref):
                _assert_same_bits(s, s_ref, f"the stashed {name}, {what}")


@cuda
@pytest.mark.parametrize("bs,steps", [(1, 1), (1, 6), (8, 6), (64, 1)])
def test_kernel_ba_fold_changes_only_the_dots_order_and_replays_in_a_graph(bs, steps):
    """On the Qwen3.6-35B shape with gate-sized random b|a dots: the kernel handed the b|a the fold used gives the
    fold's bits (only the dots moved into the kernel), that b|a is within one bf16 step plus any fp32 order's error
    of the fp64 dot, and a CUDA-graph replay of the fold is the eager launch, bit for bit."""
    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update as fused

    dev = torch.device("cuda")
    h, hv, dim, hidden = 16, 32, 128, 2048
    tokens = bs * steps
    verify = steps > 1
    gen = torch.Generator(device="cuda").manual_seed(7 + tokens)
    common, state0 = _ba_common(h, hv, dim, bs, steps, gen, dev)
    x = torch.randn(tokens, hidden, device=dev, generator=gen).to(torch.bfloat16)
    w = (torch.randn(2 * hv, hidden, device=dev, generator=gen) * 0.02).to(torch.bfloat16)

    def run(state, stash, used):
        return fused(a=None, b=None, initial_state_source=state, input_stash=stash if verify else None, ba_fold=(x, w, used), **common)

    state, stash = state0.clone(), _stash(bs, steps, h, hv, dim, dev)
    used = torch.empty(tokens, 2 * hv, dtype=torch.bfloat16, device=dev)
    out = run(state, stash, used)

    state_ref, stash_ref = state0.clone(), _stash(bs, steps, h, hv, dim, dev)
    ref = fused(a=used[:, hv:], b=used[:, :hv], initial_state_source=state_ref, input_stash=stash_ref if verify else None, **common)
    what = f"{bs} sequences of {steps}"
    _assert_same_bits(out, ref, f"the kernel on the fold's own b|a, {what}")
    _assert_same_bits(state.reshape(-1, dim), state_ref.reshape(-1, dim), f"the recurrent state, {what}")
    for s, s_ref in zip(stash, stash_ref):
        _assert_same_bits(s, s_ref, f"the stash, {what}")

    exact = x.double() @ w.double().t()
    got = used.double()
    big = torch.maximum(got.abs(), exact.abs())
    step = torch.ldexp(torch.ones_like(big), torch.frexp(big).exponent - 8)
    order = 2 * hidden * 2.0**-24 * (x.double().abs() @ w.double().abs().t())
    off = (got - exact).abs() > step + order
    assert not bool(off.any()), f"{what}: b|a {got[off][:4].tolist()} against the fp64 dot's {exact[off][:4].tolist()}"

    state_g, stash_g = state0.clone(), _stash(bs, steps, h, hv, dim, dev)
    used_g = torch.empty_like(used)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_g = run(state_g, stash_g, used_g)
    state_g.copy_(state0)
    for s in stash_g:
        s.zero_()
    used_g.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_same_bits(used_g, used, f"the graph's b|a, {what}")
    _assert_same_bits(out_g, out, f"the graph's output, {what}")
    _assert_same_bits(state_g.reshape(-1, dim), state.reshape(-1, dim), f"the graph's state, {what}")
    for s, s_ref in zip(stash_g, stash):
        _assert_same_bits(s, s_ref, f"the graph's stash, {what}")


def test_gdn_says_once_why_decode_does_not_take_b_and_a_in_the_kernel(monkeypatch):
    """One line a process, at the first decode step, naming the predicate that keeps the b|a projection's launch."""
    from types import SimpleNamespace

    from freetoken.layers.linear_attn import gdn
    from freetoken.layers.quantization.linear.unquantized import TorchLinearKernel

    said = []
    monkeypatch.setattr(gdn, "_BA_SAID", False)
    monkeypatch.setattr(gdn, "_BA_IN_KERNEL", True)
    monkeypatch.setattr(gdn.logger, "info_rank0", lambda msg, *a, **k: said.append(msg), raising=False)

    def layer(weight, bias=None, split=True):
        op = object.__new__(gdn.GatedDeltaNet)
        op._split_in_proj, op.num_v_heads = split, 4
        op.in_proj_ba = SimpleNamespace(weight=weight, bias=bias, quant_method=SimpleNamespace(kernel=TorchLinearKernel()))
        op._ba_weight, op._ba_off = False, None
        return op

    x = torch.zeros(1, 64, dtype=torch.bfloat16)
    assert layer(torch.zeros(8, 64, dtype=torch.bfloat16))._ba_weight_in_kernel(x) is None
    assert layer(torch.zeros(8, 64, dtype=torch.bfloat16), bias=torch.zeros(8))._ba_weight_in_kernel(x) is None
    assert said == ["gdn b|a in the recurrence kernel on decode: off (the b|a weight is on cpu)"]
    assert layer(torch.zeros(8, 64), bias=torch.zeros(8))._ba_weight_off(64) == "the b|a projection has a bias"
    assert layer(torch.zeros(8, 64), split=False)._ba_weight_off(64) == "b and a come out of the one input projection"
    assert layer(torch.zeros(8, 64))._ba_weight_off(64) == "the b|a weight is on cpu"
    monkeypatch.setattr(gdn, "_BA_IN_KERNEL", False)
    assert layer(torch.zeros(8, 64))._ba_weight_off(64) == "FREETOKEN_GDN_BA_IN_KERNEL=0"
    # a layer that folds decides again once in_proj_ba holds another weight tensor
    op = layer(torch.zeros(8, 64, dtype=torch.bfloat16))
    monkeypatch.setattr(op, "_ba_weight_off", lambda hidden: None)
    assert op._ba_weight_in_kernel(x) is op.in_proj_ba.weight
    op.in_proj_ba.weight = swapped = torch.zeros(8, 64, dtype=torch.bfloat16)
    assert op._ba_weight_in_kernel(x) is swapped
