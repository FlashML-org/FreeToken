"""mHC (Manifold-Constrained Hyper-Connections) unit tests.

Checks the algebraic contracts of layers/mhc.py at GLM-5.3 geometry (n=4):
Sinkhorn projection yields an (approximately) doubly-stochastic comb matrix,
pre/post mixing matches naive per-token einsums, and identity-ish weights give
the classic single-stream residual behaviour.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.layers.mhc import (
    hc_contract,
    hc_expand,
    mhc_fused_post_pre,
    mhc_post,
    mhc_pre,
)

N, HIDDEN, T = 4, 64, 9
MIX = 2 * N + N * N
EPS = 1e-6
RMS_EPS = 1e-5
POST_MULT = 2.0
SINKHORN = 20


def _weights(seed=0, device="cpu"):
    torch.manual_seed(seed)
    fn = torch.randn(MIX, N * HIDDEN, dtype=torch.float32, device=device) * 0.05
    scale = torch.randn(3, dtype=torch.float32, device=device).abs() + 0.5
    base = torch.randn(MIX, dtype=torch.float32, device=device) * 0.3
    return fn, scale, base


def _residual(seed=1, device="cpu"):
    torch.manual_seed(seed)
    return torch.randn(T, N, HIDDEN, dtype=torch.bfloat16, device=device)


def test_comb_is_doubly_stochastic():
    fn, scale, base = _weights(seed=2)
    res = _residual(seed=3)
    _, comb, _ = mhc_pre(res, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN)
    rows = comb.sum(dim=-1)
    cols = comb.sum(dim=-2)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-3)
    assert torch.allclose(cols, torch.ones_like(cols), atol=1e-3)
    assert (comb > 0).all()


def test_post_matches_naive():
    res = _residual(seed=4)
    x = torch.randn(T, HIDDEN, dtype=torch.bfloat16)
    post = torch.rand(T, N, 1, dtype=torch.float32) * POST_MULT
    comb = torch.softmax(torch.randn(T, N, N), dim=-1)
    out = mhc_post(x, res, post, comb)
    assert out.shape == (T, N, HIDDEN)

    ref = torch.zeros(T, N, HIDDEN, dtype=torch.float32)
    for t in range(T):
        for j in range(N):
            acc = post[t, j, 0] * x[t].float()
            for i in range(N):
                acc = acc + comb[t, i, j] * res[t, i].float()
            ref[t, j] = acc
    assert torch.allclose(out.float(), ref.to(torch.bfloat16).float())


def test_fused_equals_decomposed():
    fn, scale, base = _weights(seed=7)
    res = _residual(seed=8)
    x = torch.randn(T, HIDDEN, dtype=torch.bfloat16)
    post0, comb0, _ = mhc_pre(res, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN)

    r1, p1, c1, li1 = mhc_fused_post_pre(
        x, res, post0, comb0, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN
    )
    r_ref = mhc_post(x, res, post0, comb0)
    p_ref, c_ref, li_ref = mhc_pre(
        r_ref, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN
    )
    assert torch.equal(r1, r_ref)
    assert torch.equal(p1, p_ref)
    assert torch.equal(c1, c_ref)
    assert torch.equal(li1, li_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("t,hidden", [(1, 64), (9, 64), (3, 4096)])
def test_triton_fused_matches_torch(t, hidden):
    """The fused triton kernel must reproduce the decomposed torch reference
    (hc_post -> hc_pre) on every output, including GLM-5.3's real hidden size."""
    from freetoken.layers.mhc import mhc_fused_post_pre_torch
    from freetoken.kernel.triton.mhc import mhc_fused_post_pre_triton

    torch.manual_seed(11)
    mix = 2 * N + N * N
    fn = torch.randn(mix, N * hidden, dtype=torch.float32, device="cuda") * 0.05
    scale = torch.rand(3, dtype=torch.float32, device="cuda") + 0.5
    base = torch.randn(mix, dtype=torch.float32, device="cuda") * 0.3
    res = torch.randn(t, N, hidden, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(t, hidden, dtype=torch.bfloat16, device="cuda")
    post0 = torch.rand(t, N, 1, dtype=torch.float32, device="cuda") * POST_MULT
    comb0 = torch.softmax(torch.randn(t, N, N, device="cuda"), dim=-1)

    ref = mhc_fused_post_pre_torch(
        x, res, post0, comb0, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN
    )
    got = mhc_fused_post_pre_triton(
        x, res, post0, comb0, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN
    )
    names = ("residual", "post", "comb", "layer_input")
    tols = (2e-2, 2e-3, 2e-3, 2e-2)
    for name, r, g, tol in zip(names, ref, got, tols):
        assert g.shape == r.shape, (name, g.shape, r.shape)
        err = (g.float() - r.float()).abs().max().item()
        assert err < tol, f"{name}: max abs err {err}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype,tol", [(torch.float16, 2e-2), (torch.float32, 1e-4)])
def test_triton_fused_respects_input_dtype(dtype, tol):
    """--dtype float16/float32 must not be silently bf16-rounded: the kernel
    stores in the OUTPUT tensor's dtype (regression for the hard-coded
    tl.bfloat16 stores; fp32's tolerance is far below bf16's 2^-8 grid)."""
    from freetoken.layers.mhc import mhc_fused_post_pre_torch
    from freetoken.kernel.triton.mhc import mhc_fused_post_pre_triton

    torch.manual_seed(13)
    t, hidden = 4, 4096
    mix = 2 * N + N * N
    fn = torch.randn(mix, N * hidden, dtype=torch.float32, device="cuda") * 0.05
    scale = torch.rand(3, dtype=torch.float32, device="cuda") + 0.5
    base = torch.randn(mix, dtype=torch.float32, device="cuda") * 0.3
    res = torch.randn(t, N, hidden, dtype=dtype, device="cuda")
    x = torch.randn(t, hidden, dtype=dtype, device="cuda")
    post0 = torch.rand(t, N, 1, dtype=torch.float32, device="cuda") * POST_MULT
    comb0 = torch.softmax(torch.randn(t, N, N, device="cuda"), dim=-1)

    ref = mhc_fused_post_pre_torch(
        x, res, post0, comb0, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN
    )
    got = mhc_fused_post_pre_triton(
        x, res, post0, comb0, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN
    )
    assert got[0].dtype == dtype and got[3].dtype == dtype
    for name, r, g in zip(("residual", "layer_input"), (ref[0], ref[3]), (got[0], got[3])):
        err = (g.float() - r.float()).abs().max().item()
        assert err < tol, f"{name} [{dtype}]: max abs err {err}"


def test_single_pass_applies_the_previous_pre_and_hands_on_its_own():
    """Single-pass mHC (DeepSeek-V4.1): the input mix uses ``pre_mix_in``, the gates
    computed here are the next sublayer's pre; post/comb are unchanged from multi-pass."""
    from freetoken.layers.mhc import (
        mhc_fused_post_pre_single_pass,
        mhc_gates,
        mhc_mix_input,
        mhc_pre_single_pass,
    )

    fn, scale, base = _weights(seed=21)
    res = _residual(seed=22)
    x = torch.randn(T, HIDDEN, dtype=torch.bfloat16)
    pre_in = torch.rand(T, N, dtype=torch.float32)
    post0, comb0, _ = mhc_pre(res, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN)

    r1, p1, c1, pre1, li1 = mhc_fused_post_pre_single_pass(
        x, res, post0, comb0, pre_in, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN
    )
    r_ref = mhc_post(x, res, post0, comb0)
    pre_ref, p_ref, c_ref = mhc_gates(r_ref, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN)
    assert torch.equal(r1, r_ref) and torch.equal(p1, p_ref) and torch.equal(c1, c_ref)
    assert torch.equal(pre1, pre_ref)
    assert torch.equal(li1, mhc_mix_input(r_ref, pre_in))
    # the multi-pass input would have used pre_ref, not pre_in
    assert not torch.equal(li1, mhc_mix_input(r_ref, pre_ref))
    # a one-hot pre picks one stream (the model's seed for the chain)
    one_hot = torch.zeros(T, N)
    one_hot[:, 0] = 1.0
    _, _, _, li0 = mhc_pre_single_pass(res, one_hot, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN)
    assert torch.equal(li0, res[:, 0])
    # no post on the first sublayer: the streams pass through untouched
    r0, *_ = mhc_fused_post_pre_single_pass(
        x, res, None, None, one_hot, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN
    )
    assert torch.equal(r0, res)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("t,hidden,with_post", [(1, 64, True), (9, 64, True), (3, 5120, True), (3, 5120, False)])
def test_triton_single_pass_matches_torch(t, hidden, with_post):
    from freetoken.layers.mhc import mhc_fused_post_pre_single_pass_torch
    from freetoken.kernel.triton.mhc import mhc_fused_post_pre_single_pass_triton

    torch.manual_seed(23)
    mix = 2 * N + N * N
    fn = torch.randn(mix, N * hidden, dtype=torch.float32, device="cuda") * 0.05
    scale = torch.rand(3, dtype=torch.float32, device="cuda") + 0.5
    base = torch.randn(mix, dtype=torch.float32, device="cuda") * 0.3
    res = torch.randn(t, N, hidden, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(t, hidden, dtype=torch.bfloat16, device="cuda")
    pre_in = torch.rand(t, N, dtype=torch.float32, device="cuda")
    post0 = torch.rand(t, N, 1, dtype=torch.float32, device="cuda") * POST_MULT if with_post else None
    comb0 = torch.softmax(torch.randn(t, N, N, device="cuda"), dim=-1) if with_post else None

    ref = mhc_fused_post_pre_single_pass_torch(
        x, res, post0, comb0, pre_in, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN
    )
    got = mhc_fused_post_pre_single_pass_triton(
        x, res, post0, comb0, pre_in, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN
    )
    names = ("residual", "post", "comb", "pre_new", "layer_input")
    tols = (2e-2, 2e-3, 2e-3, 2e-3, 2e-2)
    for name, r, g, tol in zip(names, ref, got, tols):
        assert g.shape == r.shape, (name, g.shape, r.shape)
        err = (g.float() - r.float()).abs().max().item()
        assert err < tol, f"{name}: max abs err {err}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_triton_pre_only_matches_torch():
    """HAS_POST=False path (layer 0's standalone hc_pre through the fused kernel)."""
    from freetoken.kernel.triton.mhc import mhc_fused_post_pre_triton

    torch.manual_seed(12)
    t, hidden = 5, 128
    mix = 2 * N + N * N
    fn = torch.randn(mix, N * hidden, dtype=torch.float32, device="cuda") * 0.05
    scale = torch.rand(3, dtype=torch.float32, device="cuda") + 0.5
    base = torch.randn(mix, dtype=torch.float32, device="cuda") * 0.3
    res = torch.randn(t, N, hidden, dtype=torch.bfloat16, device="cuda")

    ref_post, ref_comb, ref_li = mhc_pre(
        res, fn, scale, base, RMS_EPS, EPS, POST_MULT, SINKHORN
    )
    got_res, got_post, got_comb, got_li = mhc_fused_post_pre_triton(
        res.new_empty(t, hidden), res, None, None, fn, scale, base,
        RMS_EPS, EPS, POST_MULT, SINKHORN,
    )
    assert torch.equal(got_res, res)  # pass-through when no post
    assert (got_post.float() - ref_post.float()).abs().max().item() < 2e-3
    assert (got_comb.float() - ref_comb.float()).abs().max().item() < 2e-3
    assert (got_li.float() - ref_li.float()).abs().max().item() < 2e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_mhc_split_preserves_chunked_results_and_graph_replay():
    from freetoken.kernel.triton.mhc import mhc_fused_post_pre_single_pass_triton as fused

    torch.manual_seed(26)
    hidden, tokens = 5120, 7
    res = torch.randn(tokens, N, hidden, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    post = torch.rand(tokens, N, 1, device="cuda")
    comb = torch.softmax(torch.randn(tokens, N, N, device="cuda"), -1)
    pre = torch.rand(tokens, N, device="cuda")
    fn = torch.randn(2 * N + N * N, N * hidden, device="cuda") * 0.05
    scale, base = torch.ones(3, device="cuda"), torch.zeros(2 * N + N * N, device="cuda")
    def run(lo, hi):
        return fused(x[lo:hi], res[lo:hi], post[lo:hi], comb[lo:hi], pre[lo:hi], fn, scale, base,
                     RMS_EPS, EPS, POST_MULT, SINKHORN)
    whole = run(0, tokens)
    chunks = [run(0, 2), run(2, tokens)]
    for i, expected in enumerate(whole):
        assert torch.equal(expected, torch.cat([chunk[i] for chunk in chunks]))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(0, tokens)
    graph.replay()
    for got, expected in zip(captured, whole):
        assert torch.equal(got, expected)
