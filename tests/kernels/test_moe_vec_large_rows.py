"""MMVQ grouped-expert GEMV past the 65535 gridDim.z cap.

``moe_vec_q`` puts the routed-row count (``tokens * top_k``) on gridDim.z, which
the hardware caps at 65535 -- only gridDim.x reaches 2^31-1. At top_k = 6 that
capped a prefill chunk at 10922 tokens (10923 * 6 = 65538), so anything larger
died on its very first launch with cudaErrorInvalidValue. The launcher now
slices z and passes each launch the base of its chunk in ``z_offset``.

"does not throw" is the cheap half of this test. The half that actually pins the
indexing down is the EQUIVALENCE check: the same routed rows computed in one
over-cap call must come out bit-identical to the concatenation of sub-cap
slices. A wrong ``z_offset`` still launches fine -- it just reads the wrong
token, the wrong expert id, or writes the wrong dst row, and only a value
comparison catches that.

The geometry is deliberately tiny (1024 hidden, 64 output rows, 8 experts) so
that the token counts, not the weights, are what makes the test big.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

GGML_IQ2_XS = 17
IQ2_XS_BLOCK_BYTES = 74
QK_K = 256

H = 1024  # ncols / hidden -- 4 IQ2_XS blocks per row
ROW_BYTES = (H // QK_K) * IQ2_XS_BLOCK_BYTES  # 296
NROWS = 64  # output features per expert
NUM_EXPERTS = 8
TOP_K = 6

# Largest token count whose routed rows still fit one launch at TOP_K = 6.
OLD_CAP_TOKENS = 65535 // TOP_K  # 10922
SUB_CAP_CHUNK = 8192  # 8192 * 6 = 49152 routed rows


def _make_bank(seed: int = 0) -> torch.Tensor:
    """``[E, NROWS, ROW_BYTES]`` uint8 IQ2_XS expert bank.

    Everything is random bytes -- the IQ2_XS grid/sign/scale nibbles are all
    total over their lookup tables -- except the fp16 super-block scale ``d`` at
    the head of each block, which is written explicitly so that random bit
    patterns cannot hand us NaN/Inf and make an exact comparison meaningless.
    """
    nb = H // QK_K
    g = torch.Generator().manual_seed(seed)
    bank = torch.randint(0, 256, (NUM_EXPERTS, NROWS, ROW_BYTES), generator=g, dtype=torch.uint8)
    d = (0.01 + 0.04 * torch.rand(NUM_EXPERTS, NROWS, nb, generator=g)).to(torch.float16)
    d_bytes = d.view(torch.uint8).reshape(NUM_EXPERTS, NROWS, nb, 2)
    for b in range(nb):
        bank[:, :, b * IQ2_XS_BLOCK_BYTES : b * IQ2_XS_BLOCK_BYTES + 2] = d_bytes[:, :, b]
    return bank.cuda()


def _make_inputs(tokens: int, seed: int = 1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(tokens, H, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    ids = torch.randint(
        0, NUM_EXPERTS, (tokens, TOP_K), generator=g, device="cuda", dtype=torch.int32
    )
    return x, ids


def _run(x, bank, ids, tokens, top_k=TOP_K):
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    y = ggml_moe_a8_vec(x, bank, ids, top_k, GGML_IQ2_XS, NROWS, tokens)
    torch.cuda.synchronize()  # surface any async launch failure here
    return y


def _run_sliced(x, bank, ids, tokens, chunk=SUB_CAP_CHUNK):
    """Same computation, but every launch stays under the old 65535 z cap."""
    assert chunk * TOP_K < 65536
    parts = []
    for lo in range(0, tokens, chunk):
        hi = min(lo + chunk, tokens)
        parts.append(_run(x[lo:hi].contiguous(), bank, ids[lo:hi].contiguous(), hi - lo))
    return torch.cat(parts, dim=0)


def _sane(y: torch.Tensor) -> None:
    assert torch.isfinite(y).all(), "kernel produced non-finite output"
    assert y.abs().sum().item() > 0, "kernel output is entirely zero -- test is vacuous"


# --------------------------------------------------------------------------- #
# 1. below the cap: behaviour must be unchanged, and absolutely correct.
# --------------------------------------------------------------------------- #


def test_small_token_count_matches_dequant_reference():
    """128 tokens (z = 768, one launch). Anchor the kernel to a dense reference
    so that the equivalence tests above the cap are comparing against something
    known to be right, not just self-consistent."""
    from freetoken.kernel.gguf import ggml_dequantize

    tokens = 128
    bank = _make_bank()
    x, ids = _make_inputs(tokens)
    y = _run(x, bank, ids, tokens)
    assert y.shape == (tokens * TOP_K, NROWS)
    _sane(y)

    flat = bank.reshape(NUM_EXPERTS * NROWS, ROW_BYTES)
    deq = ggml_dequantize(flat, GGML_IQ2_XS, NUM_EXPERTS * NROWS, H, torch.float32)
    deq = deq.reshape(NUM_EXPERTS, NROWS, H)

    flat_ids = ids.reshape(-1).long()
    xf = x.to(torch.float32)
    ref = torch.zeros(tokens * TOP_K, NROWS, device="cuda", dtype=torch.float32)
    row_token = torch.arange(tokens * TOP_K, device="cuda") // TOP_K
    for e in range(NUM_EXPERTS):
        sel = (flat_ids == e).nonzero(as_tuple=True)[0]
        if sel.numel():
            ref[sel] = xf[row_token[sel]] @ deq[e].T

    # The kernel quantizes activations to Q8_1 (8 bit / 32-element block), the
    # reference does not, so this is a closeness check, not an exact one.
    err = (y.to(torch.float32) - ref).norm() / ref.norm()
    assert err < 0.05, f"relative L2 error vs dequant reference too large: {err.item():.4f}"


def test_small_token_count_slice_equivalent():
    """Sub-cap counts take exactly one launch with z_offset == 0; slicing them
    must still be bit-identical, i.e. the fix changed nothing below the cap."""
    tokens = 128
    bank = _make_bank()
    x, ids = _make_inputs(tokens)
    full = _run(x, bank, ids, tokens)
    sliced = _run_sliced(x, bank, ids, tokens, chunk=32)
    _sane(full)
    assert torch.equal(full, sliced)


# --------------------------------------------------------------------------- #
# 2. above the cap: the whole point.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tokens", [12288, 16384])
def test_over_cap_token_counts_launch(tokens):
    """z = 73728 / 98304, both past the old 65535 cap -- these used to raise
    cudaErrorInvalidValue on the very first launch."""
    assert tokens * TOP_K > 65535
    bank = _make_bank()
    x, ids = _make_inputs(tokens)
    y = _run(x, bank, ids, tokens)
    assert y.shape == (tokens * TOP_K, NROWS)
    _sane(y)


@pytest.mark.parametrize("tokens", [12288, 16384])
def test_over_cap_matches_sub_cap_slices(tokens):
    """THE indexing test.

    One over-cap call (which the launcher internally splits at 65535 routed
    rows, a boundary that does NOT line up with the token slicing below) versus
    the concatenation of runs that each stay under the cap. Every routed row is
    an independent dot product with an identical reduction order in both paths,
    so a correct ``z_offset`` gives bit-identical results. Any drift in the token
    index, the expert id or the dst row shows up here immediately.
    """
    bank = _make_bank()
    x, ids = _make_inputs(tokens)
    full = _run(x, bank, ids, tokens)
    sliced = _run_sliced(x, bank, ids, tokens)
    _sane(full)
    assert full.shape == sliced.shape
    if not torch.equal(full, sliced):
        bad = (full != sliced).any(dim=1).nonzero(as_tuple=True)[0]
        raise AssertionError(
            f"{bad.numel()} of {full.shape[0]} routed rows differ; "
            f"first at row {bad[0].item()} (token {bad[0].item() // TOP_K}), "
            f"max abs delta {(full.to(torch.float32) - sliced.to(torch.float32)).abs().max().item()}"
        )


def test_rows_straddling_the_old_cap_are_individually_correct():
    """Pin specific tokens either side of the old 65535-row boundary against a
    single-token run of the same weights -- an independent check that does not
    rely on the slicing helper at all."""
    tokens = 12288
    bank = _make_bank()
    x, ids = _make_inputs(tokens)
    full = _run(x, bank, ids, tokens)

    # 10922 is the last token that fitted; 10923 is the one that broke it.
    for t in (0, OLD_CAP_TOKENS - 1, OLD_CAP_TOKENS, OLD_CAP_TOKENS + 1, tokens - 1):
        one = _run(x[t : t + 1].contiguous(), bank, ids[t : t + 1].contiguous(), 1)
        got = full[t * TOP_K : (t + 1) * TOP_K]
        assert torch.equal(got, one), f"token {t} mismatches its standalone run"


def test_down_projection_over_cap_matches_slices():
    """The down projection calls this same kernel with tokens = T * top_k and
    top_k = 1, so it reaches the identical z. Cover that call shape past the cap
    (z = 70000) -- and note that top_k = 1 makes ``token = z / topk`` an
    identity, a different ``z_offset`` arithmetic path from the top_k = 6 tests
    above."""
    routed = 70000
    bank = _make_bank(seed=3)
    x, _ = _make_inputs(routed, seed=5)
    g = torch.Generator(device="cuda").manual_seed(7)
    ids = torch.randint(0, NUM_EXPERTS, (routed, 1), generator=g, device="cuda", dtype=torch.int32)

    full = _run(x, bank, ids, routed, top_k=1)
    parts = []
    for lo in range(0, routed, 32768):
        hi = min(lo + 32768, routed)
        parts.append(
            _run(x[lo:hi].contiguous(), bank, ids[lo:hi].contiguous(), hi - lo, top_k=1)
        )
    sliced = torch.cat(parts, dim=0)
    _sane(full)
    assert torch.equal(full, sliced)
