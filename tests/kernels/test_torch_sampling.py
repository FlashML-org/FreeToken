"""The torch sampling fallback must select the same tokens as the kernel path.

These assert the *renormalization* semantics (which tokens survive, with what mass)
rather than the draw, so they are deterministic and arch-independent.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture
def probs():
    torch.manual_seed(0)
    return torch.softmax(torch.randn(5, 512, device="cuda"), dim=-1)


def test_top_k_keeps_exactly_the_k_largest(probs):
    from freetoken.kernel.torch_sampling import top_k_renorm_probs

    for k in (1, 3, 40, 512):
        out = top_k_renorm_probs(probs, k)
        assert torch.allclose(out.sum(-1), torch.ones(probs.shape[0], device="cuda"), atol=1e-5)
        kept = out > 0
        assert (kept.sum(-1) == k).all(), f"k={k} kept {kept.sum(-1).tolist()}"
        # the survivors are the k largest of the original row
        expect = probs.topk(k, dim=-1).indices.sort(-1).values
        assert torch.equal(kept.nonzero()[:, 1].reshape(probs.shape[0], k), expect)


def test_top_k_accepts_per_row_tensor(probs):
    from freetoken.kernel.torch_sampling import top_k_renorm_probs

    k = torch.tensor([1, 2, 3, 4, 5], device="cuda", dtype=torch.int32)
    kept = top_k_renorm_probs(probs, k) > 0
    assert kept.sum(-1).tolist() == [1, 2, 3, 4, 5]


def test_top_p_keeps_the_crossing_element(probs):
    """The prefix must *reach* p, so the element that crosses it is retained."""
    from freetoken.kernel.torch_sampling import top_p_renorm_probs

    for p in (0.1, 0.5, 0.9, 1.0):
        out = top_p_renorm_probs(probs, p)
        assert torch.allclose(out.sum(-1), torch.ones(probs.shape[0], device="cuda"), atol=1e-5)
        kept = out > 0
        for row in range(probs.shape[0]):
            ordered = probs[row].sort(descending=True).values
            n = int(kept[row].sum())
            assert ordered[:n].sum() >= p - 1e-5, f"p={p} row={row} prefix short"
            if n > 1:
                assert ordered[: n - 1].sum() < p, f"p={p} row={row} prefix longer than needed"


def test_top_p_1_keeps_everything_with_mass(probs):
    from freetoken.kernel.torch_sampling import top_p_renorm_probs

    assert ((top_p_renorm_probs(probs, 1.0) > 0) == (probs > 0)).all()


def test_draw_only_returns_tokens_the_filter_kept(probs):
    from freetoken.kernel.torch_sampling import top_k_sampling_from_probs

    k = 4
    allowed = probs.topk(k, dim=-1).indices
    for seed in range(8):
        out = top_k_sampling_from_probs(probs, k, seed=seed)
        assert out.dtype == torch.int32 and out.shape == (probs.shape[0],)
        assert (out[:, None] == allowed).any(-1).all(), f"drew a filtered-out token: {out}"


def test_draw_is_deterministic_for_a_seed(probs):
    from freetoken.kernel.torch_sampling import top_p_sampling_from_probs

    a = top_p_sampling_from_probs(probs, 0.8, seed=1234)
    b = top_p_sampling_from_probs(probs, 0.8, seed=1234)
    assert torch.equal(a, b)


def test_degenerate_row_stays_drawable():
    """An all-zero row must not produce nan or an out-of-range token."""
    from freetoken.kernel.torch_sampling import top_k_sampling_from_probs

    p = torch.zeros(2, 32, device="cuda")
    p[1, 7] = 1.0
    out = top_k_sampling_from_probs(p, 4, seed=0)
    assert out.shape == (2,)
    assert (out >= 0).all() and (out < 32).all()
    assert out[1].item() == 7
