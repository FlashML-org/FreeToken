import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _reference_top_k_renorm(probs, top_k):
    out = torch.zeros_like(probs)
    vocab_size = probs.shape[1]
    for row, k in enumerate(top_k.tolist()):
        if k >= vocab_size:
            out[row] = probs[row]
            continue
        values, indices = torch.topk(probs[row], k)
        out[row, indices] = values / values.sum()
    return out


def test_top_k_renorm_falls_back_when_too_few_candidates():
    from freetoken.kernel.triton.sampling import top_k_renorm_probs

    vocab_size = 10_000
    probs = torch.full((1, vocab_size), 0.8 / (vocab_size - 1), device="cuda")
    probs[0, 0] = 0.2
    top_k = torch.tensor([50], dtype=torch.int32, device="cuda")

    actual = top_k_renorm_probs(probs, top_k)
    expected = _reference_top_k_renorm(probs, top_k.cpu())

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(dim=-1), torch.ones(1, device="cuda"))


def test_top_k_renorm_falls_back_on_candidate_overflow():
    from freetoken.kernel.triton.sampling import top_k_renorm_probs

    vocab_size = 32_768
    probs = torch.linspace(1.0, 0.99, vocab_size, device="cuda").unsqueeze(0)
    probs /= probs.sum(dim=-1, keepdim=True)
    top_k = torch.tensor([50], dtype=torch.int32, device="cuda")

    actual = top_k_renorm_probs(probs, top_k)
    expected = _reference_top_k_renorm(probs, top_k.cpu())

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(dim=-1), torch.ones(1, device="cuda"))


def test_top_k_renorm_mixed_batch_bypasses_vocab_size_row():
    from freetoken.kernel.triton.sampling import top_k_renorm_probs

    vocab_size = 10_000
    peaky = torch.full((vocab_size,), 0.8 / (vocab_size - 1), device="cuda")
    peaky[0] = 0.2
    unfiltered = torch.linspace(1.0, 0.5, vocab_size, device="cuda")
    unfiltered /= unfiltered.sum()
    probs = torch.stack((peaky, unfiltered))
    top_k = torch.tensor([50, vocab_size], dtype=torch.int32, device="cuda")

    actual = top_k_renorm_probs(probs, top_k)
    expected = _reference_top_k_renorm(probs, top_k.cpu())

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(dim=-1), torch.ones(2, device="cuda"))
