import torch

from freetoken.kernel.triton.attention import q8_paged_attention
from freetoken.kernel.triton.q8_kv import quantize_row_q8_0_ref


def test_q8_paged_attention_matches_dequantized_reference():
    torch.manual_seed(2)
    values_k = torch.randn(3, 2, 32)
    values_v = torch.randn(3, 2, 32)
    kq, ks = quantize_row_q8_0_ref(values_k)
    vq, vs = quantize_row_q8_0_ref(values_v)
    kq, vq = kq.reshape(3, 2, 32), vq.reshape(3, 2, 32)
    ks, vs = ks.reshape(3, 2, 1), vs.reshape(3, 2, 1)
    q = torch.randn(1, 4, 32)
    out = q8_paged_attention(
        q,
        kq,
        vq,
        ks,
        vs,
        torch.tensor([0, 3], dtype=torch.int32),
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        32**-0.5,
    )
    k = kq.float() * ks.float().repeat_interleave(32, dim=-1)
    v = vq.float() * vs.float().repeat_interleave(32, dim=-1)
    scores = torch.stack([
        q[0, h].float().matmul(k[:, h % 2].transpose(0, 1)) for h in range(4)
    ]) * 32**-0.5
    ref = torch.stack([
        torch.softmax(scores[h], dim=-1).matmul(v[:, h % 2]) for h in range(4)
    ])
    assert torch.allclose(out[0].float(), ref, rtol=1e-2, atol=1e-2)


def test_q8_paged_attention_prefill_is_causal_and_handles_page_slots():
    torch.manual_seed(7)
    values_k = torch.randn(4, 1, 32)
    values_v = torch.randn(4, 1, 32)
    kq, ks = quantize_row_q8_0_ref(values_k)
    vq, vs = quantize_row_q8_0_ref(values_v)
    kq, vq = kq.reshape(4, 1, 32), vq.reshape(4, 1, 32)
    ks, vs = ks.reshape(4, 1, 1), vs.reshape(4, 1, 1)
    q = torch.randn(4, 2, 32)
    out = q8_paged_attention(
        q, kq, vq, ks, vs,
        torch.tensor([0, 4], dtype=torch.int32),
        # Non-monotonic physical slots model a page-table transition.
        torch.tensor([3, 0, 2, 1], dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
        torch.arange(4, dtype=torch.int32),
        32**-0.5,
    )
    logical_k = kq[[3, 0, 2, 1]].float() * ks[[3, 0, 2, 1]].float().repeat_interleave(32, dim=-1)
    logical_v = vq[[3, 0, 2, 1]].float() * vs[[3, 0, 2, 1]].float().repeat_interleave(32, dim=-1)
    for token in range(4):
        scores = torch.stack([
            q[token, head].float().matmul(logical_k[: token + 1, head % 1].transpose(0, 1))
            for head in range(2)
        ]) * 32**-0.5
        expected = torch.stack([
            torch.softmax(scores[head], dim=-1).matmul(logical_v[: token + 1, 0])
            for head in range(2)
        ])
        assert torch.allclose(out[token].float(), expected, rtol=1e-2, atol=1e-2)
