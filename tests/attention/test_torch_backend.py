"""Pure-PyTorch attention oracle registration and causal GQA math."""

import torch


def test_torch_backend_registered_for_full_attention():
    from freetoken.attention import AttnType, SUPPORTED_ATTENTION_BACKENDS, attention_backend_info

    assert "torch" in SUPPORTED_ATTENTION_BACKENDS.supported_names()
    info = attention_backend_info("torch")
    assert AttnType.FULL in info.supported_types
    assert info.hybrid_linear_ok


def test_torch_attention_oracle_matches_sdpa():
    tokens, query_heads, kv_heads, head_dim = 4, 8, 2, 32
    group = query_heads // kv_heads
    q = torch.randn(tokens, query_heads, head_dim)
    k = torch.randn(tokens, kv_heads, head_dim)
    v = torch.randn(tokens, kv_heads, head_dim)
    expanded_k = k.repeat_interleave(group, dim=1)
    expanded_v = v.repeat_interleave(group, dim=1)
    scale = head_dim**-0.5
    scores = torch.einsum("qhd,khd->hqk", q.float(), expanded_k.float()) * scale
    causal = torch.triu(torch.ones(tokens, tokens, dtype=torch.bool), diagonal=1)
    actual = torch.einsum(
        "hqk,khd->qhd", torch.softmax(scores.masked_fill(causal[None], float("-inf")), -1), expanded_v.float()
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        expanded_k.transpose(0, 1).unsqueeze(0),
        expanded_v.transpose(0, 1).unsqueeze(0),
        is_causal=True,
        scale=scale,
    )[0].transpose(0, 1)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_torch_attention_uses_local_query_heads_for_tp():
    from freetoken.attention.torch import TorchAttentionBackend, TorchMetadata

    class Cache:
        device = torch.device("cpu")

        def store_kv(self, k, v, out_loc, layer_id):
            return None

        def k_cache(self, layer_id):
            return torch.randn(2, 1, 4)

        def v_cache(self, layer_id):
            return torch.randn(2, 1, 4)

    backend = object.__new__(TorchAttentionBackend)
    backend.kvcache = Cache()
    backend.device = torch.device("cpu")
    # Global config has four query heads; this TP shard carries two.
    backend.num_q_heads = 4
    batch = type("Batch", (), {})()
    batch.out_loc = torch.tensor([0, 1])
    batch.attn_metadata = TorchMetadata(
        indices=torch.tensor([0, 1]),
        seqlens_q=[2],
        seqlens_k=[2],
        cached_lens=[0],
        is_decode=False,
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
    )
    q = torch.randn(2, 2, 4)
    k = torch.randn(2, 1, 4)
    v = torch.randn(2, 1, 4)

    output = backend.forward(q, k, v, 0, batch)

    assert output.shape == q.shape
