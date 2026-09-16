"""MRotaryEmbedding correctness: text degeneracy, section layouts, kernel-vs-fallback."""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

HEAD = 256
ROT = 64
SECTION = (11, 11, 10)

# independent expectations for SECTION under each layout (half = 32 slots)
REFERENCE_TABLES = {
    "contiguous": [0] * 11 + [1] * 11 + [2] * 10,
    "interleaved": [1 if i % 3 == 1 and i < 33 else 2 if i % 3 == 2 and i < 30 else 0 for i in range(32)],
    "interleaved_glm": [0, 1, 2] * 10 + [0, 1],
}

YARN = (("rope_type", "yarn"), ("factor", 4.0), ("original_max_position_embeddings", 262144))


def _hf_yarn_parameters(device):
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    class Config:
        head_dim = hidden_size = HEAD
        num_attention_heads = 1
        max_position_embeddings = 1048576
        rope_parameters = {"rope_theta": 1e7, "partial_rotary_factor": .25, **dict(YARN)}

        def standardize_rope_params(self):
            pass

    return ROPE_INIT_FUNCTIONS["yarn"](Config(), device=torch.device(device))


def test_yarn_mrope_keeps_scaled_main_and_index_frequencies():
    from freetoken.layers.rotary import MRotaryEmbedding, get_rope

    get_rope.cache_clear()
    kwargs = dict(rotary_dim=ROT, max_position=257, base=1e7, rope_scaling=YARN)
    plain = get_rope(head_dim=HEAD, **kwargs)
    main = get_rope(head_dim=HEAD, mrope_section=SECTION, **kwargs)
    index = get_rope(head_dim=128, mrope_section=SECTION, **kwargs)
    assert isinstance(main, MRotaryEmbedding) and isinstance(index, MRotaryEmbedding)
    inv_freq, amplitude = _hf_yarn_parameters("cpu")
    angles = torch.outer(torch.arange(257, dtype=torch.float32), inv_freq)
    expected = torch.cat((angles.cos(), angles.sin()), dim=-1) * amplitude
    for rope in (plain, main, index):
        torch.testing.assert_close(rope._cos_sin_cache, expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(main._cos_sin_cache, plain._cos_sin_cache, rtol=0, atol=0)
    torch.testing.assert_close(index._cos_sin_cache, plain._cos_sin_cache, rtol=0, atol=0)
    assert main._section_table.tolist() == REFERENCE_TABLES["interleaved"]
    assert amplitude > 1.0
    get_rope.cache_clear()


def test_mrope_rejects_partial_proportional_cache_layout():
    from freetoken.layers.rotary import get_rope

    with pytest.raises(ValueError, match="partial proportional"):
        get_rope(head_dim=HEAD, rotary_dim=ROT, max_position=8, base=1e7,
                 rope_scaling=(("rope_type", "proportional"),), mrope_section=SECTION)


def test_section_tables():
    from freetoken.layers.rotary import build_section_table

    for layout, expected in REFERENCE_TABLES.items():
        assert build_section_table(SECTION, layout).tolist() == expected, layout
    # GLM-V's [8,12,12]: T runs out first and H/W fill the tail
    assert build_section_table((8, 12, 12), "interleaved_glm").tolist() == [0, 1, 2] * 8 + [1, 1, 2, 1, 1, 2, 2, 2]
    with pytest.raises(ValueError, match="not representable"):
        build_section_table((8, 12, 12), "interleaved")
    with pytest.raises(ValueError, match="unknown mrope layout"):
        build_section_table(SECTION, "diagonal")


def _make(mrope: bool, layout: str = "interleaved"):
    from freetoken.layers.rotary import get_rope

    # the engine builds rope layers inside a cuda device context
    with torch.device("cuda"):
        return get_rope(
            head_dim=HEAD, rotary_dim=ROT, max_position=4096, base=1e7,
            mrope_section=SECTION if mrope else None, mrope_layout=layout,
        )


def _qk(n, heads=4, kv=2, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(n, heads * HEAD, generator=g, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(n, kv * HEAD, generator=g, dtype=torch.bfloat16, device="cuda")
    return q, k


@cuda
def test_text_positions_degenerate_to_1d_rope():
    n = 64
    pos1 = torch.arange(n, dtype=torch.int32, device="cuda")
    pos3 = pos1.unsqueeze(0).expand(3, -1).contiguous()
    q1, k1 = _qk(n)
    q3, k3 = q1.clone(), k1.clone()
    _make(False).forward(pos1, q1, k1)
    _make(True).forward(pos3, q3, k3)
    # 1-D path may use flashinfer while mrope uses the triton kernel: allclose, not bitwise
    assert torch.allclose(q1.float(), q3.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(k1.float(), k3.float(), atol=2e-2, rtol=2e-2)


def _torch_reference(pos3, q, k, table):
    """Per-axis freqs, per-slot axis selection, NeoX rotation, fp32 math."""
    half = ROT // 2
    inv = 1.0 / (1e7 ** (torch.arange(0, ROT, 2, dtype=torch.float32, device="cuda") / ROT))
    sec = torch.tensor(table, dtype=torch.long, device="cuda")
    pos = pos3[sec, :].transpose(0, 1).float()          # [n, half]
    freqs = pos * inv.unsqueeze(0)                       # [n, half]
    cos, sin = freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)
    for t in (q, k):
        v = t.view(t.shape[0], -1, HEAD)
        lo, hi = v[..., :half].float(), v[..., half:ROT].float()
        v[..., :half] = (lo * cos - hi * sin).to(v.dtype)
        v[..., half:ROT] = (hi * cos + lo * sin).to(v.dtype)


@cuda
@pytest.mark.parametrize("layout", sorted(REFERENCE_TABLES))
def test_matches_reference_semantics(layout):
    n = 97
    g = torch.Generator(device="cuda").manual_seed(7)
    pos3 = torch.randint(0, 4000, (3, n), generator=g, dtype=torch.int32, device="cuda")
    q, k = _qk(n, seed=7)
    q_ref, k_ref = q.clone(), k.clone()
    _make(True, layout).forward(pos3, q, k)
    _torch_reference(pos3, q_ref, k_ref, REFERENCE_TABLES[layout])
    assert torch.allclose(q.float(), q_ref.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(k.float(), k_ref.float(), atol=2e-2, rtol=2e-2)


@cuda
def test_kernel_matches_torch_fallback():
    from freetoken.kernel.triton.rope import (
        apply_mrope_torch_fallback,
        apply_mrope_with_cos_sin_cache_inplace,
    )

    rope = _make(True)
    n = 33
    g = torch.Generator(device="cuda").manual_seed(3)
    pos3 = torch.randint(0, 4000, (3, n), generator=g, dtype=torch.int32, device="cuda")
    q, k = _qk(n, seed=3)
    q2, k2 = q.clone(), k.clone()
    cache = rope._cos_sin_cache
    sec = rope._section_table.cuda()
    apply_mrope_with_cos_sin_cache_inplace(pos3, q, k, HEAD, cache, sec)
    apply_mrope_torch_fallback(pos3, q2, k2, HEAD, cache, sec)
    assert torch.allclose(q.float(), q2.float(), atol=1e-2, rtol=1e-2)
    assert torch.allclose(k.float(), k2.float(), atol=1e-2, rtol=1e-2)


@cuda
@pytest.mark.parametrize("head_dim", [128, HEAD], ids=["index", "attention"])
def test_yarn_mrope_cuda_matches_hf_frequencies_and_preserves_partial_tail(head_dim):
    from freetoken.layers.rotary import get_rope

    get_rope.cache_clear()
    with torch.device("cuda"):
        rope = get_rope(head_dim=head_dim, rotary_dim=ROT, max_position=4096, base=1e7,
                        rope_scaling=YARN, mrope_section=SECTION)
    gen = torch.Generator(device="cuda").manual_seed(29)
    positions = torch.randint(0, 4096, (3, 37), device="cuda", dtype=torch.int32, generator=gen)
    q = torch.randn(37, 3 * head_dim, device="cuda", dtype=torch.bfloat16, generator=gen)
    k = torch.randn(37, head_dim, device="cuda", dtype=torch.bfloat16, generator=gen)
    expected = [q.clone(), k.clone()]
    inv_freq, amplitude = _hf_yarn_parameters("cuda")
    axes = torch.tensor(REFERENCE_TABLES["interleaved"], device="cuda")
    angles = positions[axes].T.float() * inv_freq
    cos, sin = (angles.cos() * amplitude).unsqueeze(1), (angles.sin() * amplitude).unsqueeze(1)
    for tensor in expected:
        value = tensor.view(37, -1, head_dim)
        lo, hi = value[..., :ROT // 2].float(), value[..., ROT // 2:ROT].float()
        value[..., :ROT // 2] = (lo * cos - hi * sin).to(value.dtype)
        value[..., ROT // 2:ROT] = (hi * cos + lo * sin).to(value.dtype)
    rope.forward(positions, q, k)
    for actual, reference in zip((q, k), expected):
        torch.testing.assert_close(actual, reference, rtol=.02, atol=.02)
        torch.testing.assert_close(actual.view(37, -1, head_dim)[..., ROT:],
                                    reference.view(37, -1, head_dim)[..., ROT:], rtol=0, atol=0)
    get_rope.cache_clear()
