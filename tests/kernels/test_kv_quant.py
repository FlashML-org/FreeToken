from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.quant import BLOCK, FP8_E4M3, INT4, NONE, Q8_0, resolve_kv_quant

SPECS = [Q8_0, FP8_E4M3, INT4]
IDS = [spec.name for spec in SPECS]

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _kv(tokens: int, heads: int, dim: int, device="cuda", seed=0) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(tokens, heads, dim, generator=g, device=device, dtype=torch.bfloat16)


def test_bytes_per_element_amortizes_the_scale():
    # 8 bits of payload + one fp16 scale per 32 elements.
    assert Q8_0.bytes_per_element(torch.bfloat16) == 1.0 + 2 / 32
    assert FP8_E4M3.bytes_per_element(torch.bfloat16) == 1.0 + 2 / 32
    # Packed int4 stores two values per byte, with the same scale slab.
    assert INT4.bytes_per_element(torch.bfloat16) == 0.5 + 2 / 32
    assert INT4.storage_shape((7, 4, 256)) == (7, 4, 128)
    # Unquantized pools price at the compute dtype.
    assert NONE.bytes_per_element(torch.bfloat16) == 2.0
    assert NONE.bytes_per_element(torch.float32) == 4.0


def test_resolve_and_scale_shape():
    assert resolve_kv_quant(None) is NONE
    assert resolve_kv_quant("auto") is NONE
    assert resolve_kv_quant("q8_0") is Q8_0
    assert resolve_kv_quant("int4") is INT4
    assert resolve_kv_quant("q4_0") is INT4
    assert Q8_0.scale_shape((7, 4, 256)) == (7, 4, 8)
    with pytest.raises(ValueError, match="not a multiple"):
        Q8_0.scale_shape((7, 4, 100))


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_reference_roundtrip_error_is_within_the_scheme_envelope(spec):
    torch.manual_seed(0)
    """Each scheme's round-trip error must sit inside the bound its format implies.

    int8 rounds onto a uniform grid of step ``amax/127``, so the error is at most half a
    step -- except for a value at the block's extreme, which the clamp can push to a
    full step once the fp16 scale rounds down. One step is the honest envelope. e4m3
    carries 3 mantissa bits, so its error is relative, up to ~2^-4 of each value, which
    against the block's amax is bounded the same way.
    """
    x = torch.randn(64, 4, 256, dtype=torch.float32)
    q, scales = spec.quantize(x)
    back = spec.dequantize(q, scales)

    blocks = x.unflatten(-1, (x.shape[-1] // BLOCK, BLOCK))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    err = (back - x).unflatten(-1, (x.shape[-1] // BLOCK, BLOCK)).abs()
    bound = 1.0 / spec.max_magnitude if spec.is_integer else 1.0 / 2**4
    assert (err <= amax * bound + 1e-6).all()
    # And the typical error should sit well under the worst case, not at it.
    assert err.mean() <= amax.mean() * bound * 0.5


def test_int4_matches_ggml_q4_0_codes_in_element_order():
    x = torch.tensor([-8.0, -7.0, -1.0, 0.0, 1.0, 6.0, 7.0, 0.0] * 4).view(1, 1, BLOCK)
    packed, scales = INT4.quantize(x)

    # The negative extreme selects scale=1; code = value + 8. Low nibble is the even
    # logical element and high nibble is the odd logical element.
    assert scales.item() == 1.0
    assert packed[0, 0, :4].tolist() == [0x10, 0x87, 0xE9, 0x8F]
    torch.testing.assert_close(INT4.dequantize(packed, scales), x)


def test_int8_beats_fp8_on_a_flat_block_and_loses_on_a_spiky_one():
    """The tradeoff the scheme choice turns on, pinned as a test.

    Within a block, int8 spends its codes uniformly and fp8 spends them
    logarithmically. So a block of similar magnitudes favours int8, and a block where
    one outlier dwarfs the rest favours fp8 -- which is exactly why the block is 32
    elements and not a whole head.
    """
    flat = torch.full((1, 1, BLOCK), 1.0)
    flat[..., ::2] = 0.9
    spiky = torch.full((1, 1, BLOCK), 0.01)
    spiky[..., 0] = 100.0

    def rel_err(spec, x):
        back = spec.dequantize(*spec.quantize(x))
        return ((back - x).abs() / x.abs()).mean().item()

    assert rel_err(Q8_0, flat) < rel_err(FP8_E4M3, flat)
    assert rel_err(FP8_E4M3, spiky) < rel_err(Q8_0, spiky)


@cuda_only
@pytest.mark.parametrize("spec", SPECS, ids=IDS)
@pytest.mark.parametrize("head_dim", [256, 512])
def test_store_kernel_matches_the_reference_quantizer(spec, head_dim):
    from freetoken.kernel.triton.kv_quant import store_kv_quant

    tokens, heads, slots = 37, 3, 64
    k = _kv(tokens, heads, head_dim, seed=1)
    v = _kv(tokens, heads, head_dim, seed=2)
    # Scatter to non-contiguous slots: the kernel must honour the index indirection.
    indices = torch.randperm(slots, device="cuda")[:tokens].to(torch.int32)

    kc = torch.zeros(slots, heads, head_dim // spec.elements_per_byte, device="cuda", dtype=spec.storage_dtype)
    vc = torch.zeros_like(kc)
    ks = torch.zeros(slots, heads, head_dim // BLOCK, device="cuda", dtype=torch.float16)
    vs = torch.zeros_like(ks)

    store_kv_quant(kc, ks, vc, vs, indices, k, v, spec)

    idx = indices.to(torch.long)
    for src, cache, scale in ((k, kc, ks), (v, vc, vs)):
        want_q, want_s = spec.quantize(src.float())
        torch.testing.assert_close(scale[idx].float(), want_s.float(), rtol=0, atol=0)
        # Compare dequantized values: int8 is exact, fp8 codes compare through float.
        got = spec.dequantize(cache[idx].float(), scale[idx])
        torch.testing.assert_close(got, spec.dequantize(want_q.float(), want_s), rtol=0, atol=0)


@cuda_only
@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_store_kernel_leaves_untouched_slots_alone(spec):
    from freetoken.kernel.triton.kv_quant import store_kv_quant

    heads, head_dim, slots = 2, 256, 16
    k = _kv(4, heads, head_dim, seed=3)
    v = _kv(4, heads, head_dim, seed=4)
    indices = torch.tensor([1, 3, 5, 7], device="cuda", dtype=torch.int32)

    kc = torch.zeros(slots, heads, head_dim // spec.elements_per_byte, device="cuda", dtype=spec.storage_dtype)
    vc = torch.zeros_like(kc)
    ks = torch.zeros(slots, heads, head_dim // BLOCK, device="cuda", dtype=torch.float16)
    vs = torch.zeros_like(ks)
    store_kv_quant(kc, ks, vc, vs, indices, k, v, spec)

    untouched = [s for s in range(slots) if s not in {1, 3, 5, 7}]
    assert (kc[untouched].float() == 0).all()
    assert (ks[untouched] == 0).all()


@cuda_only
@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_store_kernel_handles_an_all_zero_head(spec):
    """A zero block has no max to scale by; it must store zeros, not NaNs."""
    from freetoken.kernel.triton.kv_quant import store_kv_quant

    heads, head_dim = 2, 256
    k = torch.zeros(1, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    indices = torch.zeros(1, device="cuda", dtype=torch.int32)

    kc = torch.empty(4, heads, head_dim // spec.elements_per_byte, device="cuda", dtype=spec.storage_dtype)
    vc = torch.empty_like(kc)
    ks = torch.empty(4, heads, head_dim // BLOCK, device="cuda", dtype=torch.float16)
    vs = torch.empty_like(ks)
    store_kv_quant(kc, ks, vc, vs, indices, k, v, spec)

    # Zero must dequantize back to zero. int4 encodes 0 as the offset nibble (0x88),
    # not 0x00, so compare the DEQUANTIZED values, not the raw packed bytes.
    assert (spec.dequantize(kc[0].float(), ks[0]).abs() == 0).all()
    assert torch.isfinite(ks[0]).all() and (ks[0] > 0).all()


# --------------------------------------------------------------------------------------
# Attention over a quantized pool.
#
# The gate is equivalence against the bf16 kernel fed the SAME dequantized values: that
# isolates "did the dequant path compute attention correctly" from "how much does 8-bit
# storage cost", which is a separate, looser assertion below.
# --------------------------------------------------------------------------------------


def _quantized_pool(spec, k_bf16, v_bf16):
    """Store bf16 K/V into a quantized pool; return (kq, ks, vq, vs, k_deq, v_deq)."""
    from freetoken.kernel.triton.kv_quant import store_kv_quant

    slots, heads, dim = k_bf16.shape
    epb = spec.elements_per_byte
    kq = torch.zeros(slots, heads, dim // epb, device="cuda", dtype=spec.storage_dtype)
    vq = torch.zeros_like(kq)
    ks = torch.zeros(slots, heads, dim // BLOCK, device="cuda", dtype=torch.float16)
    vs = torch.zeros_like(ks)
    indices = torch.arange(slots, device="cuda", dtype=torch.int32)
    store_kv_quant(kq, ks, vq, vs, indices, k_bf16, v_bf16, spec)
    # What the attention kernel will effectively see, in the dtype it dequantizes into.
    k_deq = spec.dequantize(kq.float(), ks).to(torch.bfloat16).reshape(slots, heads, dim)
    v_deq = spec.dequantize(vq.float(), vs).to(torch.bfloat16).reshape(slots, heads, dim)
    return kq, ks, vq, vs, k_deq, v_deq


@cuda_only
@pytest.mark.parametrize("spec", SPECS, ids=IDS)
@pytest.mark.parametrize("head_dim", [256, 512])
def test_paged_attention_over_quantized_pool(spec, head_dim):
    from freetoken.kernel.triton.attention import paged_attention

    slots, q_heads, kv_heads = 96, 8, 2
    q = _kv(6, q_heads, head_dim, seed=5)
    k = _kv(slots, kv_heads, head_dim, seed=6)
    v = _kv(slots, kv_heads, head_dim, seed=7)
    kq, ks, vq, vs, k_deq, v_deq = _quantized_pool(spec, k, v)

    indptr = torch.tensor([0, 40, 96], device="cuda", dtype=torch.int32)
    indices = torch.arange(slots, device="cuda", dtype=torch.int32)
    q_to_req = torch.tensor([0, 0, 0, 1, 1, 1], device="cuda", dtype=torch.int32)
    q_pos = torch.tensor([10, 25, 39, 5, 30, 55], device="cuda", dtype=torch.int32)
    kw = dict(indptr=indptr, indices=indices, q_to_req=q_to_req, q_positions=q_pos,
              sm_scale=head_dim**-0.5)

    got = paged_attention(q=q, k_cache=kq, v_cache=vq, k_scale=ks, v_scale=vs, **kw)
    want = paged_attention(q=q, k_cache=k_deq, v_cache=v_deq, **kw)
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


@cuda_only
@pytest.mark.parametrize("spec", SPECS, ids=IDS)
@pytest.mark.parametrize("head_dim", [256, 512])
def test_decode_attention_over_quantized_pool(spec, head_dim):
    from freetoken.kernel.triton.attention import decode_paged_attention

    slots, q_heads, kv_heads, batch = 128, 8, 2, 3
    q = _kv(batch, q_heads, head_dim, seed=8)
    k = _kv(slots, kv_heads, head_dim, seed=9)
    v = _kv(slots, kv_heads, head_dim, seed=10)
    kq, ks, vq, vs, k_deq, v_deq = _quantized_pool(spec, k, v)

    indptr = torch.tensor([0, 40, 90, 128], device="cuda", dtype=torch.int32)
    indices = torch.arange(slots, device="cuda", dtype=torch.int32)
    q_pos = torch.tensor([39, 49, 37], device="cuda", dtype=torch.int32)
    splits = 4
    logits = torch.zeros(batch, q_heads, splits, head_dim, device="cuda", dtype=torch.float32)
    lse = torch.zeros(batch, q_heads, splits, device="cuda", dtype=torch.float32)
    nsplits = torch.full((batch,), splits, device="cuda", dtype=torch.int32)
    kw = dict(indptr=indptr, indices=indices, q_positions=q_pos, attn_logits=logits,
              attn_lse=lse, num_kv_splits=nsplits, max_kv_splits=splits,
              sm_scale=head_dim**-0.5)

    got = decode_paged_attention(q=q, k_cache=kq, v_cache=vq, k_scale=ks, v_scale=vs, **kw)
    want = decode_paged_attention(q=q, k_cache=k_deq, v_cache=v_deq, **kw)
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


@cuda_only
def test_ornith_q4_tuned_decode_matches_dequantized_oracle():
    """Exercise the exact launch selected in production for Ornith.

    The former BLOCK_N=16 tuning passed generic 8-head tests but silently corrupted
    packed Q4 attention at Ornith's 16-query-head/2-KV-head geometry.
    """
    from freetoken.kernel.triton.attention import decode_paged_attention

    slots, q_heads, kv_heads, head_dim = 67, 16, 2, 256
    q = _kv(1, q_heads, head_dim, seed=81)
    k = _kv(slots, kv_heads, head_dim, seed=82)
    v = _kv(slots, kv_heads, head_dim, seed=83)
    kq, ks, vq, vs, k_deq, v_deq = _quantized_pool(INT4, k, v)
    indptr = torch.tensor([0, slots], device="cuda", dtype=torch.int32)
    indices = torch.arange(slots, device="cuda", dtype=torch.int32)
    q_pos = torch.tensor([slots - 1], device="cuda", dtype=torch.int32)

    def run(k_cache, v_cache, splits, k_scale=None, v_scale=None):
        logits = torch.empty(1, q_heads, splits, head_dim, device="cuda", dtype=torch.float32)
        lse = torch.empty(1, q_heads, splits, device="cuda", dtype=torch.float32)
        nsplits = torch.full((1,), splits, device="cuda", dtype=torch.int32)
        return decode_paged_attention(
            q, k_cache, v_cache, indptr, indices, q_pos, logits, lse, nsplits,
            splits, head_dim**-0.5, k_scale=k_scale, v_scale=v_scale,
        )

    # Exercise both architecture-specific production choices regardless of which
    # GPU runs the suite; the launch geometry, not the device name, owns correctness.
    tuned_splits = 64
    got = run(kq, vq, tuned_splits, ks, vs)
    want = run(k_deq, v_deq, 8)
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


@cuda_only
@pytest.mark.parametrize("spec", SPECS, ids=IDS)
@pytest.mark.parametrize("split", [False, True], ids=["fused", "split"])
def test_extend_attention_over_quantized_pool(spec, split):
    """Both extend paths. The split kernel reads the freshly-computed K/V in bf16 and
    only the prefix from the quantized pool, so it exercises the mixed case."""
    from freetoken.kernel.triton.attention import extend_paged_attention

    head_dim, slots, q_heads, kv_heads = 256, 64, 8, 2
    q_len, prefix = 8, 24
    q = _kv(q_len, q_heads, head_dim, seed=11)
    k = _kv(slots, kv_heads, head_dim, seed=12)
    v = _kv(slots, kv_heads, head_dim, seed=13)
    kq, ks, vq, vs, k_deq, v_deq = _quantized_pool(spec, k, v)

    qo_indptr = torch.tensor([0, q_len], device="cuda", dtype=torch.int32)
    kv_indptr = torch.tensor([0, prefix + q_len], device="cuda", dtype=torch.int32)
    kv_indices = torch.arange(prefix + q_len, device="cuda", dtype=torch.int32)
    prefix_lens = torch.tensor([prefix], device="cuda", dtype=torch.int32)
    extend = {}
    if split:
        extend = dict(k_extend=_kv(q_len, kv_heads, head_dim, seed=14),
                      v_extend=_kv(q_len, kv_heads, head_dim, seed=15))
    kw = dict(qo_indptr=qo_indptr, kv_indptr=kv_indptr, kv_indices=kv_indices,
              prefix_lens=prefix_lens, max_q_len=q_len, sm_scale=head_dim**-0.5, **extend)

    got = extend_paged_attention(q=q, k_cache=kq, v_cache=vq, k_scale=ks, v_scale=vs, **kw)
    want = extend_paged_attention(q=q, k_cache=k_deq, v_cache=v_deq, **kw)
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


@cuda_only
@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_quantized_attention_tracks_the_bf16_pool(spec):
    """The end-to-end cost of 8-bit storage: attention over a quantized pool against
    attention over the original bf16 one. This is the number that matters for quality,
    and it is looser than the kernel-equivalence gate above by construction."""
    from freetoken.kernel.triton.attention import paged_attention

    head_dim, slots, q_heads, kv_heads = 256, 128, 8, 2
    q = _kv(4, q_heads, head_dim, seed=16)
    k = _kv(slots, kv_heads, head_dim, seed=17)
    v = _kv(slots, kv_heads, head_dim, seed=18)
    kq, ks, vq, vs, _, _ = _quantized_pool(spec, k, v)

    indptr = torch.tensor([0, slots], device="cuda", dtype=torch.int32)
    indices = torch.arange(slots, device="cuda", dtype=torch.int32)
    q_to_req = torch.zeros(4, device="cuda", dtype=torch.int32)
    q_pos = torch.tensor([20, 60, 100, 127], device="cuda", dtype=torch.int32)
    kw = dict(indptr=indptr, indices=indices, q_to_req=q_to_req, q_positions=q_pos,
              sm_scale=head_dim**-0.5)

    got = paged_attention(q=q, k_cache=kq, v_cache=vq, k_scale=ks, v_scale=vs, **kw)
    ref = paged_attention(q=q, k_cache=k, v_cache=v, **kw)
    rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
    # 8-bit ~1% here; int4's 4-bit mantissa is inherently ~7x coarser, so it gets a
    # looser (still meaningful) bound. The strict per-kernel equivalence is pinned by
    # the tests above; this only whats the TOTAL storage cost.
    bound = 0.16 if spec.packed else 0.05
    assert rel < bound, f"{spec.name}: relative error {rel:.4f} vs bf16 pool (bound {bound})"
