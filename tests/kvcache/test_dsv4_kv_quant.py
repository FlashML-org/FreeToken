"""8-bit storage for the DSV4 window / compressed KV pools."""
import pytest
import torch

from freetoken.kvcache.dsv4_kv_quant import (
    BLOCK,
    BYTES_PER_ELEMENT,
    dequantize_ref,
    quantize_ref,
    scale_width,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="triton kernel")

D = 512  # DSV4-Flash head_dim


def _native_fp8() -> bool:
    return torch.cuda.get_device_capability() >= (8, 9)


def test_bytes_per_element_is_what_the_cost_model_assumes():
    assert BYTES_PER_ELEMENT == 1.0 + 2 / BLOCK == 1.0625
    assert scale_width(D) == D // BLOCK


def test_scale_width_rejects_a_ragged_head_dim():
    with pytest.raises(ValueError, match="multiple of the quant block"):
        scale_width(D + 1)


@pytest.mark.parametrize("rows", [1, 7, 512])
def test_round_trip_error_is_one_e4m3_step(rows):
    """e4m3 is a floating format, so the error is relative, not a uniform step of the
    block scale: 3 mantissa bits give at most 2**-4 relative on a normal value.

    Below the smallest normal (``scale * 2**-6``) the values are e4m3 subnormals, whose
    step is fixed at ``scale * 2**-9``, so relative error there is unbounded and only the
    absolute half-step binds. The bound is the sum of the two.
    """
    torch.manual_seed(0)
    x = torch.randn(rows, D, device="cuda", dtype=torch.bfloat16)
    q, s = quantize_ref(x)
    back = dequantize_ref(q, s)
    xf = x.float()
    step = s.float().repeat_interleave(BLOCK, dim=-1)  # the block's own scale
    bound = 2.0 ** -4 * xf.abs() + step * (2.0 ** -9) / 2
    assert ((back - xf).abs() <= bound * 1.01).all()


def test_an_all_zero_row_survives():
    x = torch.zeros(4, D, device="cuda", dtype=torch.bfloat16)
    q, s = quantize_ref(x)
    assert torch.equal(dequantize_ref(q, s), torch.zeros(4, D, device="cuda"))


def test_values_already_on_the_e4m3_grid_are_reproduced_closely():
    """The model rounds KV onto the e4m3 grid before it reaches the pool, so this is the
    regime that matters: re-blocking 64 -> 32 must not move values much."""
    torch.manual_seed(0)
    x = torch.randn(64, D, device="cuda", dtype=torch.float32)
    on_grid = x.to(torch.float8_e4m3fn).float()  # what act_quant_fp8_inplace leaves behind
    q, s = quantize_ref(on_grid)
    err = (dequantize_ref(q, s) - on_grid).abs() / on_grid.abs().clamp(min=1e-6)
    assert err.max() < 0.13, err.max()  # within one e4m3 step (2^-3)


@pytest.mark.skipif(not _native_fp8(), reason="needs native fp8 (sm_89+)")
@pytest.mark.parametrize("rows", [1, 5, 128])
def test_store_kernel_matches_the_reference(rows):
    from freetoken.kernel.triton.dsv4.kv_quant import store_kv_quant

    torch.manual_seed(0)
    slots_total = 256
    kv = torch.randn(rows, D, device="cuda", dtype=torch.bfloat16)
    slots = torch.randperm(slots_total, device="cuda")[:rows].to(torch.int32)

    pool = torch.zeros(slots_total, D, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.zeros(slots_total, scale_width(D), device="cuda", dtype=torch.float16)
    store_kv_quant(pool, scales, slots, kv)

    ref_q, ref_s = quantize_ref(kv)
    sl = slots.long()
    assert torch.equal(pool[sl].view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(scales[sl], ref_s)


@pytest.mark.skipif(not _native_fp8(), reason="needs native fp8 (sm_89+)")
def test_store_kernel_leaves_other_slots_untouched():
    from freetoken.kernel.triton.dsv4.kv_quant import store_kv_quant

    pool = torch.zeros(32, D, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.zeros(32, scale_width(D), device="cuda", dtype=torch.float16)
    kv = torch.randn(4, D, device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([1, 3, 5, 7], device="cuda", dtype=torch.int32)
    store_kv_quant(pool, scales, slots, kv)

    untouched = [i for i in range(32) if i not in (1, 3, 5, 7)]
    assert pool[untouched].view(torch.uint8).eq(0).all()
    assert scales[untouched].eq(0).all()


@pytest.mark.skipif(not _native_fp8(), reason="needs native fp8 (sm_89+)")
def test_store_kernel_handles_saturation():
    """A block whose max-abs is huge still maps that max onto MAX_MAG, not to inf."""
    from freetoken.kernel.triton.dsv4.kv_quant import store_kv_quant

    kv = torch.full((2, D), 3.0e4, device="cuda", dtype=torch.bfloat16)
    pool = torch.zeros(8, D, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.zeros(8, scale_width(D), device="cuda", dtype=torch.float16)
    store_kv_quant(pool, scales, torch.tensor([0, 1], device="cuda", dtype=torch.int32), kv)
    back = dequantize_ref(pool[:2], scales[:2])
    assert torch.isfinite(back).all()
    assert (back - kv.float()).abs().max() / 3.0e4 < 0.02


# ---------------------------------------------------------------- attention read path
@pytest.mark.skipif(not _native_fp8(), reason="needs native fp8 (sm_89+)")
@pytest.mark.parametrize("m", [1, 4])
def test_quantized_attention_tracks_the_bf16_reference(m):
    """The kernel dequantizes the gathered tile before the dot. Against the same pools
    stored bf16, the output should differ only by the storage rounding."""
    from freetoken.kernel.triton.dsv4.sparse_attn import sparse_attn_paged

    torch.manual_seed(0)
    b, h, topk, n_window = 2, 8, 64, 32
    n_win_slots, n_cmp = 128, 128

    q = torch.randn(b, m, h, D, device="cuda", dtype=torch.bfloat16)
    win = torch.randn(n_win_slots, D, device="cuda", dtype=torch.bfloat16)
    cmp = torch.randn(n_cmp, D, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(h, device="cuda", dtype=torch.float32)
    idx = torch.randint(0, 128, (b, m, topk), device="cuda", dtype=torch.int32)

    ref = sparse_attn_paged(q, win, cmp, sink, idx, n_window, D ** -0.5)

    win_q, win_s = quantize_ref(win)
    cmp_q, cmp_s = quantize_ref(cmp)
    got = sparse_attn_paged(q, win_q, cmp_q, sink, idx, n_window, D ** -0.5,
                            window_scale=win_s, cmp_scale=cmp_s)

    assert got.shape == ref.shape and got.dtype == ref.dtype
    # Attention averages over topk columns, so the per-element storage error largely
    # cancels; what survives is well inside an e4m3 step of the output's own scale.
    err = (got.float() - ref.float()).abs().max() / ref.float().abs().max()
    assert err < 0.05, err


@pytest.mark.skipif(not _native_fp8(), reason="needs native fp8 (sm_89+)")
def test_quantized_attention_honours_masked_columns():
    """-1 columns must stay masked: a scale load for a masked column must not resurrect it."""
    from freetoken.kernel.triton.dsv4.sparse_attn import sparse_attn_paged

    torch.manual_seed(0)
    b, m, h, topk, n_window = 1, 1, 4, 32, 16
    q = torch.randn(b, m, h, D, device="cuda", dtype=torch.bfloat16)
    win = torch.randn(64, D, device="cuda", dtype=torch.bfloat16)
    cmp = torch.randn(64, D, device="cuda", dtype=torch.bfloat16)
    sink = torch.zeros(h, device="cuda", dtype=torch.float32)
    idx = torch.full((b, m, topk), -1, device="cuda", dtype=torch.int32)
    idx[..., :4] = torch.arange(4, device="cuda", dtype=torch.int32)

    win_q, win_s = quantize_ref(win)
    cmp_q, cmp_s = quantize_ref(cmp)
    ref = sparse_attn_paged(q, win, cmp, sink, idx, n_window, D ** -0.5)
    got = sparse_attn_paged(q, win_q, cmp_q, sink, idx, n_window, D ** -0.5,
                            window_scale=win_s, cmp_scale=cmp_s)
    assert torch.isfinite(got.float()).all()
    err = (got.float() - ref.float()).abs().max() / ref.float().abs().max().clamp(min=1e-6)
    assert err < 0.05, err


@pytest.mark.skipif(not _native_fp8(), reason="needs native fp8 (sm_89+)")
def test_quantized_attention_on_the_splitk_decode_path():
    """The decode path is a different kernel with its own gather, so it needs its own
    check: m == 1 and topk > MIN_TILES_PER_SPLIT * BLOCK_T is what selects it."""
    from freetoken.kernel.triton.dsv4.sparse_attn import sparse_attn_paged, split_count

    torch.manual_seed(0)
    b, m, h, topk, n_window = 1, 1, 16, 512, 128   # DSV4's index_topk
    assert split_count(b, m, h, topk, "cuda") > 1, "shape did not select the split-k kernel"

    q = torch.randn(b, m, h, D, device="cuda", dtype=torch.bfloat16)
    win = torch.randn(256, D, device="cuda", dtype=torch.bfloat16)
    cmp = torch.randn(256, D, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(h, device="cuda", dtype=torch.float32)
    idx = torch.randint(0, 256, (b, m, topk), device="cuda", dtype=torch.int32)

    ref = sparse_attn_paged(q, win, cmp, sink, idx, n_window, D ** -0.5)
    win_q, win_s = quantize_ref(win)
    cmp_q, cmp_s = quantize_ref(cmp)
    got = sparse_attn_paged(q, win_q, cmp_q, sink, idx, n_window, D ** -0.5,
                            window_scale=win_s, cmp_scale=cmp_s)
    err = (got.float() - ref.float()).abs().max() / ref.float().abs().max()
    assert err < 0.05, err


@pytest.mark.skipif(not _native_fp8(), reason="needs native fp8 (sm_89+)")
def test_quantized_attention_respects_cmp_counts():
    """cmp_counts bounds the compressed half from device memory; the dequant must not
    read past it into stale slots."""
    from freetoken.kernel.triton.dsv4.sparse_attn import sparse_attn_paged

    torch.manual_seed(0)
    b, m, h, topk, n_window = 1, 1, 16, 512, 128
    q = torch.randn(b, m, h, D, device="cuda", dtype=torch.bfloat16)
    win = torch.randn(256, D, device="cuda", dtype=torch.bfloat16)
    cmp = torch.randn(256, D, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(h, device="cuda", dtype=torch.float32)
    idx = torch.randint(0, 256, (b, m, topk), device="cuda", dtype=torch.int32)
    counts = torch.full((b, m), 40, device="cuda", dtype=torch.int32)

    win_q, win_s = quantize_ref(win)
    cmp_q, cmp_s = quantize_ref(cmp)
    ref = sparse_attn_paged(q, win, cmp, sink, idx, n_window, D ** -0.5, cmp_counts=counts)
    got = sparse_attn_paged(q, win_q, cmp_q, sink, idx, n_window, D ** -0.5,
                            cmp_counts=counts, window_scale=win_s, cmp_scale=cmp_s)
    err = (got.float() - ref.float()).abs().max() / ref.float().abs().max()
    assert err < 0.05, err
