"""ISO store/dequant CUDA kernel tests (kernel/csrc/jit/iso_store.cu)."""

import pytest
import torch

from freetoken.kernel import iso

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


@pytest.mark.parametrize("fmt", ["iso3", "iso4"])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_store_dequant_matches_reference(fmt, head_dim):
    torch.manual_seed(5)
    dev = "cuda"
    n, heads = 48, 2
    slots = 96
    rb = iso.packed_row_bytes(head_dim, fmt)
    kc = torch.zeros(slots, heads * rb, dtype=torch.uint8, device=dev)
    vc = torch.zeros_like(kc)
    k = (torch.randn(n, heads, head_dim, device=dev) * 2.5).to(torch.bfloat16)
    v = (torch.randn(n, heads, head_dim, device=dev) * 1.5).to(torch.bfloat16)
    idx = torch.randperm(slots, device=dev, dtype=torch.int32)[:n]

    iso.iso_store_cache(kc, vc, idx, k.view(n, -1), v.view(n, -1), heads, head_dim, fmt)
    torch.cuda.synchronize()

    ref_k = iso.quantize_ref(k.float().reshape(n * heads, head_dim), fmt)
    got_k = kc[idx.long()].view(n, heads * rb)
    eq = (got_k == ref_k.view(torch.uint8).to(dev).view(n, heads * rb)).float().mean().item()
    assert eq == 1.0, f"packed bytes diverge: {eq * 100:.3f}% match"

    kd, vd = iso.iso_dequant_rows(kc, vc, idx, heads, head_dim, fmt)
    ref_deq = iso.dequantize_ref(ref_k, fmt, head_dim).to(dev)
    assert torch.allclose(kd.float(), ref_deq.float().view(n, -1), atol=2e-2, rtol=2e-2)
    cos = torch.nn.functional.cosine_similarity(k.float().view(n, -1), kd.float(), dim=-1)
    assert cos.min() > 0.95


@pytest.mark.parametrize("fmt", ["iso3", "iso4"])
def test_store_deterministic_and_index_dtype(fmt):
    torch.manual_seed(6)
    dev = "cuda"
    n, heads, head_dim = 16, 2, 128
    slots = 32
    rb = iso.packed_row_bytes(head_dim, fmt)
    k = torch.randn(n, heads * head_dim, device=dev).to(torch.bfloat16)
    v = torch.randn(n, heads * head_dim, device=dev).to(torch.bfloat16)
    idx32 = torch.randperm(slots, device=dev, dtype=torch.int32)[:n]
    idx64 = idx32.long()

    def run(indices):
        kc = torch.zeros(slots, heads * rb, dtype=torch.uint8, device=dev)
        vc = torch.zeros_like(kc)
        iso.iso_store_cache(kc, vc, indices, k, v, heads, head_dim, fmt)
        torch.cuda.synchronize()
        return kc.clone(), vc.clone()

    a = run(idx32)
    b = run(idx32)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    c = run(idx64)
    assert torch.equal(a[0], c[0]) and torch.equal(a[1], c[1])
