"""ISO attention kernel tests (kernel/csrc/jit/iso_attention.cu).

Oracle: plain torch attention over python-reference-dequantized packed KV
(tight tolerance), plus a looser check against the original bf16 KV.
"""

import pytest
import torch

from freetoken.kernel import iso

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


def _torch_attn(q, k, v, scale, causal_offset=0):
    """q [N, Hq, D] fp32, k/v [T, Hkv, D] fp32 -> [N, Hq, D] fp32.

    causal_offset: query token i attends keys [0, causal_offset + i].
    """
    n, hq, d = q.shape
    hkv = k.shape[1]
    g = hq // hkv
    kk = k.repeat_interleave(g, dim=1)
    vv = v.repeat_interleave(g, dim=1)
    scores = torch.einsum("nhd,thd->nht", q, kk) * scale
    t = k.shape[0]
    i = torch.arange(n, device=q.device).unsqueeze(1)
    j = torch.arange(t, device=q.device).unsqueeze(0)
    mask = j <= (causal_offset + i)
    scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
    probs = scores.softmax(dim=-1)
    return torch.einsum("nht,thd->nhd", probs, vv)


def _make_pool(slots, heads, head_dim, fmt, dev):
    rb = iso.packed_row_bytes(head_dim, fmt)
    return (
        torch.zeros(slots, heads * rb, dtype=torch.uint8, device=dev),
        torch.zeros(slots, heads * rb, dtype=torch.uint8, device=dev),
    )


def _store(k, v, idx, heads, head_dim, fmt, kc, vc):
    n = k.shape[0]
    iso.iso_store_cache(kc, vc, idx, k.view(n, -1), v.view(n, -1), heads, head_dim, fmt)


def _deq(kc, vc, idx, heads, head_dim, fmt):
    k, v = iso.iso_dequant_rows(kc, vc, idx, heads, head_dim, fmt)
    return k.float().view(-1, heads, head_dim), v.float().view(-1, heads, head_dim)


@pytest.mark.parametrize("fmt", ["iso3", "iso4"])
@pytest.mark.parametrize("hq,hkv,d", [(4, 2, 256), (8, 2, 128), (16, 8, 128)])
def test_decode(fmt, hq, hkv, d):
    torch.manual_seed(9)
    dev = "cuda"
    lens = [1, 37, 200]
    total = sum(lens)
    slots = total + 16
    kc, vc = _make_pool(slots, hkv, d, fmt, dev)
    k = (torch.randn(total, hkv, d, device=dev) * 2).to(torch.bfloat16)
    v = (torch.randn(total, hkv, d, device=dev) * 2).to(torch.bfloat16)
    idx = torch.randperm(slots, device=dev, dtype=torch.int32)[:total]
    _store(k, v, idx, hkv, d, fmt, kc, vc)

    indptr = torch.tensor([0, lens[0], lens[0] + lens[1], total], dtype=torch.int32, device=dev)
    q = (torch.randn(len(lens), hq * d, device=dev)).to(torch.bfloat16)
    out = torch.empty_like(q)
    scale = d ** -0.5
    iso.iso_attention_decode(q, out, kc, vc, indptr, idx, hq, hkv, d, scale, fmt)
    torch.cuda.synchronize()

    kd, vd = _deq(kc, vc, idx, hkv, d, fmt)
    for r, ln in enumerate(lens):
        s = int(indptr[r])
        # decode: the single query is the LAST token -> causal_offset = ln - 1
        ref_deq = _torch_attn(q[r].float().view(1, hq, d), kd[s : s + ln], vd[s : s + ln],
                              scale, causal_offset=ln - 1)
        got = out[r].float().view(1, hq, d)
        cos = torch.nn.functional.cosine_similarity(ref_deq.view(-1), got.view(-1), dim=0)
        assert cos > 0.9999, f"decode vs dequant-oracle cos={cos}"
        ref_orig = _torch_attn(q[r].float().view(1, hq, d),
                               k[s : s + ln].float(), v[s : s + ln].float(), scale,
                               causal_offset=ln - 1)
        cos2 = torch.nn.functional.cosine_similarity(ref_orig.view(-1), got.view(-1), dim=0)
        # quantization-limited sanity bound (peaked softmax on random data);
        # the precise mechanics gate is the dequant-oracle check above
        min_cos2 = 0.90 if fmt == "iso3" else 0.93
        assert cos2 > min_cos2, f"decode vs orig-oracle cos={cos2}"


@pytest.mark.parametrize("fmt", ["iso3", "iso4"])
@pytest.mark.parametrize("hq,hkv,d", [(4, 2, 256), (8, 2, 128)])
def test_extend(fmt, hq, hkv, d):
    torch.manual_seed(10)
    dev = "cuda"
    ctxs = [0, 50, 130]
    news = [5, 1, 40]
    total_ctx = sum(ctxs)
    slots = total_ctx + 8
    kc, vc = _make_pool(slots, hkv, d, fmt, dev)
    kp = (torch.randn(total_ctx, hkv, d, device=dev) * 2).to(torch.bfloat16)
    vp = (torch.randn(total_ctx, hkv, d, device=dev) * 2).to(torch.bfloat16)
    pidx = torch.randperm(slots, device=dev, dtype=torch.int32)[:total_ctx]
    _store(kp, vp, pidx, hkv, d, fmt, kc, vc)

    n_new = sum(news)
    ke = (torch.randn(n_new, hkv, d, device=dev) * 2).to(torch.bfloat16)
    ve = (torch.randn(n_new, hkv, d, device=dev) * 2).to(torch.bfloat16)
    q = torch.randn(n_new, hq * d, device=dev).to(torch.bfloat16)
    out = torch.empty_like(q)

    qo_indptr = torch.tensor([0, news[0], news[0] + news[1], n_new], dtype=torch.int32, device=dev)
    kv_indptr = torch.tensor([0, ctxs[0], ctxs[0] + ctxs[1], total_ctx], dtype=torch.int32, device=dev)
    scale = d ** -0.5
    iso.iso_attention_extend(q, out, kc, vc, ke.view(n_new, -1), ve.view(n_new, -1),
                             qo_indptr, kv_indptr, pidx, hq, hkv, d, scale,
                             max(news), fmt)
    torch.cuda.synchronize()

    kd, vd = _deq(kc, vc, pidx, hkv, d, fmt)
    for r, (ctx, nw) in enumerate(zip(ctxs, news)):
        cs, qs_ = int(kv_indptr[r]), int(qo_indptr[r])
        k_all = torch.cat([kd[cs : cs + ctx], ke[qs_ : qs_ + nw].float()])
        v_all = torch.cat([vd[cs : cs + ctx], ve[qs_ : qs_ + nw].float()])
        ref = _torch_attn(q[qs_ : qs_ + nw].float().view(nw, hq, d), k_all, v_all,
                          scale, causal_offset=ctx)
        got = out[qs_ : qs_ + nw].float().view(nw, hq, d)
        cos = torch.nn.functional.cosine_similarity(ref.reshape(-1), got.reshape(-1), dim=0)
        assert cos > 0.9999, f"extend req{r} vs dequant-oracle cos={cos}"
