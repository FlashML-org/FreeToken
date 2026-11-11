"""Hand-written GQA-fused extend (prefill) attention for pre-sm70.

The Triton split kernel cannot fuse the 8 query heads sharing a kv head into one
program: 64 fused rows need ~80 KiB of staging and Triton hoists the loop-invariant q
tile wholesale (48 KiB block limit). This CUDA kernel keeps q (64 rows = 8 heads x 8
tokens, 32 KiB) in shared memory, walks K/V in BN=8 tiles (4 KiB each) and the fp32
accumulator (64 per thread) in registers -- every K/V tile is loaded once per
(kv_head, m_block) and serves all 8 heads. Measured on a GTX 1080 Ti at a 120k-token
prefix (q=1536, 16Q/2KV heads, D=256): 2754 -> 1494 ms per layer (1.84x), outputs
bit-identical to the Triton kernel at fp16.

Constraints (dispatch-guarded by the caller): GROUP == 8, head_dim == 256, fp16,
no sliding window, no sinks. Compilation is lazy and cached by torch's extension
cache; any build failure surfaces as an ImportError so the caller falls back to the
Triton kernel.
"""
from __future__ import annotations

import os

import torch

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdint>

#define QROWS 64
#define BN 8
#define DHT 256
#define NTHREADS 256

// thread t: warp w = t/32 owns rows w*8..w*8+7; lane ds = (t%32)*8 dim slice.
// q in smem [64][256+pad], k/v tiles [BN][256+pad]; acc[8][8] fp32 in regs.
__global__ void gqa_extend_kernel(
    const half* __restrict__ q, const half* __restrict__ k_ext,
    const half* __restrict__ v_ext, const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    const int* __restrict__ qo_indptr, const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices, const int* __restrict__ prefix_lens,
    float sm_scale, half* __restrict__ o,
    int HQ, int HKV, int num_mblocks,
    long sqt, long sqh, long skcs, long skch, long skes, long skeh,
    long svcs, long svch, long sves, long sveh, long sot, long soh)
{
    const int seq = blockIdx.x, kv_head = blockIdx.y, mb = blockIdx.z;
    const int GROUP = HQ / HKV;
    const int TOKENS = QROWS / GROUP;   // tokens per CTA (8 for GROUP=8)

    const int q_start = qo_indptr[seq];
    const int q_len = qo_indptr[seq+1] - q_start;
    const int kv_start = kv_indptr[seq];
    const int prefix_len = prefix_lens[seq];

    const int tid = threadIdx.x;
    const int rg = tid / 32;            // row group 0..7
    const int ds = (tid % 32) * 8;      // dim slice base
    const int lane0 = (tid % 32) == 0;

    __shared__ half s_q[QROWS][DHT + 8];
    __shared__ half s_k[BN][DHT + 8];
    __shared__ half s_v[BN][DHT + 8];
    __shared__ float s_m[QROWS], s_l[QROWS], s_alpha[QROWS];
    __shared__ float s_p[QROWS][BN];

    // cooperative q load: row r -> (h_local = r/TOKENS, tok = mb*TOKENS + r%TOKENS)
    for (int i = tid; i < QROWS * DHT; i += NTHREADS) {
        int r = i / DHT, d = i % DHT;
        int tok = mb * TOKENS + (r % TOKENS);
        bool ok = (r / TOKENS < GROUP) && (tok < q_len);
        size_t off = (size_t)(q_start + min(tok, q_len - 1)) * sqt
                     + (size_t)(kv_head * GROUP + r / TOKENS) * sqh + d;
        s_q[r][d] = ok ? q[off] : __float2half(0.f);
    }
    float acc[8][8];
#pragma unroll
    for (int rr = 0; rr < 8; rr++)
#pragma unroll
        for (int dd = 0; dd < 8; dd++) acc[rr][dd] = 0.f;
    if (tid < QROWS) { s_m[tid] = -1e30f; s_l[tid] = 1.f; s_alpha[tid] = 1.f; }
    __syncthreads();

    const int q_end_block = min(q_len, (mb + 1) * TOKENS);

    for (int phase = 0; phase < 2; phase++) {
        int n_iters;
        if (phase == 0) n_iters = prefix_len;          // paged cache part
        else            n_iters = q_end_block;         // extend part (causal)
        if (phase == 1 && q_end_block <= 0) break;

        for (int n0 = 0; n0 < n_iters; n0 += BN) {
            for (int i = tid; i < BN * DHT; i += NTHREADS) {
                int n = i / DHT, d = i % DHT;
                int pos = n0 + n;
                bool ok = pos < n_iters;
                half kk = __float2half(0.f), vv = __float2half(0.f);
                if (ok) {
                    if (phase == 0) {
                        int slot = kv_indices[kv_start + pos];
                        kk = k_cache[(size_t)slot * skcs + kv_head * skch + d];
                        vv = v_cache[(size_t)slot * svcs + kv_head * svch + d];
                    } else {
                        kk = k_ext[(size_t)(q_start + pos) * skes + kv_head * skeh + d];
                        vv = v_ext[(size_t)(q_start + pos) * sves + kv_head * sveh + d];
                    }
                }
                s_k[n][d] = kk; s_v[n][d] = vv;
            }
            __syncthreads();

            float sc[8][BN];
#pragma unroll
            for (int rr = 0; rr < 8; rr++) {
#pragma unroll
                for (int n = 0; n < BN; n++) {
                    float s = 0.f;
#pragma unroll
                    for (int dd = 0; dd < 8; dd++)
                        s += __half2float(s_q[rg*8+rr][ds+dd]) * __half2float(s_k[n][ds+dd]);
                    sc[rr][n] = s;
                }
            }
#pragma unroll
            for (int rr = 0; rr < 8; rr++) {
#pragma unroll
                for (int n = 0; n < BN; n++) {
                    float v = sc[rr][n];
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1)
                        v += __shfl_down_sync(0xffffffffu, v, off);
                    if (lane0) s_p[rg*8+rr][n] = v * sm_scale;
                }
            }
            __syncthreads();
            if (tid < QROWS) {
                int r = tid;
                int tok = mb * TOKENS + (r % TOKENS);
                float m_old = s_m[r], m_new = m_old;
#pragma unroll
                for (int n = 0; n < BN; n++) {
                    int pos = n0 + n;
                    bool vis = (n0 + n) < n_iters;
                    if (phase == 1) vis = vis && (pos <= tok);   // causal
                    if (!vis) { s_p[r][n] = -1e30f; continue; }
                    m_new = fmaxf(m_new, s_p[r][n]);
                }
                float alpha = __expf(m_old - m_new);
                float l = s_l[r] * alpha;
#pragma unroll
                for (int n = 0; n < BN; n++) {
                    float p = __expf(s_p[r][n] - m_new);
                    s_p[r][n] = p;
                    l += p;
                }
                s_m[r] = m_new; s_l[r] = l; s_alpha[r] = alpha;
            }
            __syncthreads();
#pragma unroll
            for (int rr = 0; rr < 8; rr++) {
                float a = s_alpha[rg*8+rr];
#pragma unroll
                for (int dd = 0; dd < 8; dd++) {
                    float s = acc[rr][dd] * a;
#pragma unroll
                    for (int n = 0; n < BN; n++)
                        s += s_p[rg*8+rr][n] * __half2float(s_v[n][ds+dd]);
                    acc[rr][dd] = s;
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int rr = 0; rr < 8; rr++) {
        int r = rg*8+rr;
        int h_local = r / TOKENS;
        int tok = mb * TOKENS + (r % TOKENS);
        if (tok >= q_len || h_local >= GROUP) continue;
        float inv_l = 1.f / s_l[r];
        half* dst = o + (size_t)(q_start + tok) * sot
                    + (size_t)(kv_head * GROUP + h_local) * soh;
#pragma unroll
        for (int dd = 0; dd < 8; dd++)
            dst[ds+dd] = __float2half(acc[rr][dd] * inv_l);
    }
}

torch::Tensor gqa_extend(
    torch::Tensor q, torch::Tensor k_ext, torch::Tensor v_ext,
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor qo_indptr, torch::Tensor kv_indptr, torch::Tensor kv_indices,
    torch::Tensor prefix_lens, double sm_scale, int64_t stream)
{
    const int HQ = q.size(1), HKV = k_cache.size(1);
    const int nseq = qo_indptr.numel() - 1;
    const int max_q = q.size(0);
    const int GROUP = HQ / HKV;
    TORCH_CHECK(GROUP == 8 && 64 % GROUP == 0, "kernel specialized for GROUP=8");
    auto o = torch::empty_like(q);
    const int TOKENS = 64 / GROUP;
    int max_qlen = 0;
    for (int i = 0; i < nseq; i++)
        max_qlen = std::max(max_qlen, (int)(qo_indptr[i+1].item<int>() - qo_indptr[i].item<int>()));
    const int num_mb = (max_qlen + TOKENS - 1) / TOKENS;
    dim3 grid(nseq, HKV, num_mb);
    auto qs = q.strides(); auto ks = k_cache.strides(); auto es = k_ext.strides();
    auto vs = v_cache.strides(); auto ves = v_ext.strides(); auto os = o.strides();
    gqa_extend_kernel<<<grid, NTHREADS, 0, (cudaStream_t)stream>>>(
        (const half*)q.data_ptr(), (const half*)k_ext.data_ptr(),
        (const half*)v_ext.data_ptr(), (const half*)k_cache.data_ptr(),
        (const half*)v_cache.data_ptr(),
        qo_indptr.data_ptr<int>(), kv_indptr.data_ptr<int>(),
        kv_indices.data_ptr<int>(), prefix_lens.data_ptr<int>(),
        (float)sm_scale, (half*)o.data_ptr(), HQ, HKV, num_mb,
        qs[0], qs[1], ks[0], ks[1], es[0], es[1],
        vs[0], vs[1], ves[0], ves[1], os[0], os[1]);
    return o;
}
"""

_CPP_SRC = "torch::Tensor gqa_extend(torch::Tensor q, torch::Tensor k_ext, torch::Tensor v_ext, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor qo_indptr, torch::Tensor kv_indptr, torch::Tensor kv_indices, torch::Tensor prefix_lens, double sm_scale, int64_t stream);"

_mod = None


def _build():
    global _mod
    if _mod is not None:
        return _mod
    from torch.utils.cpp_extension import load_inline

    _mod = load_inline(
        name="freetoken_gqa_extend_pascal_v4",
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        functions=["gqa_extend"],
        extra_cuda_cflags=["-O3", "--use_fast_math",
                           "-gencode=arch=compute_61,code=sm_61"],
        verbose=False,
    )
    return _mod


def gqa_extend_paged(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices,
                     prefix_lens, max_q_len, sm_scale, k_extend=None, v_extend=None):
    """Same contract as extend_paged_attention's split path (k_extend/v_extend required)."""
    assert k_extend is not None and v_extend is not None
    return _build().gqa_extend(q, k_extend, v_extend, k_cache, v_cache,
                               qo_indptr, kv_indptr, kv_indices, prefix_lens,
                               float(sm_scale), torch.cuda.current_stream().cuda_stream)
