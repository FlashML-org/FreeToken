// IsoQuant (ISO3/ISO4) paged attention kernels (decode + extend).
//
// Decode: one threadblock per (request, kv_head); Q heads of the GQA group
// share the dequantized K/V pairs. Online softmax in fp32. K/V are dequantized
// on the fly from the packed pool (iso_common.cuh).
//
// Extend (prefill): one threadblock per (request, kv_head, q_index); the query
// token attends the packed prefix [0, ctx) plus the bf16 extend tokens
// [0, q_index] (causal). New tokens are NOT read from the packed pool — the
// backend stores them after attention (deferred quantization), so prefill
// attention never consumes freshly-quantized values.

#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>
#include <freetoken/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

#include "iso_common.cuh"

namespace {

using iso::IsoBlockBytes;

constexpr float kNegInf = -1e30f;

// Online-softmax state for one query head, kept in shared memory.
struct SoftmaxState {
  float m;      // running max
  float l;      // running sum
  float p;      // last token weight
  float alpha;  // last rescale factor
};

// Block reduce of per-thread partials for GT query heads, then softmax state
// update (by the first GT threads). kThreads = 64*NB (2 or 4 warps).
template <int GT, int kNumWarps>
__device__ __forceinline__ void block_softmax_update(
    float (&partial)[GT], float scale, SoftmaxState *states,
    float *red,  // [kNumWarps * GT]
    int tid) {
  const int lane = tid & 31;
  const int warp = tid >> 5;
#pragma unroll
  for (int g = 0; g < GT; ++g) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      partial[g] += __shfl_xor_sync(0xffffffffu, partial[g], off);
  }
  if (lane == 0)
#pragma unroll
    for (int g = 0; g < GT; ++g) red[warp * GT + g] = partial[g];
  __syncthreads();
  if (tid < GT) {
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < kNumWarps; ++w) s += red[w * GT + tid];
    s *= scale;
    SoftmaxState &st = states[tid];
    const float m_new = fmaxf(st.m, s);
    const float alpha = expf(st.m - m_new);
    const float p = expf(s - m_new);
    st.l = st.l * alpha + p;
    st.m = m_new;
    st.p = p;
    st.alpha = alpha;
  }
  __syncthreads();
}

template <int GT>
__device__ __forceinline__ void accumulate_v(float (&acc0)[GT], float (&acc1)[GT],
                                             float vf0, float vf1,
                                             const SoftmaxState *states) {
#pragma unroll
  for (int g = 0; g < GT; ++g) {
    acc0[g] = acc0[g] * states[g].alpha + states[g].p * vf0;
    acc1[g] = acc1[g] * states[g].alpha + states[g].p * vf1;
  }
}

template <int GT>
__device__ __forceinline__ void write_out_pair(__nv_bfloat16 *out_row, int g,
                                               int j0, float a0, float a1,
                                               float l) {
  const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
  __nv_bfloat162 val = __floats2bfloat162_rn(a0 * inv_l, a1 * inv_l);
  *reinterpret_cast<__nv_bfloat162 *>(out_row + j0) = val;
}

// ---------------------------------------------------------------------------
// Decode: q (R, Hq*D) bf16 against packed paged KV.
// ---------------------------------------------------------------------------

struct IsoAttnDecodeParams {
  const __nv_bfloat16 *__restrict__ q;
  __nv_bfloat16 *__restrict__ out;
  const uint8_t *__restrict__ k_cache;
  const uint8_t *__restrict__ v_cache;
  const int32_t *__restrict__ kv_indptr;
  const int32_t *__restrict__ kv_indices;
  std::size_t q_row_bytes;
  std::size_t out_row_bytes;
  std::size_t cache_row_bytes;
  int nheads_q;
  int nheads_kv;
  int nb;
  float scale;
};

template <int FMT, int NB, int GT>
__global__ __launch_bounds__(64 * NB) void //
    iso_attn_decode_kernel(const __grid_constant__ IsoAttnDecodeParams params) {
  constexpr int D = NB * iso::kIsoBlock;
  constexpr int BB = IsoBlockBytes<FMT>::value;
  constexpr int kThreads = 64 * NB;
  constexpr int kNumWarps = kThreads / 32;

  const int req = blockIdx.x / params.nheads_kv;
  const int kvh = blockIdx.x % params.nheads_kv;
  const int gq = params.nheads_q / params.nheads_kv;
  if (gq > GT) return;  // host dispatches a sufficient GT

  const int tid = threadIdx.x;
  const int b = tid >> 6;         // 128-block index (0..NB-1)
  const int u = tid & 63;         // pair index within the block
  const int j0 = b * iso::kIsoBlock + 2 * u;

  extern __shared__ float smem[];
  float *qs = smem;                          // [GT][D]
  float *red = qs + GT * D;                  // [kNumWarps][GT]
  SoftmaxState *states =                     // [GT]
      reinterpret_cast<SoftmaxState *>(red + kNumWarps * GT);

  const __nv_bfloat16 *q_row = reinterpret_cast<const __nv_bfloat16 *>(
      reinterpret_cast<const uint8_t *>(params.q) + req * params.q_row_bytes);
  for (int idx = tid; idx < gq * D; idx += kThreads) {
    const int g = idx / D, e = idx % D;
    qs[idx] = __bfloat162float(q_row[(kvh * gq + g) * D + e]);
  }
  if (tid < gq) states[tid] = {kNegInf, 0.0f, 0.0f, 0.0f};
  __syncthreads();

  float acc0[GT], acc1[GT];
#pragma unroll
  for (int g = 0; g < GT; ++g) { acc0[g] = 0.0f; acc1[g] = 0.0f; }

  const int kv_start = params.kv_indptr[req];
  const int kv_end = params.kv_indptr[req + 1];
  for (int t = kv_start; t < kv_end; ++t) {
    const std::size_t slot = static_cast<std::size_t>(params.kv_indices[t]);
    const uint8_t *krow = params.k_cache + slot * params.cache_row_bytes +
                          kvh * (params.nb * BB);
    const uint8_t *vrow = params.v_cache + slot * params.cache_row_bytes +
                          kvh * (params.nb * BB);
    float kf0, kf1, vf0, vf1;
    iso::iso_dequant_pair<FMT>(krow + b * BB, u, kf0, kf1);
    iso::iso_dequant_pair<FMT>(vrow + b * BB, u, vf0, vf1);

    float partial[GT];
#pragma unroll
    for (int g = 0; g < GT; ++g)
      partial[g] = kf0 * qs[g * D + j0] + kf1 * qs[g * D + j0 + 1];
    block_softmax_update<GT, kNumWarps>(partial, params.scale, states, red, tid);
    accumulate_v<GT>(acc0, acc1, vf0, vf1, states);
    __syncthreads();
  }

  __nv_bfloat16 *out_row = reinterpret_cast<__nv_bfloat16 *>(
      reinterpret_cast<uint8_t *>(params.out) + req * params.out_row_bytes);
#pragma unroll
  for (int g = 0; g < GT; ++g) {
    if (g < gq)
      write_out_pair<GT>(out_row + (kvh * gq + g) * D, g, j0, acc0[g], acc1[g],
                         states[g].l);
  }
}

// ---------------------------------------------------------------------------
// Extend: q (total_new, Hq*D) bf16 against packed prefix + bf16 extend tokens.
// gridDim.y = max extend length; q token i attends prefix + extend[0..=i].
// ---------------------------------------------------------------------------

struct IsoAttnExtendParams {
  const __nv_bfloat16 *__restrict__ q;
  __nv_bfloat16 *__restrict__ out;
  const uint8_t *__restrict__ k_cache;
  const uint8_t *__restrict__ v_cache;
  const __nv_bfloat16 *__restrict__ k_ext;
  const __nv_bfloat16 *__restrict__ v_ext;
  const int32_t *__restrict__ qo_indptr;
  const int32_t *__restrict__ kv_indptr;   // prefix only
  const int32_t *__restrict__ kv_indices;  // prefix slots
  std::size_t q_row_bytes;
  std::size_t out_row_bytes;
  std::size_t cache_row_bytes;
  std::size_t ext_row_bytes;
  int nheads_q;
  int nheads_kv;
  int nb;
  float scale;
};

template <int FMT, int NB, int GT>
__global__ __launch_bounds__(64 * NB) void //
    iso_attn_extend_kernel(const __grid_constant__ IsoAttnExtendParams params) {
  constexpr int D = NB * iso::kIsoBlock;
  constexpr int BB = IsoBlockBytes<FMT>::value;
  constexpr int kThreads = 64 * NB;
  constexpr int kNumWarps = kThreads / 32;

  const int req = blockIdx.x / params.nheads_kv;
  const int kvh = blockIdx.x % params.nheads_kv;
  const int qi = blockIdx.y;
  const int q_start = params.qo_indptr[req];
  const int q_end = params.qo_indptr[req + 1];
  const int n_new = q_end - q_start;
  if (qi >= n_new) return;
  const int gq = params.nheads_q / params.nheads_kv;
  if (gq > GT) return;

  const int tid = threadIdx.x;
  const int b = tid >> 6;
  const int u = tid & 63;
  const int j0 = b * iso::kIsoBlock + 2 * u;

  extern __shared__ float smem[];
  float *qs = smem;
  float *red = qs + GT * D;
  SoftmaxState *states =
      reinterpret_cast<SoftmaxState *>(red + kNumWarps * GT);

  const __nv_bfloat16 *q_row = reinterpret_cast<const __nv_bfloat16 *>(
      reinterpret_cast<const uint8_t *>(params.q) +
      static_cast<std::size_t>(q_start + qi) * params.q_row_bytes);
  for (int idx = tid; idx < gq * D; idx += kThreads) {
    const int g = idx / D, e = idx % D;
    qs[idx] = __bfloat162float(q_row[(kvh * gq + g) * D + e]);
  }
  if (tid < gq) states[tid] = {kNegInf, 0.0f, 0.0f, 0.0f};
  __syncthreads();

  float acc0[GT], acc1[GT];
#pragma unroll
  for (int g = 0; g < GT; ++g) { acc0[g] = 0.0f; acc1[g] = 0.0f; }

  // 1) packed prefix
  const int kv_start = params.kv_indptr[req];
  const int kv_end = params.kv_indptr[req + 1];
  for (int t = kv_start; t < kv_end; ++t) {
    const std::size_t slot = static_cast<std::size_t>(params.kv_indices[t]);
    const uint8_t *krow = params.k_cache + slot * params.cache_row_bytes +
                          kvh * (params.nb * BB);
    const uint8_t *vrow = params.v_cache + slot * params.cache_row_bytes +
                          kvh * (params.nb * BB);
    float kf0, kf1, vf0, vf1;
    iso::iso_dequant_pair<FMT>(krow + b * BB, u, kf0, kf1);
    iso::iso_dequant_pair<FMT>(vrow + b * BB, u, vf0, vf1);

    float partial[GT];
#pragma unroll
    for (int g = 0; g < GT; ++g)
      partial[g] = kf0 * qs[g * D + j0] + kf1 * qs[g * D + j0 + 1];
    block_softmax_update<GT, kNumWarps>(partial, params.scale, states, red, tid);
    accumulate_v<GT>(acc0, acc1, vf0, vf1, states);
    __syncthreads();
  }

  // 2) bf16 extend tokens [0..=qi] (causal)
  const __nv_bfloat16 *k_ext_rows = params.k_ext + kvh * D;
  const __nv_bfloat16 *v_ext_rows = params.v_ext + kvh * D;
  for (int t = 0; t <= qi; ++t) {
    const std::size_t row = static_cast<std::size_t>(q_start + t);
    const __nv_bfloat16 *krow = reinterpret_cast<const __nv_bfloat16 *>(
        reinterpret_cast<const uint8_t *>(k_ext_rows) + row * params.ext_row_bytes);
    const __nv_bfloat16 *vrow = reinterpret_cast<const __nv_bfloat16 *>(
        reinterpret_cast<const uint8_t *>(v_ext_rows) + row * params.ext_row_bytes);
    const float kf0 = __bfloat162float(krow[j0]);
    const float kf1 = __bfloat162float(krow[j0 + 1]);
    const float vf0 = __bfloat162float(vrow[j0]);
    const float vf1 = __bfloat162float(vrow[j0 + 1]);

    float partial[GT];
#pragma unroll
    for (int g = 0; g < GT; ++g)
      partial[g] = kf0 * qs[g * D + j0] + kf1 * qs[g * D + j0 + 1];
    block_softmax_update<GT, kNumWarps>(partial, params.scale, states, red, tid);
    accumulate_v<GT>(acc0, acc1, vf0, vf1, states);
    __syncthreads();
  }

  __nv_bfloat16 *out_row = reinterpret_cast<__nv_bfloat16 *>(
      reinterpret_cast<uint8_t *>(params.out) +
      static_cast<std::size_t>(q_start + qi) * params.out_row_bytes);
#pragma unroll
  for (int g = 0; g < GT; ++g) {
    if (g < gq)
      write_out_pair<GT>(out_row + (kvh * gq + g) * D, g, j0, acc0[g], acc1[g],
                         states[g].l);
  }
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------

inline int gt_dispatch_target(int gq) {
  int gt = 1;
  while (gt < gq) gt <<= 1;
  return gt;
}

template <int fmt_bits, int nb, int gt>
struct IsoAttnDecodeKernel {
  static void run(const tvm::ffi::TensorView q, const tvm::ffi::TensorView out,
                  const tvm::ffi::TensorView k_cache,
                  const tvm::ffi::TensorView v_cache,
                  const tvm::ffi::TensorView kv_indptr,
                  const tvm::ffi::TensorView kv_indices, int64_t nheads_q,
                  int64_t nheads_kv, int64_t head_dim, double scale) {
    using namespace host;
    auto R = SymbolicSize{"R"};
    auto E = SymbolicSize{"E"};
    auto B = SymbolicSize{"B"};
    auto X = SymbolicSize{"X"};
    auto Y = SymbolicSize{"Y"};
    auto W = SymbolicSize{"W"};
    auto device_ = SymbolicDevice{};

    TensorMatcher({R, E})
        .with_strides({Y, 1})
        .with_device<kDLCUDA>(device_)
        .verify(q);
    TensorMatcher({R, E})
        .with_strides({W, 1})
        .with_device<kDLCUDA>(device_)
        .verify(out);
    TensorMatcher({-1, B})
        .with_strides({X, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype<uint8_t>()
        .verify(k_cache)
        .verify(v_cache);
    TensorMatcher({-1}).with_device<kDLCUDA>(device_).with_dtype<int32_t>().verify(kv_indptr);
    TensorMatcher({-1}).with_device<kDLCUDA>(device_).with_dtype<int32_t>().verify(kv_indices);
    RuntimeCheck(head_dim == nb * iso::kIsoBlock, "head_dim mismatch with template nb");
    RuntimeCheck(gt_dispatch_target(static_cast<int>(nheads_q / nheads_kv)) == gt,
                 "gt template mismatch");

    const auto device = device_.unwrap();
    constexpr int kThreads = 64 * nb;
    constexpr int kNumWarps = kThreads / 32;
    const std::size_t smem =
        sizeof(float) * (gt * head_dim + kNumWarps * gt) + sizeof(SoftmaxState) * gt;

    const auto params = IsoAttnDecodeParams{
        .q = static_cast<const __nv_bfloat16 *>(q.data_ptr()),
        .out = static_cast<__nv_bfloat16 *>(out.data_ptr()),
        .k_cache = static_cast<const uint8_t *>(k_cache.data_ptr()),
        .v_cache = static_cast<const uint8_t *>(v_cache.data_ptr()),
        .kv_indptr = static_cast<const int32_t *>(kv_indptr.data_ptr()),
        .kv_indices = static_cast<const int32_t *>(kv_indices.data_ptr()),
        .q_row_bytes = static_cast<std::size_t>(Y.unwrap()) * sizeof(__nv_bfloat16),
        .out_row_bytes = static_cast<std::size_t>(W.unwrap()) * sizeof(__nv_bfloat16),
        .cache_row_bytes = static_cast<std::size_t>(X.unwrap()),
        .nheads_q = static_cast<int>(nheads_q),
        .nheads_kv = static_cast<int>(nheads_kv),
        .nb = static_cast<int>(nb),
        .scale = static_cast<float>(scale),
    };

    const auto num_reqs = static_cast<std::size_t>(R.unwrap());
    LaunchKernel(num_reqs * static_cast<std::size_t>(nheads_kv), kThreads, device,
                 smem)(iso_attn_decode_kernel<fmt_bits, nb, gt>, params);
  }
};

template <int fmt_bits, int nb, int gt>
struct IsoAttnExtendKernel {
  static void run(const tvm::ffi::TensorView q, const tvm::ffi::TensorView out,
                  const tvm::ffi::TensorView k_cache,
                  const tvm::ffi::TensorView v_cache,
                  const tvm::ffi::TensorView k_ext,
                  const tvm::ffi::TensorView v_ext,
                  const tvm::ffi::TensorView qo_indptr,
                  const tvm::ffi::TensorView kv_indptr,
                  const tvm::ffi::TensorView kv_indices, int64_t nheads_q,
                  int64_t nheads_kv, int64_t head_dim, double scale,
                  int64_t max_extend) {
    using namespace host;
    auto R = SymbolicSize{"R"};
    auto E = SymbolicSize{"E"};
    auto B = SymbolicSize{"B"};
    auto X = SymbolicSize{"X"};
    auto Y = SymbolicSize{"Y"};
    auto Z = SymbolicSize{"Z"};
    auto W = SymbolicSize{"W"};
    auto device_ = SymbolicDevice{};

    TensorMatcher({-1, E})
        .with_strides({Y, 1})
        .with_device<kDLCUDA>(device_)
        .verify(q);
    TensorMatcher({-1, E})
        .with_strides({W, 1})
        .with_device<kDLCUDA>(device_)
        .verify(out);
    TensorMatcher({-1, -1})
        .with_strides({Z, 1})
        .with_device<kDLCUDA>(device_)
        .verify(k_ext)
        .verify(v_ext);
    TensorMatcher({-1, B})
        .with_strides({X, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype<uint8_t>()
        .verify(k_cache)
        .verify(v_cache);
    TensorMatcher({R}).with_device<kDLCUDA>(device_).with_dtype<int32_t>().verify(qo_indptr);
    TensorMatcher({R}).with_device<kDLCUDA>(device_).with_dtype<int32_t>().verify(kv_indptr);
    TensorMatcher({-1}).with_device<kDLCUDA>(device_).with_dtype<int32_t>().verify(kv_indices);
    RuntimeCheck(head_dim == nb * iso::kIsoBlock, "head_dim mismatch with template nb");
    RuntimeCheck(gt_dispatch_target(static_cast<int>(nheads_q / nheads_kv)) == gt,
                 "gt template mismatch");

    const auto device = device_.unwrap();
    constexpr int kThreads = 64 * nb;
    constexpr int kNumWarps = kThreads / 32;
    const std::size_t smem =
        sizeof(float) * (gt * head_dim + kNumWarps * gt) + sizeof(SoftmaxState) * gt;

    const auto params = IsoAttnExtendParams{
        .q = static_cast<const __nv_bfloat16 *>(q.data_ptr()),
        .out = static_cast<__nv_bfloat16 *>(out.data_ptr()),
        .k_cache = static_cast<const uint8_t *>(k_cache.data_ptr()),
        .v_cache = static_cast<const uint8_t *>(v_cache.data_ptr()),
        .k_ext = static_cast<const __nv_bfloat16 *>(k_ext.data_ptr()),
        .v_ext = static_cast<const __nv_bfloat16 *>(v_ext.data_ptr()),
        .qo_indptr = static_cast<const int32_t *>(qo_indptr.data_ptr()),
        .kv_indptr = static_cast<const int32_t *>(kv_indptr.data_ptr()),
        .kv_indices = static_cast<const int32_t *>(kv_indices.data_ptr()),
        .q_row_bytes = static_cast<std::size_t>(Y.unwrap()) * sizeof(__nv_bfloat16),
        .out_row_bytes = static_cast<std::size_t>(W.unwrap()) * sizeof(__nv_bfloat16),
        .cache_row_bytes = static_cast<std::size_t>(X.unwrap()),
        .ext_row_bytes = static_cast<std::size_t>(Z.unwrap()) * sizeof(__nv_bfloat16),
        .nheads_q = static_cast<int>(nheads_q),
        .nheads_kv = static_cast<int>(nheads_kv),
        .nb = static_cast<int>(nb),
        .scale = static_cast<float>(scale),
    };

    const auto num_reqs = static_cast<std::size_t>(R.unwrap() - 1);
    dim3 grid(static_cast<unsigned>(num_reqs * static_cast<std::size_t>(nheads_kv)),
              static_cast<unsigned>(max_extend));
    LaunchKernel(grid, kThreads, device, smem)(
        iso_attn_extend_kernel<fmt_bits, nb, gt>, params);
  }
};

} // namespace
