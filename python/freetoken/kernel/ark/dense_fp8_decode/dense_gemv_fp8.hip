// Radeon Operator Library — dense fp8 block-scale GEMV (HIP, W8A16, decode M=1).
//
// For the DeepSeek-V3-style fp8-block dense linears (GDN in_proj/out_proj, lm_head): weight [N,K] fp8-e4m3
// with a per-128x128-block bf16 scale (weight_scale_inv [N/128, K/128]); activation x [1,K] bf16 stays
// bf16 (weight-only fp8). One wave per output row reduces over K in 128-element blocks, each block's partial
// scaled by weight_scale_inv[n/128][kb] before the single final warp reduce (the block scale is a scalar,
// so it factors out of the reduction). Vectorized: each lane reads a uint (4 fp8 weights) + a float2 of
// 4 bf16 activations per step (K%128==0, N any). Mirrors moe_decode_gemv_fp8 stage2 minus expert/h/atomic.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

namespace radeon_ops {
namespace hip_native {

using bf16 = __hip_bfloat16;

__device__ __forceinline__ float dg_wred(float x) {
  for (int o = warpSize / 2; o > 0; o >>= 1) x += __shfl_xor(x, o);
  return x;
}

__device__ __forceinline__ float dg_e4m3_to_f32(unsigned char v) {
#if defined(__gfx1201__)
  return __builtin_amdgcn_cvt_f32_fp8((int)v,0);
#else
  unsigned short bits = (unsigned short)((unsigned)(v & 0x80) << 8) |
                        (unsigned short)((unsigned)(v & 0x7F) << 7);
  __half hf;
  __builtin_memcpy(&hf, &bits, sizeof(hf));
  return __half2float(hf) * 256.0f;
#endif
}

// Split-K: NW warps per output row, each striding over a 1/NW share of the K-blocks (shorter serial
// reduction chain at large K), combined across warps via LDS. NW=1 is the plain wave-per-output path.
template <int NW>
__global__ void __launch_bounds__(NW * 32) dense_gemv_fp8_k(const bf16* __restrict__ x,
                                 const unsigned char* __restrict__ w, const bf16* __restrict__ wscale,
                                 bf16* __restrict__ out, int N, int K, int wscale_stride) {
  int n = blockIdx.x;
  if (n >= N) return;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned char* wr = w + (size_t)n * K;
  const bf16* sc = wscale + (size_t)(n / 128) * wscale_stride;   // 128 output rows share one scale row
  int nkb = K / 128;
  float acc = 0.f;                                               // per-lane running sum (this warp's blocks)
  for (int kb = warp; kb < nkb; kb += NW) {
    int idx = kb * 128 + lane * 4;                              // 32 lanes x 4 fp8 = 128 weights/block
    unsigned int wv = *(const unsigned int*)(wr + idx);
    float2 xv = *(const float2*)(x + idx);
    const bf16* xp = (const bf16*)&xv;
    float p = 0.f;
    #pragma unroll
    for (int b = 0; b < 4; ++b) p += dg_e4m3_to_f32((wv >> (8 * b)) & 0xFF) * __bfloat162float(xp[b]);
    acc += p * __bfloat162float(sc[kb]);
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor(acc, o);    // reduce this warp's lanes
  if (NW == 1) {
    if (lane == 0) out[n] = (bf16)acc;
    return;
  }
  __shared__ float red[NW];
  if (lane == 0) red[warp] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < NW; ++i) s += red[i];
    out[n] = (bf16)s;
  }
}

template <int ROWS>
__global__ void __launch_bounds__(ROWS * 32) dense_gemv_fp8_rows(const bf16* __restrict__ x,
                                 const unsigned char* __restrict__ w, const bf16* __restrict__ wscale,
                                 bf16* __restrict__ out, int N, int K, int wscale_stride) {
  int n = blockIdx.x*ROWS+(threadIdx.x>>5);
  if (n >= N) return;
  int warp = 0, lane = threadIdx.x & 31;
  const unsigned char* wr = w + (size_t)n * K;
  const bf16* sc = wscale + (size_t)(n / 128) * wscale_stride;   // 128 output rows share one scale row
  int nkb = K / 128;
  float acc = 0.f;                                               // per-lane running sum (this warp's blocks)
  for (int kb = warp; kb < nkb; kb += 1) {
    int idx = kb * 128 + lane * 4;                              // 32 lanes x 4 fp8 = 128 weights/block
    unsigned int wv = *(const unsigned int*)(wr + idx);
    float2 xv = *(const float2*)(x + idx);
    const bf16* xp = (const bf16*)&xv;
    float p = 0.f;
    #pragma unroll
    for (int b = 0; b < 4; ++b) p += dg_e4m3_to_f32((wv >> (8 * b)) & 0xFF) * __bfloat162float(xp[b]);
    acc += p * __bfloat162float(sc[kb]);
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor(acc, o);    // reduce this warp's lanes
  if (1 == 1) {
    if (lane == 0) out[n] = (bf16)acc;
    return;
  }
  __shared__ float red[1];
  if (lane == 0) red[warp] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < 1; ++i) s += red[i];
    out[n] = (bf16)s;
  }
}

void launch_dense_gemv_fp8(const bf16* x, const unsigned char* w, const bf16* wscale, bf16* out,
                           int N, int K, int wscale_stride, int nw, hipStream_t stream) {
  (void)K;
  // auto split-K by N: small N underfills the GPU at NW=1, so split each output across more warps; large
  // N already has plenty of output waves -> NW=1. NOTE this op WINS at moderate N (e.g. GDN out_proj
  // K=4096 N=2048 ~2x both arches) but LOSES to triton's tuned big-N GEMV at large N (e.g. N>=16384 on
  // gfx1201) -- the caller should gate on N and keep triton for large-N projections (lm_head, in_proj).
  if (nw <= 0) nw = (N <= 2048) ? 4 : (N <= 8192 ? 2 : 1);
  dim3 grid((unsigned)N);
  if (nw >= 4)      hipLaunchKernelGGL((dense_gemv_fp8_k<4>), grid, dim3(128), 0, stream, x, w, wscale, out, N, K, wscale_stride);
  else if (nw == 2) hipLaunchKernelGGL((dense_gemv_fp8_k<2>), grid, dim3(64),  0, stream, x, w, wscale, out, N, K, wscale_stride);
  else              hipLaunchKernelGGL((dense_gemv_fp8_rows<4>), dim3((N+3)/4), dim3(128), 0, stream, x, w, wscale, out, N, K, wscale_stride);
}

}  // namespace hip_native
}  // namespace radeon_ops

// Dense fp8 block-scale GEMV (W8A16, M=1). x [1,K] bf16; w [N,K] fp8-e4m3; wscale [N/128,K/128] bf16
// (weight_scale_inv; wscale_stride = its dim-0/per-output-block stride); out [1,N] bf16. K % 128 == 0.
extern "C" int radeon_dense_gemv_fp8(const void* x, const void* w, const void* wscale, void* out,
                                     int N, int K, int wscale_stride, int nw, void* stream) {
  radeon_ops::hip_native::launch_dense_gemv_fp8(
      (const __hip_bfloat16*)x, (const unsigned char*)w, (const __hip_bfloat16*)wscale,
      (__hip_bfloat16*)out, N, K, wscale_stride, nw, (hipStream_t)stream);
  return (int)hipGetLastError();
}
