// Separate gfx1100 GGUF MoE candidate module.  Legacy gguf_kernel.cu remains
// independently loadable so candidate compile failure cannot strand fallback.
#if defined(USE_ROCM)
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
using cudaStream_t = hipStream_t;
#else
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif
#include <torch/all.h>
#include <torch/extension.h>
#include <string>

#if defined(USE_ROCM)
#define GGUF_DEVICE_GUARD(device) c10::cuda::OptionalCUDAGuard device_guard(device)
#define GGUF_CURRENT_STREAM() c10::cuda::getCurrentCUDAStream()
#else
#define GGUF_DEVICE_GUARD(device) at::cuda::OptionalCUDAGuard device_guard(device)
#define GGUF_CURRENT_STREAM() at::cuda::getCurrentCUDAStream()
#endif

// Keep include order aligned with gguf_kernel.cu. These are the active JIT
// headers; ignored/generated *.hip sidecars are not dependencies.
#include "dispatch.h"
#if defined(USE_ROCM)
#include "ggml-common_hip.h"
#include "vecdotq_hip.cuh"
#include "moe_vec_gfx1100_hip.cuh"
#else
#include "ggml-common.h"
#include "vecdotq.cuh"
#include "moe_vec_gfx1100.cuh"
#endif

template <typename scalar_t>
static __global__ void quantize_q8_1_gfx1100(
    const scalar_t* __restrict__ x, void* __restrict__ vy,
    const int kx, const int kx_padded) {
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;
  block_q8_1* y = static_cast<block_q8_1*>(vy);
  const int ib = i_padded / QK8_1;
  const int iqs = i_padded % QK8_1;
  const float xi = ix < kx ? static_cast<float>(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, SGLANG_SHFL_XOR_SYNC_WIDTH(uint32_t(-1), amax, mask, 32));
    sum += SGLANG_SHFL_XOR_SYNC_WIDTH(uint32_t(-1), sum, mask, 32);
  }

  const float d = amax / 127;
  y[ib].qs[iqs] = amax == 0.0f ? 0 : static_cast<int8_t>(roundf(xi / d));
  if (iqs == 0) {
    y[ib].ds.x = __float2half(d);
    y[ib].ds.y = __float2half(sum);
  }
}

template <typename scalar_t>
static void quantize_row_q8_1_gfx1100(
    const scalar_t* x, void* vy, const int kx, const int ky,
    cudaStream_t stream) {
  const int64_t kx_padded = (kx + 512 - 1) / 512 * 512;
  const int block_num_x = (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) /
      CUDA_QUANTIZE_BLOCK_SIZE;
  constexpr int max_block_size = 65535;
  for (int off = 0; off < ky; off += max_block_size) {
    const int num_blocks_y = std::min(ky, off + max_block_size) - off;
    quantize_q8_1_gfx1100<scalar_t><<<
        dim3(block_num_x, num_blocks_y, 1), dim3(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1),
        0, stream>>>(
        &x[off * kx], static_cast<int32_t*>(vy) + off * (kx_padded / 32 * 9),
        kx, kx_padded);
  }
}

torch::Tensor ggml_moe_a8_vec_gfx1100(
    torch::Tensor X, torch::Tensor W, torch::Tensor topk_ids,
    int64_t top_k, int64_t type, int64_t row, int64_t tokens) {
  TORCH_CHECK(X.is_cuda() && W.is_cuda() && topk_ids.is_cuda(),
              "gfx1100 GGUF MoE candidate requires CUDA/HIP tensors");
  TORCH_CHECK(X.is_contiguous() && W.is_contiguous() && topk_ids.is_contiguous(),
              "gfx1100 GGUF MoE candidate requires contiguous tensors");
  TORCH_CHECK(type == 8 || type == 12,
              "gfx1100 candidate supports Q8_0 (8) and Q4_K (12), got ", type);
  TORCH_CHECK(X.dim() == 2 && W.dim() == 3 && topk_ids.dim() == 2,
              "invalid gfx1100 GGUF MoE tensor ranks");
  const int col = X.sizes()[1];
  const int padded = (col + 512 - 1) / 512 * 512;
  const GGUF_DEVICE_GUARD(device_of(X));
  auto output_options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::zeros({tokens * top_k, row}, output_options);
  auto quant_options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({tokens, padded / 32 * 9}, quant_options);
  cudaStream_t stream = GGUF_CURRENT_STREAM().stream();

  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_moe_a8_vec_gfx1100", [&] {
    quantize_row_q8_1_gfx1100<scalar_t>(
        static_cast<scalar_t*>(X.data_ptr()), quant_X.data_ptr(), col, tokens, stream);
    if (type == 12) {
      moe_vec_q4_K_q8_1_gfx1100<scalar_t>(
          W.data_ptr(), quant_X.data_ptr(), static_cast<scalar_t*>(Y.data_ptr()),
          static_cast<int*>(topk_ids.data_ptr()), top_k, tokens, col, row,
          quant_X.stride(0), stream);
    } else {
      moe_vec_q8_0_q8_1_gfx1100<scalar_t>(
          W.data_ptr(), quant_X.data_ptr(), static_cast<scalar_t*>(Y.data_ptr()),
          static_cast<int*>(topk_ids.data_ptr()), top_k, tokens, col, row,
          quant_X.stride(0), stream);
    }
  });
  return Y;
}

torch::Tensor ggml_moe_mmvq_id(
    torch::Tensor X, torch::Tensor W, torch::Tensor topk_ids,
    int64_t top_k, int64_t type, int64_t row, int64_t tokens,
    int64_t expert_stride_bytes, int64_t row_stride_bytes,
    const std::string& id_space,
    torch::Tensor output = torch::Tensor(),
    torch::Tensor quant_X_input = torch::Tensor()) {
  TORCH_CHECK(X.is_cuda() && W.is_cuda() && topk_ids.is_cuda(),
              "ggml_moe_mmvq_id requires CUDA/HIP tensors");
  TORCH_CHECK(X.is_contiguous() && topk_ids.is_contiguous(),
              "ggml_moe_mmvq_id requires contiguous activation/ID tensors");
  TORCH_CHECK(W.dim() == 3 && W.scalar_type() == torch::kUInt8 && W.stride(2) == 1,
              "ggml_moe_mmvq_id requires packed uint8 [E,rows,bytes] weights");
  TORCH_CHECK(X.dim() == 2 && topk_ids.dim() == 2 && topk_ids.scalar_type() == torch::kInt,
              "invalid ggml_moe_mmvq_id tensor ranks or ID dtype");
  TORCH_CHECK(tokens == X.size(0) && top_k == topk_ids.size(1),
              "ggml_moe_mmvq_id token/top-k shape mismatch");
  TORCH_CHECK(type == 8 || type == 12 || type == 13 || type == 14,
              "ggml_moe_mmvq_id supports Q4_K/Q5_K/Q6_K/Q8_0, got ", type);
  TORCH_CHECK(id_space == "raw" || id_space == "slot",
              "ggml_moe_mmvq_id id_space must be raw or slot");
  TORCH_CHECK(expert_stride_bytes == W.stride(0) && row_stride_bytes == W.stride(1),
              "GGUF expert/row strides must match the supplied packed bank");
  const int col = X.size(1);
  const int padded = (col + 512 - 1) / 512 * 512;
  const GGUF_DEVICE_GUARD(device_of(X));
  auto output_options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y;
  if (output.defined()) {
    TORCH_CHECK(output.is_cuda() && output.is_contiguous() && output.device() == W.device(),
                "ggml_moe_mmvq_id output must be contiguous on bank device");
    TORCH_CHECK(output.scalar_type() == X.scalar_type() &&
                    output.sizes() == torch::IntArrayRef({tokens * top_k, row}),
                "ggml_moe_mmvq_id output shape/dtype mismatch");
    Y = output;
    Y.zero_();
  } else {
    Y = torch::zeros({tokens * top_k, row}, output_options);
  }
  auto quant_options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X;
  if (quant_X_input.defined()) {
    TORCH_CHECK(quant_X_input.is_cuda() && quant_X_input.is_contiguous() &&
                    quant_X_input.device() == W.device() &&
                    quant_X_input.scalar_type() == torch::kInt32 &&
                    quant_X_input.sizes() == torch::IntArrayRef({tokens, padded / 32 * 9}),
                "ggml_moe_mmvq_id quant_X shape/device/dtype mismatch");
    quant_X = quant_X_input;
  } else {
    quant_X = torch::empty({tokens, padded / 32 * 9}, quant_options);
  }
  cudaStream_t stream = GGUF_CURRENT_STREAM().stream();
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_moe_mmvq_id", [&] {
    quantize_row_q8_1_gfx1100<scalar_t>(
        static_cast<scalar_t*>(X.data_ptr()), quant_X.data_ptr(), col, tokens, stream);
    if (type == 12) {
      moe_vec_q4_K_q8_1_gfx1100_id<scalar_t>(
          W.data_ptr(), quant_X.data_ptr(), static_cast<scalar_t*>(Y.data_ptr()),
          static_cast<const int*>(topk_ids.data_ptr()), top_k, tokens, col, row,
          quant_X.stride(0), expert_stride_bytes, row_stride_bytes, stream);
    } else if (type == 13) {
      moe_vec_q5_K_q8_1_gfx1100_id<scalar_t>(
          W.data_ptr(), quant_X.data_ptr(), static_cast<scalar_t*>(Y.data_ptr()),
          static_cast<const int*>(topk_ids.data_ptr()), top_k, tokens, col, row,
          quant_X.stride(0), expert_stride_bytes, row_stride_bytes, stream);
    } else if (type == 14) {
      moe_vec_q6_K_q8_1_gfx1100_id<scalar_t>(
          W.data_ptr(), quant_X.data_ptr(), static_cast<scalar_t*>(Y.data_ptr()),
          static_cast<const int*>(topk_ids.data_ptr()), top_k, tokens, col, row,
          quant_X.stride(0), expert_stride_bytes, row_stride_bytes, stream);
    } else {
      moe_vec_q8_0_q8_1_gfx1100_id<scalar_t>(
          W.data_ptr(), quant_X.data_ptr(), static_cast<scalar_t*>(Y.data_ptr()),
          static_cast<const int*>(topk_ids.data_ptr()), top_k, tokens, col, row,
          quant_X.stride(0), expert_stride_bytes, row_stride_bytes, stream);
    }
  });
  return Y;
}

template <typename scalar_t>
static __global__ void moe_gate_up_swiglu_id_kernel(
    const uint8_t* __restrict__ weights,
    const int32_t* __restrict__ quant_x,
    scalar_t* __restrict__ output,
    const int* __restrict__ ids,
    const int top_k,
    const int ncols,
    const int nrows,
    const int quant_x_stride,
    const int64_t expert_stride_bytes,
    const int64_t row_stride_bytes,
    const int experts) {
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  const int route = blockIdx.y;
  const int expert = ids[route];
  if (row >= nrows || expert < 0 || expert >= experts) return;

  const int token = route / top_k;
  const int blocks_per_row = ncols / QK_K;
  const int lanes_per_chunk = QI4_K / VDR_Q4_K_Q8_1_MMVQ;
  const int blocks_per_wave = VDR_Q4_K_Q8_1_MMVQ * WARP_SIZE / QI4_K;
  const uint8_t* expert_base = weights + expert * expert_stride_bytes;
  const uint8_t* gate_base = expert_base + row * row_stride_bytes;
  const uint8_t* up_base = expert_base + (row + nrows) * row_stride_bytes;
  const block_q8_1* x = reinterpret_cast<const block_q8_1*>(
      quant_x + token * quant_x_stride);
  float gate = 0.0f;
  float up = 0.0f;
  for (int i = threadIdx.x / lanes_per_chunk; i < blocks_per_row;
       i += blocks_per_wave) {
    const int iqs = VDR_Q4_K_Q8_1_MMVQ * (threadIdx.x % lanes_per_chunk);
    gate += vec_dot_q4_K_q8_1(gate_base + i * sizeof(block_q4_K), &x[i], iqs);
    up += vec_dot_q4_K_q8_1(up_base + i * sizeof(block_q4_K), &x[i], iqs);
  }
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    gate += SGLANG_SHFL_XOR_SYNC(uint32_t(-1), gate, mask);
    up += SGLANG_SHFL_XOR_SYNC(uint32_t(-1), up, mask);
  }
  if (threadIdx.x == 0) {
    const float silu = gate / (1.0f + expf(-gate));
    output[route * nrows + row] = static_cast<scalar_t>(silu * up);
  }
}

torch::Tensor ggml_moe_gate_up_swiglu_id(
    torch::Tensor X, torch::Tensor W, torch::Tensor topk_ids,
    int64_t top_k, int64_t nrows, int64_t tokens,
    int64_t expert_stride_bytes, int64_t row_stride_bytes,
    const std::string& id_space, torch::Tensor output = torch::Tensor(),
    torch::Tensor quant_X_input = torch::Tensor()) {
  TORCH_CHECK(X.is_cuda() && W.is_cuda() && topk_ids.is_cuda(),
              "ggml_moe_gate_up_swiglu_id requires CUDA/HIP tensors");
  TORCH_CHECK(X.is_contiguous() && topk_ids.is_contiguous(),
              "ggml_moe_gate_up_swiglu_id requires contiguous activation/ID tensors");
  TORCH_CHECK(W.dim() == 3 && W.scalar_type() == torch::kUInt8 && W.stride(2) == 1,
              "ggml_moe_gate_up_swiglu_id requires packed uint8 [E,2I,bytes] weights");
  TORCH_CHECK(X.dim() == 2 && topk_ids.dim() == 2 && topk_ids.scalar_type() == torch::kInt,
              "invalid ggml_moe_gate_up_swiglu_id tensor ranks or ID dtype");
  TORCH_CHECK(tokens == X.size(0) && top_k == topk_ids.size(1) &&
                  nrows * 2 == W.size(1),
              "ggml_moe_gate_up_swiglu_id shape mismatch");
  TORCH_CHECK(id_space == "raw" || id_space == "slot",
              "ggml_moe_gate_up_swiglu_id id_space must be raw or slot");
  const int col = X.size(1);
  TORCH_CHECK(col > 0 && col % QK_K == 0 && row_stride_bytes >=
                  (col / QK_K) * static_cast<int64_t>(sizeof(block_q4_K)) &&
                  expert_stride_bytes >= W.size(1) * row_stride_bytes,
              "ggml_moe_gate_up_swiglu_id requires aligned columns and valid strides");
  const GGUF_DEVICE_GUARD(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y;
  if (output.defined()) {
    TORCH_CHECK(output.is_cuda() && output.is_contiguous() && output.device() == W.device() &&
                    output.scalar_type() == X.scalar_type() &&
                    output.sizes() == torch::IntArrayRef({tokens * top_k, nrows}),
                "ggml_moe_gate_up_swiglu_id output shape/device/dtype mismatch");
    Y = output;
    Y.zero_();
  } else {
    Y = torch::zeros({tokens * top_k, nrows}, options);
  }
  const int padded = (col + 512 - 1) / 512 * 512;
  auto quant_options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X;
  if (quant_X_input.defined()) {
    TORCH_CHECK(quant_X_input.is_cuda() && quant_X_input.is_contiguous() &&
                    quant_X_input.device() == W.device() &&
                    quant_X_input.scalar_type() == torch::kInt32 &&
                    quant_X_input.sizes() == torch::IntArrayRef({tokens, padded / 32 * 9}),
                "ggml_moe_gate_up_swiglu_id quant_X shape/device/dtype mismatch");
    quant_X = quant_X_input;
  } else {
    quant_X = torch::empty({tokens, padded / 32 * 9}, quant_options);
  }
  cudaStream_t stream = GGUF_CURRENT_STREAM().stream();
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_moe_gate_up_swiglu_id", [&] {
    quantize_row_q8_1_gfx1100<scalar_t>(
        static_cast<const scalar_t*>(X.data_ptr()), quant_X.data_ptr(), col, tokens, stream);
    moe_gate_up_swiglu_id_kernel<scalar_t><<<
        dim3((nrows + 3) / 4, tokens * top_k, 1), dim3(WARP_SIZE, 4, 1), 0, stream>>>(
        W.data_ptr<uint8_t>(), quant_X.data_ptr<int32_t>(),
        static_cast<scalar_t*>(Y.data_ptr()), static_cast<const int*>(topk_ids.data_ptr()),
        top_k, col, nrows, quant_X.stride(0), expert_stride_bytes, row_stride_bytes,
        W.size(0));
  });
  return Y;
}

static __device__ __forceinline__ void mmvdq_get_scale_min_k4(
    int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

template <int type>
static __device__ __forceinline__ float mmvdq_value(
    const uint8_t* block, int index);

template <>
static __device__ __forceinline__ float mmvdq_value<12>(
    const uint8_t* block, int index) {
  const block_q4_K* q = reinterpret_cast<const block_q4_K*>(block);
  const int il = index / 64;
  const int local = index & 63;
  const bool high = local >= 32;
  const int in_group = high ? local - 32 : local;
  const int ir = in_group / 4;
  const int l = in_group & 3;
  const int scale_index = 2 * il + (high ? 1 : 0);
  uint8_t scale, minimum;
  mmvdq_get_scale_min_k4(scale_index, q->scales, scale, minimum);
  const half d = __hmul(__low2half(q->dm), __int2half_rn(scale));
  const half m = __hmul(__high2half(q->dm), __int2half_rn(minimum));
  const uint8_t packed = q->qs[32 * il + 4 * ir + l];
  const int value = high ? packed >> 4 : packed & 0xF;
  return __half2float(__hsub(__hmul(d, __int2half_rn(value)), m));
}

template <>
static __device__ __forceinline__ float mmvdq_value<13>(
    const uint8_t* block, int index) {
  const block_q5_K* q = reinterpret_cast<const block_q5_K*>(block);
  const int il = index / 64;
  const int local = index & 63;
  const bool high = local >= 32;
  const int in_group = high ? local - 32 : local;
  const int ir = in_group / 2;
  const int l = in_group & 1;
  const int scale_index = 2 * il + (high ? 1 : 0);
  uint8_t scale, minimum;
  mmvdq_get_scale_min_k4(scale_index, q->scales, scale, minimum);
  const half d = __hmul(__low2half(q->dm), __int2half_rn(scale));
  const half m = __hmul(__high2half(q->dm), __int2half_rn(minimum));
  const int high_bit = high ? 1 : 0;
  const uint8_t packed = q->qs[32 * il + 2 * ir + l];
  const uint8_t high_bits = q->qh[2 * ir + l];
  const int value = (high ? packed >> 4 : packed & 0xF) +
      ((high_bits & (1u << (2 * il + high_bit))) ? 16 : 0);
  return __half2float(__hsub(__hmul(d, __int2half_rn(value)), m));
}

template <>
static __device__ __forceinline__ float mmvdq_value<14>(
    const uint8_t* block, int index) {
  const block_q6_K* q = reinterpret_cast<const block_q6_K*>(block);
  const int ip = index / 128;
  const int local = index & 127;
  const int group = local / 32;
  const int pos = local & 31;
  const uint8_t ql = q->ql[64 * ip + (group & 1) * 32 + pos];
  const uint8_t qh = q->qh[32 * ip + pos];
  const int low = group < 2 ? ql & 0xF : ql >> 4;
  const int value = low | (((qh >> (2 * group)) & 3) << 4);
  const int scale_index = 8 * ip + (pos / 16) + 2 * group;
  return __half2float(__hmul(
      q->d, __int2half_rn(q->scales[scale_index] * (value - 32))));
}

template <typename scalar_t, int type, int block_bytes>
static __global__ void moe_mmvdq_id_kernel(
    const uint8_t* __restrict__ weights,
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ output,
    const int* __restrict__ ids,
    const int top_k,
    const int ncols,
    const int nrows,
    const int64_t expert_stride_bytes,
    const int64_t row_stride_bytes,
    const int experts) {
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  const int route = blockIdx.y;
  const int expert = ids[route];
  if (row >= nrows || expert < 0 || expert >= experts) return;

  const int token = route / top_k;
  const uint8_t* row_data = weights + expert * expert_stride_bytes +
      row * row_stride_bytes;
  float sum = 0.0f;
  for (int k = threadIdx.x; k < ncols; k += blockDim.x) {
    const uint8_t* block = row_data + (k / QK_K) * block_bytes;
    sum += mmvdq_value<type>(block, k & (QK_K - 1)) *
        static_cast<float>(x[token * ncols + k]);
  }

  const int lane = threadIdx.x;
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    sum += SGLANG_SHFL_XOR_SYNC(uint32_t(-1), sum, mask);
  }
  if (lane == 0) output[route * nrows + row] = static_cast<scalar_t>(sum);
}

torch::Tensor ggml_moe_mmvdq_id(
    torch::Tensor X, torch::Tensor W, torch::Tensor topk_ids,
    int64_t top_k, int64_t type, int64_t row, int64_t tokens,
    int64_t expert_stride_bytes, int64_t row_stride_bytes,
    const std::string& id_space, torch::Tensor output = torch::Tensor()) {
  TORCH_CHECK(X.is_cuda() && W.is_cuda() && topk_ids.is_cuda(),
              "ggml_moe_mmvdq_id requires CUDA/HIP tensors");
  TORCH_CHECK(X.is_contiguous() && topk_ids.is_contiguous(),
              "ggml_moe_mmvdq_id requires contiguous activation/ID tensors");
  TORCH_CHECK(W.dim() == 3 && W.scalar_type() == torch::kUInt8 && W.stride(2) == 1,
              "ggml_moe_mmvdq_id requires packed uint8 [E,rows,bytes] weights");
  TORCH_CHECK(X.dim() == 2 && topk_ids.dim() == 2 && topk_ids.scalar_type() == torch::kInt,
              "invalid ggml_moe_mmvdq_id tensor ranks or ID dtype");
  TORCH_CHECK(tokens == X.size(0) && top_k == topk_ids.size(1),
              "ggml_moe_mmvdq_id token/top-k shape mismatch");
  TORCH_CHECK(type == 12 || type == 13 || type == 14,
              "ggml_moe_mmvdq_id supports Q4_K/Q5_K/Q6_K, got ", type);
  TORCH_CHECK(id_space == "raw" || id_space == "slot",
              "ggml_moe_mmvdq_id id_space must be raw or slot");
  TORCH_CHECK(X.scalar_type() == torch::kFloat || X.scalar_type() == torch::kHalf ||
              X.scalar_type() == torch::kBFloat16,
              "ggml_moe_mmvdq_id supports F32/F16/BF16 activations");
  const int col = X.size(1);
  TORCH_CHECK(col > 0 && col % QK_K == 0 && row == W.size(1),
              "ggml_moe_mmvdq_id requires QK_K-aligned columns and matching rows");
  const int block_bytes = type == 12 ? sizeof(block_q4_K) :
      (type == 13 ? sizeof(block_q5_K) : sizeof(block_q6_K));
  TORCH_CHECK(row_stride_bytes >= (col / QK_K) * block_bytes &&
                  expert_stride_bytes >= row * row_stride_bytes,
              "GGUF MMVDQ strides are smaller than packed row extent");
  const GGUF_DEVICE_GUARD(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y;
  if (output.defined()) {
    TORCH_CHECK(output.is_cuda() && output.is_contiguous() && output.device() == W.device() &&
                    output.scalar_type() == X.scalar_type() &&
                    output.sizes() == torch::IntArrayRef({tokens * top_k, row}),
                "ggml_moe_mmvdq_id output shape/device/dtype mismatch");
    Y = output;
    Y.zero_();
  } else {
    Y = torch::zeros({tokens * top_k, row}, options);
  }
  cudaStream_t stream = GGUF_CURRENT_STREAM().stream();
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_moe_mmvdq_id", [&] {
    const scalar_t* x = static_cast<const scalar_t*>(X.data_ptr());
    scalar_t* y = static_cast<scalar_t*>(Y.data_ptr());
    const int* ids = static_cast<const int*>(topk_ids.data_ptr());
    if (type == 12) {
      moe_mmvdq_id_kernel<scalar_t, 12, sizeof(block_q4_K)><<<
          dim3((row + 3) / 4, tokens * top_k, 1), dim3(WARP_SIZE, 4, 1), 0, stream>>>(
          W.data_ptr<uint8_t>(), x, y, ids, top_k, col, row,
          expert_stride_bytes, row_stride_bytes, W.size(0));
    } else if (type == 13) {
      moe_mmvdq_id_kernel<scalar_t, 13, sizeof(block_q5_K)><<<
          dim3((row + 3) / 4, tokens * top_k, 1), dim3(WARP_SIZE, 4, 1), 0, stream>>>(
          W.data_ptr<uint8_t>(), x, y, ids, top_k, col, row,
          expert_stride_bytes, row_stride_bytes, W.size(0));
    } else {
      moe_mmvdq_id_kernel<scalar_t, 14, sizeof(block_q6_K)><<<
          dim3((row + 3) / 4, tokens * top_k, 1), dim3(WARP_SIZE, 4, 1), 0, stream>>>(
          W.data_ptr<uint8_t>(), x, y, ids, top_k, col, row,
          expert_stride_bytes, row_stride_bytes, W.size(0));
    }
  });
  return Y;
}

void bind_gguf_moe_gfx1100(pybind11::module_& m) {
  m.def("ggml_moe_a8_vec_gfx1100", &ggml_moe_a8_vec_gfx1100, "");
  m.def("ggml_moe_mmvq_id",
        [](torch::Tensor X, torch::Tensor W, torch::Tensor ids,
           int64_t top_k, int64_t type, int64_t row, int64_t tokens,
           int64_t expert_stride, int64_t row_stride, const std::string& id_space) {
          return ggml_moe_mmvq_id(X, W, ids, top_k, type, row, tokens,
                                  expert_stride, row_stride, id_space);
        }, "");
  m.def("ggml_moe_mmvq_id_workspace",
        [](torch::Tensor X, torch::Tensor W, torch::Tensor ids,
           int64_t top_k, int64_t type, int64_t row, int64_t tokens,
           int64_t expert_stride, int64_t row_stride, const std::string& id_space,
           torch::Tensor output, torch::Tensor quant_X) {
          return ggml_moe_mmvq_id(X, W, ids, top_k, type, row, tokens,
                                  expert_stride, row_stride, id_space, output, quant_X);
        }, "");
  m.def("ggml_moe_mmvdq_id",
        [](torch::Tensor X, torch::Tensor W, torch::Tensor ids,
           int64_t top_k, int64_t type, int64_t row, int64_t tokens,
           int64_t expert_stride, int64_t row_stride, const std::string& id_space) {
          return ggml_moe_mmvdq_id(X, W, ids, top_k, type, row, tokens,
                                  expert_stride, row_stride, id_space);
        }, "");
  m.def("ggml_moe_mmvdq_id_workspace",
        [](torch::Tensor X, torch::Tensor W, torch::Tensor ids,
           int64_t top_k, int64_t type, int64_t row, int64_t tokens,
           int64_t expert_stride, int64_t row_stride, const std::string& id_space,
           torch::Tensor output) {
          return ggml_moe_mmvdq_id(X, W, ids, top_k, type, row, tokens,
                                  expert_stride, row_stride, id_space, output);
        }, "");
  m.def("ggml_moe_gate_up_swiglu_id",
        [](torch::Tensor X, torch::Tensor W, torch::Tensor ids,
           int64_t top_k, int64_t nrows, int64_t tokens,
           int64_t expert_stride, int64_t row_stride, const std::string& id_space) {
          return ggml_moe_gate_up_swiglu_id(X, W, ids, top_k, nrows, tokens,
                                            expert_stride, row_stride, id_space);
        }, "");
  m.def("ggml_moe_gate_up_swiglu_id_workspace",
        [](torch::Tensor X, torch::Tensor W, torch::Tensor ids,
           int64_t top_k, int64_t nrows, int64_t tokens,
           int64_t expert_stride, int64_t row_stride, const std::string& id_space,
           torch::Tensor output, torch::Tensor quant_X) {
          return ggml_moe_gate_up_swiglu_id(X, W, ids, top_k, nrows, tokens,
                                            expert_stride, row_stride, id_space,
                                            output, quant_X);
        }, "");
}

#ifndef FREETOKEN_GGUF_NO_PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  bind_gguf_moe_gfx1100(m);
}
#endif
