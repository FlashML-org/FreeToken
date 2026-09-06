// Adapted from
// https://github.com/vllm-project/vllm/blob/755ed7b05be4743237d3339c4ff8c22bcaae04f4/csrc/quantization/gguf/gguf_kernel.cu
// Algorithm cross-check: llama.cpp 7e4c0a968 (b10434),
// ggml/src/ggml-cuda/mmvq.cu and ggml/src/ggml-cuda/ggml-cuda.cu.
// Local HIP wrappers below preserve the GGUF packed Q4_K/Q5_K/Q6_K/Q8_0
// contracts; native strided Q5_K/Q6_K is FreeToken-specific cache glue.
#if defined(USE_ROCM)
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#else
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif
#include <torch/all.h>

#if defined(USE_ROCM)
#define GGUF_DEVICE_GUARD(device) c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device)
#define GGUF_CURRENT_STREAM() c10::hip::getCurrentHIPStreamMasqueradingAsCUDA()
#else
#define GGUF_DEVICE_GUARD(device) at::cuda::OptionalCUDAGuard device_guard(device)
#define GGUF_CURRENT_STREAM() at::cuda::getCurrentCUDAStream()
#endif

// dont use clang-format here, it breaks the include order
// clang-format off
#include "dispatch.h"

#if defined(USE_ROCM)
// These are checked-in HIP translations. Do not rely on ignored/generated *.hip
// sidecars; this selector is the active JIT source contract.
#include "ggml-common_hip.h"
#include "vecdotq_hip.cuh"
#include "dequantize_hip.cuh"
#include "mmvq_hip.cuh"
#include "mmq_hip.cuh"
#include "moe_hip.cuh"
#include "moe_vec_hip.cuh"
#else
#include "ggml-common.h"
#include "vecdotq.cuh"
#include "dequantize.cuh"
#include "mmvq.cuh"
#include "mmq.cuh"
#include "moe.cuh"
#include "moe_vec.cuh"
#endif
// clang-format off

// Q8 gemv
template <typename scalar_t>
static __global__ void
quantize_q8_1(const scalar_t* __restrict__ x, void* __restrict__ vy, const int kx, const int kx_padded) {
  const auto ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const auto iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1* y = (block_q8_1*)vy;

  const int ib = i_padded / QK8_1;   // block index
  const int iqs = i_padded % QK8_1;  // quant index

  const float xi = ix < kx ? static_cast<float>(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, SGLANG_SHFL_XOR_SYNC_WIDTH(uint32_t(-1), amax, mask, 32));
    sum += SGLANG_SHFL_XOR_SYNC_WIDTH(uint32_t(-1), sum, mask, 32);
  }

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  y[ib].ds.x = __float2half(d);
  y[ib].ds.y = __float2half(sum);
}

template <typename scalar_t>
static void quantize_row_q8_1_cuda(const scalar_t* x, void* vy, const int kx, const int ky, cudaStream_t stream) {
  const int64_t kx_padded = (kx + 512 - 1) / 512 * 512;
  const int block_num_x = (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  constexpr int MAX_BLOCK_SIZE = 65535;
  for (int off = 0; off < ky; off += MAX_BLOCK_SIZE) {
    const int num_blocks_y = std::min(ky, off + MAX_BLOCK_SIZE) - off;
    const dim3 num_blocks(block_num_x, num_blocks_y, 1);
    const dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(
        &x[off * kx], (int32_t*)vy + off * (kx_padded / 32 * 9), kx, kx_padded);
  }
}

torch::Tensor ggml_dequantize(
    torch::Tensor W,  // quant weight
    int64_t type,
    int64_t m,
    int64_t n,
    std::optional<at::ScalarType> const& dtype) {
  const GGUF_DEVICE_GUARD(device_of(W));
  auto dtype_ = dtype.value_or(torch::kFloat16);
  auto options = torch::TensorOptions().dtype(dtype_).device(W.device());
  at::Tensor DW = torch::empty({m, n}, options);
  cudaStream_t stream = GGUF_CURRENT_STREAM().stream();

  DISPATCH_FLOAT_TYPES(DW.scalar_type(), "ggml_dequantize", [&] {
    auto to_cuda = ggml_get_to_cuda<scalar_t>(type);
    to_cuda((void*)W.data_ptr(), (scalar_t*)DW.data_ptr(), m * n, stream);
  });

  return DW;
}

torch::Tensor ggml_mul_mat_vec_a8(
    torch::Tensor W,  // quant weight
    torch::Tensor X,  // input
    int64_t type,
    int64_t row) {
  int col = X.sizes()[1];
  int vecs = X.sizes()[0];
  const int padded = (col + 512 - 1) / 512 * 512;
  const GGUF_DEVICE_GUARD(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({vecs, row}, options);
  cudaStream_t stream = GGUF_CURRENT_STREAM().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({vecs, padded / 32 * 9}, options);
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_mul_mat_vec_a8", [&] {
    quantize_row_q8_1_cuda<scalar_t>((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, vecs, stream);
    switch (type) {
      case 2:
        mul_mat_vec_q4_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 3:
        mul_mat_vec_q4_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 6:
        mul_mat_vec_q5_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 7:
        mul_mat_vec_q5_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 8:
        mul_mat_vec_q8_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 10:
        mul_mat_vec_q2_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 11:
        mul_mat_vec_q3_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 12:
        mul_mat_vec_q4_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 13:
        mul_mat_vec_q5_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 14:
        mul_mat_vec_q6_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 16:
        mul_mat_vec_iq2_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 17:
        mul_mat_vec_iq2_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 18:
        mul_mat_vec_iq3_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 19:
        mul_mat_vec_iq1_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 20:
        mul_mat_vec_iq4_nl_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 21:
        mul_mat_vec_iq3_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 22:
        mul_mat_vec_iq2_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 23:
        mul_mat_vec_iq4_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 29:
        mul_mat_vec_iq1_m_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
    }
  });
  return Y;
}

torch::Tensor ggml_mul_mat_a8(
    torch::Tensor W,  // quant weight
    torch::Tensor X,  // input
    int64_t type,
    int64_t row) {
  int col = X.sizes()[1];
  int padded = (col + 512 - 1) / 512 * 512;
  int batch = X.sizes()[0];
  const GGUF_DEVICE_GUARD(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({batch, row}, options);
  cudaStream_t stream = GGUF_CURRENT_STREAM().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({batch, padded / 32 * 9}, options);
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_mul_mat_a8", [&] {
    quantize_row_q8_1_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, batch, stream);

    switch (type) {
      case 2:
        ggml_mul_mat_q4_0_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 3:
        ggml_mul_mat_q4_1_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 6:
        ggml_mul_mat_q5_0_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 7:
        ggml_mul_mat_q5_1_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 8:
        ggml_mul_mat_q8_0_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 10:
        ggml_mul_mat_q2_K_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 11:
        ggml_mul_mat_q3_K_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 12:
        ggml_mul_mat_q4_K_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 13:
        ggml_mul_mat_q5_K_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 14:
        ggml_mul_mat_q6_K_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
    }
  });
  return Y;
}

torch::Tensor ggml_moe_a8(
    torch::Tensor X,  // input
    torch::Tensor W,  // expert weights
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded,
    int64_t type,
    int64_t row,
    int64_t top_k,
    int64_t tokens) {
  int col = X.sizes()[1];
  int padded = (col + 512 - 1) / 512 * 512;
  const GGUF_DEVICE_GUARD(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({tokens * top_k, row}, options);
  cudaStream_t stream = GGUF_CURRENT_STREAM().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({tokens, padded / 32 * 9}, options);
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_moe_a8", [&] {
    quantize_row_q8_1_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, tokens, stream);
    switch (type) {
      case 2:
        ggml_moe_q4_0_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 3:
        ggml_moe_q4_1_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 6:
        ggml_moe_q5_0_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 7:
        ggml_moe_q5_1_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 8:
        ggml_moe_q8_0_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 10:
        ggml_moe_q2_K_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 11:
        ggml_moe_q3_K_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 12:
        ggml_moe_q4_K_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 13:
        ggml_moe_q5_K_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 14:
        ggml_moe_q6_K_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
    }
  });
  return Y;
}

torch::Tensor ggml_moe_a8_vec(
    torch::Tensor X,  // input
    torch::Tensor W,  // expert weights
    torch::Tensor topk_ids,
    int64_t top_k,
    int64_t type,
    int64_t row,
    int64_t tokens,
    torch::Tensor output = torch::Tensor(),
    torch::Tensor quant_X_input = torch::Tensor()) {
  int col = X.sizes()[1];
  const int padded = (col + 512 - 1) / 512 * 512;
  const GGUF_DEVICE_GUARD(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y;
  if (output.defined()) {
    TORCH_CHECK(output.is_cuda() && output.is_contiguous(),
                "ggml_moe_a8_vec output must be contiguous CUDA/HIP tensor");
    TORCH_CHECK(output.device() == W.device() && output.dtype() == X.dtype(),
                "ggml_moe_a8_vec output device/dtype must match input/output");
    TORCH_CHECK(output.sizes() == torch::IntArrayRef({tokens * top_k, row}),
                "ggml_moe_a8_vec output shape mismatch");
    Y = output;
    // The legacy vector kernels skip invalid/padded routes. Preserve their old zero-fill
    // contract while allowing callers to reuse fixed graph-address output storage.
    Y.zero_();
  } else {
    Y = torch::zeros({tokens * top_k, row}, options);
  }
  cudaStream_t stream = GGUF_CURRENT_STREAM().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X;
  if (quant_X_input.defined()) {
    TORCH_CHECK(quant_X_input.is_cuda() && quant_X_input.is_contiguous(),
                "ggml_moe_a8_vec quant_X must be contiguous CUDA/HIP tensor");
    TORCH_CHECK(quant_X_input.device() == W.device() &&
                    quant_X_input.scalar_type() == torch::kInt32,
                "ggml_moe_a8_vec quant_X device/dtype mismatch");
    TORCH_CHECK(quant_X_input.sizes() == torch::IntArrayRef({tokens, padded / 32 * 9}),
                "ggml_moe_a8_vec quant_X shape mismatch");
    quant_X = quant_X_input;
  } else {
    quant_X = torch::empty({tokens, padded / 32 * 9}, options);
  }
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_moe_vec_a8", [&] {
    quantize_row_q8_1_cuda<scalar_t>((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, tokens, stream);
    switch (type) {
      case 2:
        moe_vec_q4_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 3:
        moe_vec_q4_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 6:
        moe_vec_q5_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 7:
        moe_vec_q5_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 8:
        moe_vec_q8_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 10:
        moe_vec_q2_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 11:
        moe_vec_q3_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 12:
        moe_vec_q4_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 13:
        moe_vec_q5_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 14:
        moe_vec_q6_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 16:
        moe_vec_iq2_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 17:
        moe_vec_iq2_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 18:
        moe_vec_iq3_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 19:
        moe_vec_iq1_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 20:
        moe_vec_iq4_nl_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 21:
        moe_vec_iq3_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 22:
        moe_vec_iq2_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 23:
        moe_vec_iq4_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 29:
        moe_vec_iq1_m_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
    }
  });
  return Y;
}

torch::Tensor ggml_moe_a8_vec_strided(
    torch::Tensor X, torch::Tensor W, torch::Tensor topk_ids,
    int64_t top_k, int64_t type, int64_t row, int64_t tokens,
    int64_t expert_stride_bytes, int64_t row_stride_bytes,
    torch::Tensor output = torch::Tensor(),
    torch::Tensor quant_X_input = torch::Tensor()) {
  TORCH_CHECK(X.is_cuda() && W.is_cuda() && topk_ids.is_cuda(),
              "ggml_moe_a8_vec_strided requires CUDA/HIP tensors");
  TORCH_CHECK(X.is_contiguous() && W.is_contiguous() && topk_ids.is_contiguous(),
              "ggml_moe_a8_vec_strided requires contiguous tensors");
  TORCH_CHECK(type == 8 || type == 12 || type == 13 || type == 14,
              "ggml_moe_a8_vec_strided supports Q8_0 (8), Q4_K (12), Q5_K (13), and Q6_K (14), got ", type);
  TORCH_CHECK(X.dim() == 2 && W.dim() == 3 && topk_ids.dim() == 2,
              "invalid ggml_moe_a8_vec_strided tensor ranks");
  TORCH_CHECK(tokens == X.size(0) && top_k == topk_ids.size(1),
              "ggml_moe_a8_vec_strided token/top-k shape mismatch");
  TORCH_CHECK(expert_stride_bytes == W.stride(0) && row_stride_bytes == W.stride(1),
              "weight strides must match contiguous uint8 tensor");
  const int col = X.size(1);
  const int padded = (col + 512 - 1) / 512 * 512;
  const GGUF_DEVICE_GUARD(device_of(X));
  auto output_options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y;
  if (output.defined()) {
    TORCH_CHECK(output.is_cuda() && output.is_contiguous(),
                "ggml_moe_a8_vec_strided output must be contiguous CUDA/HIP tensor");
    TORCH_CHECK(output.device() == W.device() && output.dtype() == X.dtype(),
                "ggml_moe_a8_vec_strided output device/dtype mismatch");
    TORCH_CHECK(output.sizes() == torch::IntArrayRef({tokens * top_k, row}),
                "ggml_moe_a8_vec_strided output shape mismatch");
    Y = output;
    Y.zero_();
  } else {
    Y = torch::zeros({tokens * top_k, row}, output_options);
  }
  auto quant_options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X;
  if (quant_X_input.defined()) {
    TORCH_CHECK(quant_X_input.is_cuda() && quant_X_input.is_contiguous(),
                "ggml_moe_a8_vec_strided quant_X must be contiguous CUDA/HIP tensor");
    TORCH_CHECK(quant_X_input.device() == W.device() &&
                    quant_X_input.scalar_type() == torch::kInt32,
                "ggml_moe_a8_vec_strided quant_X device/dtype mismatch");
    TORCH_CHECK(quant_X_input.sizes() == torch::IntArrayRef({tokens, padded / 32 * 9}),
                "ggml_moe_a8_vec_strided quant_X shape mismatch");
    quant_X = quant_X_input;
  } else {
    quant_X = torch::empty({tokens, padded / 32 * 9}, quant_options);
  }
  cudaStream_t stream = GGUF_CURRENT_STREAM().stream();
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_moe_vec_a8_strided", [&] {
    quantize_row_q8_1_cuda<scalar_t>(
        (scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, tokens, stream);
    if (type == 8) {
      moe_vec_q8_0_q8_1_strided_cuda<scalar_t>(
          (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(),
          (int*)topk_ids.data_ptr(), top_k, tokens, col, row,
          quant_X.stride(0), expert_stride_bytes, row_stride_bytes, stream);
    } else if (type == 12) {
      moe_vec_q4_K_q8_1_strided_cuda<scalar_t>(
          (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(),
          (int*)topk_ids.data_ptr(), top_k, tokens, col, row,
          quant_X.stride(0), expert_stride_bytes, row_stride_bytes, stream);
    } else if (type == 13) {
      moe_vec_q5_K_q8_1_strided_cuda<scalar_t>(
          (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(),
          (int*)topk_ids.data_ptr(), top_k, tokens, col, row,
          quant_X.stride(0), expert_stride_bytes, row_stride_bytes, stream);
    } else {
      moe_vec_q6_K_q8_1_strided_cuda<scalar_t>(
          (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(),
          (int*)topk_ids.data_ptr(), top_k, tokens, col, row,
          quant_X.stride(0), expert_stride_bytes, row_stride_bytes, stream);
    }
  });
  return Y;
}

int64_t ggml_moe_get_block_size(int64_t type) {
  switch (type) {
    case 2:
      return MOE_X_Q4_0;
    case 3:
      return MOE_X_Q4_1;
    case 6:
      return MOE_X_Q5_0;
    case 7:
      return MOE_X_Q5_1;
    case 8:
      return MOE_X_Q8_0;
    case 10:
      return MOE_X_Q2_K;
    case 11:
      return MOE_X_Q3_K;
    case 12:
      return MOE_X_Q4_K;
    case 13:
      return MOE_X_Q5_K;
    case 14:
      return MOE_X_Q6_K;
  }
  return 0;
}

// ---- FreeToken pybind bindings (donor registers these via TORCH_LIBRARY; we
// expose them through torch.utils.cpp_extension.load's pybind module instead) ----
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ggml_dequantize", &ggml_dequantize, "");
  m.def("ggml_mul_mat_vec_a8", &ggml_mul_mat_vec_a8, "");
  m.def("ggml_mul_mat_a8", &ggml_mul_mat_a8, "");
  m.def("ggml_moe_a8", &ggml_moe_a8, "");
  m.def("ggml_moe_a8_vec",
        [](torch::Tensor X, torch::Tensor W, torch::Tensor topk_ids,
           int64_t top_k, int64_t type, int64_t row, int64_t tokens) {
          return ggml_moe_a8_vec(X, W, topk_ids, top_k, type, row, tokens);
        }, "",
        py::arg("X"), py::arg("W"), py::arg("topk_ids"), py::arg("top_k"),
        py::arg("type"), py::arg("row"), py::arg("tokens"));
  m.def("ggml_moe_a8_vec",
        [](torch::Tensor X, torch::Tensor W, torch::Tensor topk_ids,
           int64_t top_k, int64_t type, int64_t row, int64_t tokens,
           torch::Tensor output) {
          return ggml_moe_a8_vec(X, W, topk_ids, top_k, type, row, tokens, output);
        }, "",
        py::arg("X"), py::arg("W"), py::arg("topk_ids"), py::arg("top_k"),
        py::arg("type"), py::arg("row"), py::arg("tokens"), py::arg("output"));
  m.def("ggml_moe_a8_vec_workspace",
        [](torch::Tensor X, torch::Tensor W, torch::Tensor topk_ids,
           int64_t top_k, int64_t type, int64_t row, int64_t tokens,
           torch::Tensor output, torch::Tensor quant_X) {
          return ggml_moe_a8_vec(
              X, W, topk_ids, top_k, type, row, tokens, output, quant_X);
        }, "",
        py::arg("X"), py::arg("W"), py::arg("topk_ids"), py::arg("top_k"),
        py::arg("type"), py::arg("row"), py::arg("tokens"), py::arg("output"),
        py::arg("quant_X"));
  m.def("ggml_moe_a8_vec_strided",
        [](torch::Tensor X, torch::Tensor W, torch::Tensor topk_ids,
           int64_t top_k, int64_t type, int64_t row, int64_t tokens,
           int64_t expert_stride_bytes, int64_t row_stride_bytes) {
          return ggml_moe_a8_vec_strided(
              X, W, topk_ids, top_k, type, row, tokens,
              expert_stride_bytes, row_stride_bytes);
        }, "",
        py::arg("X"), py::arg("W"), py::arg("topk_ids"), py::arg("top_k"),
        py::arg("type"), py::arg("row"), py::arg("tokens"),
        py::arg("expert_stride_bytes"), py::arg("row_stride_bytes"));
  m.def("ggml_moe_a8_vec_strided",
        [](torch::Tensor X, torch::Tensor W, torch::Tensor topk_ids,
           int64_t top_k, int64_t type, int64_t row, int64_t tokens,
           int64_t expert_stride_bytes, int64_t row_stride_bytes,
           torch::Tensor output) {
          return ggml_moe_a8_vec_strided(
              X, W, topk_ids, top_k, type, row, tokens,
              expert_stride_bytes, row_stride_bytes, output);
        }, "",
        py::arg("X"), py::arg("W"), py::arg("topk_ids"), py::arg("top_k"),
        py::arg("type"), py::arg("row"), py::arg("tokens"),
        py::arg("expert_stride_bytes"), py::arg("row_stride_bytes"),
        py::arg("output"));
  m.def("ggml_moe_a8_vec_strided_workspace",
        [](torch::Tensor X, torch::Tensor W, torch::Tensor topk_ids,
           int64_t top_k, int64_t type, int64_t row, int64_t tokens,
           int64_t expert_stride_bytes, int64_t row_stride_bytes,
           torch::Tensor output, torch::Tensor quant_X) {
          return ggml_moe_a8_vec_strided(
              X, W, topk_ids, top_k, type, row, tokens,
              expert_stride_bytes, row_stride_bytes, output, quant_X);
        }, "",
        py::arg("X"), py::arg("W"), py::arg("topk_ids"), py::arg("top_k"),
        py::arg("type"), py::arg("row"), py::arg("tokens"),
        py::arg("expert_stride_bytes"), py::arg("row_stride_bytes"),
        py::arg("output"), py::arg("quant_X"));
  m.def("ggml_moe_get_block_size", &ggml_moe_get_block_size, "");
}
