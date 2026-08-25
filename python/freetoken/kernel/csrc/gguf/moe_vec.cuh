// copied from
// https://github.com/vllm-project/vllm/blob/4492e3a55428e161ca8db381edc28263e5da4c8d/csrc/quantization/gguf/moe_vec.cuh
// copied and adapted from
// https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/mmvq.cu
template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static __global__ void moe_vec_q(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* topk_ids,
    const int topk,
    const int ncols,
    const int nrows,
    const int token_stride) {
  const auto row = blockIdx.x * blockDim.y + threadIdx.y;

  const auto token = blockIdx.z / topk;
  const auto expert = (topk_ids)[blockIdx.z];

  if (row >= nrows) {
    return;
  }

  const int blocks_per_row = ncols / qk;
  const int blocks_per_warp = vdr * WARP_SIZE / qi;

  // partial sum for each thread
  float tmp = 0.0f;

  const block_q_t* x = ((const block_q_t*)vx) + expert * nrows * blocks_per_row;
  const block_q8_1* y = (const block_q8_1*)(((const int*)vy) + token * token_stride);

  for (auto i = threadIdx.x / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
    const int ibx = row * blocks_per_row + i;  // x block index

    const int iby = i * (qk / QK8_1);  // y block index that aligns with ibx

    const int iqs = vdr * (threadIdx.x % (qi / vdr));  // x block quant index when casting the quants to int

    tmp += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
  }

  // sum up partial sums and write back result
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    tmp += SGLANG_SHFL_XOR_SYNC(uint32_t(-1), tmp, mask);
  }

  if (threadIdx.x == 0) {
    dst[blockIdx.z * nrows + row] = tmp;
  }
}


// CUDA caps gridDim.z at 65535 on every architecture. The routed-pair index
// (token * top_k + slot) is carried in z, so a prefill wider than 65535 / top_k tokens
// overflows it and cudaLaunchKernel fails with cudaErrorInvalidConfiguration, surfaced by
// torch as "CUDA error: invalid argument". At top_k 8 that ceiling is only 8191 tokens,
// which any long prompt crosses; it is not architecture- or OS-specific.
//
// Fixed by launching in token-aligned chunks rather than by reshaping the grid. Moving the
// pair index into x would also fit, but it would cost the locality this kernel is built
// around: with the row index in x, consecutive blocks walk rows of the SAME expert and
// reuse its weight pages. Chunking keeps that intact and leaves the kernel untouched.
//
// Chunks are whole tokens so the kernel's own token = blockIdx.z / topk arithmetic stays
// valid against the offset pointers.
#define MOE_VEC_MAX_GRID_Z 65535

template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda>
static void moe_vec_launch(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int top_k,
    const int tokens,
    const int ncols,
    const int nrows,
    const int token_stride,
    cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  // top_k is the router's experts-per-token (<= 128 in practice), so this is >= 511.
  const int tokens_per_chunk = MOE_VEC_MAX_GRID_Z / (top_k > 0 ? top_k : 1);
  for (int t0 = 0; t0 < tokens; t0 += tokens_per_chunk) {
    const int nt = (tokens - t0) < tokens_per_chunk ? (tokens - t0) : tokens_per_chunk;
    const dim3 block_nums(block_num_y, 1, nt * top_k);
    moe_vec_q<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda>
        <<<block_nums, block_dims, 0, stream>>>(
            vx,
            (const void*)(((const int*)vy) + (size_t)t0 * token_stride),
            dst + (size_t)t0 * top_k * nrows,
            topk_ids + (size_t)t0 * top_k,
            top_k, ncols, nrows, token_stride);
  }
}

template <typename scalar_t>
static void moe_vec_q4_0_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int top_k,
    const int tokens,
    const int ncols,
    const int nrows,
    const int token_stride,
    cudaStream_t stream) {
  moe_vec_launch<scalar_t, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
      vx, vy, dst, topk_ids, top_k, tokens, ncols, nrows, token_stride, stream);
}

template <typename scalar_t>
static void moe_vec_q4_1_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int top_k,
    const int tokens,
    const int ncols,
    const int nrows,
    const int token_stride,
    cudaStream_t stream) {
  moe_vec_launch<scalar_t, QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
      vx, vy, dst, topk_ids, top_k, tokens, ncols, nrows, token_stride, stream);
}

template <typename scalar_t>
static void moe_vec_q5_0_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int top_k,
    const int tokens,
    const int ncols,
    const int nrows,
    const int token_stride,
    cudaStream_t stream) {
  moe_vec_launch<scalar_t, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
      vx, vy, dst, topk_ids, top_k, tokens, ncols, nrows, token_stride, stream);
}

template <typename scalar_t>
static void moe_vec_q5_1_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int top_k,
    const int tokens,
    const int ncols,
    const int nrows,
    const int token_stride,
    cudaStream_t stream) {
  moe_vec_launch<scalar_t, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
      vx, vy, dst, topk_ids, top_k, tokens, ncols, nrows, token_stride, stream);
}

template <typename scalar_t>
static void moe_vec_q8_0_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int top_k,
    const int tokens,
    const int ncols,
    const int nrows,
    const int token_stride,
    cudaStream_t stream) {
  moe_vec_launch<scalar_t, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
      vx, vy, dst, topk_ids, top_k, tokens, ncols, nrows, token_stride, stream);
}

template <typename scalar_t>
static void moe_vec_q2_K_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int top_k,
    const int tokens,
    const int ncols,
    const int nrows,
    const int token_stride,
    cudaStream_t stream) {
  moe_vec_launch<scalar_t, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
      vx, vy, dst, topk_ids, top_k, tokens, ncols, nrows, token_stride, stream);
}

template <typename scalar_t>
static void moe_vec_q3_K_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int top_k,
    const int tokens,
    const int ncols,
    const int nrows,
    const int token_stride,
    cudaStream_t stream) {
  moe_vec_launch<scalar_t, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
      vx, vy, dst, topk_ids, top_k, tokens, ncols, nrows, token_stride, stream);
}

template <typename scalar_t>
static void moe_vec_q4_K_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int top_k,
    const int tokens,
    const int ncols,
    const int nrows,
    const int token_stride,
    cudaStream_t stream) {
  moe_vec_launch<scalar_t, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
      vx, vy, dst, topk_ids, top_k, tokens, ncols, nrows, token_stride, stream);
}

template <typename scalar_t>
static void moe_vec_q5_K_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int top_k,
    const int tokens,
    const int ncols,
    const int nrows,
    const int token_stride,
    cudaStream_t stream) {
  moe_vec_launch<scalar_t, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
      vx, vy, dst, topk_ids, top_k, tokens, ncols, nrows, token_stride, stream);
}

template <typename scalar_t>
static void moe_vec_q6_K_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int top_k,
    const int tokens,
    const int ncols,
    const int nrows,
    const int token_stride,
    cudaStream_t stream) {
  moe_vec_launch<scalar_t, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
      vx, vy, dst, topk_ids, top_k, tokens, ncols, nrows, token_stride, stream);
}

template <typename scalar_t>
static void moe_vec_iq2_xxs_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int top_k,
    const int tokens,
    const int ncols,
    const int nrows,
    const int token_stride,
    cudaStream_t stream) {
  moe_vec_launch<scalar_t, QK_K, QI2_XXS, block_iq2_xxs, 1, vec_dot_iq2_xxs_q8_1>(
      vx, vy, dst, topk_ids, top_k, tokens, ncols, nrows, token_stride, stream);
}

template <typename scalar_t>
static void moe_vec_iq2_xs_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int top_k,
    const int tokens,
    const int ncols,
    const int nrows,
    const int token_stride,
    cudaStream_t stream) {
  moe_vec_launch<scalar_t, QK_K, QI2_XS, block_iq2_xs, 1, vec_dot_iq2_xs_q8_1>(
      vx, vy, dst, topk_ids, top_k, tokens, ncols, nrows, token_stride, stream);
}

template <typename scalar_t>
static void moe_vec_iq2_s_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int top_k,
    const int tokens,
    const int ncols,
    const int nrows,
    const int token_stride,
    cudaStream_t stream) {
  moe_vec_launch<scalar_t, QK_K, QI2_S, block_iq2_s, 1, vec_dot_iq2_s_q8_1>(
      vx, vy, dst, topk_ids, top_k, tokens, ncols, nrows, token_stride, stream);
}

template <typename scalar_t>
static void moe_vec_iq3_xxs_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int top_k,
    const int tokens,
    const int ncols,
    const int nrows,
    const int token_stride,
    cudaStream_t stream) {
  moe_vec_launch<scalar_t, QK_K, QI3_XXS, block_iq3_xxs, 1, vec_dot_iq3_xxs_q8_1>(
      vx, vy, dst, topk_ids, top_k, tokens, ncols, nrows, token_stride, stream);
}

template <typename scalar_t>
static void moe_vec_iq1_s_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int top_k,
    const int tokens,
    const int ncols,
    const int nrows,
    const int token_stride,
    cudaStream_t stream) {
  moe_vec_launch<scalar_t, QK_K, QI1_S, block_iq1_s, 1, vec_dot_iq1_s_q8_1>(
      vx, vy, dst, topk_ids, top_k, tokens, ncols, nrows, token_stride, stream);
}

template <typename scalar_t>
static void moe_vec_iq1_m_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int top_k,
    const int tokens,
    const int ncols,
    const int nrows,
    const int token_stride,
    cudaStream_t stream) {
  moe_vec_launch<scalar_t, QK_K, QI1_M, block_iq1_m, 1, vec_dot_iq1_m_q8_1>(
      vx, vy, dst, topk_ids, top_k, tokens, ncols, nrows, token_stride, stream);
}

template <typename scalar_t>
static void moe_vec_iq4_nl_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int top_k,
    const int tokens,
    const int ncols,
    const int nrows,
    const int token_stride,
    cudaStream_t stream) {
  moe_vec_launch<scalar_t, QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>(
      vx, vy, dst, topk_ids, top_k, tokens, ncols, nrows, token_stride, stream);
}

template <typename scalar_t>
static void moe_vec_iq4_xs_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int top_k,
    const int tokens,
    const int ncols,
    const int nrows,
    const int token_stride,
    cudaStream_t stream) {
  moe_vec_launch<scalar_t, QK_K, QI4_XS, block_iq4_xs, 1, vec_dot_iq4_xs_q8_1>(
      vx, vy, dst, topk_ids, top_k, tokens, ncols, nrows, token_stride, stream);
}

template <typename scalar_t>
static void moe_vec_iq3_s_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int top_k,
    const int tokens,
    const int ncols,
    const int nrows,
    const int token_stride,
    cudaStream_t stream) {
  moe_vec_launch<scalar_t, QK_K, QI3_XS, block_iq3_s, 1, vec_dot_iq3_s_q8_1>(
      vx, vy, dst, topk_ids, top_k, tokens, ncols, nrows, token_stride, stream);
}
