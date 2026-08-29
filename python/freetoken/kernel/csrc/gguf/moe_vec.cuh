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

#if defined(USE_ROCM)
// Compute one Q4_0 row against four Q8_1 packed words that the caller has
// already loaded.  The two-row HIP kernel invokes this twice per lane with two
// adjacent Q4_0 rows but one common activation block.  Keeping the activation
// words in the caller avoids issuing the same four global Q8 loads for both
// rows.  The DP4A order and Q4_0 scale correction match vec_dot_q4_0_q8_1,
// preserving the exact native GGUF arithmetic contract.
static __device__ __forceinline__ float moe_vec_q4_0_dot_shared_q8(
    const block_q4_0* __restrict__ row,
    const int u0,
    const int u1,
    const int u2,
    const int u3,
    const half2 ds8,
    const int iqs) {
  const int v0 = get_int_from_uint8(row->qs, iqs);
  const int v1 = get_int_from_uint8(row->qs, iqs + 1);
  int sumi = 0;
  sumi = __dp4a(v0 & 0x0F0F0F0F, u0, sumi);
  sumi = __dp4a((v0 >> 4) & 0x0F0F0F0F, u1, sumi);
  sumi = __dp4a(v1 & 0x0F0F0F0F, u2, sumi);
  sumi = __dp4a((v1 >> 4) & 0x0F0F0F0F, u3, sumi);
  const float2 ds8f = __half22float2(ds8);
  return __half2float(row->d) * (sumi * ds8f.x - (8 * 2 / QI4_0) * ds8f.y);
}

// The HIP launcher is defined after the CUDA-compatible wrapper so the
// generic wrappers remain grouped by quantization format below.
template <typename scalar_t>
static void moe_vec_q4_0_q8_1_hip_two_rows_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int top_k,
    const int tokens,
    const int ncols,
    const int nrows,
    const int token_stride,
    cudaStream_t stream);
#endif

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
#if defined(USE_ROCM)
  // Route AMD builds through the one-wave/two-row specialization above.  CUDA
  // retains the established generic implementation until it has independent
  // NVIDIA evidence, so this HIP experiment cannot alter CUDA behavior.
  moe_vec_q4_0_q8_1_hip_two_rows_cuda<scalar_t>(
      vx, vy, dst, topk_ids, top_k, tokens, ncols, nrows, token_stride, stream);
#else
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
#endif
}

#if defined(USE_ROCM)
// HIP Q4_0 MoE specialization derived from the current llama.cpp MMVQ row
// structure.  Unlike the older GGML_CUDA_MMV_Y=2 experiment, this launch uses
// one 32-lane wave for two rows, rather than two independent waves.  The two
// float accumulators share the same packed Q4_0 activation block and expert
// selection, reducing grid work while preserving FreeToken's existing packed
// bank layout, route indexing, and BF16 output contract.
template <typename scalar_t>
__launch_bounds__(WARP_SIZE, 1)
static __global__ void moe_vec_q4_0_hip_two_rows(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* __restrict__ topk_ids,
    const int topk,
    const int ncols,
    const int nrows,
    const int token_stride) {
  // X indexes adjacent pairs of output rows.  Y is the flattened
  // token/top-k route index, matching the former Z dimension exactly.
  const int row0 = 2 * blockIdx.x;
  const int route = blockIdx.y;
  if (row0 >= nrows) {
    return;
  }

  const int token = route / topk;
  const int expert = topk_ids[route];
  const int blocks_per_row = ncols / QK4_0;
  const int blocks_per_wave = VDR_Q4_0_Q8_1_MMVQ * WARP_SIZE / QI4_0;
  const block_q4_0* x = ((const block_q4_0*)vx) + expert * nrows * blocks_per_row;
  const block_q8_1* y = (const block_q8_1*)(((const int*)vy) + token * token_stride);

  // Each lane owns the same packed-Q4 range for both rows.  Keeping the
  // reductions separate preserves the original arithmetic for each result.
  float tmp0 = 0.0f;
  float tmp1 = 0.0f;
  for (int i = threadIdx.x / (QI4_0 / VDR_Q4_0_Q8_1_MMVQ); i < blocks_per_row;
       i += blocks_per_wave) {
    const int iby = i * (QK4_0 / QK8_1);
    const int iqs = VDR_Q4_0_Q8_1_MMVQ * (threadIdx.x % (QI4_0 / VDR_Q4_0_Q8_1_MMVQ));
    // Both output rows consume one identical Q8_1 activation block.  Load its
    // four packed words once per lane and pass them to the two row products.
    // This is deliberately confined to the HIP two-row specialization: the
    // generic CUDA and non-Q4 routes retain their established helper calls.
    const block_q8_1* y_block = &y[iby];
    const int u0 = get_int_from_int8_aligned(y_block->qs, iqs);
    const int u1 = get_int_from_int8_aligned(y_block->qs, iqs + QI4_0);
    const int u2 = get_int_from_int8_aligned(y_block->qs, iqs + 1);
    const int u3 = get_int_from_int8_aligned(y_block->qs, iqs + 1 + QI4_0);
    tmp0 += moe_vec_q4_0_dot_shared_q8(&x[row0 * blocks_per_row + i], u0, u1, u2, u3, y_block->ds, iqs);
    if (row0 + 1 < nrows) {
      tmp1 += moe_vec_q4_0_dot_shared_q8(
          &x[(row0 + 1) * blocks_per_row + i], u0, u1, u2, u3, y_block->ds, iqs);
    }
  }

  // A wave-level XOR reduction leaves the same sum in every lane.  Lane zero
  // writes row zero and lane one writes row one, avoiding shared memory.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    tmp0 += SGLANG_SHFL_XOR_SYNC(uint32_t(-1), tmp0, mask);
    tmp1 += SGLANG_SHFL_XOR_SYNC(uint32_t(-1), tmp1, mask);
  }
  if (threadIdx.x == 0) {
    dst[route * nrows + row0] = tmp0;
  }
  if (threadIdx.x == 1 && row0 + 1 < nrows) {
    dst[route * nrows + row0 + 1] = tmp1;
  }
}

template <typename scalar_t>
static void moe_vec_q4_0_q8_1_hip_two_rows_cuda(
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
  // One block now covers two rows and one route.  ``tokens * top_k`` remains
  // the complete flattened routing domain used by the original launcher.
  const dim3 block_nums((nrows + 1) / 2, tokens * top_k, 1);
  const dim3 block_dims(WARP_SIZE, 1, 1);
  moe_vec_q4_0_hip_two_rows<scalar_t>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}
#endif

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
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
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
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
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
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
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
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
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
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
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
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
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
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
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
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
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
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
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
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI2_XXS, block_iq2_xxs, 1, vec_dot_iq2_xxs_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
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
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI2_XS, block_iq2_xs, 1, vec_dot_iq2_xs_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
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
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI2_S, block_iq2_s, 1, vec_dot_iq2_s_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
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
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI3_XXS, block_iq3_xxs, 1, vec_dot_iq3_xxs_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
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
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI1_S, block_iq1_s, 1, vec_dot_iq1_s_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
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
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI1_M, block_iq1_m, 1, vec_dot_iq1_m_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
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
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
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
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI4_XS, block_iq4_xs, 1, vec_dot_iq4_xs_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
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
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI3_XS, block_iq3_s, 1, vec_dot_iq3_s_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}
