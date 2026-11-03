#include "common.cuh"
#include "mmid.cuh"

struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const { return data & 0x003FFFFF; }
    __device__ uint32_t iex_used() const { return data >> 22; }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// The generic path passes 0, which needs no padding since it never groups lanes by token.
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1,
        const bool write_inverse, const bool use_shared) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;
    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        for (int it = 0; it < n_tokens; ++it) {
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += (unsigned) expert_used < (unsigned) expert; // FreeToken: unsigned, so an inactive route (negative id) sorts after every expert
            }
        }
    } else {
        constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;
        static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;
            const int iex = threadIdx.x % neu_padded;
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            nex_prev += (unsigned) expert_used < (unsigned) expert; // FreeToken: unsigned, so an inactive route (negative id) sorts after every expert
            if (use_shared) {
                const int match = expert_used == expert;
                int rank = match;
                for (int offset = 1; offset < warp_size; offset *= 2) {
                    const int prior = __shfl_up_sync(0xFFFFFFFF, rank, offset, warp_size);
                    if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                        rank += prior;
                    }
                }
                const int count = __shfl_sync(0xFFFFFFFF, rank, warp_size - 1, warp_size);
                if (match) {
                    store[it_compact + rank - 1] = mm_ids_helper_store(it, iex);
                }
                it_compact += count;
            }
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);
    if (use_shared) {
        ggml_cuda_syncwarp();
        for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
            const mm_ids_helper_store route = store[itc];
            const int it = route.it();
            const int iex = route.iex_used();
            const int dst = nex_prev + itc;
            ids_dst[dst] = it*n_expert_used + iex;
            if (write_inverse) {
                ids_src1[it*n_expert_used + iex] = dst;
            } else {
                ids_src1[dst] = it*sis1 + iex % nchannels_y;
            }
        }
    } else {
        // Count first so every route can write directly to its final expert slice.
        // A token may use the same expert more than once; a per-token compact row loses routes.
        if constexpr (n_expert_used_template == 0) {
            for (int it = 0; it < n_tokens; ++it) {
                for (int iex0 = 0; iex0 < n_expert_used; iex0 += warp_size) {
                    const int iex = iex0 + threadIdx.x;
                    const int match = iex < n_expert_used && ids[it*si1 + iex] == expert;
                    int rank = match;
                    for (int offset = 1; offset < warp_size; offset *= 2) {
                        const int prior = __shfl_up_sync(0xFFFFFFFF, rank, offset, warp_size);
                        if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                            rank += prior;
                        }
                    }
                    const int count = __shfl_sync(0xFFFFFFFF, rank, warp_size - 1, warp_size);
                    if (match) {
                        const int dst = nex_prev + it_compact + rank - 1;
                        ids_dst[dst] = it*n_expert_used + iex;
                        if (write_inverse) {
                            ids_src1[it*n_expert_used + iex] = dst;
                        } else {
                            ids_src1[dst] = it*sis1 + iex % nchannels_y;
                        }
                    }
                    it_compact += count;
                }
            }
        } else {
            constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;
            for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
                const int it = it0 + threadIdx.x / neu_padded;
                const int iex = threadIdx.x % neu_padded;
                const int match = iex < n_expert_used && it < n_tokens && ids[it*si1 + iex] == expert;
                int rank = match;
                for (int offset = 1; offset < warp_size; offset *= 2) {
                    const int prior = __shfl_up_sync(0xFFFFFFFF, rank, offset, warp_size);
                    if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                        rank += prior;
                    }
                }
                const int count = __shfl_sync(0xFFFFFFFF, rank, warp_size - 1, warp_size);
                if (match) {
                    const int dst = nex_prev + it_compact + rank - 1;
                    ids_dst[dst] = it*n_expert_used + iex;
                    if (write_inverse) {
                        ids_src1[it*n_expert_used + iex] = dst;
                    } else {
                        ids_src1[dst] = it*sis1 + iex % nchannels_y;
                    }
                }
                it_compact += count;
            }
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    const size_t needed_shared = (size_t) n_tokens*n_expert_used_var*sizeof(mm_ids_helper_store);
    const bool use_shared = n_expert_used_template != 0 && n_tokens < (1 << 22) &&
                            n_expert_used_var < (1 << 10) && needed_shared <= smpbo;
    if (use_shared) {
        CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);
    }
    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, use_shared ? needed_shared : 0, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse, use_shared);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
    }
}

// Lists the (expert, column tile) pairs that hold columns for a flat MoE grid, in expert order.
// The other ntiles grid columns get tile jt_none of expert 0, which starts past its columns, so mul_mat_q skips them.
template <int nthreads>
__launch_bounds__(nthreads, 1)
static __global__ void mm_ids_tile_map(
        const int32_t * __restrict__ expert_bounds, int2 * __restrict__ tile_map, const int n_experts, const int J, const int ntiles, const int jt_none) {
    __shared__ int scan[nthreads];

    int start = 0; // First tile of the current chunk of experts.
    for (int e0 = 0; e0 < n_experts; e0 += nthreads) {
        const int e = e0 + threadIdx.x;
        const int n = e < n_experts ? (expert_bounds[e + 1] - expert_bounds[e] + J - 1) / J : 0;

        scan[threadIdx.x] = n;
        __syncthreads();
#pragma unroll
        for (int offset = 1; offset < nthreads; offset *= 2) {
            const int add = threadIdx.x >= offset ? scan[threadIdx.x - offset] : 0;
            __syncthreads();
            scan[threadIdx.x] += add;
            __syncthreads();
        }

        const int first = start + scan[threadIdx.x] - n;
        for (int jt = 0; jt < n; ++jt) {
            tile_map[first + jt] = make_int2(e, jt);
        }
        start += scan[nthreads - 1];
        __syncthreads();
    }

    for (int t = start + threadIdx.x; t < ntiles; t += nthreads) {
        tile_map[t] = make_int2(0, jt_none);
    }
}

void ggml_cuda_launch_mm_ids_tile_map(
        const int32_t * expert_bounds, int2 * tile_map, const int n_experts, const int J, const int ntiles, const int jt_none, cudaStream_t stream) {
    constexpr int nthreads = 512;
    mm_ids_tile_map<nthreads><<<1, nthreads, 0, stream>>>(expert_bounds, tile_map, n_experts, J, ntiles, jt_none);
}
