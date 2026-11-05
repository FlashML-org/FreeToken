#include "common.cuh"
#include "mmid.cuh"

static constexpr int mm_ids_nthreads         = 256;
static constexpr int mm_ids_routes_per_block = 8*mm_ids_nthreads;
static constexpr int mm_ids_one_block_routes = 4096; // one block beats the three launches up to here (RTX PRO 6000 sweep)

// Route r is slot r % n_expert_used of token r / n_expert_used. A negative (inactive) id is >= n_experts as unsigned, so it gets no compact row.
static __device__ __forceinline__ int mm_ids_route_expert(
        const int32_t * __restrict__ ids, const int r, const int n_expert_used, const int si1, const int n_experts) {
    const int expert = ids[(r / n_expert_used)*si1 + r % n_expert_used];
    return (unsigned) expert < (unsigned) n_experts ? expert : -1;
}

// Exclusive prefix sum of v over the block, with the block total in total. Every thread must call it.
static __device__ __forceinline__ int mm_ids_block_scan(const int v, int & total) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = mm_ids_nthreads/warp_size;
    __shared__ int warp_sums[nwarps];

    const int x = warp_prefix_inclusive_sum<int, warp_size>(v);
    if (threadIdx.x % warp_size == warp_size - 1) {
        warp_sums[threadIdx.x / warp_size] = x;
    }
    __syncthreads();
    int before = 0;
    total = 0;
#pragma unroll
    for (int w = 0; w < nwarps; ++w) {
        before += w < (int) threadIdx.x / warp_size ? warp_sums[w] : 0;
        total  += warp_sums[w];
    }
    __syncthreads();
    return before + x - v;
}

// count[e] += the routes in [r0, r1) that go to expert e
static __device__ __forceinline__ void mm_ids_count_routes(
        const int32_t * __restrict__ ids, int * count, const int r0, const int r1, const int n_experts, const int n_expert_used, const int si1) {
    for (int r = r0 + threadIdx.x; r < r1; r += mm_ids_nthreads) {
        const int expert = mm_ids_route_expert(ids, r, n_expert_used, si1, n_experts);
        if (expert >= 0) {
            atomicAdd(&count[expert], 1);
        }
    }
}

// Turns the route count per expert into the expert's first row and writes expert_bounds.
static __device__ __forceinline__ void mm_ids_first_rows(int * count, int32_t * __restrict__ expert_bounds, const int n_experts) {
    int start = 0;
    for (int e0 = 0; e0 < n_experts; e0 += mm_ids_nthreads) {
        const int e = e0 + threadIdx.x;
        int total;
        const int prefix = mm_ids_block_scan(e < n_experts ? count[e] : 0, total);
        if (e < n_experts) {
            count[e]         = start + prefix;
            expert_bounds[e] = start + prefix;
        }
        start += total;
    }
    if (threadIdx.x == 0) {
        expert_bounds[n_experts] = start;
    }
    __syncthreads();
}

// Puts each route in [r0, r1) at the next free row of its expert, taken from next_row.
static __device__ __forceinline__ void mm_ids_scatter_routes(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int * next_row,
        const int r0, const int r1, const int n_experts, const int n_expert_used, const int nchannels_y, const int si1, const int sis1,
        const bool write_inverse) {
    for (int r = r0 + threadIdx.x; r < r1; r += mm_ids_nthreads) {
        const int expert = mm_ids_route_expert(ids, r, n_expert_used, si1, n_experts);
        if (expert < 0) {
            continue;
        }
        const int row = atomicAdd(&next_row[expert], 1);
        ids_dst[row] = r;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[r] = row;
        } else {
            ids_src1[row] = (r / n_expert_used)*sis1 + (r % n_expert_used) % nchannels_y;
        }
    }
}

// The whole sort in one block, so a small batch takes a single launch.
static __global__ void mm_ids_sort_one_block(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_routes, const int n_experts, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    extern __shared__ int data_mm_ids[];
    for (int e = threadIdx.x; e < n_experts; e += mm_ids_nthreads) {
        data_mm_ids[e] = 0;
    }
    __syncthreads();
    mm_ids_count_routes(ids, data_mm_ids, 0, n_routes, n_experts, n_expert_used, si1);
    __syncthreads();
    mm_ids_first_rows(data_mm_ids, expert_bounds, n_experts);
    mm_ids_scatter_routes(ids, ids_src1, ids_dst, data_mm_ids, 0, n_routes, n_experts, n_expert_used, nchannels_y, si1, sis1, write_inverse);
}

// counts[b*n_experts + e]: the routes of block b that go to expert e
static __global__ void mm_ids_count(
        const int32_t * __restrict__ ids, int32_t * __restrict__ counts, const int n_routes, const int n_experts, const int n_expert_used, const int si1) {
    extern __shared__ int data_mm_ids[];
    for (int e = threadIdx.x; e < n_experts; e += mm_ids_nthreads) {
        data_mm_ids[e] = 0;
    }
    __syncthreads();
    const int r0 = blockIdx.x*mm_ids_routes_per_block;
    mm_ids_count_routes(ids, data_mm_ids, r0, min(r0 + mm_ids_routes_per_block, n_routes), n_experts, n_expert_used, si1);
    __syncthreads();
    for (int e = threadIdx.x; e < n_experts; e += mm_ids_nthreads) {
        counts[(int64_t) blockIdx.x*n_experts + e] = data_mm_ids[e];
    }
}

// Turns counts[b][e] into the number of expert e's rows taken by blocks before b, so each expert's rows follow token order block by block.
static __global__ void mm_ids_bounds(int32_t * __restrict__ counts, int32_t * __restrict__ expert_bounds, const int n_experts, const int nblocks) {
    constexpr int nload = 8; // loads in flight per expert
    extern __shared__ int data_mm_ids[];
    for (int e = threadIdx.x; e < n_experts; e += mm_ids_nthreads) {
        int n = 0;
        for (int b0 = 0; b0 < nblocks; b0 += nload) {
            int c[nload];
#pragma unroll
            for (int i = 0; i < nload; ++i) {
                c[i] = b0 + i < nblocks ? counts[(int64_t) (b0 + i)*n_experts + e] : 0;
            }
#pragma unroll
            for (int i = 0; i < nload; ++i) {
                if (b0 + i < nblocks) {
                    counts[(int64_t) (b0 + i)*n_experts + e] = n;
                }
                n += c[i];
            }
        }
        data_mm_ids[e] = n;
    }
    __syncthreads();
    mm_ids_first_rows(data_mm_ids, expert_bounds, n_experts);
}

static __global__ void mm_ids_scatter(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ counts, const int n_routes, const int n_experts,
        const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    extern __shared__ int data_mm_ids[];
    for (int e = threadIdx.x; e < n_experts; e += mm_ids_nthreads) {
        data_mm_ids[e] = expert_bounds[e] + counts[(int64_t) blockIdx.x*n_experts + e];
    }
    __syncthreads();
    const int r0 = blockIdx.x*mm_ids_routes_per_block;
    mm_ids_scatter_routes(ids, ids_src1, ids_dst, data_mm_ids, r0, min(r0 + mm_ids_routes_per_block, n_routes),
        n_experts, n_expert_used, nchannels_y, si1, sis1, write_inverse);
}

static int mm_ids_nblocks(const int n_tokens, const int n_expert_used) {
    const int64_t n_routes = (int64_t) n_tokens*n_expert_used;
    // the last block's route indices must stay below INT_MAX
    GGML_ASSERT(n_routes <= INT_MAX - mm_ids_routes_per_block);
    return n_routes <= mm_ids_one_block_routes ? 1 : (n_routes + mm_ids_routes_per_block - 1) / mm_ids_routes_per_block;
}

int64_t ggml_cuda_mm_ids_helper_scratch_size(const int n_experts, const int n_tokens, const int n_expert_used) {
    const int nblocks = mm_ids_nblocks(n_tokens, n_expert_used);
    return nblocks > 1 ? (int64_t) nblocks*n_experts : 0;
}

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
// FreeToken: a counting sort over blocks of routes; within an expert the blocks keep token order, the routes inside one block do not.
void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        int32_t * __restrict__ scratch, const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y,
        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    const int nblocks  = mm_ids_nblocks(n_tokens, n_expert_used);
    const int n_routes = n_tokens*n_expert_used;
    const size_t nbytes_shared = n_experts*sizeof(int);
    GGML_ASSERT(nbytes_shared <= 32*1024 && "mm_ids_helper keeps one int per expert in shared memory");

    if (nblocks == 1) {
        mm_ids_sort_one_block<<<1, mm_ids_nthreads, nbytes_shared, stream>>>(
            ids, ids_src1, ids_dst, expert_bounds, n_routes, n_experts, n_expert_used, nchannels_y, si1, sis1, write_inverse);
        return;
    }
    mm_ids_count<<<nblocks, mm_ids_nthreads, nbytes_shared, stream>>>(ids, scratch, n_routes, n_experts, n_expert_used, si1);
    mm_ids_bounds<<<1, mm_ids_nthreads, nbytes_shared, stream>>>(scratch, expert_bounds, n_experts, nblocks);
    mm_ids_scatter<<<nblocks, mm_ids_nthreads, nbytes_shared, stream>>>(
        ids, ids_src1, ids_dst, expert_bounds, scratch, n_routes, n_experts, n_expert_used, nchannels_y, si1, sis1, write_inverse);
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
