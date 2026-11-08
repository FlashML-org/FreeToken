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
// With tile_map it also lists the (expert, column tile) pairs that hold columns for a flat MoE grid, in expert order.
// The other ntiles grid columns get tile jt_none of expert 0, which starts past its columns, so mul_mat_q skips them.
static __device__ __forceinline__ void mm_ids_first_rows(
        int * count, int32_t * __restrict__ expert_bounds, int2 * __restrict__ tile_map, const int J, const int ntiles, const int jt_none,
        const int n_experts) {
    int start      = 0;
    int start_tile = 0;
    for (int e0 = 0; e0 < n_experts; e0 += mm_ids_nthreads) {
        const int e = e0 + threadIdx.x;
        const int n = e < n_experts ? count[e] : 0;
        int total;
        const int prefix = mm_ids_block_scan(n, total);
        if (e < n_experts) {
            count[e]         = start + prefix;
            expert_bounds[e] = start + prefix;
        }
        start += total;
        if (tile_map) {
            const int ntile = (n + J - 1) / J;
            int total_tiles;
            const int first = start_tile + mm_ids_block_scan(ntile, total_tiles);
            for (int jt = 0; jt < ntile; ++jt) {
                tile_map[first + jt] = make_int2(e, jt);
            }
            start_tile += total_tiles;
        }
    }
    if (threadIdx.x == 0) {
        expert_bounds[n_experts] = start;
    }
    if (tile_map) {
        for (int t = start_tile + threadIdx.x; t < ntiles; t += mm_ids_nthreads) {
            tile_map[t] = make_int2(0, jt_none);
        }
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
        int2 * __restrict__ tile_map, const int J, const int ntiles, const int jt_none,
        const int n_routes, const int n_experts, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    extern __shared__ int data_mm_ids[];
    for (int e = threadIdx.x; e < n_experts; e += mm_ids_nthreads) {
        data_mm_ids[e] = 0;
    }
    __syncthreads();
    mm_ids_count_routes(ids, data_mm_ids, 0, n_routes, n_experts, n_expert_used, si1);
    __syncthreads();
    mm_ids_first_rows(data_mm_ids, expert_bounds, tile_map, J, ntiles, jt_none, n_experts);
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
static __global__ void mm_ids_bounds(
        int32_t * __restrict__ counts, int32_t * __restrict__ expert_bounds, int2 * __restrict__ tile_map, const int J, const int ntiles,
        const int jt_none, const int n_experts, const int nblocks) {
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
    mm_ids_first_rows(data_mm_ids, expert_bounds, tile_map, J, ntiles, jt_none, n_experts);
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
// FreeToken: a counting sort over blocks of routes. One warp per expert over every route costs n_experts x n_tokens and bounds n_tokens by shared memory.
// Within an expert the blocks keep token order, the routes inside one block do not.
// The flat MoE grid does not see that order (one tile per CUDA block), but a stream-k regular grid (ncols_max > 0) would, in its fp32 summation order.
void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        int32_t * __restrict__ scratch, int2 * __restrict__ tile_map, const int J, const int ntiles, const int jt_none,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y,
        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    const int nblocks  = mm_ids_nblocks(n_tokens, n_expert_used);
    const int n_routes = n_tokens*n_expert_used;
    // FreeToken: an offloaded MoE passes its slot count as n_experts, so the cap is the device's opt-in limit less mm_ids_block_scan's static warp_sums.
    const size_t max_shared    = ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo - mm_ids_nthreads/32*sizeof(int);
    const size_t nbytes_shared = n_experts*sizeof(int);
    GGML_ASSERT(nbytes_shared <= max_shared && "mm_ids_helper keeps one int per expert in shared memory");
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_sort_one_block, max_shared);
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_count, max_shared);
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_bounds, max_shared);
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_scatter, max_shared);

    if (nblocks == 1) {
        mm_ids_sort_one_block<<<1, mm_ids_nthreads, nbytes_shared, stream>>>(
            ids, ids_src1, ids_dst, expert_bounds, tile_map, J, ntiles, jt_none, n_routes, n_experts, n_expert_used, nchannels_y, si1, sis1, write_inverse);
        return;
    }
    mm_ids_count<<<nblocks, mm_ids_nthreads, nbytes_shared, stream>>>(ids, scratch, n_routes, n_experts, n_expert_used, si1);
    mm_ids_bounds<<<1, mm_ids_nthreads, nbytes_shared, stream>>>(scratch, expert_bounds, tile_map, J, ntiles, jt_none, n_experts, nblocks);
    mm_ids_scatter<<<nblocks, mm_ids_nthreads, nbytes_shared, stream>>>(
        ids, ids_src1, ids_dst, expert_bounds, scratch, n_routes, n_experts, n_expert_used, nchannels_y, si1, sis1, write_inverse);
}
