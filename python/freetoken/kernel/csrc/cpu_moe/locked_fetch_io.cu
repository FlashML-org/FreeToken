#include <stdexcept>
#include <string>

#include <freetoken/hip_compat.h>

#include "locked_fetch_io.h"

namespace freetoken::cpu_moe {
namespace {

__device__ inline int64_t clamp_rows(const int64_t* num_indices, int max_rows) {
  const int64_t n = *num_indices;
  return n < 0 ? 0 : (n > max_rows ? max_rows : n);
}

// One block. The rows land before the doorbell, which the CPU reads with acquire.
__global__ void publish_kernel(const int64_t* num_indices, const int32_t* src_indices, int max_rows, int64_t* meta_count,
                               int32_t* meta_rows, volatile int64_t* ready, volatile int64_t* done, int64_t task) {
  const int64_t n = clamp_rows(num_indices, max_rows);
  if (n == 0) {
    if (threadIdx.x == 0) {
      *done = 1;
      __threadfence_system();
    }
    return;
  }
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) meta_rows[i] = src_indices[i];
  if (threadIdx.x == 0) *meta_count = n;
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    *done = 0;
    __threadfence_system();
    *ready = task;
    __threadfence_system();
  }
}

__device__ inline int64_t load_acquire_system(const volatile int64_t* p) {
#if defined(__HIP_PLATFORM_AMD__)
  const int64_t v = *p;
  __threadfence_system();
  return v;
#else
  int64_t v;
  asm volatile("ld.acquire.sys.global.s64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
#endif
}

// grid (blocks_per_bank, num_banks). Rows are multiples of 16 bytes (the fused copy plan checks it).
__global__ void gather_kernel(const int64_t* num_indices, int max_rows, const int32_t* evict_slots, const int64_t* dst_ptrs,
                              const int64_t* staging_ptrs, const int64_t* row_bytes, const volatile int64_t* done, bool spin) {
  if (spin) {
    if (threadIdx.x == 0)
      while (load_acquire_system(done) < 1) {
      }
    __syncthreads();
    __threadfence_system();
  }
  const int b = blockIdx.y;
  const int64_t rb = row_bytes[b];
  const int64_t n = clamp_rows(num_indices, max_rows);
  if (rb == 0 || n == 0) return;
  const int64_t words = rb / 16;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t r = 0; r < n; ++r) {
    const uint4* src = reinterpret_cast<const uint4*>(staging_ptrs[b]) + r * words;
    uint4* dst = reinterpret_cast<uint4*>(reinterpret_cast<char*>(dst_ptrs[b]) + static_cast<int64_t>(evict_slots[r]) * rb);
    for (int64_t w = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; w < words; w += stride) dst[w] = src[w];
  }
}

void check(cudaError_t rc, const char* what) {
  if (rc != cudaSuccess) throw std::runtime_error(std::string("locked fetch: ") + what + ": " + cudaGetErrorString(rc));
}

}  // namespace

void locked_publish(uintptr_t stream, const int64_t* num_indices, const int32_t* src_indices, int max_rows, int64_t* meta_count,
                    int32_t* meta_rows, volatile int64_t* ready, volatile int64_t* done, int64_t task) {
  publish_kernel<<<1, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(num_indices, src_indices, max_rows, meta_count, meta_rows, ready,
                                                                       done, task);
  check(cudaGetLastError(), "publish launch");
}

void locked_gather(uintptr_t stream, const int64_t* num_indices, int max_rows, const int32_t* evict_slots, const int64_t* dst_ptrs,
                   const int64_t* staging_ptrs, const int64_t* row_bytes, int num_banks, int blocks_per_bank, const volatile int64_t* done,
                   bool spin) {
  const dim3 grid(static_cast<unsigned>(blocks_per_bank), static_cast<unsigned>(num_banks));
  gather_kernel<<<grid, 1024, 0, reinterpret_cast<cudaStream_t>(stream)>>>(num_indices, max_rows, evict_slots, dst_ptrs, staging_ptrs,
                                                                          row_bytes, done, spin);
  check(cudaGetLastError(), "gather launch");
}

}  // namespace freetoken::cpu_moe
