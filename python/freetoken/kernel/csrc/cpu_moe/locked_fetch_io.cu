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

}  // namespace freetoken::cpu_moe
