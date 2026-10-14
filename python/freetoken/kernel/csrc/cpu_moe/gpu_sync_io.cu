#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <freetoken/hip_compat.h>

#include "gpu_sync_io.h"

namespace freetoken::cpu_moe {
namespace {

__device__ inline float bf16_bits_to_float(uint16_t b) { return __uint_as_float(static_cast<uint32_t>(b) << 16); }

// Round to nearest even, the same rounding as a torch bf16 add.
__device__ inline uint16_t float_to_bf16_bits(float f) {
  const uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
  return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// One block. Copies x/ids/w into the IO rows, then done = 0 and ready = task.
// A step without a CPU route writes done = 1 and rings nothing: the coordinator stays asleep and the GPU wait passes at once.
__global__ void publish_kernel(const uint16_t* __restrict__ x, const int32_t* __restrict__ ids, const float* __restrict__ w,
                               int64_t n_x, int64_t n_ids, uint16_t* io_x, int32_t* io_ids, float* io_w,
                               volatile int64_t* ready, volatile int64_t* done, int64_t task, int32_t* has_cpu) {
  __shared__ int any;
  if (threadIdx.x == 0) any = 0;
  __syncthreads();
  for (int64_t i = threadIdx.x; i < n_ids; i += blockDim.x)
    if (ids[i] >= 0) any = 1;
  __syncthreads();
  if (!any) {
    if (threadIdx.x == 0) {
      *has_cpu = 0;
      *done = 1;
      __threadfence_system();
    }
    return;
  }
  const bool vec = (reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(io_x)) % 16 == 0 && n_x % 8 == 0;
  if (vec) {
    for (int64_t i = threadIdx.x; i < n_x / 8; i += blockDim.x)
      reinterpret_cast<uint4*>(io_x)[i] = reinterpret_cast<const uint4*>(x)[i];
  } else {
    for (int64_t i = threadIdx.x; i < n_x; i += blockDim.x) io_x[i] = x[i];
  }
  for (int64_t i = threadIdx.x; i < n_ids; i += blockDim.x) {
    io_ids[i] = ids[i];
    io_w[i] = w[i];
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    *has_cpu = 1;
    *done = 0;
    __threadfence_system();
    *ready = task;
    __threadfence_system();
  }
}

// out = CPU rows, or out += CPU rows with accumulate; a step publish skipped contributes zeros.
// io_y is volatile: the same host rows hold a new step on every call.
__global__ void combine_kernel(uint16_t* out, const volatile uint16_t* io_y, const int32_t* has_cpu, int64_t n, bool accumulate) {
  const bool on = *has_cpu != 0;
  if (accumulate && !on) return;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    float v = accumulate ? bf16_bits_to_float(out[i]) : 0.0f;
    if (on) v += bf16_bits_to_float(io_y[i]);
    out[i] = float_to_bf16_bits(v);
  }
}

cudaError_t launch_publish(cudaStream_t stream, const uint16_t* x, const int32_t* ids, const float* w, int64_t n_x, int64_t n_ids,
                           uint16_t* io_x, int32_t* io_ids, float* io_w, volatile int64_t* ready, volatile int64_t* done, int64_t task,
                           int32_t* has_cpu) {
  publish_kernel<<<1, 256, 0, stream>>>(x, ids, w, n_x, n_ids, io_x, io_ids, io_w, ready, done, task, has_cpu);
  return cudaGetLastError();
}

cudaError_t launch_combine(cudaStream_t stream, uint16_t* out, const uint16_t* io_y, const int32_t* has_cpu, int64_t n, bool accumulate) {
  if (n == 0) return cudaSuccess;
  const auto blocks = static_cast<unsigned>(std::min<int64_t>((n + 255) / 256, 1024));
  combine_kernel<<<blocks, 256, 0, stream>>>(out, io_y, has_cpu, n, accumulate);
  return cudaGetLastError();
}

void check(cudaError_t rc, const char* what) {
  if (rc != cudaSuccess) throw std::runtime_error(std::string("CPU MoE kernel IO: ") + what + ": " + cudaGetErrorString(rc));
}

template <typename T>
T* device_address(const void* host) {
  void* dev = nullptr;
  check(cudaHostGetDevicePointer(&dev, const_cast<void*>(host), 0), "device address of the IO rows");
  return static_cast<T*>(dev);
}

}  // namespace

KernelIo::KernelIo(uintptr_t ready, uintptr_t done, const void* x, const int32_t* ids, const float* w, const void* y, int hidden,
                   int top_k)
    : ready_(reinterpret_cast<volatile int64_t*>(ready)),
      done_(reinterpret_cast<volatile int64_t*>(done)),
      x_(device_address<uint16_t>(x)),
      ids_(device_address<int32_t>(ids)),
      w_(device_address<float>(w)),
      y_(device_address<const uint16_t>(y)),
      hidden_(hidden),
      top_k_(top_k) {
  check(cudaMalloc(reinterpret_cast<void**>(&has_cpu_), sizeof(int32_t)), "has_cpu allocation");
  const cudaError_t rc = cudaMemset(has_cpu_, 0, sizeof(int32_t));
  if (rc != cudaSuccess) {
    (void)cudaFree(has_cpu_);
    check(rc, "has_cpu reset");
  }
}

KernelIo::~KernelIo() { (void)cudaFree(has_cpu_); }

void KernelIo::publish(uintptr_t stream, const void* x, const int32_t* ids, const float* w, int num_tokens, uint64_t task) {
  check(launch_publish(reinterpret_cast<cudaStream_t>(stream), static_cast<const uint16_t*>(x), ids, w,
                       static_cast<int64_t>(num_tokens) * hidden_, static_cast<int64_t>(num_tokens) * top_k_, x_, ids_, w_, ready_, done_,
                       static_cast<int64_t>(task), has_cpu_),
        "publish launch");
}

void KernelIo::combine(uintptr_t stream, void* out, int num_tokens, bool accumulate) {
  check(launch_combine(reinterpret_cast<cudaStream_t>(stream), static_cast<uint16_t*>(out), y_, has_cpu_,
                       static_cast<int64_t>(num_tokens) * hidden_, accumulate),
        "combine launch");
}

// A ringing step, its combine and a skipped step on scratch mapped memory, checking every word they write.
std::string KernelIo::probe(uintptr_t stream_handle) {
  struct alignas(16) Area {
    int64_t ready, done;
    int32_t has_cpu, pad[3];
    uint16_t x[8], io_x[8], io_y[8], out[8];
    int32_t ids[2], io_ids[2];
    float w[2], io_w[2];
  };
  const auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
  void* host = nullptr;
  void* dev = nullptr;
  if (cudaHostAlloc(&host, sizeof(Area), cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess)
    return "pinned scratch allocation failed";
  std::string reason;
  auto ok = [&](cudaError_t rc, const char* what) {
    if (rc != cudaSuccess && reason.empty()) reason = std::string(what) + ": " + cudaGetErrorString(rc);
    return rc == cudaSuccess;
  };
  auto fail = [&](const char* why) {
    if (reason.empty()) reason = why;
  };
  if (ok(cudaHostGetDevicePointer(&dev, host, 0), "device address of the scratch")) {
    auto* h = static_cast<Area*>(host);
    auto* d = static_cast<Area*>(dev);
    std::memset(h, 0, sizeof(Area));
    for (int i = 0; i < 8; ++i) {
      h->x[i] = static_cast<uint16_t>(0x3f80 + i);
      h->io_y[i] = 0x3f80;  // 1.0
      h->out[i] = 0x4000;  // 2.0
    }
    h->ids[0] = 3;
    h->ids[1] = -1;
    h->w[0] = 0.25f;
    h->w[1] = 0.75f;
    auto publish = [&] {
      return launch_publish(stream, d->x, d->ids, d->w, 8, 2, d->io_x, d->io_ids, d->io_w, &d->ready, &d->done, 5, &d->has_cpu);
    };

    if (ok(publish(), "publish launch") && ok(cudaStreamSynchronize(stream), "publish")) {
      if (h->ready != 5 || h->done != 0 || h->has_cpu != 1) fail("publish wrote wrong handshake words");
      if (std::memcmp(h->io_x, h->x, sizeof(h->x)) || std::memcmp(h->io_ids, h->ids, sizeof(h->ids)) ||
          std::memcmp(h->io_w, h->w, sizeof(h->w)))
        fail("publish wrote wrong IO rows");
    }
    if (reason.empty() && ok(launch_combine(stream, d->out, d->io_y, &d->has_cpu, 8, true), "combine launch") &&
        ok(cudaStreamSynchronize(stream), "combine")) {
      for (int i = 0; i < 8; ++i)
        if (h->out[i] != 0x4040) fail("combine returned a wrong sum");  // 3.0
    }
    if (reason.empty()) {
      h->ready = 0;
      h->ids[0] = -1;
      if (ok(publish(), "skip launch") && ok(launch_combine(stream, d->out, d->io_y, &d->has_cpu, 8, false), "skip combine launch") &&
          ok(cudaStreamSynchronize(stream), "skipped step")) {
        bool zeros = true;
        for (int i = 0; i < 8; ++i) zeros = zeros && h->out[i] == 0;
        if (h->ready != 0 || h->done != 1 || h->has_cpu != 0 || !zeros) fail("a skipped step wrote wrong words");
      }
    }
  }
  (void)cudaFreeHost(host);
  return reason;
}

}  // namespace freetoken::cpu_moe
