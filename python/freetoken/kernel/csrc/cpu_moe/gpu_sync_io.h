// Kernel IO for the GPU <-> CPU handshake (gpu_sync.h): kernels on the executor's mapped pinned IO rows in place of the copies around submit and wait.
// One source for both runtimes: nvcc builds gpu_sync_io.cu on CUDA, hipcc on ROCm through freetoken/hip_compat.h.
#pragma once

#include <cstdint>
#include <string>

namespace freetoken::cpu_moe {

class KernelIo {
 public:
  // ready, done: device addresses of GpuSync's words. x, ids, w, y: host addresses of the pinned IO rows. Throws if setup fails.
  KernelIo(uintptr_t ready, uintptr_t done, const void* x, const int32_t* ids, const float* w, const void* y, int hidden, int top_k);
  ~KernelIo();
  KernelIo(const KernelIo&) = delete;
  KernelIo& operator=(const KernelIo&) = delete;

  // Copies the step's device inputs into the IO rows and does gpu_submit's writes, or skips the step when no route goes to the CPU.
  // x [n, hidden] bf16, ids [n, top_k] int32 (< 0 = not a CPU route), w [n, top_k] fp32.
  void publish(uintptr_t stream, const void* x, const int32_t* ids, const float* w, int num_tokens, uint64_t task);
  // Follows gpu_wait(): writes the CPU rows to out [n, hidden] bf16, or adds them to it; a skipped step gives zeros.
  void combine(uintptr_t stream, void* out, int num_tokens, bool accumulate);

  // Whether both kernels run and write the right words on this device: "" or the reason.
  static std::string probe(uintptr_t stream);

 private:
  volatile int64_t* ready_;
  volatile int64_t* done_;
  uint16_t* x_;
  int32_t* ids_;
  float* w_;
  const uint16_t* y_;
  int hidden_;
  int top_k_;
  int32_t* has_cpu_ = nullptr;  // device word: whether the last published step has CPU routes
};

}  // namespace freetoken::cpu_moe
