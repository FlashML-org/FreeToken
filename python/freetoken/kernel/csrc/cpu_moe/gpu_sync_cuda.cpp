#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cuda_runtime_api.h>

#include "gpu_sync.h"

namespace freetoken::cpu_moe {
namespace {

// =====================================================================================
// CUDA stream memory operations (driver API, resolved via dlopen -- no link-time or
// toolchain dependence). The GPU side of the flag handshake: submit = WRITE_VALUE
// (done=0 then ready=task word), sync = WAIT_VALUE(done >= 1). The wait is
// executed by the GPU front-end (no SM-resident kernel), so GPU "utilization" stays
// truthful during CPU compute windows -- a resident spin kernel pinned it at 99%,
// which laptop CPU/GPU dynamic power schedulers answered by clamping the CPU's max
// frequency (GEMV workers -1.5x: the reported edge regression). Availability is
// probed functionally at startup (memops_probe); anything unsupported (some driver models,
// vGPU, old drivers) falls back to the cudaLaunchHostFunc path.
#if defined(_WIN32)
static void* cumemop_dlopen() { return (void*)::LoadLibraryA("nvcuda.dll"); }
static void* cumemop_dlsym(void* h, const char* n) {
  return (void*)::GetProcAddress((HMODULE)h, n);
}
#else
static void* cumemop_dlopen() {
  void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
  if (h == nullptr) h = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
  return h;
}
static void* cumemop_dlsym(void* h, const char* n) { return dlsym(h, n); }
#endif

using cuMemOp64_fn = int (*)(void* stream, unsigned long long addr, unsigned long long value,
                             unsigned int flags);
static cuMemOp64_fn g_cu_write64 = nullptr;
static cuMemOp64_fn g_cu_wait64 = nullptr;
static constexpr unsigned int kCuWaitValueGeq = 0x0;   // CU_STREAM_WAIT_VALUE_GEQ
static constexpr unsigned int kCuWriteDefault = 0x0;   // CU_STREAM_WRITE_VALUE_DEFAULT

static bool cumemop_resolve() {
  static bool resolved = [] {
    void* h = cumemop_dlopen();
    if (h == nullptr) return false;
    // 11.7+ made the v2 entry points the default; older drivers export only the v1
    // names with the same signature.
    g_cu_write64 = reinterpret_cast<cuMemOp64_fn>(cumemop_dlsym(h, "cuStreamWriteValue64_v2"));
    if (g_cu_write64 == nullptr)
      g_cu_write64 = reinterpret_cast<cuMemOp64_fn>(cumemop_dlsym(h, "cuStreamWriteValue64"));
    g_cu_wait64 = reinterpret_cast<cuMemOp64_fn>(cumemop_dlsym(h, "cuStreamWaitValue64_v2"));
    if (g_cu_wait64 == nullptr)
      g_cu_wait64 = reinterpret_cast<cuMemOp64_fn>(cumemop_dlsym(h, "cuStreamWaitValue64"));
    return g_cu_write64 != nullptr && g_cu_wait64 != nullptr;
  }();
  return resolved;
}

// GPU side of the flag handshake (see the block comment above): enqueued on the
// caller's (possibly capturing) stream; the WAIT immediate is the constant 1,
// replay-safe under CUDA graphs.
// The startup probe validates EAGER memops; a driver could still reject them at graph
// capture time. Those enqueue errors would otherwise be swallowed here and surface only
// as a later EndCapture failure -- log the first CUresult so triage is one step.
static void cumemop_check(int rc, const char* what) {
  static std::atomic<bool> warned{false};
  if (rc != 0 && !warned.exchange(true)) {
    std::fprintf(stderr,
                 "[freetoken/cpu_moe] %s failed with CUresult=%d (first occurrence; "
                 "subsequent errors are not repeated). If this happened during CUDA "
                 "graph capture, the driver lacks capture support for stream memops -- "
                 "set FREETOKEN_CPU_MOE_FLAG_SYNC=0.\n",
                 what, rc);
  }
}

}  // namespace

// ready and done live on separate cache lines of one mapped-pinned block; with UVA the GPU memops and the coordinator use the same addresses.
struct GpuSync::Impl {
  void* block = nullptr;
  void* device_block = nullptr;
  volatile int64_t* ready = nullptr;
  volatile int64_t* done = nullptr;
};

GpuSync::GpuSync() : impl_(std::make_unique<Impl>()) {
  if (!cumemop_resolve()) throw std::runtime_error("CPU MoE flag sync: CUDA stream memops are unavailable");
  const cudaError_t rc = cudaHostAlloc(&impl_->block, 128, cudaHostAllocPortable | cudaHostAllocMapped);
  if (rc != cudaSuccess)
    throw std::runtime_error(std::string("CPU MoE flag allocation failed: ") + cudaGetErrorString(rc));
  std::memset(impl_->block, 0, 128);
  impl_->ready = static_cast<volatile int64_t*>(impl_->block);
  impl_->done = impl_->ready + 8;
  const cudaError_t dev_rc = cudaHostGetDevicePointer(&impl_->device_block, impl_->block, 0);
  if (dev_rc != cudaSuccess) {
    cudaFreeHost(impl_->block);
    throw std::runtime_error(std::string("CPU MoE flag device address: ") + cudaGetErrorString(dev_rc));
  }
}

GpuSync::~GpuSync() { cudaFreeHost(impl_->block); }

// Functional probe on a scratch pinned int64: enqueue WRITE(7) + WAIT(>=7) + sync.
// Returns true only if the whole memop path works on THIS stream/device/driver.
bool GpuSync::probe(uintptr_t stream) {
  if (!cumemop_resolve()) return false;
  void* scratch = nullptr;
  if (cudaHostAlloc(&scratch, sizeof(int64_t), cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess)
    return false;
  *static_cast<int64_t*>(scratch) = 0;
  auto* s = reinterpret_cast<void*>(stream);
  const auto addr = reinterpret_cast<unsigned long long>(scratch);
  const bool ok = g_cu_write64(s, addr, 7ULL, kCuWriteDefault) == 0 && g_cu_wait64(s, addr, 7ULL, kCuWaitValueGeq) == 0 &&
                  cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)) == cudaSuccess;
  // a failed enqueue can leave the WRITE pending on the scratch word
  cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
  cudaFreeHost(scratch);
  return ok;
}

bool GpuSync::host_func_capture_safe() { return true; }

void GpuSync::gpu_submit(uintptr_t stream, uint64_t task) {
  auto* s = reinterpret_cast<void*>(stream);
  // Order matters and is preserved by the front end: reset done BEFORE raising ready,
  // so the coordinator's completion write for THIS step can never be wiped.
  cumemop_check(g_cu_write64(s, reinterpret_cast<unsigned long long>(impl_->done), 0ULL, kCuWriteDefault),
                "cuStreamWriteValue64(done)");
  cumemop_check(g_cu_write64(s, reinterpret_cast<unsigned long long>(impl_->ready), task, kCuWriteDefault),
                "cuStreamWriteValue64(ready)");
}

void GpuSync::gpu_wait(uintptr_t stream) {
  cumemop_check(g_cu_wait64(reinterpret_cast<void*>(stream), reinterpret_cast<unsigned long long>(impl_->done), 1ULL,
                            kCuWaitValueGeq),
                "cuStreamWaitValue64(done)");
}

uint64_t GpuSync::cpu_poll() {
  // Acquire: everything the GPU made visible before writing ready -- the D2H input copies -- is visible to the worker pool after this read.
  const int64_t task = flag_load_acquire(impl_->ready);
  if (task == 0) return 0;
  flag_store_release(impl_->ready, 0);  // consume this step's doorbell
  return static_cast<uint64_t>(task);
}

void GpuSync::cpu_done() { flag_store_release(impl_->done, 1); }

bool GpuSync::pending() const { return flag_load_acquire(impl_->ready) != 0 && flag_load_acquire(impl_->done) == 0; }

uintptr_t GpuSync::ready_word() const { return reinterpret_cast<uintptr_t>(impl_->device_block); }

uintptr_t GpuSync::done_word() const { return reinterpret_cast<uintptr_t>(impl_->device_block) + 8 * sizeof(int64_t); }

}  // namespace freetoken::cpu_moe
