// GPU <-> CPU handshake for the CPU MoE coordinator, one implementation per GPU runtime (gpu_sync_cuda.cpp, gpu_sync_hip.cpp).
//
// A backend keeps two contracts, in eager mode and in graph replay:
// 1. submit, GPU -> CPU: after gpu_submit(stream, task), cpu_poll() returns task once, and every earlier write of the stream is visible to the CPU.
// 2. wait, CPU -> GPU: gpu_wait(stream) holds later work of the stream until cpu_done(), and that work sees every CPU write made before cpu_done().
// task is a nonzero word, opaque here; the executor encodes the layer and token count in it, as a constant baked into each captured graph.
// One task is in flight at a time: decode runs on one stream and every layer waits before the next one signals.
// KernelIo (gpu_sync_io.h) may replace the copies around submit and wait: its kernel copies the inputs into the IO rows and does gpu_submit's writes itself.
// It skips a step without a CPU route: done = 1 and no ready, so cpu_poll() never sees the step and gpu_wait() passes at once.
#pragma once

#include <cstdint>
#include <memory>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace freetoken::cpu_moe {

// Portable ordering for the flag handshake: "ready observed => the DMA'd inputs that
// preceded the bump are visible" and "y stores are visible before done". Plain
// volatile loads lean on x86 TSO; acquire/release makes it hold on aarch64 too
// (GH200/Jetson) at zero x86 cost. (MSVC branch is x86-only today: compiler barrier
// + TSO.)
inline int64_t flag_load_acquire(const volatile int64_t* p) {
#if defined(_MSC_VER)
  const int64_t v = *p;
  _ReadWriteBarrier();
  return v;
#else
  return __atomic_load_n(const_cast<const int64_t*>(p), __ATOMIC_ACQUIRE);
#endif
}

inline void flag_store_release(volatile int64_t* p, int64_t v) {
#if defined(_MSC_VER)
  _ReadWriteBarrier();
  *p = v;
#else
  __atomic_store_n(const_cast<int64_t*>(p), v, __ATOMIC_RELEASE);
#endif
}

class GpuSync {
 public:
  // Allocates the handshake memory; throws if this runtime cannot.
  GpuSync();
  ~GpuSync();
  GpuSync(const GpuSync&) = delete;
  GpuSync& operator=(const GpuSync&) = delete;

  // Whether both contracts hold on this device and stream, including graph replay where the runtime needs a real check.
  static bool probe(uintptr_t stream);
  // Whether the cudaLaunchHostFunc fallback is replayed correctly from a captured graph.
  static bool host_func_capture_safe();

  // 1. submit. gpu_submit() is enqueued on `stream`, which may be capturing; cpu_poll() returns 0 when nothing is submitted.
  void gpu_submit(uintptr_t stream, uint64_t task);
  uint64_t cpu_poll();

  // 2. wait. cpu_done() also unblocks the stream on teardown and from the watchdog.
  void gpu_wait(uintptr_t stream);
  void cpu_done();

  // Submitted but not polled yet (watchdog only).
  bool pending() const;

  // Device addresses of the two words, for KernelIo.
  uintptr_t ready_word() const;
  uintptr_t done_word() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace freetoken::cpu_moe
