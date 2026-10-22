// HIP stream memory operations for the CPU MoE flag handshake.
//
// HIP WAIT addresses must come from hipMallocSignalMemory, one 64-bit word per allocation.
//
// ROCm 10 records hipStreamWrite/WaitValue64 into the graph under stream capture, one node per call.
// ROCm 7.14 runs them while the stream is being captured but records nothing, so there we add explicit batch-memop nodes and splice them into the stream's dependency set.
// Which of the two applies is measured once at runtime (records_native_memops), not read from the version number.
// The probe then does a real capture + instantiate + replay through the chosen path.
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>

#include <freetoken/hip_compat.h>

#include "gpu_sync.h"

namespace freetoken::cpu_moe {

namespace {

void hip_check(hipError_t rc, const char* what) {
  if (rc != hipSuccess)
    throw std::runtime_error(std::string("CPU MoE HIP flag sync: ") + what + " failed: " + hipGetErrorString(rc));
}

// Capability failures must not poison eager fallback or hide a caller's earlier error.
struct ProbeErrorScope {
  ProbeErrorScope() { hip_check(hipPeekAtLastError(), "pending error before probe"); }
  ~ProbeErrorScope() { (void)hipGetLastError(); }
};

hipError_t alloc_signal(uint64_t** out) {
  void* storage = nullptr;
  const hipError_t rc = hipExtMallocWithFlags(&storage, sizeof(uint64_t), hipMallocSignalMemory);
  if (rc == hipSuccess) {
    *out = static_cast<uint64_t*>(storage);
    __atomic_store_n(*out, 0ULL, __ATOMIC_RELEASE);
  }
  return rc;
}

void free_signal(uint64_t* signal) {
  if (signal != nullptr) (void)hipFree(signal);
}

volatile int64_t* as_flag(uint64_t* signal) { return reinterpret_cast<volatile int64_t*>(signal); }

bool capturing(hipStream_t stream) {
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  hip_check(hipStreamIsCapturing(stream, &status), "capture status query");
  return status == hipStreamCaptureStatusActive;
}

// Whether this runtime records a native hipStreamWriteValue64 issued under capture as a graph node.
bool records_native_memops() {
  static const bool native = [] {
    ProbeErrorScope errors;
    uint64_t* word = nullptr;
    hipStream_t stream = nullptr;
    hipGraph_t graph = nullptr;
    hipError_t rc = alloc_signal(&word);
    if (rc == hipSuccess) rc = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    if (rc == hipSuccess) rc = hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed);
    const bool capture_started = rc == hipSuccess;
    if (rc == hipSuccess) rc = hipStreamWriteValue64(stream, word, 1, hipStreamWriteValueDefault);
    if (capture_started) {
      const hipError_t end_rc = hipStreamEndCapture(stream, &graph);
      if (rc == hipSuccess) rc = end_rc;
    }
    size_t nodes = 0;
    if (rc == hipSuccess) rc = hipGraphGetNodes(graph, nullptr, &nodes);
    if (graph != nullptr) (void)hipGraphDestroy(graph);
    if (stream != nullptr) {
      (void)hipStreamSynchronize(stream);
      (void)hipStreamDestroy(stream);
    }
    free_signal(word);
    return rc == hipSuccess && nodes == 1;
  }();
  return native;
}

// Appends one batch-memop node to the capture of `stream`; the node keeps a pointer to `ops`.
hipError_t add_capture_memop_node(hipStream_t stream, hipStreamBatchMemOpParams* ops, unsigned int count) {
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  hipGraph_t graph = nullptr;
  const hipGraphNode_t* dependencies = nullptr;
  size_t dependency_count = 0;
  hipError_t rc = hipStreamGetCaptureInfo_v2(stream, &status, nullptr, &graph, &dependencies, &dependency_count);
  if (rc != hipSuccess) return rc;
  if (status != hipStreamCaptureStatusActive || graph == nullptr) return hipErrorIllegalState;

  hipCtx_t context = nullptr;
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
  rc = hipCtxGetCurrent(&context);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
  if (rc != hipSuccess) return rc;

  hipBatchMemOpNodeParams params{};
  params.ctx = context;
  params.count = count;
  params.paramArray = ops;
  hipGraphNode_t node = nullptr;
  rc = hipGraphAddBatchMemOpNode(&node, graph, dependencies, dependency_count, &params);
  if (rc != hipSuccess) return rc;
  rc = hipStreamUpdateCaptureDependencies(stream, &node, 1, hipStreamSetCaptureDependencies);
  // EndCapture rejects a node that was added but never joined to the stream.
  if (rc != hipSuccess && node != nullptr) (void)hipGraphDestroyNode(node);
  return rc;
}

hipStreamBatchMemOpParams write_op(uint64_t* address, uint64_t value) {
  hipStreamBatchMemOpParams op{};
  op.writeValue.operation = hipStreamMemOpWriteValue64;
  op.writeValue.address = reinterpret_cast<hipDeviceptr_t>(address);
  op.writeValue.value64 = value;
  return op;
}

hipStreamBatchMemOpParams wait_op(uint64_t* address) {
  hipStreamBatchMemOpParams op{};
  op.waitValue.operation = hipStreamMemOpWaitValue64;
  op.waitValue.address = reinterpret_cast<hipDeviceptr_t>(address);
  op.waitValue.value64 = 1;
  op.waitValue.flags = hipStreamWaitValueGte;
  return op;
}

bool needs_explicit_nodes(hipStream_t stream) { return capturing(stream) && !records_native_memops(); }

// node_op is null for a native call, else the parameter slot of an explicit node, which the node keeps a pointer to.
// The ops of one batch node run in parallel, so each write gets its own node and consecutive writes stay in order.
hipError_t write_value(hipStream_t stream, uint64_t* address, uint64_t value, hipStreamBatchMemOpParams* node_op) {
  if (node_op == nullptr) return hipStreamWriteValue64(stream, address, value, hipStreamWriteValueDefault);
  *node_op = write_op(address, value);
  return add_capture_memop_node(stream, node_op, 1);
}

hipError_t wait_done(hipStream_t stream, uint64_t* done, hipStreamBatchMemOpParams* node_op) {
  if (node_op == nullptr) return hipStreamWaitValue64(stream, done, 1, hipStreamWaitValueGte, UINT64_MAX);
  *node_op = wait_op(done);
  return add_capture_memop_node(stream, node_op, 1);
}

}  // namespace

struct GpuSync::Impl {
  uint64_t* ready = nullptr;  // signal memory
  uint64_t* done = nullptr;
  // Explicit nodes keep these addresses, so one parameter pair per distinct task, never moved (std::map nodes are stable).
  std::map<uint64_t, std::array<hipStreamBatchMemOpParams, 2>> submit_ops;  // done = 0, ready = task
  hipStreamBatchMemOpParams wait_op{};
};

// The memop-free mode serves only the locked fetch, which ROCm keeps off: every GpuSync here needs memops.
GpuSync::GpuSync(bool) : impl_(std::make_unique<Impl>()) {
  hipError_t rc = alloc_signal(&impl_->ready);
  if (rc == hipSuccess) rc = alloc_signal(&impl_->done);
  if (rc != hipSuccess) {
    free_signal(impl_->ready);
    free_signal(impl_->done);
    hip_check(rc, "signal allocation");
  }
  // Measure now: the measurement captures and allocates, which a graph capture in progress would reject.
  (void)records_native_memops();
}

GpuSync::~GpuSync() {
  free_signal(impl_->ready);
  free_signal(impl_->done);
}

bool GpuSync::probe(uintptr_t /*stream*/) {
  ProbeErrorScope errors;
  int device = 0;
  int can_wait = 0;
  if (hipGetDevice(&device) != hipSuccess ||
      hipDeviceGetAttribute(&can_wait, hipDeviceAttributeCanUseStreamWaitValue, device) != hipSuccess || can_wait == 0)
    return false;
  const bool explicit_nodes = !records_native_memops();

  uint64_t* ready = nullptr;
  uint64_t* done = nullptr;
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  std::array<hipStreamBatchMemOpParams, 2> node_ops{};
  hipError_t rc = alloc_signal(&ready);
  if (rc == hipSuccess) rc = alloc_signal(&done);
  if (rc == hipSuccess) rc = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
  // A runtime that runs the WAIT during capture instead of recording it must not block here.
  if (rc == hipSuccess) __atomic_store_n(done, 1ULL, __ATOMIC_RELEASE);
  if (rc == hipSuccess) rc = hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed);
  const bool capture_started = rc == hipSuccess;
  if (rc == hipSuccess) rc = write_value(stream, ready, 1, explicit_nodes ? &node_ops[0] : nullptr);
  if (rc == hipSuccess) rc = wait_done(stream, done, explicit_nodes ? &node_ops[1] : nullptr);
  // End the capture even on failure so its stream and graph are not left active during cleanup.
  if (capture_started) {
    const hipError_t end_rc = hipStreamEndCapture(stream, &graph);
    if (rc == hipSuccess) rc = end_rc;
  }
  if (rc == hipSuccess) rc = hipStreamSynchronize(stream);
  size_t nodes = 0;
  if (rc == hipSuccess) rc = hipGraphGetNodes(graph, nullptr, &nodes);
  if (rc == hipSuccess && nodes != 2) rc = hipErrorNotSupported;
  if (rc == hipSuccess) rc = hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

  bool served = false;
  if (rc == hipSuccess) {
    __atomic_store_n(ready, 0ULL, __ATOMIC_RELEASE);
    __atomic_store_n(done, 0ULL, __ATOMIC_RELEASE);
    std::thread responder([&] {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (__atomic_load_n(ready, __ATOMIC_ACQUIRE) == 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
      served = __atomic_load_n(ready, __ATOMIC_ACQUIRE) == 1;
      // The probe graph never resets done, so this also releases a launch that arrives after the deadline.
      __atomic_store_n(done, 1ULL, __ATOMIC_RELEASE);
    });
    rc = hipGraphLaunch(graph_exec, stream);
    if (rc == hipSuccess) rc = hipStreamSynchronize(stream);
    responder.join();
  }
  const bool ok = rc == hipSuccess && served;

  if (graph_exec != nullptr) (void)hipGraphExecDestroy(graph_exec);
  if (graph != nullptr) (void)hipGraphDestroy(graph);
  if (stream != nullptr) (void)hipStreamDestroy(stream);
  free_signal(ready);
  free_signal(done);
  return ok;
}

void GpuSync::gpu_submit(uintptr_t stream, uint64_t task) {
  auto s = reinterpret_cast<hipStream_t>(stream);
  hipStreamBatchMemOpParams* ops = needs_explicit_nodes(s) ? impl_->submit_ops[task].data() : nullptr;
  // Reset done before raising ready, so the coordinator's completion for this step can never be wiped.
  hip_check(write_value(s, impl_->done, 0, ops), "write done");
  hip_check(write_value(s, impl_->ready, task, ops ? ops + 1 : nullptr), "write ready");
}

void GpuSync::gpu_wait(uintptr_t stream) {
  auto s = reinterpret_cast<hipStream_t>(stream);
  hip_check(wait_done(s, impl_->done, needs_explicit_nodes(s) ? &impl_->wait_op : nullptr), "wait done");
}

bool GpuSync::host_func_capture_safe() { return false; }

uint64_t GpuSync::cpu_poll() {
  const int64_t task = flag_load_acquire(as_flag(impl_->ready));
  if (task == 0) return 0;
  flag_store_release(as_flag(impl_->ready), 0);
  return static_cast<uint64_t>(task);
}

void GpuSync::cpu_done() { flag_store_release(as_flag(impl_->done), 1); }

bool GpuSync::pending() const {
  return flag_load_acquire(as_flag(impl_->ready)) != 0 && flag_load_acquire(as_flag(impl_->done)) == 0;
}

uintptr_t GpuSync::ready_word() const { return reinterpret_cast<uintptr_t>(impl_->ready); }

uintptr_t GpuSync::done_word() const { return reinterpret_cast<uintptr_t>(impl_->done); }

}  // namespace freetoken::cpu_moe
