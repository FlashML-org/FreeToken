#include <hip/hip_runtime.h>
#include <dlfcn.h>

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace {
std::atomic<int> begins{0}, ends{0}, injected{0}, removed_nodes{0}, last_end_rc{0};

bool mode(const char* value) {
  const char* selected = std::getenv("FREETOKEN_TEST_HIP_PROBE_FAULT");
  return selected != nullptr && std::strcmp(selected, value) == 0;
}

template <typename Function>
Function real(const char* name) {
  return reinterpret_cast<Function>(dlsym(RTLD_NEXT, name));
}
}  // namespace

extern "C" hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode capture_mode) {
  const hipError_t rc = real<decltype(&hipStreamBeginCapture)>("hipStreamBeginCapture")(stream, capture_mode);
  if (rc == hipSuccess) ++begins;
  return rc;
}

extern "C" hipError_t hipStreamWriteValue64(hipStream_t stream, void* address, uint64_t value, unsigned flags) {
  // Force the compatibility path on runtimes that normally record native writes.
  if (mode("dependency") && begins == 1) {
    ++injected;
    return hipSuccess;
  }
  return real<decltype(&hipStreamWriteValue64)>("hipStreamWriteValue64")(stream, address, value, flags);
}

extern "C" hipError_t hipStreamUpdateCaptureDependencies(
    hipStream_t stream, hipGraphNode_t* dependencies, size_t count, unsigned flags) {
  if (mode("dependency")) {
    ++injected;
    return hipErrorNotSupported;
  }
  return real<decltype(&hipStreamUpdateCaptureDependencies)>("hipStreamUpdateCaptureDependencies")(
      stream, dependencies, count, flags);
}

extern "C" hipError_t hipGraphDestroyNode(hipGraphNode_t node) {
  const hipError_t rc = real<decltype(&hipGraphDestroyNode)>("hipGraphDestroyNode")(node);
  if (rc == hipSuccess) ++removed_nodes;
  return rc;
}

extern "C" hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* graph) {
  ++ends;
  hipError_t rc = real<decltype(&hipStreamEndCapture)>("hipStreamEndCapture")(stream, graph);
  if (mode("end-capture") && ends == 2 && rc == hipSuccess) {
    // A real invalid API call also sets HIP's thread-local last-error state.
    rc = real<decltype(&hipGraphGetNodes)>("hipGraphGetNodes")(*graph, nullptr, nullptr);
    ++injected;
  }
  last_end_rc = static_cast<int>(rc);
  return rc;
}

extern "C" int probe_test_injected() { return injected; }
extern "C" int probe_test_begins() { return begins; }
extern "C" int probe_test_ends() { return ends; }
extern "C" int probe_test_removed_nodes() { return removed_nodes; }
extern "C" int probe_test_last_end_rc() { return last_end_rc; }
extern "C" int probe_test_peek_error() {
  return static_cast<int>(real<decltype(&hipPeekAtLastError)>("hipPeekAtLastError")());
}
extern "C" int probe_test_make_error() {
  return static_cast<int>(real<decltype(&hipGraphGetNodes)>("hipGraphGetNodes")(nullptr, nullptr, nullptr));
}
