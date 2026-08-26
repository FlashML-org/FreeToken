// Minimal stub of <cuda_runtime_api.h> for CPU-ONLY builds.
//
// FreeToken's CPU MoE / pinned-tensor C++ extensions only use a handful of
// CUDA runtime symbols (host-function graph nodes + pinned-host allocation).
// On a machine with no CUDA toolkit we substitute no-op implementations so the
// pure-C++ compute kernels (AVX-512 bf16 GEMV) still compile and link.
//
// This is NOT a CUDA implementation -- it is a build-time shim that lets the
// CPU decode-fallback path build and run without an NVIDIA toolchain. The
// actual GEMV compute never calls these; they exist only for the graph-glue
// and pinned-buffer APIs the upstream sources reference.
//
// Activated by defining FREETOKEN_CPU_ONLY before including this header
// (setup.py passes -DFREETOKEN_CPU_ONLY on the CPU-only build path).

#ifndef FREETOKEN_STUB_CUDA_RUNTIME_H
#define FREETOKEN_STUB_CUDA_RUNTIME_H

// CUDART_CB is a calling-convention attribute CUDA applies to host-function
// callbacks (e.g. `static void CUDART_CB submit_cb(void*)`). It is empty on
// non-Windows platforms, so we define it as nothing for the CPU-only stub.
#ifndef CUDART_CB
#define CUDART_CB
#endif

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* cudaStream_t;
typedef void* cudaEvent_t;
typedef void* cudaGraph_t;
typedef void* cudaGraphNode_t;
typedef int cudaError_t;

#define cudaSuccess 0
#define cudaErrorNotReady 600

// --- enum flags / attributes used by pinned_tensor.cpp ---------------------
#define cudaHostAllocDefault 0x00
#define cudaHostAllocPortable 0x01
#define cudaHostAllocMapped 0x02
#define cudaHostAllocWriteCombined 0x04
#define cudaHostRegisterDefault 0x00
#define cudaHostRegisterPortable 0x01
#define cudaHostRegisterMapped 0x02
#define cudaHostRegisterIoMemory 0x04
#define cudaDevAttrMaxThreadsPerBlock 1
#define cudaDevAttrMaxSharedMemoryPerBlock 8
#define cudaDevAttrUnifiedAddressing 41
#define cudaDevAttrCanUseHostPointerForRegisteredMem 53
#define cudaDevAttrComputeCapabilityMajor 75
#define cudaDevAttrComputeCapabilityMinor 76

// Opaque handle for the host-function callback signature.
typedef void (*cudaHostFn_t)(void* userData);

static inline const char* cudaGetErrorString(cudaError_t) { return "cpu-only stub"; }

static inline cudaError_t cudaGetDevice(int* /*device*/) { return cudaSuccess; }
static inline cudaError_t cudaGetDeviceCount(int* count) { *count = 0; return cudaSuccess; }
static inline cudaError_t cudaDeviceGetAttribute(int* value, int /*attr*/, int /*device*/) {
    *value = 0; return cudaSuccess;
}
static inline cudaError_t cudaDriverGetVersion(int* ver) { *ver = 0; return cudaSuccess; }
static inline cudaError_t cudaRuntimeGetVersion(int* ver) { *ver = 0; return cudaSuccess; }

// Pinned host memory: on CPU-only we fall back to plain malloc (the CPU
// executor reads from ordinary host buffers anyway).
static inline cudaError_t cudaMallocHost(void** ptr, size_t size) {
    *ptr = malloc(size); return (*ptr) ? cudaSuccess : 1;
}
static inline cudaError_t cudaFreeHost(void* ptr) { free(ptr); return cudaSuccess; }
static inline cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int /*flags*/) {
    return cudaMallocHost(ptr, size);
}
static inline cudaError_t cudaHostRegister(void* /*ptr*/, size_t /*size*/, unsigned int /*flags*/) {
    return cudaSuccess;
}
static inline cudaError_t cudaHostGetDevicePointer(void** p, void* h, unsigned int /*flags*/) {
    *p = h; return cudaSuccess;
}

// Stream / event / graph glue -- no-ops (the CPU path drives work via the
// in-process worker pool, not a CUDA stream).
static inline cudaError_t cudaStreamCreate(cudaStream_t* s) { *s = nullptr; return cudaSuccess; }
static inline cudaError_t cudaStreamSynchronize(cudaStream_t /*s*/) { return cudaSuccess; }
static inline cudaError_t cudaStreamAddCallback(cudaStream_t /*s*/, cudaHostFn_t /*cb*/,
                                                void* /*data*/, unsigned int /*flags*/) {
    return cudaSuccess;
}
static inline cudaError_t cudaLaunchHostFunc(cudaStream_t /*s*/, cudaHostFn_t cb, void* data) {
    if (cb) cb(data);  // run immediately on the calling thread (CPU-only)
    return cudaSuccess;
}
static inline cudaError_t cudaEventCreate(cudaEvent_t* e) { *e = nullptr; return cudaSuccess; }
static inline cudaError_t cudaEventRecord(cudaEvent_t /*e*/, cudaStream_t /*s*/) { return cudaSuccess; }
static inline cudaError_t cudaEventSynchronize(cudaEvent_t /*e*/) { return cudaSuccess; }
static inline cudaError_t cudaEventDestroy(cudaEvent_t /*e*/) { return cudaSuccess; }
static inline cudaError_t cudaStreamDestroy(cudaStream_t /*s*/) { return cudaSuccess; }
static inline cudaError_t cudaDeviceSynchronize(void) { return cudaSuccess; }

#ifdef __cplusplus
}
#endif

#endif  // FREETOKEN_STUB_CUDA_RUNTIME_H
