#pragma once

// Lets pinned_tensor.cpp and cpu_moe_ext.cpp call the CUDA Runtime API names they
// were written against while actually linking HIP on ROCm builds. Only the calls
// those two files use are covered -- this is not a general CUDA/HIP compat layer.
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime_api.h>

// CUDA's host-callback calling-convention annotation; empty on POSIX (matches
// cuda_runtime_api.h's own definition there). hipHostFn_t has no such annotation.
#define CUDART_CB

using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
constexpr hipError_t cudaSuccess = hipSuccess;
constexpr unsigned int cudaHostAllocPortable = hipHostMallocPortable;
constexpr unsigned int cudaHostAllocMapped = hipHostMallocMapped;
constexpr unsigned int cudaHostRegisterPortable = hipHostRegisterPortable;
constexpr unsigned int cudaHostRegisterMapped = hipHostRegisterMapped;
constexpr hipDeviceAttribute_t cudaDevAttrUnifiedAddressing =
    hipDeviceAttributeUnifiedAddressing;
constexpr hipDeviceAttribute_t cudaDevAttrCanUseHostPointerForRegisteredMem =
    hipDeviceAttributeCanUseHostPointerForRegisteredMem;

inline hipError_t cudaMallocHost(void **ptr, size_t size) {
  return hipHostMalloc(ptr, size, hipHostMallocDefault);
}
inline hipError_t cudaFreeHost(void *ptr) { return hipHostFree(ptr); }
inline hipError_t cudaHostAlloc(void **ptr, size_t size, unsigned int flags) {
  return hipHostMalloc(ptr, size, flags);
}
inline hipError_t cudaGetDevice(int *device) { return hipGetDevice(device); }
inline hipError_t cudaDeviceGetAttribute(int *value, hipDeviceAttribute_t attr,
                                         int device) {
  return hipDeviceGetAttribute(value, attr, device);
}
inline hipError_t cudaHostGetDevicePointer(void **devPtr, void *hostPtr,
                                           unsigned int flags) {
  return hipHostGetDevicePointer(devPtr, hostPtr, flags);
}
inline hipError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags) {
  return hipHostRegister(ptr, size, flags);
}
inline hipError_t cudaDriverGetVersion(int *v) { return hipDriverGetVersion(v); }
inline const char *cudaGetErrorString(hipError_t e) { return hipGetErrorString(e); }
inline hipError_t cudaStreamSynchronize(hipStream_t s) {
  return hipStreamSynchronize(s);
}
inline hipError_t cudaLaunchHostFunc(hipStream_t s, hipHostFn_t fn, void *data) {
  return hipLaunchHostFunc(s, fn, data);
}

#else
#include <cuda_runtime_api.h>
#endif
