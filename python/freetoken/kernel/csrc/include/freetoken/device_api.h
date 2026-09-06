#pragma once

// Small runtime seam for source files shared by nvcc and hipcc.
#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
#include <hip/hip_runtime.h>
namespace freetoken::device {
using Error = hipError_t;
inline constexpr Error kErrorSuccess = hipSuccess;
using Stream = hipStream_t;
inline const char* error_string(Error error) { return hipGetErrorString(error); }
}
#define DEVICE_MEMCPY_ASYNC hipMemcpyAsync
#define DEVICE_MEMCPY_HOST_TO_DEVICE hipMemcpyHostToDevice
#define DEVICE_STREAM_SYNCHRONIZE hipStreamSynchronize
#define DEVICE_LAUNCH_HOST_FUNC hipLaunchHostFunc
#else
#include <cuda_runtime_api.h>
namespace freetoken::device {
using Error = cudaError_t;
inline constexpr Error kErrorSuccess = cudaSuccess;
using Stream = cudaStream_t;
inline const char* error_string(Error error) { return cudaGetErrorString(error); }
}
#define DEVICE_MEMCPY_ASYNC cudaMemcpyAsync
#define DEVICE_MEMCPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
#define DEVICE_STREAM_SYNCHRONIZE cudaStreamSynchronize
#define DEVICE_LAUNCH_HOST_FUNC cudaLaunchHostFunc
#endif
