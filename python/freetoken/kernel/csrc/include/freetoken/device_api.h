// Device API seam for the tvm-ffi JIT/store/index kernels (and future GPU kernels).
//
// FreeToken's hand-written kernels are compiled once, guarded by `#if defined(USE_HIP)`
// (set by kernel/_toolchain.py `_rocm_cflags` and by setup.py for the torch extensions).
// This header maps the small set of runtime calls those kernels use to the active
// backend, so a kernel body written against these macros compiles under both CUDA and
// HIP without `#if` sprinkling at every call site.
//
// Every macro here must expand to a no-op-safe, backend-correct call. Prefer the 1:1
// HIP counterparts (the HIP runtime is API-compatible at this level).
#pragma once

#include <cstdint>
#include <cstdlib>

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

namespace freetoken::device {

#if defined(USE_HIP)
using Error = hipError_t;
inline constexpr Error kErrorSuccess = hipSuccess;
inline const char* error_string(Error e) { return hipGetErrorString(e); }
using Device = hipDevice_t;
using Stream = hipStream_t;
using DeviceMemPtr = hipDeviceptr_t;
using HostFn = hipHostFn_t;
#else
using Error = cudaError_t;
inline constexpr Error kErrorSuccess = cudaSuccess;
inline const char* error_string(Error e) { return cudaGetErrorString(e); }
using Device = int;
using Stream = cudaStream_t;
using DeviceMemPtr = void*;
using HostFn = void (*)(void*);
#endif

}  // namespace freetoken::device

// ---- allocation / copy / sync / launch (backend-agnostic call sites) ----
#if defined(USE_HIP)
#define DEVICE_MALLOC(ptr, n) hipMalloc((void**)(ptr), (n))
#define DEVICE_FREE(ptr)                 hipFree(ptr)
#define DEVICE_MEMCPY_ASYNC(dst, src, n, kind, stream) \
  hipMemcpyAsync((dst), (src), (n), (kind), (stream))
#define DEVICE_MEMCPY_DEVICE_TO_DEVICE   hipMemcpyDeviceToDevice
#define DEVICE_MEMCPY_DEVICE_TO_HOST     hipMemcpyDeviceToHost
#define DEVICE_MEMCPY_HOST_TO_DEVICE     hipMemcpyHostToDevice
#define DEVICE_SYNCTHREADS()             __syncthreads()
#define DEVICE_LAUNCH_HOST_FUNC(stream, fn, data) \
  hipLaunchHostFunc((stream), (fn), (data))
#define DEVICE_STREAM_SYNCHRONIZE(stream) hipStreamSynchronize(stream)
#define DEVICE_ATOMIC_ADD(ptr, val)      atomicAdd((ptr), (val))
#define DEVICE_ATOMIC_MAX(ptr, val)      atomicMax((ptr), (val))
#define DEVICE_ATOMIC_MIN(ptr, val)      atomicMin((ptr), (val))
#define DEVICE_ATOMIC_EXCHANGE(ptr, val) atomicExch((ptr), (val))
#define DEVICE_THREAD_IDX_X              threadIdx.x
#define DEVICE_BLOCK_DIM_X               blockDim.x
#define DEVICE_BLOCK_IDX_X               blockIdx.x
#define DEVICE_GRID_DIM_X                gridDim.x
#else
#define DEVICE_MALLOC(ptr, n)            cudaMalloc((void**)(ptr), (n))
#define DEVICE_FREE(ptr)                 cudaFree(ptr)
#define DEVICE_MEMCPY_ASYNC(dst, src, n, kind, dir) \
  cudaMemcpyAsync((dst), (src), (n), (kind), (dir))
#define DEVICE_MEMCPY_DEVICE_TO_DEVICE   cudaMemcpyDeviceToDevice
#define DEVICE_MEMCPY_DEVICE_TO_HOST     cudaMemcpyDeviceToHost
#define DEVICE_MEMCPY_HOST_TO_DEVICE     cudaMemcpyHostToDevice
#define DEVICE_SYNCTHREADS()             __syncthreads()
#define DEVICE_LAUNCH_HOST_FUNC(stream, fn, data) \
  cudaLaunchHostFunc((stream), (fn), (data))
#define DEVICE_STREAM_SYNCHRONIZE(stream) cudaStreamSynchronize(stream)
#define DEVICE_ATOMIC_ADD(ptr, val)      atomicAdd((ptr), (val))
#define DEVICE_ATOMIC_MAX(ptr, val)      atomicMax((ptr), (val))
#define DEVICE_ATOMIC_MIN(ptr, val)      atomicMin((ptr), (val))
#define DEVICE_ATOMIC_EXCHANGE(ptr, val) atomicExch((ptr), (val))
#define DEVICE_THREAD_IDX_X              threadIdx.x
#define DEVICE_BLOCK_DIM_X               blockDim.x
#define DEVICE_BLOCK_IDX_X               blockIdx.x
#define DEVICE_GRID_DIM_X                gridDim.x
#endif

// The FFI kernels pin tensors to kDLCUDA today; on ROCm the tvm-ffi device type is
// kDLROCM. Kernels that resolve a device must accept both (see tensor.h). This macro
// picks the backend's DLDevice code at call time.
#if defined(USE_HIP)
#define DEVICE_DLDEVICE_ROCM 10  // kDLROCM
#define DEVICE_DLDEVICE_CUDA 2   // kDLCUDA
#define DEVICE_ACTIVE_DLDEVICE DEVICE_DLDEVICE_ROCM
#else
#define DEVICE_DLDEVICE_ROCM 10
#define DEVICE_DLDEVICE_CUDA 2
#define DEVICE_ACTIVE_DLDEVICE DEVICE_DLDEVICE_CUDA
#endif
