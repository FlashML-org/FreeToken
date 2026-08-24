#pragma once

// HIP compatibility shim: maps CUDA runtime API names to HIP equivalents so
// the same C++ source compiles under both nvcc and hipcc.  Include this instead
// of <cuda_runtime_api.h> directly when the file needs the runtime API.
//
// On NVIDIA platforms the CUDA headers are included as-is and every macro below
// resolves to the original CUDA symbol, so there is zero overhead.
//
// Supported ROCm targets:
//   gfx1100 — RX 7900 XTX / XT
//   gfx1101 — RX 7900 GRE
//   gfx1102 — RX 7700 / XT
//   gfx1103 — RX 7600 / XT
//   gfx1200 — RX 9060 family
//   gfx1201 — RX 9070 family / Radeon AI PRO R9700

#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)

#define FREETOKEN_USE_ROCM 1

// --- HIP runtime headers ---
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// --- API name mapping (CUDA -> HIP) ---
// HIP already defines most cuda* names as macros that expand to hip* equivalents
// via hip_runtime.h, but a few are missing or differ in signature.  Define them
// here so call-sites stay unchanged.

#ifndef cudaSuccess
#define cudaSuccess hipSuccess
#endif

#ifndef cudaError_t
#define cudaError_t hipError_t
#endif

#ifndef cudaGetErrorString
#define cudaGetErrorString hipGetErrorString
#endif

#ifndef cudaGetLastError
#define cudaGetLastError hipGetLastError
#endif

#ifndef cudaMallocHost
#define cudaMallocHost hipMallocHost
#endif

#ifndef cudaFreeHost
#define cudaFreeHost hipFreeHost
#endif

#ifndef cudaHostAlloc
#define cudaHostAlloc hipHostMalloc
#endif

#ifndef cudaHostRegister
#define cudaHostRegister hipHostRegister
#endif

#ifndef cudaHostRegisterPortable
#define cudaHostRegisterPortable hipHostRegisterPortable
#endif

#ifndef cudaHostRegisterMapped
#define cudaHostRegisterMapped hipHostRegisterMapped
#endif

#ifndef cudaHostAllocPortable
#define cudaHostAllocPortable hipHostMallocPortable
#endif

#ifndef cudaHostAllocMapped
#define cudaHostAllocMapped hipHostMallocMapped
#endif

#ifndef cudaHostGetDevicePointer
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#endif

#ifndef cudaGetDevice
#define cudaGetDevice hipGetDevice
#endif

#ifndef cudaDriverGetVersion
#define cudaDriverGetVersion hipDriverGetVersion
#endif

#ifndef cudaDeviceGetAttribute
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#endif

#ifndef cudaDevAttrUnifiedAddressing
#define cudaDevAttrUnifiedAddressing hipDeviceAttributeUnifiedAddressing
#endif

#ifndef cudaDevAttrCanUseHostPointerForRegisteredMem
// HIP does not expose this attribute; assume UVA identity on ROCm (true on Linux).
// TODO(ROCm): re-enable proper UVA query if HIP adds this attribute.
#define cudaDevAttrCanUseHostPointerForRegisteredMem hipDeviceAttributeUnifiedAddressing
#endif

#ifndef cudaFuncSetAttribute
#define cudaFuncSetAttribute hipFuncSetAttribute
#endif

#ifndef cudaFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#endif

#ifndef cudaLaunchKernelEx
// ROCm 7 exposes the CUDA-compatible extended launch configuration through HIP.
#define cudaLaunchKernelEx hipLaunchKernelEx
#endif

#ifndef cudaLaunchConfig_t
#define cudaLaunchConfig_t hipLaunchConfig_t
#endif

#ifndef cudaLaunchAttribute
#define cudaLaunchAttribute hipLaunchAttribute
#endif

#ifndef cudaLaunchAttributeProgrammaticStreamSerialization
// PDL (Programmatic Dependent Launch) is NVIDIA-specific.
// TODO(ROCm): PDL has no ROCm equivalent — disabled, may affect overlap scheduling latency.
#define cudaLaunchAttributeProgrammaticStreamSerialization 0
#endif

#ifndef cudaStream_t
#define cudaStream_t hipStream_t
#endif

#ifndef cudaStreamSynchronize
#define cudaStreamSynchronize hipStreamSynchronize
#endif

#ifndef cudaLaunchHostFunc
#define cudaLaunchHostFunc hipLaunchHostFunc
#endif

#ifndef CUDART_CB
#define CUDART_CB
#endif

#ifndef __grid_constant__
#define __grid_constant__
#endif

#ifndef dim3
// HIP already provides dim3; this is a no-op guard.
#endif

#else  // NVIDIA CUDA path

#define FREETOKEN_USE_ROCM 0

#include <cuda_runtime_api.h>

#endif
