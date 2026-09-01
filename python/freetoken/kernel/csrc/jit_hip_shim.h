// HIP compatibility shim for FreeToken's JIT-compiled kernels.
// Force-included (-include) into kernel TUs built with the HIP toolchain.
// Maps the CUDA launch API surface used by csrc/include/freetoken/utils.cuh
// onto HIP equivalents. PDL (programmatic dependent launch) attributes have no
// HIP equivalent on this runtime; every kernel served on AMD builds with
// use_pdl=false, so the attribute path stays compile-only dead code and the
// launcher drops attrs instead of setting them.
#pragma once
#include <hip/hip_runtime.h>
#include <cstddef>
#include <utility>

#ifndef __HIP_PLATFORM_AMD__
#error "This shim is only meaningful for the AMD HIP toolchain"
#endif

typedef hipStream_t cudaStream_t;
typedef hipError_t cudaError_t;
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrUnifiedAddressing hipDeviceAttributeUnifiedAddressing
#define cudaDevAttrCanUseHostPointerForRegisteredMem hipDeviceAttributeCanUseHostPointerForRegisteredMem
#define cudaMallocHost hipMallocHost
#define cudaHostAlloc hipHostAlloc
#define cudaFreeHost hipFreeHost
#define cudaHostRegister hipHostRegister
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaHostAllocPortable hipHostAllocPortable
#define cudaHostAllocMapped hipHostAllocMapped
#define cudaHostRegisterPortable hipHostRegisterPortable
#define cudaHostRegisterMapped hipHostRegisterMapped
#define cudaGetDevice hipGetDevice
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaLaunchHostFunc hipLaunchHostFunc
#define CUDART_CB

enum cudaLaunchAttributeID_shim {
  cudaLaunchAttributeProgrammaticStreamSerialization = 99,
};

struct cudaLaunchAttribute {
  int id;
  union {
    int programmaticStreamSerializationAllowed;
  } val;
};

struct cudaLaunchConfig_t {
  dim3 gridDim;
  dim3 blockDim;
  size_t dynamicSmemBytes;
  cudaStream_t stream;
  cudaLaunchAttribute* attrs;
  unsigned int numAttrs;
};

template <typename... KernelArgs, typename... Params>
static inline hipError_t cudaLaunchKernelEx(const cudaLaunchConfig_t* config,
                                            void (*kernel)(KernelArgs...),
                                            Params&&... args) {
  hipLaunchConfig_t hcfg;
  hcfg.gridDim = config->gridDim;
  hcfg.blockDim = config->blockDim;
  hcfg.dynamicSmemBytes = config->dynamicSmemBytes;
  hcfg.stream = config->stream;
  hcfg.attrs = nullptr;
  hcfg.numAttrs = 0;
  return hipLaunchKernelEx(&hcfg, kernel, std::forward<Params>(args)...);
}

// ROCm 7.2 HIP does not provide the CUDA 11.4+ __grid_constant__ parameter
// annotation; an empty define keeps by-value kernel parameters valid.
#ifndef __grid_constant__
#define __grid_constant__
#endif

#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaGetLastError hipGetLastError
