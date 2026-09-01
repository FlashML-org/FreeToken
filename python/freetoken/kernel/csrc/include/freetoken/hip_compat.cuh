#pragma once

// HIP compatibility shim for CUDA-flavored csrc headers.
// Included only when built with -D__HIP_PLATFORM_AMD__=1 (see utils.cuh).
//
// Uses hipLaunchKernel (not hipLaunchKernelExC) so this compiles on ROCm 6.x
// and TheRock 10.1 / HIP 7.x. PDL / programmatic-stream attrs are NVIDIA-only.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <tuple>

#ifndef __always_inline
#define __always_inline inline __attribute__((always_inline))
#endif

#ifndef __grid_constant__
#define __grid_constant__
#endif

using cudaError_t = ::hipError_t;
using cudaStream_t = ::hipStream_t;

inline constexpr auto cudaSuccess = ::hipSuccess;
inline constexpr auto cudaFuncAttributeMaxDynamicSharedMemorySize =
    ::hipFuncAttributeMaxDynamicSharedMemorySize;

[[nodiscard]] inline auto cudaGetErrorString(::cudaError_t e) -> const char * {
  return ::hipGetErrorString(e);
}

inline auto cudaGetLastError() -> ::cudaError_t { return ::hipGetLastError(); }

inline constexpr auto cudaDevAttrUnifiedAddressing =
    ::hipDeviceAttributeUnifiedAddressing;
inline constexpr auto cudaDevAttrCanUseHostPointerForRegisteredMem =
    ::hipDeviceAttributeCanUseHostPointerForRegisteredMem;

template <typename... A>
inline auto cudaGetDevice(A &&...args) -> ::cudaError_t {
  return ::hipGetDevice(args...);
}
template <typename... A>
inline auto cudaDeviceGetAttribute(A &&...args) -> ::cudaError_t {
  return ::hipDeviceGetAttribute(args...);
}
template <typename... A>
inline auto cudaSetDevice(A &&...args) -> ::cudaError_t {
  return ::hipSetDevice(args...);
}
template <typename... A>
inline auto cudaHostGetDevicePointer(A &&...args) -> ::cudaError_t {
  return ::hipHostGetDevicePointer(args...);
}

template <typename F>
inline auto cudaFuncSetAttribute(F *func, ::hipFuncAttribute attr, int value)
    -> ::cudaError_t {
  return ::hipFuncSetAttribute(reinterpret_cast<const void *>(func), attr,
                               value);
}

// Own launch-config type: hipLaunchConfig_t is not present on all ROCm versions.
struct cudaLaunchConfig_t {
  dim3 gridDim{};
  dim3 blockDim{};
  std::size_t dynamicSmemBytes = 0;
  cudaStream_t stream = nullptr;
  void *attrs = nullptr;
  unsigned int numAttrs = 0;
};

// Placeholder so utils.cuh can keep an attrs cache field on HIP.
struct cudaLaunchAttribute {
  int id = 0;
  struct {
    int programmaticStreamSerializationAllowed = 0;
  } val;
};

// Extended launch mapped onto portable hipLaunchKernel.
template <typename F, typename... Args>
inline auto cudaLaunchKernelEx(const cudaLaunchConfig_t *config, F func,
                               Args &&...args) -> ::cudaError_t {
  auto storage = std::make_tuple(args...);
  return [&]<std::size_t... I>(std::index_sequence<I...>) {
    void *params[] = {const_cast<void *>(
        static_cast<const void *>(&std::get<I>(storage)))...};
    return ::hipLaunchKernel(reinterpret_cast<const void *>(func),
                             config->gridDim, config->blockDim, params,
                             config->dynamicSmemBytes, config->stream);
  }(std::index_sequence_for<Args...>{});
}
