#pragma once

#include <freetoken/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <concepts>
#include <cstddef>
#include <source_location>
#include <type_traits>

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
// HIP has no __grid_constant__ (a CUDA read-only-constant optimization). Define it
// empty so `const __grid_constant__ Params params` compiles as a plain by-value
// parameter, which is correct (just without the CUDA constant-cache hint).
#ifndef __grid_constant__
#define __grid_constant__
#endif
#endif

namespace device {

inline constexpr auto kWarpThreads = 32u;

template <std::integral T, std::integral U>
__always_inline __device__ constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}

namespace pointer {

// we only allow void * pointer arithmetic for safety

template <typename T, std::integral... U>
__always_inline __device__ auto offset(T *ptr, U... offset) -> void * {
  static_assert(std::is_same_v<T, void>,
                "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<char *>(ptr) + (... + offset);
}

template <typename T, std::integral... U>
__always_inline __device__ auto offset(const T *ptr, U... offset) -> const
    void * {
  static_assert(std::is_same_v<T, void>,
                "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<const char *>(ptr) + (... + offset);
}

} // namespace pointer

namespace PDL {

// Programmatic Dependent Launch is a CUDA-only optimization (griddepcontrol).
// HIP has no equivalent; the wait/launch are no-ops there. PDL is optional in the
// kernels (use_pdl defaults to false), so dropping it is purely a perf change.
template <bool kUsePDL> __always_inline __device__ void wait() {
#if !defined(USE_HIP)
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
#endif
}

template <bool kUsePDL> __always_inline __device__ void launch() {
#if !defined(USE_HIP)
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.launch_dependents;" :::);
  }
#endif
}

} // namespace PDL

} // namespace device

namespace host {

#if defined(USE_HIP)
inline auto
HIP_CHECK(::hipError_t error,
          std::source_location location = std::source_location::current())
    -> void {
  if (error != ::hipSuccess) [[unlikely]] {
    ::host::panic(location, "HIP error: ", ::hipGetErrorString(error));
  }
}

inline auto
HIP_CHECK(std::source_location location = std::source_location::current())
    -> void {
  return HIP_CHECK(::hipGetLastError(), location);
}
#else
inline auto
CUDA_CHECK(::cudaError_t error,
           std::source_location location = std::source_location::current())
    -> void {
  if (error != ::cudaSuccess) [[unlikely]] {
    ::host::panic(location, "CUDA error: ", ::cudaGetErrorString(error));
  }
}

inline auto
CUDA_CHECK(std::source_location location = std::source_location::current())
    -> void {
  return CUDA_CHECK(::cudaGetLastError(), location);
}
#endif

#if defined(USE_HIP)
template <auto F> inline void set_smem_once(std::size_t smem_size) {
  static const auto last_smem_size = [&] {
    HIP_CHECK(::hipFuncSetAttribute(
        F, ::hipFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    return smem_size;
  }();
  RuntimeCheck(
      smem_size <= last_smem_size,
      "Dynamic shared memory size exceeds the previously set maximum size: ",
      last_smem_size, " bytes");
}
#else
template <auto F> inline void set_smem_once(std::size_t smem_size) {
  static const auto last_smem_size = [&] {
    CUDA_CHECK(::cudaFuncSetAttribute(
        F, ::cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    return smem_size;
  }();
  RuntimeCheck(
      smem_size <= last_smem_size,
      "Dynamic shared memory size exceeds the previously set maximum size: ",
      last_smem_size, " bytes");
}
#endif

#if defined(USE_HIP)
struct LaunchKernel {
public:
  explicit LaunchKernel(dim3 grid_dim, dim3 block_dim, DLDevice device,
                        std::size_t dynamic_shared_mem_bytes = 0) noexcept
      : m_grid(grid_dim), m_block(block_dim),
        m_stream(resolve_device(device)), m_smem(dynamic_shared_mem_bytes) {}

  explicit LaunchKernel(dim3 grid_dim, dim3 block_dim, hipStream_t stream,
                        std::size_t dynamic_shared_mem_bytes = 0) noexcept
      : m_grid(grid_dim), m_block(block_dim), m_stream(stream),
        m_smem(dynamic_shared_mem_bytes) {}

  static auto resolve_device(DLDevice device) -> hipStream_t {
    return static_cast<hipStream_t>(
        ::TVMFFIEnvGetStream(device.device_type, device.device_id));
  }

  LaunchKernel(const LaunchKernel &) = delete;
  LaunchKernel &operator=(const LaunchKernel &) = delete;

  template <typename T, typename... Args>
  auto operator()(T &&kernel, Args &&...args) const -> void {
    // hipLaunchKernel takes a void** args array (pointers to each argument value),
    // unlike cudaLaunchKernelEx's variadic form. The array is consumed at launch.
    void *arg_array[sizeof...(Args)] = {
        const_cast<void *>(static_cast<const void *>(&args))...};
    HIP_CHECK(::hipLaunchKernel(reinterpret_cast<const void *>(kernel), m_grid,
                                m_block, arg_array, m_smem, m_stream));
  }

  auto with_attr(bool /*use_pdl*/) -> LaunchKernel & {
    // HIP has no programmatic dependent launch / launch attributes; PDL is a
    // CUDA-only optimization and is dropped here.
    return *this;
  }

private:
  dim3 m_grid;
  dim3 m_block;
  hipStream_t m_stream;
  std::size_t m_smem;
};
#else
struct LaunchKernel {
public:
  explicit LaunchKernel(dim3 grid_dim, dim3 block_dim, DLDevice device,
                        std::size_t dynamic_shared_mem_bytes = 0) noexcept
      : m_config(s_make_config(grid_dim, block_dim, resolve_device(device),
                               dynamic_shared_mem_bytes)) {}

  explicit LaunchKernel(dim3 grid_dim, dim3 block_dim, cudaStream_t stream,
                        std::size_t dynamic_shared_mem_bytes = 0) noexcept
      : m_config(s_make_config(grid_dim, block_dim, stream,
                               dynamic_shared_mem_bytes)) {}

  static auto resolve_device(DLDevice device) -> cudaStream_t {
    return static_cast<cudaStream_t>(
        ::TVMFFIEnvGetStream(device.device_type, device.device_id));
  }

  LaunchKernel(const LaunchKernel &) = delete;
  LaunchKernel &operator=(const LaunchKernel &) = delete;

  template <typename T, typename... Args>
  auto operator()(T &&kernel, Args &&...args) const -> void {
    CUDA_CHECK(
        ::cudaLaunchKernelEx(&m_config, kernel, std::forward<Args>(args)...));
  }

  auto with_attr(bool use_pdl) -> LaunchKernel & {
    if (use_pdl) {
      m_attr_cache.id = ::cudaLaunchAttributeProgrammaticStreamSerialization;
      m_attr_cache.val.programmaticStreamSerializationAllowed = 1;
      m_config.attrs = &m_attr_cache;
      m_config.numAttrs = 1;
    } else {
      m_config.numAttrs = 0;
    }
    return *this;
  }

private:
  static auto s_make_config(dim3 grid_dim, dim3 block_dim, cudaStream_t stream,
                            std::size_t smem) -> cudaLaunchConfig_t {
    auto config = ::cudaLaunchConfig_t{};
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    config.numAttrs = 0;
    return config;
  }
  cudaLaunchConfig_t m_config;
  cudaLaunchAttribute m_attr_cache;
};
#endif

} // namespace host
