#pragma once

#include <freetoken/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <concepts>
#include <cstddef>
#include <source_location>
#include <type_traits>

// nvcc implicitly pulls in the CUDA runtime for .cu translation units; hipcc does
// not do the equivalent for HIP, so it must be included explicitly here. On the
// HIP path there is no cudaLaunchKernelEx/cudaLaunchConfig_t equivalent (that API
// is Hopper PDL-specific), so LaunchKernel gets its own HIP-side definition below
// instead of a name-aliasing shim -- see PDL below for why that also means
// with_attr(true) is a no-op on this path.
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>

using cudaError_t = hipError_t;
constexpr hipError_t cudaSuccess = hipSuccess;
using cudaStream_t = hipStream_t;

inline const char *cudaGetErrorString(hipError_t e) { return hipGetErrorString(e); }
inline hipError_t cudaGetLastError() { return hipGetLastError(); }
inline hipError_t cudaFuncSetAttribute(const void *func, hipFuncAttribute attr,
                                       int value) {
  return hipFuncSetAttribute(func, attr, value);
}
constexpr hipFuncAttribute cudaFuncAttributeMaxDynamicSharedMemorySize =
    hipFuncAttributeMaxDynamicSharedMemorySize;

// CUDA-only kernel-parameter annotation (passes large by-value params via constant
// memory instead of copying them into local/generic memory first); HIP has no
// equivalent attribute, so this just falls back to an ordinary by-value parameter.
#define __grid_constant__
#else
#include <cuda_runtime.h>
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

// Programmatic Dependent Launch is a Hopper-only CUDA hardware feature; the PTX
// below has no HIP/ROCm equivalent. Callers gate kUsePDL off for non-Hopper CUDA
// targets already, and LaunchKernel::with_attr is a no-op on HIP (see below), so
// this stays unconditionally a no-op there rather than a compile failure.
template <bool kUsePDL> __always_inline __device__ void wait() {
#if !(defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__))
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
#endif
}

template <bool kUsePDL> __always_inline __device__ void launch() {
#if !(defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__))
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.launch_dependents;" :::);
  }
#endif
}

} // namespace PDL

} // namespace device

namespace host {

inline auto
CUDA_CHECK(::cudaError_t error,
           std::source_location location = std::source_location::current())
    -> void {
  if (error != ::cudaSuccess) {
    [[unlikely]];
    ::host::panic(location, "CUDA error: ", ::cudaGetErrorString(error));
  }
}

inline auto
CUDA_CHECK(std::source_location location = std::source_location::current())
    -> void {
  return CUDA_CHECK(::cudaGetLastError(), location);
}

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

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)

// HIP has no cudaLaunchKernelEx/cudaLaunchConfig_t analog (that API only exists to
// carry Hopper PDL attributes, which ROCm hardware has no equivalent for), so this
// launches via the plain triple-chevron form instead. with_attr(true) is therefore
// a no-op here -- there is no attribute to carry.
struct LaunchKernel {
public:
  explicit LaunchKernel(dim3 grid_dim, dim3 block_dim, DLDevice device,
                        std::size_t dynamic_shared_mem_bytes = 0) noexcept
      : m_grid_dim(grid_dim), m_block_dim(block_dim),
        m_smem(dynamic_shared_mem_bytes), m_stream(resolve_device(device)) {}

  explicit LaunchKernel(dim3 grid_dim, dim3 block_dim, cudaStream_t stream,
                        std::size_t dynamic_shared_mem_bytes = 0) noexcept
      : m_grid_dim(grid_dim), m_block_dim(block_dim),
        m_smem(dynamic_shared_mem_bytes), m_stream(stream) {}

  static auto resolve_device(DLDevice device) -> cudaStream_t {
    return static_cast<cudaStream_t>(
        ::TVMFFIEnvGetStream(device.device_type, device.device_id));
  }

  LaunchKernel(const LaunchKernel &) = delete;
  LaunchKernel &operator=(const LaunchKernel &) = delete;

  template <typename T, typename... Args>
  auto operator()(T &&kernel, Args &&...args) const -> void {
    kernel<<<m_grid_dim, m_block_dim, m_smem, m_stream>>>(
        std::forward<Args>(args)...);
    CUDA_CHECK(::cudaGetLastError());
  }

  auto with_attr(bool /*use_pdl*/) -> LaunchKernel & { return *this; }

private:
  dim3 m_grid_dim;
  dim3 m_block_dim;
  std::size_t m_smem;
  cudaStream_t m_stream;
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
