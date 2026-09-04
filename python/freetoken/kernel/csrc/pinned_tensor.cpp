#include <cstdint>
#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime_api.h>
#endif
#include <torch/extension.h>

// host alloc flags (mapped+portable) shared by both backends
#if defined(USE_HIP)
#define DEVICE_HOST_ALLOC_FLAGS_MAPPED \
  (hipHostMallocPortable | hipHostMallocMapped)
#define DEVICE_HOST_REGISTER_FLAGS_MAPPED \
  (hipHostRegisterPortable | hipHostRegisterMapped)
#else
#define DEVICE_HOST_ALLOC_FLAGS_MAPPED \
  (cudaHostAllocPortable | cudaHostAllocMapped)
#define DEVICE_HOST_REGISTER_FLAGS_MAPPED \
  (cudaHostRegisterPortable | cudaHostRegisterMapped)
#endif

namespace {

void free_pinned(void *ptr) {
  if (ptr != nullptr) {
#if defined(USE_HIP)
    hipHostFree(ptr);
#else
    cudaFreeHost(ptr);
#endif
  }
}

torch::Tensor create_pinned_tensor_like(torch::Tensor input) {
  TORCH_CHECK(input.device().is_cpu(), "Input tensor must be on CPU");
  TORCH_CHECK(input.layout() == torch::kStrided,
              "Input tensor must have strided layout");

  const auto sizes = input.sizes().vec();
  const auto strides = input.strides().vec();
  const int64_t itemsize = input.element_size();
  TORCH_CHECK(itemsize > 0, "Input tensor element size must be positive");

  const bool is_empty = input.numel() == 0;
  uint64_t storage_elements = is_empty ? 0 : 1;
  for (int64_t i = 0; i < static_cast<int64_t>(sizes.size()); ++i) {
    TORCH_CHECK(strides[i] >= 0, "Negative strides are not supported");
    if (!is_empty) {
      storage_elements += static_cast<uint64_t>(sizes[i] - 1) *
                          static_cast<uint64_t>(strides[i]);
    }
  }

  const uint64_t nbytes = storage_elements * static_cast<uint64_t>(itemsize);
  const size_t alloc_nbytes = static_cast<size_t>(nbytes == 0 ? 1 : nbytes);

  void *data_ptr = nullptr;
#if defined(USE_HIP)
  const hipError_t alloc_err =
      hipMallocHost(&data_ptr, alloc_nbytes);
  TORCH_CHECK(alloc_err == hipSuccess,
              "hipMallocHost failed: ", hipGetErrorString(alloc_err));
#else
  const cudaError_t alloc_err = cudaMallocHost(&data_ptr, alloc_nbytes);
  TORCH_CHECK(alloc_err == cudaSuccess,
              "cudaMallocHost failed: ", cudaGetErrorString(alloc_err));
#endif

  auto options = input.options().device(torch::kCPU).pinned_memory(true);

  return torch::from_blob(data_ptr, sizes, strides, free_pinned, options);
}

torch::Tensor alloc_pinned_tensor(std::vector<int64_t> sizes,
                                  at::ScalarType dtype) {
  int64_t numel = 1;
  for (const int64_t s : sizes) {
    TORCH_CHECK(s >= 0, "Sizes must be non-negative");
    numel *= s;
  }

  const uint64_t nbytes =
      static_cast<uint64_t>(numel) * c10::elementSize(dtype);
  const size_t alloc_nbytes = static_cast<size_t>(nbytes == 0 ? 1 : nbytes);

  // Portable + mapped: the offload gather kernel reads these banks straight
  // from host memory (zero-copy), which requires device-mapped pinned pages.
  void *data_ptr = nullptr;
#if defined(USE_HIP)
  const hipError_t alloc_err = hipHostMalloc(
      &data_ptr, alloc_nbytes, DEVICE_HOST_ALLOC_FLAGS_MAPPED);
  TORCH_CHECK(alloc_err == hipSuccess,
              "hipHostMalloc failed: ", hipGetErrorString(alloc_err));
#else
  const cudaError_t alloc_err = cudaHostAlloc(
      &data_ptr, alloc_nbytes, cudaHostAllocPortable | cudaHostAllocMapped);
  TORCH_CHECK(alloc_err == cudaSuccess,
              "cudaHostAlloc failed: ", cudaGetErrorString(alloc_err));
#endif

  auto options = torch::TensorOptions()
                     .dtype(dtype)
                     .device(torch::kCPU)
                     .pinned_memory(true);

  return torch::from_blob(data_ptr, sizes, free_pinned, options);
}

// Pinned host memory is GPU-dereferenceable at its host VA only where UVA identity
// holds (Linux; not Windows/WDDM, where cudaHostRegister'd memory maps to a different
// device address). Zero-copy consumers resolve bank base addresses through these.
bool host_ptr_identity() {
  int device = 0;
#if defined(USE_HIP)
  const hipError_t err = hipGetDevice(&device);
  TORCH_CHECK(err == hipSuccess, "hipGetDevice failed: ", hipGetErrorString(err));
  int uva = 0, reg = 0;
  hipDeviceGetAttribute(&uva, hipDeviceAttributeUnifiedAddressing, device);
  hipDeviceGetAttribute(
      &reg, hipDeviceAttributeCanUseHostPointerForRegisteredMem, device);
#else
  const cudaError_t err = cudaGetDevice(&device);
  TORCH_CHECK(err == cudaSuccess, "cudaGetDevice failed: ", cudaGetErrorString(err));
  int uva = 0, reg = 0;
  cudaDeviceGetAttribute(&uva, cudaDevAttrUnifiedAddressing, device);
  cudaDeviceGetAttribute(&reg, cudaDevAttrCanUseHostPointerForRegisteredMem, device);
#endif
  return uva == 1 && reg == 1;
}

int64_t host_device_ptr(int64_t host_ptr) {
  void *dev_ptr = nullptr;
#if defined(USE_HIP)
  const hipError_t err = hipHostGetDevicePointer(
      &dev_ptr, reinterpret_cast<void *>(host_ptr), 0);
  TORCH_CHECK(err == hipSuccess,
              "hipHostGetDevicePointer failed (host memory must be pinned+mapped): ",
              hipGetErrorString(err));
#else
  const cudaError_t err =
      cudaHostGetDevicePointer(&dev_ptr, reinterpret_cast<void *>(host_ptr), 0);
  TORCH_CHECK(err == cudaSuccess,
              "cudaHostGetDevicePointer failed (host memory must be pinned+mapped): ",
              cudaGetErrorString(err));
#endif
  return reinterpret_cast<int64_t>(dev_ptr);
}

void host_register(int64_t addr, int64_t nbytes) {
#if defined(USE_HIP)
  const hipError_t err =
      hipHostRegister(reinterpret_cast<void *>(addr), static_cast<size_t>(nbytes),
                      DEVICE_HOST_REGISTER_FLAGS_MAPPED);
  TORCH_CHECK(err == hipSuccess,
              "hipHostRegister failed: ", hipGetErrorString(err));
#else
  const cudaError_t err =
      cudaHostRegister(reinterpret_cast<void *>(addr), static_cast<size_t>(nbytes),
                       cudaHostRegisterPortable | cudaHostRegisterMapped);
  TORCH_CHECK(err == cudaSuccess,
              "cudaHostRegister failed: ", cudaGetErrorString(err));
#endif
}

// CUDA-only: the maximum CUDA version the NVIDIA driver supports, used to gate
// driver-JIT kernels. On ROCm there is no CUDA driver; return 0 (== "no driver").
int64_t driver_cuda_version() {
#if defined(USE_HIP)
  return 0;
#else
  int version = 0;  // stays 0 when no driver is installed
  const cudaError_t err = cudaDriverGetVersion(&version);
  TORCH_CHECK(err == cudaSuccess,
              "cudaDriverGetVersion failed: ", cudaGetErrorString(err));
  return version;
#endif
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("create_pinned_tensor_like", &create_pinned_tensor_like,
        "Create an exact-size CPU pinned tensor with input's size/stride/dtype");
  m.def("alloc_pinned_tensor", &alloc_pinned_tensor,
        "Allocate an exact-size, uninitialized CPU pinned tensor");
  m.def("host_ptr_identity", &host_ptr_identity,
        "True if the GPU dereferences pinned host memory at its host VA (UVA identity)");
  m.def("host_device_ptr", &host_device_ptr,
        "Device-visible alias of a pinned+mapped host address");
  m.def("host_register", &host_register,
        "cudaHostRegister/hipHostRegister an existing host range as portable+mapped");
  m.def("driver_cuda_version", &driver_cuda_version,
        "Max CUDA version the installed NVIDIA driver supports (0 if none / ROCm)");
}
