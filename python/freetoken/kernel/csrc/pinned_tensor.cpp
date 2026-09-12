#include <cstdint>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

namespace {

void free_pinned(void *ptr) {
  if (ptr != nullptr) {
    cudaFreeHost(ptr);
  }
}

// A failed CUDA runtime call leaves its status in the per-thread "last error" slot, so
// the next unrelated C10_CUDA_CHECK anywhere in the process reports it instead of its
// own result. Consume it before throwing to keep the failure local to this call. It
// matters most for host_device_ptr, which drivers may legitimately reject (pre-sm_70
// validates registration where newer arches let UVA degenerate the lookup to identity).
void check_cuda(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    cudaGetLastError();
    TORCH_CHECK(false, what, ": ", cudaGetErrorString(err));
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
  const cudaError_t alloc_err = cudaMallocHost(&data_ptr, alloc_nbytes);
  check_cuda(alloc_err, "cudaMallocHost failed");

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
  const cudaError_t alloc_err = cudaHostAlloc(
      &data_ptr, alloc_nbytes, cudaHostAllocPortable | cudaHostAllocMapped);
  check_cuda(alloc_err, "cudaHostAlloc failed");

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
  const cudaError_t err = cudaGetDevice(&device);
  check_cuda(err, "cudaGetDevice failed");
  int uva = 0, reg = 0;
  cudaDeviceGetAttribute(&uva, cudaDevAttrUnifiedAddressing, device);
  cudaDeviceGetAttribute(&reg, cudaDevAttrCanUseHostPointerForRegisteredMem, device);
  return uva == 1 && reg == 1;
}

int64_t host_device_ptr(int64_t host_ptr) {
  void *dev_ptr = nullptr;
  const cudaError_t err =
      cudaHostGetDevicePointer(&dev_ptr, reinterpret_cast<void *>(host_ptr), 0);
  check_cuda(err,
             "cudaHostGetDevicePointer failed (host memory must be pinned+mapped)");
  return reinterpret_cast<int64_t>(dev_ptr);
}

void host_register(int64_t addr, int64_t nbytes) {
  const cudaError_t err =
      cudaHostRegister(reinterpret_cast<void *>(addr), static_cast<size_t>(nbytes),
                       cudaHostRegisterPortable | cudaHostRegisterMapped);
  check_cuda(err, "cudaHostRegister failed");
}

int64_t driver_cuda_version() {
  int version = 0;  // stays 0 when no driver is installed
  const cudaError_t err = cudaDriverGetVersion(&version);
  check_cuda(err, "cudaDriverGetVersion failed");
  return version;
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
        "cudaHostRegister an existing host range as portable+mapped");
  m.def("driver_cuda_version", &driver_cuda_version,
        "Max CUDA version the installed NVIDIA driver supports (0 if none)");
}
