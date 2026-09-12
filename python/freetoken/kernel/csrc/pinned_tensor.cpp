#include <cstdint>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

namespace {

// A failed runtime call latches its status in the calling thread until someone
// reads it. TORCH_CHECK reports the error but does not clear it, so the next
// unrelated CUDA call -- torch's, in practice -- reports *this* failure instead
// of its own. Drain it before raising. The failures these calls can originate
// are non-sticky, so draining leaves the context usable; a sticky error latched
// elsewhere will simply re-latch on the next call, which is correct.
#define FT_CUDA_CHECK(expr, ...)                                               \
  do {                                                                         \
    const cudaError_t ft_err_ = (expr);                                        \
    if (ft_err_ != cudaSuccess) {                                              \
      [[maybe_unused]] const cudaError_t ft_drained_ = cudaGetLastError();      \
      TORCH_CHECK(false, __VA_ARGS__, cudaGetErrorString(ft_err_));            \
    }                                                                          \
  } while (0)

void free_pinned(void *ptr) {
  if (ptr == nullptr) {
    return;
  }
  // A from_blob deleter runs during GC, with no exception to attribute a failure
  // to -- so this drains and never throws. Left latched it would be the worst
  // case of the bug above: a stale status with no visible origin at all.
  if (cudaFreeHost(ptr) != cudaSuccess) {
    [[maybe_unused]] const cudaError_t drained = cudaGetLastError();
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
  FT_CUDA_CHECK(cudaMallocHost(&data_ptr, alloc_nbytes),
                "cudaMallocHost failed: ");

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
  FT_CUDA_CHECK(cudaHostAlloc(&data_ptr, alloc_nbytes,
                              cudaHostAllocPortable | cudaHostAllocMapped),
                "cudaHostAlloc failed: ");

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
  FT_CUDA_CHECK(cudaGetDevice(&device), "cudaGetDevice failed: ");
  // An unqueryable attribute stays 0 and answers "no identity", which is the safe
  // answer: device_ptr() then goes through host_device_ptr(), the real translation,
  // correct on every platform. Only the latch is new here -- do not raise, or an
  // attribute query that used to degrade gracefully becomes fatal on the offload path.
  int uva = 0, reg = 0;
  if (cudaDeviceGetAttribute(&uva, cudaDevAttrUnifiedAddressing, device) != cudaSuccess) {
    [[maybe_unused]] const cudaError_t drained = cudaGetLastError();
    uva = 0;
  }
  if (cudaDeviceGetAttribute(&reg, cudaDevAttrCanUseHostPointerForRegisteredMem,
                             device) != cudaSuccess) {
    [[maybe_unused]] const cudaError_t drained = cudaGetLastError();
    reg = 0;
  }
  return uva == 1 && reg == 1;
}

int64_t host_device_ptr(int64_t host_ptr) {
  void *dev_ptr = nullptr;
  FT_CUDA_CHECK(
      cudaHostGetDevicePointer(&dev_ptr, reinterpret_cast<void *>(host_ptr), 0),
      "cudaHostGetDevicePointer failed (host memory must be pinned+mapped): ");
  return reinterpret_cast<int64_t>(dev_ptr);
}

void host_register(int64_t addr, int64_t nbytes) {
  FT_CUDA_CHECK(cudaHostRegister(reinterpret_cast<void *>(addr),
                                 static_cast<size_t>(nbytes),
                                 cudaHostRegisterPortable | cudaHostRegisterMapped),
                "cudaHostRegister failed: ");
}

int64_t driver_cuda_version() {
  int version = 0;  // stays 0 when no driver is installed
  FT_CUDA_CHECK(cudaDriverGetVersion(&version), "cudaDriverGetVersion failed: ");
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
