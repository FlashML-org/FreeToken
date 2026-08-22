#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>
#include <freetoken/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <concepts>
#include <cstddef>
#include <cstdint>

namespace {

struct StoreKernelParams {
  void *__restrict__ k_cache;
  void *__restrict__ v_cache;
  const void *__restrict__ indices;
  const void *__restrict__ k;
  const void *__restrict__ v;
  std::size_t kv_cache_stride;
  std::size_t kv_input_stride;
  std::size_t length;
};

template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kElementSize, std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void //
    store_kv_cache(const FT_GRID_CONSTANT StoreKernelParams params) {
  using namespace device;

  constexpr auto kWarpPerBlock =
      static_cast<unsigned>(kNumThreads / kWarpThreads);
  static_assert(kNumThreads % kWarpThreads == 0);

  const auto &[k_cache, v_cache, indices, k, v, kv_cache_stride,
               kv_input_stride, length] = params;
  const auto warp_id =
      (threadIdx.x / kWarpThreads) + blockIdx.x * kWarpPerBlock;
  PDL::wait<kUsePDL>();

  // each warp handles one element
  if (warp_id < length) {
    const auto pos = static_cast<const T *>(indices)[warp_id];
    const auto dst_k = pointer::offset(k_cache, pos * kv_cache_stride);
    const auto src_k = pointer::offset(k, warp_id * kv_input_stride);
    warp::copy<kElementSize>(dst_k, src_k);
    const auto dst_v = pointer::offset(v_cache, pos * kv_cache_stride);
    const auto src_v = pointer::offset(v, warp_id * kv_input_stride);
    warp::copy<kElementSize>(dst_v, src_v);
  }

  PDL::launch<kUsePDL>();
}

template <std::size_t element_size, // depends on data type and embedding dim
          std::size_t num_threads = 128,   // number of threads per block
          std::size_t max_concurrency = 1, // max blocks per SM
          bool use_pdl = false>
struct StoreKernel {
  static void run(const tvm::ffi::TensorView k_cache,
                  const tvm::ffi::TensorView v_cache,
                  const tvm::ffi::TensorView indices,
                  const tvm::ffi::TensorView k, const tvm::ffi::TensorView v) {
    using namespace host;
    auto D = SymbolicSize{"D"}; // element size
    auto L = SymbolicSize{"L"}; // length
    auto X = SymbolicSize{"X"}; // stride kv cache
    auto Y = SymbolicSize{"Y"}; // stride kv input
    auto indices_dtype_ = SymbolicDType{};
    auto dtype_ = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    TensorMatcher({-1, D}) //
        .with_strides({X, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype(dtype_)
        .verify(k_cache)
        .verify(v_cache);
    TensorMatcher({L, D}) //
        .with_strides({Y, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype(dtype_)
        .verify(k)
        .verify(v);
    TensorMatcher({L}) //
        .with_device<kDLCUDA>(device_)
        .with_dtype<int32_t, int64_t>(indices_dtype_)
        .verify(indices);

    const auto dtype_size = dtype_bytes(dtype_.unwrap());
    RuntimeCheck(element_size == dtype_size * D.unwrap());

    const auto device = device_.unwrap();
    const auto use_int32 = indices_dtype_.unwrap().bits == 32;
    const auto length = static_cast<std::size_t>(L.unwrap());
    const auto kv_cache_stride = X.unwrap() * dtype_size;
    const auto kv_input_stride = Y.unwrap() * dtype_size;

    const auto params = StoreKernelParams{
        .k_cache = k_cache.data_ptr(),
        .v_cache = v_cache.data_ptr(),
        .indices = indices.data_ptr(),
        .k = k.data_ptr(),
        .v = v.data_ptr(),
        .kv_cache_stride = kv_cache_stride,
        .kv_input_stride = kv_input_stride,
        .length = length,
    };

    constexpr auto kWarpPerBlock = num_threads / 32;
    static_assert(num_threads % 32 == 0);
    const auto num_blocks = div_ceil(length, kWarpPerBlock);
    const auto kernel = use_int32
                            ? store_kv_cache<num_threads, max_concurrency,
                                             use_pdl, element_size, int32_t>
                            : store_kv_cache<num_threads, max_concurrency,
                                             use_pdl, element_size, int64_t>;
    LaunchKernel(num_blocks, num_threads, device)
        .with_attr(use_pdl)(kernel, params);
  }
};

} // namespace
