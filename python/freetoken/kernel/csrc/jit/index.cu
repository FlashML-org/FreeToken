#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>
#include <freetoken/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/tuple.h>

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace {

struct IndexKernelParams {
  void *__restrict__ output;
  const void *__restrict__ weight;
  const void *__restrict__ indice;
  std::size_t num_warps;
};

struct MaskedKernelParams {
  IndexKernelParams params;
  std::size_t start;
  std::size_t length;
};

template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kElementSize, std::size_t kNumSplits, std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void //
    index_kernel(const FT_GRID_CONSTANT IndexKernelParams params) {
  using namespace device;
  constexpr auto kSize = kElementSize;
  constexpr auto kSizePerWarp = kSize / kNumSplits;
  constexpr auto kWarpPerBlock = static_cast<unsigned>(kNumThreads / 32);

  static_assert(kNumThreads % 32 == 0);
  static_assert(std::has_single_bit(kNumSplits));
  static_assert(kElementSize % kNumSplits == 0);

  const auto &[output, weight, indices_, num_warps] = params;
  const auto indices = static_cast<const T *>(indices_);
  const auto warp_id =
      (threadIdx.x / kWarpThreads) + blockIdx.x * kWarpPerBlock;
  PDL::wait<kUsePDL>();

  if (warp_id < num_warps) {
    const auto pos = indices[warp_id / kNumSplits];
    const auto dst = pointer::offset(output, warp_id * kSizePerWarp);
    const auto src = pointer::offset(weight, pos * kSize,
                                     (warp_id % kNumSplits) * kSizePerWarp);
    warp::copy<kSizePerWarp>(dst, src);
  }

  PDL::launch<kUsePDL>();
}

template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kElementSize, std::size_t kNumSplits, std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void //
    masked_index_kernel(
        const FT_GRID_CONSTANT MaskedKernelParams mask_params) {
  using namespace device;
  constexpr auto kSize = kElementSize;
  constexpr auto kSizePerWarp = kSize / kNumSplits;
  constexpr auto kWarpPerBlock = static_cast<unsigned>(kNumThreads / 32);

  static_assert(kNumThreads % 32 == 0);
  static_assert(std::has_single_bit(kNumSplits));
  static_assert(kElementSize % kNumSplits == 0);

  const auto &[params, start, length] = mask_params;
  const auto &[output, weight, indices_, num_warps] = params;
  const auto indices = static_cast<const T *>(indices_);
  const auto warp_id =
      (threadIdx.x / kWarpThreads) + blockIdx.x * kWarpPerBlock;

  PDL::wait<kUsePDL>();

  if (warp_id < num_warps) {
    const auto pos = indices[warp_id / kNumSplits] - start;
    const auto dst = pointer::offset(output, warp_id * kSizePerWarp);
    if (pos < length) {
      const auto src = pointer::offset(weight, pos * kSize,
                                       (warp_id % kNumSplits) * kSizePerWarp);
      warp::copy<kSizePerWarp>(dst, src);
    } else {
      warp::reset<kSizePerWarp>(dst);
    }
  }

  PDL::launch<kUsePDL>();
}

template <std::size_t element_size,   // depends on data type and embedding dim
          std::size_t num_splits = 1, // how many warps handles one element
          std::size_t num_threads = 128,   // number of threads per block
          std::size_t max_concurrency = 1, // max blocks per SM
          bool use_pdl = false>
struct IndexKernel {
  static void run(const tvm::ffi::TensorView weights,
                  const tvm::ffi::TensorView indices,
                  const tvm::ffi::TensorView output,
                  tvm::ffi::Optional<tvm::ffi::Tuple<int, int>> mask_opts) {
    using namespace host;
    auto D = SymbolicSize{"D"}; // embedding size
    auto L = SymbolicSize{"L"}; // num indices
    auto device_ = SymbolicDevice{};
    auto weights_dtype_ = SymbolicDType{};
    auto indices_dtype_ = SymbolicDType{};

    TensorMatcher({-1, D}) //
        .with_dtype(weights_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(weights);
    TensorMatcher({L, D}) //
        .with_dtype(weights_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(output);
    TensorMatcher({L}) //
        .with_dtype<int32_t, int64_t>(indices_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(indices);

    const auto device = device_.unwrap();
    const auto use_int32 = indices_dtype_.unwrap().bits == 32;
    const auto num_indices = L.unwrap();
    const auto entry_size = dtype_bytes(weights_dtype_.unwrap()) * D.unwrap();
    RuntimeCheck(entry_size == element_size,
                 "IndexKernel: element_size mismatch.");

    constexpr auto kWarpPerBlock = num_threads / 32;
    const auto num_warps = num_splits * num_indices;
    const auto num_blocks = div_ceil(num_warps, kWarpPerBlock);
    const auto params = IndexKernelParams{
        .output = static_cast<char *>(output.data_ptr()),
        .weight = static_cast<const char *>(weights.data_ptr()),
        .indice = indices.data_ptr(),
        .num_warps = num_warps,
    };

    if (mask_opts.has_value()) {
      const auto &obj = mask_opts.value();
      const auto [start, length] = obj;
      const auto m_params = MaskedKernelParams{
          .params = params,
          .start = static_cast<std::size_t>(start),
          .length = static_cast<std::size_t>(length),
      };
      const auto kernel =
          use_int32 ? masked_index_kernel<num_threads, max_concurrency, use_pdl,
                                          element_size, num_splits, int32_t>
                    : masked_index_kernel<num_threads, max_concurrency, use_pdl,
                                          element_size, num_splits, int64_t>;
      LaunchKernel(num_blocks, num_threads, device)
          .with_attr(use_pdl)(kernel, m_params);
    } else {
      const auto kernel =
          use_int32 ? index_kernel<num_threads, max_concurrency, use_pdl,
                                   element_size, num_splits, int32_t>
                    : index_kernel<num_threads, max_concurrency, use_pdl,
                                   element_size, num_splits, int64_t>;
      LaunchKernel(num_blocks, num_threads, device)
          .with_attr(use_pdl)(kernel, params);
    }
  }
};

} // namespace
