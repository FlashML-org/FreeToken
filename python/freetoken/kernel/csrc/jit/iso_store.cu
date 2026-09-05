// IsoQuant (ISO3/ISO4) KV-cache pack/unpack kernels (quantize-on-write).
// See iso_common.cuh for the algorithm and block layouts.

#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>
#include <freetoken/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <concepts>
#include <cstddef>
#include <cstdint>

#include "iso_common.cuh"

namespace {

using iso::kIsoBlock;
using iso::IsoBlockBytes;

struct IsoStoreParams {
  uint8_t *__restrict__ k_cache;
  uint8_t *__restrict__ v_cache;
  const void *__restrict__ indices;
  const __nv_bfloat16 *__restrict__ k;
  const __nv_bfloat16 *__restrict__ v;
  std::size_t cache_row_bytes;  // bytes per token row in the packed caches
  std::size_t in_row_bytes;     // bytes per token row in the bf16 inputs
  std::size_t length;           // number of tokens
  int nheads;
  int nb;  // 128-value blocks per head (head_dim / 128)
};

template <int kNumThreads, int FMT, std::integral T>
__global__ __launch_bounds__(kNumThreads) void //
    iso_store_kernel(const __grid_constant__ IsoStoreParams params) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const std::size_t wid = blockIdx.x * (kNumThreads / 32) + warp;
  const std::size_t total = params.length * static_cast<std::size_t>(params.nheads);
  if (wid >= total) return;
  const std::size_t tok = wid / params.nheads;
  const int head = static_cast<int>(wid % params.nheads);
  constexpr int BB = IsoBlockBytes<FMT>::value;

  const auto pos = static_cast<const T *>(params.indices)[tok];
  const __nv_bfloat16 *src_k = reinterpret_cast<const __nv_bfloat16 *>(
      reinterpret_cast<const uint8_t *>(params.k) + tok * params.in_row_bytes) +
      head * params.nb * kIsoBlock;
  const __nv_bfloat16 *src_v = reinterpret_cast<const __nv_bfloat16 *>(
      reinterpret_cast<const uint8_t *>(params.v) + tok * params.in_row_bytes) +
      head * params.nb * kIsoBlock;
  uint8_t *dst_k = params.k_cache + static_cast<std::size_t>(pos) * params.cache_row_bytes +
                   head * (params.nb * BB);
  uint8_t *dst_v = params.v_cache + static_cast<std::size_t>(pos) * params.cache_row_bytes +
                   head * (params.nb * BB);
  for (int b = 0; b < params.nb; ++b) {
    iso::iso_quantize_block<FMT>(src_k + b * kIsoBlock, dst_k + b * BB, lane);
    iso::iso_quantize_block<FMT>(src_v + b * kIsoBlock, dst_v + b * BB, lane);
  }
}

struct IsoDequantParams {
  __nv_bfloat16 *__restrict__ k_out;
  __nv_bfloat16 *__restrict__ v_out;
  const uint8_t *__restrict__ k_cache;
  const uint8_t *__restrict__ v_cache;
  const void *__restrict__ indices;
  std::size_t cache_row_bytes;
  std::size_t out_row_bytes;
  std::size_t length;
  int nheads;
  int nb;
};

template <int kNumThreads, int FMT, std::integral T>
__global__ __launch_bounds__(kNumThreads) void //
    iso_dequant_kernel(const __grid_constant__ IsoDequantParams params) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const std::size_t wid = blockIdx.x * (kNumThreads / 32) + warp;
  const std::size_t total = params.length * static_cast<std::size_t>(params.nheads);
  if (wid >= total) return;
  const std::size_t tok = wid / params.nheads;
  const int head = static_cast<int>(wid % params.nheads);
  constexpr int BB = IsoBlockBytes<FMT>::value;

  const auto pos = static_cast<const T *>(params.indices)[tok];
  const uint8_t *src_k = params.k_cache + static_cast<std::size_t>(pos) * params.cache_row_bytes +
                         head * (params.nb * BB);
  const uint8_t *src_v = params.v_cache + static_cast<std::size_t>(pos) * params.cache_row_bytes +
                         head * (params.nb * BB);
  __nv_bfloat16 *dst_k = reinterpret_cast<__nv_bfloat16 *>(
      reinterpret_cast<uint8_t *>(params.k_out) + tok * params.out_row_bytes) +
      head * params.nb * kIsoBlock;
  __nv_bfloat16 *dst_v = reinterpret_cast<__nv_bfloat16 *>(
      reinterpret_cast<uint8_t *>(params.v_out) + tok * params.out_row_bytes) +
      head * params.nb * kIsoBlock;
  for (int b = 0; b < params.nb; ++b) {
    iso::iso_dequantize_block<FMT>(src_k + b * BB, dst_k + b * kIsoBlock, lane);
    iso::iso_dequantize_block<FMT>(src_v + b * BB, dst_v + b * kIsoBlock, lane);
  }
}

template <int fmt_bits, int num_threads = 128>
struct IsoStoreKernel {
  static void run(const tvm::ffi::TensorView k_cache,
                  const tvm::ffi::TensorView v_cache,
                  const tvm::ffi::TensorView indices,
                  const tvm::ffi::TensorView k, const tvm::ffi::TensorView v,
                  int64_t nheads, int64_t head_dim) {
    using namespace host;
    auto R = SymbolicSize{"R"};  // packed bytes per token row
    auto L = SymbolicSize{"L"};  // tokens to store
    auto E = SymbolicSize{"E"};  // elements per token row (nheads * head_dim)
    auto X = SymbolicSize{"X"};
    auto Y = SymbolicSize{"Y"};
    auto indices_dtype_ = SymbolicDType{};
    auto bf16_dtype_ = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    TensorMatcher({-1, R})
        .with_strides({X, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype<uint8_t>()
        .verify(k_cache)
        .verify(v_cache);
    TensorMatcher({L, E})
        .with_strides({Y, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype(bf16_dtype_)
        .verify(k)
        .verify(v);
    TensorMatcher({L})
        .with_device<kDLCUDA>(device_)
        .with_dtype<int32_t, int64_t>(indices_dtype_)
        .verify(indices);
    {
      const auto dt = bf16_dtype_.unwrap();
      RuntimeCheck(dt.code == kDLBfloat && dt.bits == 16, "k/v must be bf16");
    }

    constexpr int BB = IsoBlockBytes<fmt_bits>::value;
    const std::size_t nb = static_cast<std::size_t>(head_dim) / kIsoBlock;
    RuntimeCheck(head_dim % kIsoBlock == 0, "head_dim must be divisible by 128");
    RuntimeCheck(R.unwrap() == static_cast<int64_t>(nheads * nb * BB),
                 "packed cache row size mismatch");

    const auto device = device_.unwrap();
    const auto use_int32 = indices_dtype_.unwrap().bits == 32;
    const auto length = static_cast<std::size_t>(L.unwrap());

    const auto params = IsoStoreParams{
        .k_cache = static_cast<uint8_t *>(k_cache.data_ptr()),
        .v_cache = static_cast<uint8_t *>(v_cache.data_ptr()),
        .indices = indices.data_ptr(),
        .k = static_cast<const __nv_bfloat16 *>(k.data_ptr()),
        .v = static_cast<const __nv_bfloat16 *>(v.data_ptr()),
        .cache_row_bytes = static_cast<std::size_t>(X.unwrap()),
        .in_row_bytes = static_cast<std::size_t>(Y.unwrap()) * sizeof(__nv_bfloat16),
        .length = length,
        .nheads = static_cast<int>(nheads),
        .nb = static_cast<int>(nb),
    };

    const auto num_warps = length * static_cast<std::size_t>(nheads);
    const auto num_blocks = div_ceil(num_warps, std::size_t{num_threads / 32});
    const auto kernel = use_int32
                            ? iso_store_kernel<num_threads, fmt_bits, int32_t>
                            : iso_store_kernel<num_threads, fmt_bits, int64_t>;
    LaunchKernel(num_blocks, num_threads, device)(kernel, params);
  }
};

template <int fmt_bits, int num_threads = 128>
struct IsoDequantKernel {
  static void run(const tvm::ffi::TensorView k_out,
                  const tvm::ffi::TensorView v_out,
                  const tvm::ffi::TensorView k_cache,
                  const tvm::ffi::TensorView v_cache,
                  const tvm::ffi::TensorView indices, int64_t nheads,
                  int64_t head_dim) {
    using namespace host;
    auto R = SymbolicSize{"R"};
    auto L = SymbolicSize{"L"};
    auto E = SymbolicSize{"E"};
    auto X = SymbolicSize{"X"};
    auto Y = SymbolicSize{"Y"};
    auto indices_dtype_ = SymbolicDType{};
    auto bf16_dtype_ = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    TensorMatcher({-1, R})
        .with_strides({X, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype<uint8_t>()
        .verify(k_cache)
        .verify(v_cache);
    TensorMatcher({L, E})
        .with_strides({Y, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype(bf16_dtype_)
        .verify(k_out)
        .verify(v_out);
    TensorMatcher({L})
        .with_device<kDLCUDA>(device_)
        .with_dtype<int32_t, int64_t>(indices_dtype_)
        .verify(indices);
    {
      const auto dt = bf16_dtype_.unwrap();
      RuntimeCheck(dt.code == kDLBfloat && dt.bits == 16, "k/v must be bf16");
    }

    constexpr int BB = IsoBlockBytes<fmt_bits>::value;
    const std::size_t nb = static_cast<std::size_t>(head_dim) / kIsoBlock;
    RuntimeCheck(head_dim % kIsoBlock == 0, "head_dim must be divisible by 128");

    const auto device = device_.unwrap();
    const auto use_int32 = indices_dtype_.unwrap().bits == 32;
    const auto length = static_cast<std::size_t>(L.unwrap());

    const auto params = IsoDequantParams{
        .k_out = static_cast<__nv_bfloat16 *>(k_out.data_ptr()),
        .v_out = static_cast<__nv_bfloat16 *>(v_out.data_ptr()),
        .k_cache = static_cast<const uint8_t *>(k_cache.data_ptr()),
        .v_cache = static_cast<const uint8_t *>(v_cache.data_ptr()),
        .indices = indices.data_ptr(),
        .cache_row_bytes = static_cast<std::size_t>(X.unwrap()),
        .out_row_bytes = static_cast<std::size_t>(Y.unwrap()) * sizeof(__nv_bfloat16),
        .length = length,
        .nheads = static_cast<int>(nheads),
        .nb = static_cast<int>(nb),
    };

    const auto num_warps = length * static_cast<std::size_t>(nheads);
    const auto num_blocks = div_ceil(num_warps, std::size_t{num_threads / 32});
    const auto kernel = use_int32
                            ? iso_dequant_kernel<num_threads, fmt_bits, int32_t>
                            : iso_dequant_kernel<num_threads, fmt_bits, int64_t>;
    LaunchKernel(num_blocks, num_threads, device)(kernel, params);
  }
};

} // namespace
