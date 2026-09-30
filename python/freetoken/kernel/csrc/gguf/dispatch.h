// Minimal AT_DISPATCH helper for the vendored GGUF kernels (borrowed from
// sgl-kernel csrc/quantization/gguf, which are ports of llama.cpp). The donor
// pulls these macros from its large include/utils.h; we only need the float
// dispatch, so vendor just that to keep the JIT compile self-contained.
#pragma once

#include <ATen/Dispatch.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Warp-shuffle wrappers the donor pulls from sgl-kernel's utils.h (CUDA variants).
// ROCm requires a 64-bit warp mask (amd_warp_sync_functions static_assert); the
// vLLM/sgl donor passes uint32 masks. Widen to uint64_t on HIP.
#ifdef __HIP_PLATFORM_AMD__
#define SGLANG_WARP_MASK(mask) ((uint64_t)(mask))
#else
#define SGLANG_WARP_MASK(mask) (mask)
#endif
#ifndef SGLANG_SHFL_XOR_SYNC
#define SGLANG_SHFL_XOR_SYNC(mask, var, lane_mask) __shfl_xor_sync(SGLANG_WARP_MASK(mask), (var), (lane_mask))
#endif
#ifndef SGLANG_SHFL_XOR_SYNC_WIDTH
#define SGLANG_SHFL_XOR_SYNC_WIDTH(mask, var, lane_mask, width) \
  __shfl_xor_sync(SGLANG_WARP_MASK(mask), (var), (lane_mask), (width))
#endif

#define DISPATCH_CASE_FLOAT_TYPES(...)                 \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define DISPATCH_FLOAT_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_FLOAT_TYPES(__VA_ARGS__))
