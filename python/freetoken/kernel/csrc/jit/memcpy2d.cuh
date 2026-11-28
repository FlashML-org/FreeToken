#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

// Strided copies on the copy engines: each row of ``desc`` (CPU int64 [n, 6]) is one
// cudaMemcpy2DAsync of ``height`` rows of ``width`` bytes, ``spitch`` apart at ``src``
// and ``dpitch`` apart at ``dst``; ``src == 0`` zeroes the destination rows instead.
// Addresses are UVA (device memory, or registered host memory), so the kind is Default.
struct Memcpy2D {
    static void run(tvm::ffi::TensorView desc, int64_t stream_handle) {
        using namespace host;
        auto N = SymbolicSize{"copies"};
        auto dtype = SymbolicDType{};
        TensorMatcher({N, 6}).with_dtype<int64_t>(dtype).with_device<kDLCPU>().verify(desc);
        const auto* d = static_cast<const int64_t*>(desc.data_ptr());
        const auto n = static_cast<std::size_t>(N.unwrap());
#if FREETOKEN_USE_ROCM
        const auto stream = reinterpret_cast<::hipStream_t>(stream_handle);
#else
        const auto stream = reinterpret_cast<::cudaStream_t>(stream_handle);
#endif
        for (std::size_t i = 0; i < n; ++i) {
            auto* dst = reinterpret_cast<void*>(d[6 * i]);
            const auto dpitch = static_cast<std::size_t>(d[6 * i + 1]);
            const auto* src = reinterpret_cast<const void*>(d[6 * i + 2]);
            const auto spitch = static_cast<std::size_t>(d[6 * i + 3]);
            const auto width = static_cast<std::size_t>(d[6 * i + 4]);
            const auto height = static_cast<std::size_t>(d[6 * i + 5]);
#if FREETOKEN_USE_ROCM
            const auto status = src == nullptr
                ? ::hipMemset2DAsync(dst, dpitch, 0, width, height, stream)
                : ::hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, ::hipMemcpyDefault, stream);
            if (status != ::hipSuccess) {
                ::host::panic(std::source_location::current(), "hipMemcpy2DAsync failed: ", ::hipGetErrorString(status));
            }
#else
            if (src == nullptr) {
                CUDA_CHECK(::cudaMemset2DAsync(dst, dpitch, 0, width, height, stream));
            } else {
                CUDA_CHECK(::cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, ::cudaMemcpyDefault, stream));
            }
#endif
        }
    }
};
