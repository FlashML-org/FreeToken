#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

// Host wrapper over cudaMemcpyBatchAsync (CUDA >= 13.0, the 8-argument signature;
// 12.8/12.9 carried an extra failIdx parameter): enqueue N independent
// pointer-to-pointer copies with ONE runtime call, on an explicit (non-legacy)
// stream. Callers hand pre-resolved raw addresses; copies within a batch are
// unordered, so entries must be pairwise independent.
// ROCm uses hipMemcpyBatchAsync with its nine-argument signature (failIdx).
struct BatchMemcpy {
    static void run(
        tvm::ffi::TensorView dst_ptrs,
        tvm::ffi::TensorView src_ptrs,
        tvm::ffi::TensorView sizes,
        int64_t stream_handle
    ) {
#if FREETOKEN_USE_ROCM || CUDART_VERSION >= 13000
        using namespace host;
        auto N = SymbolicSize{"batch length"};
        auto ptr_dtype = SymbolicDType{};
        TensorMatcher({N})
            .template with_dtype<int64_t>(ptr_dtype)
            .template with_device<kDLCPU>()
            .verify(dst_ptrs)
            .verify(src_ptrs)
            .verify(sizes);
        const auto n = static_cast<std::size_t>(N.unwrap());
        if (n == 0) {
            return;
        }
        RuntimeCheck(stream_handle != 0, "batch memcpy requires an explicit non-NULL stream");
#if FREETOKEN_USE_ROCM
        auto attr = ::hipMemcpyAttributes{};
        attr.srcAccessOrder = ::hipMemcpySrcAccessOrderStream;
        std::size_t attr_idx = 0;
        std::size_t fail_idx = SIZE_MAX;
        const auto status = ::hipMemcpyBatchAsync(
            reinterpret_cast<void**>(dst_ptrs.data_ptr()),
            reinterpret_cast<void**>(src_ptrs.data_ptr()),
            reinterpret_cast<std::size_t*>(sizes.data_ptr()),
            n,
            &attr,
            &attr_idx,
            1,
            &fail_idx,
            reinterpret_cast<::hipStream_t>(stream_handle)
        );
        if (status != ::hipSuccess) {
            ::host::panic(
                std::source_location::current(),
                "hipMemcpyBatchAsync failed (failIdx=", fail_idx, "): ", ::hipGetErrorString(status)
            );
        }
#else
        auto attr = ::cudaMemcpyAttributes{};
        attr.srcAccessOrder = ::cudaMemcpySrcAccessOrderStream;
        std::size_t attr_idx = 0;
        CUDA_CHECK(::cudaMemcpyBatchAsync(
            reinterpret_cast<void* const*>(dst_ptrs.data_ptr()),
            reinterpret_cast<const void* const*>(src_ptrs.data_ptr()),
            reinterpret_cast<const std::size_t*>(sizes.data_ptr()),
            n,
            &attr,
            &attr_idx,
            1,
            reinterpret_cast<::cudaStream_t>(stream_handle)
        ));
#endif
#else
        ::host::panic(
            std::source_location::current(),
            "this cudaMemcpyBatchAsync binding requires CUDA >= 13.0 at build time"
        );
#endif
    }
};
