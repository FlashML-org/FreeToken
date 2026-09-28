#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdio>
#include <cstddef>
#include <cstdint>

// Host wrapper over cudaMemcpyBatchAsync (CUDA >= 13.0, the 8-argument signature;
// 12.8/12.9 carried an extra failIdx parameter): enqueue N independent
// pointer-to-pointer copies with ONE runtime call, on an explicit (non-legacy)
// stream. Callers hand pre-resolved raw addresses; copies within a batch are
// unordered, so entries must be pairwise independent.
struct BatchMemcpy {
    static void run(
        tvm::ffi::TensorView dst_ptrs,
        tvm::ffi::TensorView src_ptrs,
        tvm::ffi::TensorView sizes,
        int64_t stream_handle
    ) {
#if CUDART_VERSION >= 13000
        using namespace host;
        auto N = SymbolicSize{"batch length"};
        auto ptr_dtype = SymbolicDType{};
        TensorMatcher({N})
            .with_dtype<int64_t>(ptr_dtype)
            .with_device<kDLCPU>()
            .verify(dst_ptrs)
            .verify(src_ptrs)
            .verify(sizes);
        const auto n = static_cast<std::size_t>(N.unwrap());
        if (n == 0) {
            return;
        }
        RuntimeCheck(stream_handle != 0, "cudaMemcpyBatchAsync rejects the legacy NULL stream");
        auto attr = ::cudaMemcpyAttributes{};
        attr.srcAccessOrder = ::cudaMemcpySrcAccessOrderStream;
        std::size_t attr_idx = 0;
        const auto rc = ::cudaMemcpyBatchAsync(
            reinterpret_cast<void* const*>(dst_ptrs.data_ptr()),
            reinterpret_cast<const void* const*>(src_ptrs.data_ptr()),
            reinterpret_cast<const std::size_t*>(sizes.data_ptr()),
            n,
            &attr,
            &attr_idx,
            1,
            reinterpret_cast<::cudaStream_t>(stream_handle)
        );
        if (rc != ::cudaSuccess) {
            // 驱动拒绝时打印全部描述符摘要：条目数与各数组首尾值，逐条目挑出
            // NULL/零长——真实服务上该错误与动态钉住相关且难复现，现场摘要胜过盲猜。
            const auto* d = reinterpret_cast<void* const*>(dst_ptrs.data_ptr());
            const auto* s = reinterpret_cast<const void* const*>(src_ptrs.data_ptr());
            const auto* z = reinterpret_cast<const std::size_t*>(sizes.data_ptr());
            std::fprintf(stderr,
                "cudaMemcpyBatchAsync failed (%s): N=%zu stream=%p\n"
                "  dst[0]=%p dst[N-1]=%p  src[0]=%p src[N-1]=%p  size[0]=%zu size[N-1]=%zu\n",
                ::cudaGetErrorString(rc), n, reinterpret_cast<void*>(stream_handle),
                d[0], d[n - 1], s[0], s[n - 1], z[0], z[n - 1]);
            for (std::size_t i = 0; i < n; ++i) {
                if (d[i] == nullptr || s[i] == nullptr || z[i] == 0) {
                    std::fprintf(stderr, "  BAD entry %zu: dst=%p src=%p size=%zu\n",
                                 i, d[i], s[i], z[i]);
                }
            }
            std::fflush(stderr);
        }
        CUDA_CHECK(rc);
#else
        ::host::panic(
            std::source_location::current(),
            "this cudaMemcpyBatchAsync binding requires CUDA >= 13.0 at build time"
        );
#endif
    }
};
