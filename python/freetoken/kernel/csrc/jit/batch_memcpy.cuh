#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

#if defined(USE_HIP)
// HIP has no cudaMemcpyBatchAsync equivalent. Implement a per-copy grid kernel: one
// block per copy, each block cooperatively copying its (src, dst, nbytes) triple.
// The host pointer arrays are staged to device memory for the launch. Copies within
// a batch are unordered (as with cudaMemcpyBatchAsync), so blocks may run in any order.
__global__ void batch_memcpy_kernel(const void* const* srcs, void* const* dsts,
                                    const std::size_t* sizes, std::size_t n) {
    const std::size_t i = blockIdx.x;
    if (i >= n) {
        return;
    }
    const char* s = static_cast<const char*>(srcs[i]);
    char* d = static_cast<char*>(dsts[i]);
    const std::size_t sz = sizes[i];
    for (std::size_t j = threadIdx.x; j < sz; j += blockDim.x) {
        d[j] = s[j];
    }
}
#endif

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
#if defined(USE_HIP)
        using namespace host;
        auto N = SymbolicSize{"batch length"};
        auto ptr_dtype = SymbolicDType{};
        TensorMatcher({N})
            .with_dtype<int64_t>(ptr_dtype)
            .with_device<kDLCPU>()
            .verify(dst_ptrs)
            .verify(src_ptrs)
            .verify(sizes);
        auto n = static_cast<std::size_t>(N.unwrap());
        if (n == 0) {
            return;
        }
        RuntimeCheck(stream_handle != 0, "batch_memcpy rejects the legacy NULL stream");
        auto stream = reinterpret_cast<hipStream_t>(stream_handle);

        const void* const* srcs = reinterpret_cast<const void* const*>(src_ptrs.data_ptr());
        void* const* dsts = reinterpret_cast<void* const*>(dst_ptrs.data_ptr());
        const std::size_t* sizes_arr = reinterpret_cast<const std::size_t*>(sizes.data_ptr());

        // Stage the host pointer arrays to device memory for the kernel. Use
        // stream-ordered allocation so the free is ordered after the (async) kernel
        // launch -- a plain hipFree here could free memory the kernel is still reading.
        void* d_srcs = nullptr;
        void* d_dsts = nullptr;
        void* d_sizes = nullptr;
        HIP_CHECK(hipMallocAsync(&d_srcs, n * sizeof(void*), stream));
        HIP_CHECK(hipMallocAsync(&d_dsts, n * sizeof(void*), stream));
        HIP_CHECK(hipMallocAsync(&d_sizes, n * sizeof(std::size_t), stream));
        HIP_CHECK(hipMemcpy(d_srcs, srcs, n * sizeof(void*), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_dsts, dsts, n * sizeof(void*), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_sizes, sizes_arr, n * sizeof(std::size_t), hipMemcpyHostToDevice));

        void* args[4] = {&d_srcs, &d_dsts, &d_sizes, &n};
        HIP_CHECK(hipLaunchKernel(
            reinterpret_cast<const void*>(batch_memcpy_kernel), dim3(n), dim3(256), args, 0,
            stream));

        HIP_CHECK(hipFreeAsync(d_srcs, stream));
        HIP_CHECK(hipFreeAsync(d_dsts, stream));
        HIP_CHECK(hipFreeAsync(d_sizes, stream));
#elif CUDART_VERSION >= 13000
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
#else
        ::host::panic(
            std::source_location::current(),
            "this cudaMemcpyBatchAsync binding requires CUDA >= 13.0 at build time"
        );
#endif
    }
};
