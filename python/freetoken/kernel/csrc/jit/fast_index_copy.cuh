#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>
#include <tvm/ffi/object.h>

namespace device {

template <typename T, std::size_t N>
struct device_vec {
    T data[N];
};

}  // namespace device

namespace details {

template <std::size_t kUnit>
inline constexpr auto get_mem_package() {
    if constexpr (kUnit == 16) {
    return uint4{};
    } else if constexpr (kUnit == 8) {
    return uint2{};
    } else if constexpr (kUnit == 4) {
    return uint1{};
    } else {
    static_assert(kUnit == 16 || kUnit == 8 || kUnit == 4, "Unsupported memory package size");
    }
}

// The L1::no_allocate hint keeps this streaming host gather from evicting L1, but the
// modifier is sm_70+ and Pascal's ptxas rejects it outright. It is a cache hint only, so
// pre-sm_70 issues the plain load and loses nothing but the anti-pollution optimisation.
// (st.global.wt below assembles fine on sm_6x, so the stores are left alone.)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#define FT_LD_STREAM "ld.global."
#else
#define FT_LD_STREAM "ld.global.L1::no_allocate."
#endif

__always_inline __device__ auto load_nc(const uint1* __restrict__ src) -> uint1 {
    uint32_t tmp;
    asm volatile(FT_LD_STREAM "b32 %0,[%1];" : "=r"(tmp) : "l"(src));
    return uint1{tmp};
}

__always_inline __device__ auto load_nc(const uint2* __restrict__ src) -> uint2 {
    uint32_t tmp0, tmp1;
    asm volatile(FT_LD_STREAM "v2.b32 {%0,%1},[%2];" : "=r"(tmp0), "=r"(tmp1) : "l"(src));
    return uint2{tmp0, tmp1};
}

__always_inline __device__ auto load_nc(const uint4* __restrict__ src) -> uint4 {
    uint32_t tmp0, tmp1, tmp2, tmp3;
    asm volatile(FT_LD_STREAM "v4.b32 {%0,%1,%2,%3},[%4];" : "=r"(tmp0), "=r"(tmp1), "=r"(tmp2), "=r"(tmp3) : "l"(src));
    return uint4{tmp0, tmp1, tmp2, tmp3};
}

__always_inline __device__ void store_nc(uint1* __restrict__ dst, const uint1& value) {
    uint32_t tmp = value.x;
    asm volatile("st.global.wt.b32 [%0],%1;" ::"l"(dst), "r"(tmp));
}

__always_inline __device__ void store_nc(uint2* __restrict__ dst, const uint2& value) {
    uint32_t tmp0 = value.x;
    uint32_t tmp1 = value.y;
    asm volatile("st.global.wt.v2.b32 [%0],{%1,%2};" ::"l"(dst), "r"(tmp0), "r"(tmp1));
}

__always_inline __device__ void store_nc(uint4* __restrict__ dst, const uint4& value) {
    uint32_t tmp0 = value.x;
    uint32_t tmp1 = value.y;
    uint32_t tmp2 = value.z;
    uint32_t tmp3 = value.w;
    asm volatile("st.global.wt.v4.b32 [%0],{%1,%2,%3,%4};" ::"l"(dst), "r"(tmp0), "r"(tmp1), "r"(tmp2), "r"(tmp3));
}

__always_inline __device__ void wait_flag_clear(const int32_t* __restrict__ flag_ptr) {
    // Exponential backoff to avoid hammering a global atomic in a tight loop.
    auto* flag = reinterpret_cast<int*>(const_cast<int32_t*>(flag_ptr));
    uint32_t sleep_ns = 128;
    while (atomicAdd(flag, 0) > 0) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(sleep_ns);
#endif
        sleep_ns = sleep_ns < 2048 ? (sleep_ns << 1) : 2048;
    }
}

template <std::size_t kUnit>
using mem_package_t = decltype(get_mem_package<kUnit>());

template <std::size_t kBytes, std::size_t kUnit, std::size_t kThreads>
__always_inline __device__ auto load_vec(const void* __restrict__ src) {
    using Package = mem_package_t<kUnit>;
    constexpr auto kBytesPerLoop = sizeof(Package) * kThreads;
    constexpr auto kLoopCount = kBytes / kBytesPerLoop;
    static_assert(kBytes % kBytesPerLoop == 0, "kBytes must be multiple of 128 bytes");

    const auto src_packed = static_cast<const Package*>(src);
    const auto lane_id = threadIdx.x % kThreads;
    device::device_vec<Package, kLoopCount> vec;

#pragma unroll kLoopCount
    for (std::size_t i = 0; i < kLoopCount; ++i) {
        const auto j = i * kThreads + lane_id;
        vec.data[i] = load_nc(src_packed + j);
    }

    return vec;
}

template <std::size_t kBytes, std::size_t kUnit, std::size_t kThreads, typename Tp>
__always_inline __device__ void store_vec(void* __restrict__ dst, const Tp& vec) {
    using Package = mem_package_t<kUnit>;
    constexpr auto kBytesPerLoop = sizeof(Package) * kThreads;
    constexpr auto kLoopCount = kBytes / kBytesPerLoop;
    static_assert(kBytes % kBytesPerLoop == 0, "kBytes must be multiple of 128 bytes");
    static_assert(std::is_same_v<Tp, device::device_vec<Package, kLoopCount>>);

    const auto dst_packed = static_cast<Package*>(dst);
    const auto lane_id = threadIdx.x % kThreads;

#pragma unroll kLoopCount
    for (std::size_t i = 0; i < kLoopCount; ++i) {
        const auto j = i * kThreads + lane_id;
        details::store_nc(dst_packed + j, vec.data[i]);
    }
}

}



// Pinned host memory is GPU-dereferenceable at its host VA only where UVA identity
// holds (Linux). On Windows/WDDM, cudaHostRegister'd memory maps to a different device
// address, so host-resident tensors are translated here -- the one point their pointer
// enters kernel params. Cached once per process: FreeToken pins one CUDA device per
// process (set at engine launch).
inline bool host_ptr_identity() {
    static const bool identity = [] {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) {
            return false;  // fail closed: translate (and surface errors), don't assume identity
        }
        int uva = 0, reg = 0;
        cudaDeviceGetAttribute(&uva, cudaDevAttrUnifiedAddressing, device);
        cudaDeviceGetAttribute(&reg, cudaDevAttrCanUseHostPointerForRegisteredMem, device);
        return uva == 1 && reg == 1;
    }();
    return identity;
}

inline void* device_alias(void* ptr, DLDevice dev) {
    if (dev.device_type == kDLCUDA || host_ptr_identity()) {
        return ptr;
    }
    void* mapped = nullptr;
    const auto err = cudaHostGetDevicePointer(&mapped, ptr, 0);
    host::RuntimeCheck(err == cudaSuccess,
        "fast_index_copy: host tensor must be pinned+mapped (cudaHostGetDevicePointer: ",
        cudaGetErrorString(err), ")");
    return mapped;
}

struct IndexKernelParams {
    void* __restrict__ dst;
    const void* __restrict__ indices_dst;
    void* __restrict__ src;
    const void* __restrict__ indices_src;
    std::size_t length;
    const int64_t* __restrict__ valid_length;
};

/*
Each worker has `kWorkerThreads` threads and handles `kWorkersFeatures` features for one index.
For one index that num_feat is large, we split it and use multiple workers to handle it in parallel.

cost of pre index read = [read index] + [copy kWorkersFeatures data]

For num_feat be small (<=1024) while length is large, set kWorkerThreads = 8:
    - so a warp (32 threads) can handle 4 indices in parallel.
    - cost one step to read 4 indices (reduce index read cost)

For num_feat be large (>2048), while length is small, set kWorkerThreads = 32, kWorkersFeatures=1024:
    - so enough threads parrallel to copy data for one index.

In one worker:
unroll `kUnrollCount` times to copy data.
each thread will use sizeof(Dtype) * kUnrollCount Bytes in one iteration.

launch (kNumBlocks, kNumThreads)

*/

template <
    typename IdType,
    std::size_t kFeatureBytes,
    std::size_t kWorkerThreads,
    std::size_t kWorkersFeatures,
    std::size_t kNumThreads, // should equal to blockDim.x
    std::size_t kNumBlocks,  // should equal to gridDim.x
    std::size_t kMaxOccupancy,
    bool kWaitOnFlag
>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void fast_index_copy(
    IndexKernelParams params,
    int32_t* sync_flag_ptr
) {
    using namespace device;
    static_assert(kNumThreads % kWorkerThreads == 0);
    constexpr auto kWorkersPerBlock = kNumThreads / kWorkerThreads;
    constexpr auto kWorkers = kWorkersPerBlock * kNumBlocks;
    
    const auto& [
        dst_ptr, indices_dst, src_ptr, indices_src,
        length, valid_length_ptr
    ] = params;

    const auto length_limit = valid_length_ptr ? static_cast<std::size_t>(valid_length_ptr[0]) : length;
    
    static_assert(kFeatureBytes % kWorkersFeatures == 0, "kFeatureBytes must be multiple of kWorkersFeatures");
    const auto kWorkersPerIndex = ((kFeatureBytes + kWorkersFeatures - 1) / kWorkersFeatures); // TODO: support not divisible
    
    const auto worker_id = blockIdx.x * kWorkersPerBlock + threadIdx.x / kWorkerThreads;

    constexpr auto kGranularity = 128 / kWorkerThreads;


    const auto total_work_items = length_limit * kWorkersPerIndex;
    const auto max_loops = (total_work_items + kWorkers - 1) / kWorkers;

    // Keep loop trip count identical across workers so future synchronization points
    // (e.g. block/warp polling logic) can be inserted safely.
    for (std::size_t loop = 0; loop < max_loops; ++loop) {
        if constexpr (kWaitOnFlag) {
            // One thread per block polls the shared flag and gates this loop.
            // All threads must execute this barrier the same number of times.
            if (threadIdx.x == 0) {
                details::wait_flag_clear(sync_flag_ptr);
            }
            __syncthreads();
        }

        const auto i = worker_id + loop * kWorkers;
        if (i >= total_work_items) {
            continue;
        }

        const auto index_id = i / kWorkersPerIndex;
        const auto index_subid = i % kWorkersPerIndex;
        const auto pos_src = static_cast<const IdType*>(indices_src)[index_id];
        const auto pos_dst = static_cast<const IdType*>(indices_dst)[index_id];

        const auto col = index_subid * kWorkersFeatures;
        const auto src_base = pointer::offset(src_ptr, pos_src * kFeatureBytes + col);
        const auto dst_base = pointer::offset(dst_ptr, pos_dst * kFeatureBytes + col);

        const auto vec = details::load_vec<kWorkersFeatures, kGranularity, kWorkerThreads>(src_base);
        details::store_vec<kWorkersFeatures, kGranularity, kWorkerThreads>(dst_base, vec);
    }
}

__global__ void update_copy_flag_kernel(int32_t* flag_ptr, int delta) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd(reinterpret_cast<int*>(flag_ptr), delta);
    }
}

inline auto get_sync_flag_ptr(
    tvm::ffi::TensorView sync_flag,
    host::SymbolicDevice& device
) -> int32_t* {
    auto flag_dtype = host::SymbolicDType{};
    host::TensorMatcher({1})
        .with_dtype<int32_t>(flag_dtype)
        .with_device<kDLCUDA>(device)
        .verify(sync_flag);
    return static_cast<int32_t*>(sync_flag.data_ptr());
}

/// Manually update the sync flag by delta. Use for signaling normal-priority
/// workers to pause (delta > 0) or resume (delta < 0).
inline void update_copy_flag(tvm::ffi::TensorView sync_flag, int32_t delta) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto* flag_ptr = get_sync_flag_ptr(sync_flag, device);
    LaunchKernel(1, 1, device.unwrap())(::update_copy_flag_kernel, flag_ptr, delta);
}


template <
    std::size_t kFeatureBytes,
    std::size_t kWorkerThreads,
    std::size_t kWorkersFeatures,
    std::size_t kNumThreads, // should equal to blockDim.x
    std::size_t kNumBlocks,  // should equal to gridDim.x
    std::size_t kMaxOccupancy
>
struct FastIndexCopyKernel {

    template <typename IdType>
    static constexpr auto _kernel_nowait = fast_index_copy<
        IdType,
        kFeatureBytes,
        kWorkerThreads,
        kWorkersFeatures,
        kNumThreads,
        kNumBlocks,
        kMaxOccupancy,
        false
    >;

    template <typename IdType>
    static constexpr auto _kernel_wait = fast_index_copy<
        IdType,
        kFeatureBytes,
        kWorkerThreads,
        kWorkersFeatures,
        kNumThreads,
        kNumBlocks,
        kMaxOccupancy,
        true
    >;

    enum class PriorityMode {
        kDefault,
        kHigh,
        kNormal
    };

    static void run_impl(
        tvm::ffi::TensorView dst,
        tvm::ffi::TensorView dst_indices,
        tvm::ffi::TensorView src,
        tvm::ffi::TensorView src_indices,
        tvm::ffi::Optional<tvm::ffi::TensorView> num_indices,
        tvm::ffi::Optional<tvm::ffi::TensorView> sync_flag,
        PriorityMode mode
    ) {
        using namespace host;
        auto D = SymbolicSize{"feature dimension"};
        auto L = SymbolicSize{"indices length"};

        auto data_dtype = SymbolicDType{};
        auto indices_dtype = SymbolicDType{};
        auto device = SymbolicDevice{};
        auto num_indices_dtype = SymbolicDType{};

        TensorMatcher({-1, D})
        .with_dtype(data_dtype)
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .verify(src);

        TensorMatcher({-1, D})
        .with_dtype(data_dtype)
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .verify(dst);

        TensorMatcher({L})
        .with_dtype<int32_t, int64_t>(indices_dtype)
        .with_device<kDLCUDA>(device)
        .verify(src_indices)
        .verify(dst_indices);

        const int64_t* num_indices_data_ptr = nullptr;
        if (num_indices.has_value()) {
            const auto num_indices_tensor = num_indices.value();
            TensorMatcher({1})
                .with_dtype<int64_t>(num_indices_dtype)
                .with_device<kDLCUDA>(device)
                .verify(num_indices_tensor);

            num_indices_data_ptr = static_cast<const int64_t*>(num_indices_tensor.data_ptr());
        }

        // verify dimension match
        const auto dtype_size = dtype_bytes(data_dtype.unwrap());
        const auto element_bytes = D.unwrap() * dtype_size;
        RuntimeCheck(kFeatureBytes == element_bytes, "HicacheKernel: cache dimension mismatch.");

        const auto dst_ptr = device_alias(dst.data_ptr(), dst.device());
        const auto src_ptr = device_alias(src.data_ptr(), src.device());
        const auto dst_indices_ptr = dst_indices.data_ptr();
        const auto src_indices_ptr = src_indices.data_ptr();
        const auto length = static_cast<std::size_t>(L.unwrap());
        const auto use_int32 = indices_dtype.unwrap().bits == 32;
        const auto _device = device.unwrap();

        const auto params = IndexKernelParams{
            dst_ptr,
            dst_indices_ptr,
            src_ptr,
            src_indices_ptr,
            length,
            num_indices_data_ptr
        };

        int32_t* sync_flag_ptr = nullptr;
        if (mode == PriorityMode::kHigh || mode == PriorityMode::kNormal) {
            RuntimeCheck(sync_flag.has_value(), "sync_flag is required for high/normal priority modes.");
            sync_flag_ptr = get_sync_flag_ptr(sync_flag.value(), device);
        }

        if (mode == PriorityMode::kHigh) {
            LaunchKernel(1, 1, _device)(update_copy_flag_kernel, sync_flag_ptr, 1);
            try {
                const auto kernel = use_int32 ? _kernel_nowait<int32_t> : _kernel_nowait<int64_t>;
                LaunchKernel(kNumBlocks, kNumThreads, _device)(kernel, params, static_cast<int32_t*>(nullptr));
            } catch (...) {
                LaunchKernel(1, 1, _device)(update_copy_flag_kernel, sync_flag_ptr, -1);
                throw;
            }
            LaunchKernel(1, 1, _device)(update_copy_flag_kernel, sync_flag_ptr, -1);
            return;
        }

        if (mode == PriorityMode::kNormal) {
            const auto kernel = use_int32 ? _kernel_wait<int32_t> : _kernel_wait<int64_t>;
            LaunchKernel(kNumBlocks, kNumThreads, _device)(kernel, params, sync_flag_ptr);
            return;
        }

        const auto kernel = use_int32 ? _kernel_nowait<int32_t> : _kernel_nowait<int64_t>;
        LaunchKernel(kNumBlocks, kNumThreads, _device)(kernel, params, static_cast<int32_t*>(nullptr));
    }


    static void run(
        tvm::ffi::TensorView dst,
        tvm::ffi::TensorView dst_indices,
        tvm::ffi::TensorView src,
        tvm::ffi::TensorView src_indices,
        tvm::ffi::Optional<tvm::ffi::TensorView> num_indices
    ) {
        run_impl(
            dst,
            dst_indices,
            src,
            src_indices,
            num_indices,
            tvm::ffi::Optional<tvm::ffi::TensorView>{},
            PriorityMode::kDefault
        );
    }

    static void run_high(
        tvm::ffi::TensorView dst,
        tvm::ffi::TensorView dst_indices,
        tvm::ffi::TensorView src,
        tvm::ffi::TensorView src_indices,
        tvm::ffi::Optional<tvm::ffi::TensorView> num_indices,
        tvm::ffi::TensorView sync_flag
    ) {
        run_impl(dst, dst_indices, src, src_indices, num_indices, sync_flag, PriorityMode::kHigh);
    }

    static void run_normal(
        tvm::ffi::TensorView dst,
        tvm::ffi::TensorView dst_indices,
        tvm::ffi::TensorView src,
        tvm::ffi::TensorView src_indices,
        tvm::ffi::Optional<tvm::ffi::TensorView> num_indices,
        tvm::ffi::TensorView sync_flag
    ) {
        run_impl(dst, dst_indices, src, src_indices, num_indices, sync_flag, PriorityMode::kNormal);
    }
};


// ---------------------------------------------------------------------------
// Multi-bank fused index copy. The offload cache copies the SAME rows
// (dst_indices=evict_slots <- src_indices) for every registered bank, but the
// banks have distinct per-row feature byte sizes, so the single-bank kernel above
// needs one launch per bank (e.g. 6 banks * 36 layers = 216 launches/decode step,
// all near-empty at a warm/full cache). This fuses every bank into one launch:
// block b copies bank `b = blockIdx.x / kBlocksPerBank`, grid-striding over that
// bank's (num_indices * feat/16) 16-byte units. Pointers + feature sizes are passed
// as small device arrays (built once by the cache), so it stays CUDA-graph capturable.
struct MultiIndexCopyParams {
    const int64_t* __restrict__ dst_ptrs;     // [B] device, each base addr of a bank slot cache
    const int64_t* __restrict__ src_ptrs;     // [B] device, each GPU-visible base addr of a bank host source
    const int64_t* __restrict__ feat_bytes;   // [B] device, per-row bytes (multiple of 16)
    const void* __restrict__ dst_indices;     // [L]
    const void* __restrict__ src_indices;     // [L]
    const int64_t* __restrict__ valid_length; // [1] or null
    int64_t length;                           // max L
    int num_banks;
};

template <typename IdType, std::size_t kNumThreads, std::size_t kBlocksPerBank>
__global__ __launch_bounds__(kNumThreads) void fast_index_copy_multi(
    const FT_GRID_CONSTANT MultiIndexCopyParams p
) {
    const int b = static_cast<int>(blockIdx.x / kBlocksPerBank);
    if (b >= p.num_banks) {
        return;
    }
    const int blk = static_cast<int>(blockIdx.x % kBlocksPerBank);
    const auto* src = reinterpret_cast<const uint8_t*>(p.src_ptrs[b]);
    auto* dst = reinterpret_cast<uint8_t*>(p.dst_ptrs[b]);
    const int64_t feat = p.feat_bytes[b];
    const int64_t n = p.valid_length ? p.valid_length[0] : p.length;
    const int64_t units = feat >> 4;  // 16-byte (uint4) units per row; feat % 16 == 0
    const int64_t total = n * units;
    const auto* di = static_cast<const IdType*>(p.dst_indices);
    const auto* si = static_cast<const IdType*>(p.src_indices);
    const int64_t stride = static_cast<int64_t>(kBlocksPerBank) * kNumThreads;
    for (int64_t u = static_cast<int64_t>(blk) * kNumThreads + threadIdx.x; u < total; u += stride) {
        const int64_t row = u / units;
        const int64_t col = (u - row * units) << 4;  // byte offset within the row
        const int64_t pd = static_cast<int64_t>(di[row]);
        const int64_t ps = static_cast<int64_t>(si[row]);
        const uint4 v = *reinterpret_cast<const uint4*>(src + ps * feat + col);
        *reinterpret_cast<uint4*>(dst + pd * feat + col) = v;
    }
}

template <std::size_t kNumThreads, std::size_t kBlocksPerBank>
struct MultiIndexCopyKernel {
    static void run(
        tvm::ffi::TensorView dst_ptrs,
        tvm::ffi::TensorView src_ptrs,
        tvm::ffi::TensorView feat_bytes,
        tvm::ffi::TensorView dst_indices,
        tvm::ffi::TensorView src_indices,
        tvm::ffi::Optional<tvm::ffi::TensorView> num_indices
    ) {
        using namespace host;
        auto device = SymbolicDevice{};
        auto B = SymbolicSize{"num_banks"};
        auto L = SymbolicSize{"indices length"};
        auto ptr_dtype = SymbolicDType{};
        auto indices_dtype = SymbolicDType{};
        auto num_indices_dtype = SymbolicDType{};

        TensorMatcher({B}).with_dtype<int64_t>(ptr_dtype).with_device<kDLCUDA>(device)
            .verify(dst_ptrs).verify(src_ptrs).verify(feat_bytes);
        TensorMatcher({L}).with_dtype<int32_t, int64_t>(indices_dtype).with_device<kDLCUDA>(device)
            .verify(dst_indices).verify(src_indices);

        const int64_t* valid_length = nullptr;
        if (num_indices.has_value()) {
            TensorMatcher({1}).with_dtype<int64_t>(num_indices_dtype).with_device<kDLCUDA>(device)
                .verify(num_indices.value());
            valid_length = static_cast<const int64_t*>(num_indices.value().data_ptr());
        }

        const int num_banks = static_cast<int>(B.unwrap());
        const auto params = MultiIndexCopyParams{
            static_cast<const int64_t*>(dst_ptrs.data_ptr()),
            static_cast<const int64_t*>(src_ptrs.data_ptr()),
            static_cast<const int64_t*>(feat_bytes.data_ptr()),
            dst_indices.data_ptr(),
            src_indices.data_ptr(),
            valid_length,
            static_cast<int64_t>(L.unwrap()),
            num_banks,
        };
        const auto use_int32 = indices_dtype.unwrap().bits == 32;
        const auto kernel = use_int32
            ? fast_index_copy_multi<int32_t, kNumThreads, kBlocksPerBank>
            : fast_index_copy_multi<int64_t, kNumThreads, kBlocksPerBank>;
        LaunchKernel(static_cast<std::size_t>(kBlocksPerBank) * num_banks, kNumThreads,
                     device.unwrap())(kernel, params);
    }
};
