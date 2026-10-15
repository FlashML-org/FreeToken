// The ggml-base and ggml-cuda.cu host symbols that the vendored ggml-cuda kernels reference.
// Tensor descriptors built by ggml_ops.cu never have a backend buffer, and temporary buffers come from tvm-ffi's env allocator (torch's caching allocator under torch).

#include "common.cuh"

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <cstdarg>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

struct TypeInfo {
    const char * name = nullptr;
    int64_t blck_size = 0;
    size_t type_size = 0;
    bool quantized = false;
};

const TypeInfo * type_infos() {
    static TypeInfo t[GGML_TYPE_COUNT];
    static const bool init = [] {
        auto set = [](ggml_type type, const char * name, int64_t blck, size_t size, bool quantized) {
            t[type] = {name, blck, size, quantized};
        };
        set(GGML_TYPE_F32, "f32", 1, sizeof(float), false);
        set(GGML_TYPE_F16, "f16", 1, sizeof(ggml_fp16_t), false);
        set(GGML_TYPE_BF16, "bf16", 1, sizeof(ggml_bf16_t), false);
        set(GGML_TYPE_F64, "f64", 1, sizeof(double), false);
        set(GGML_TYPE_I8, "i8", 1, sizeof(int8_t), false);
        set(GGML_TYPE_I16, "i16", 1, sizeof(int16_t), false);
        set(GGML_TYPE_I32, "i32", 1, sizeof(int32_t), false);
        set(GGML_TYPE_I64, "i64", 1, sizeof(int64_t), false);
        set(GGML_TYPE_Q1_0, "q1_0", QK1_0, sizeof(block_q1_0), true);
        set(GGML_TYPE_Q2_0, "q2_0", QK2_0, sizeof(block_q2_0), true);
        set(GGML_TYPE_Q4_0, "q4_0", QK4_0, sizeof(block_q4_0), true);
        set(GGML_TYPE_Q4_1, "q4_1", QK4_1, sizeof(block_q4_1), true);
        set(GGML_TYPE_Q5_0, "q5_0", QK5_0, sizeof(block_q5_0), true);
        set(GGML_TYPE_Q5_1, "q5_1", QK5_1, sizeof(block_q5_1), true);
        set(GGML_TYPE_Q8_0, "q8_0", QK8_0, sizeof(block_q8_0), true);
        set(GGML_TYPE_Q8_1, "q8_1", QK8_1, sizeof(block_q8_1), true);
        set(GGML_TYPE_MXFP4, "mxfp4", QK_MXFP4, sizeof(block_mxfp4), true);
        set(GGML_TYPE_NVFP4, "nvfp4", QK_NVFP4, sizeof(block_nvfp4), true);
        set(GGML_TYPE_TQ1_0, "tq1_0", QK_K, sizeof(block_tq1_0), true);
        set(GGML_TYPE_TQ2_0, "tq2_0", QK_K, sizeof(block_tq2_0), true);
        set(GGML_TYPE_Q2_K, "q2_K", QK_K, sizeof(block_q2_K), true);
        set(GGML_TYPE_Q3_K, "q3_K", QK_K, sizeof(block_q3_K), true);
        set(GGML_TYPE_Q4_K, "q4_K", QK_K, sizeof(block_q4_K), true);
        set(GGML_TYPE_Q5_K, "q5_K", QK_K, sizeof(block_q5_K), true);
        set(GGML_TYPE_Q6_K, "q6_K", QK_K, sizeof(block_q6_K), true);
        set(GGML_TYPE_Q8_K, "q8_K", QK_K, sizeof(block_q8_K), true);
        set(GGML_TYPE_IQ2_XXS, "iq2_xxs", QK_K, sizeof(block_iq2_xxs), true);
        set(GGML_TYPE_IQ2_XS, "iq2_xs", QK_K, sizeof(block_iq2_xs), true);
        set(GGML_TYPE_IQ2_S, "iq2_s", QK_K, sizeof(block_iq2_s), true);
        set(GGML_TYPE_IQ3_XXS, "iq3_xxs", QK_K, sizeof(block_iq3_xxs), true);
        set(GGML_TYPE_IQ3_S, "iq3_s", QK_K, sizeof(block_iq3_s), true);
        set(GGML_TYPE_IQ1_S, "iq1_s", QK_K, sizeof(block_iq1_s), true);
        set(GGML_TYPE_IQ1_M, "iq1_m", QK_K, sizeof(block_iq1_m), true);
        set(GGML_TYPE_IQ4_NL, "iq4_nl", QK4_NL, sizeof(block_iq4_nl), true);
        set(GGML_TYPE_IQ4_XS, "iq4_xs", QK_K, sizeof(block_iq4_xs), true);
        return true;
    }();
    (void) init;
    return t;
}

const TypeInfo & type_info(ggml_type type) {
    if (type < 0 || type >= GGML_TYPE_COUNT || type_infos()[type].name == nullptr) {
        throw std::invalid_argument("unknown ggml type " + std::to_string((int) type));
    }
    return type_infos()[type];
}

struct EnvPool : ggml_cuda_pool {
    explicit EnvPool(int device) : device(device) {}

    void * alloc(size_t size, size_t * actual_size) override {
        *actual_size = size;
        if (size == 0) {
            return nullptr;
        }
#ifdef GGML_USE_HIP
        constexpr DLDeviceType device_type = kDLROCM;
#else
        constexpr DLDeviceType device_type = kDLCUDA;
#endif
        auto buf = tvm::ffi::Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, tvm::ffi::Shape({(int64_t) size}), DLDataType{kDLUInt, 8, 1},
                                                DLDevice{device_type, device});
        void * ptr = buf.data_ptr();
        live.emplace(ptr, std::move(buf));
        return ptr;
    }

    void free(void * ptr, size_t) override { live.erase(ptr); }

    int device;
    std::unordered_map<void *, tvm::ffi::Tensor> live;
};

}  // namespace

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    char msg[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);
    throw std::runtime_error(std::string("ggml: ") + file + ":" + std::to_string(line) + ": " + msg);
}

void ggml_log_internal(enum ggml_log_level level, const char * format, ...) {
    if (level < GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_CONT) {
        return;
    }
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
}

int64_t ggml_blck_size(enum ggml_type type) { return type_info(type).blck_size; }

size_t ggml_type_size(enum ggml_type type) { return type_info(type).type_size; }

const char * ggml_type_name(enum ggml_type type) { return type_info(type).name; }

bool ggml_is_quantized(enum ggml_type type) { return type_info(type).quantized; }

size_t ggml_row_size(enum ggml_type type, int64_t ne) { return ggml_type_size(type) * ne / ggml_blck_size(type); }

size_t ggml_element_size(const struct ggml_tensor * tensor) { return ggml_type_size(tensor->type); }

int64_t ggml_nelements(const struct ggml_tensor * tensor) {
    return tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

size_t ggml_nbytes(const struct ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0) {
            return 0;
        }
    }
    size_t nbytes;
    const size_t blck_size = ggml_blck_size(tensor->type);
    if (blck_size == 1) {
        nbytes = ggml_type_size(tensor->type);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1) * tensor->nb[i];
        }
    } else {
        nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1) * tensor->nb[i];
        }
    }
    return nbytes;
}

bool ggml_is_matrix(const struct ggml_tensor * tensor) { return tensor->ne[2] == 1 && tensor->ne[3] == 1; }

bool ggml_is_contiguous(const struct ggml_tensor * tensor) {
    size_t next_nb = ggml_type_size(tensor->type);
    if (tensor->ne[0] != ggml_blck_size(tensor->type) && tensor->nb[0] != next_nb) {
        return false;
    }
    next_nb *= tensor->ne[0] / ggml_blck_size(tensor->type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        if (tensor->ne[i] != 1 && tensor->nb[i] != next_nb) {
            return false;
        }
        next_nb *= tensor->ne[i];
    }
    return true;
}

bool ggml_is_contiguously_allocated(const struct ggml_tensor * tensor) {
    return ggml_nbytes(tensor) == ggml_nelements(tensor) * ggml_type_size(tensor->type) / ggml_blck_size(tensor->type);
}

bool ggml_are_same_shape(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
    return t0->ne[0] == t1->ne[0] && t0->ne[1] == t1->ne[1] && t0->ne[2] == t1->ne[2] && t0->ne[3] == t1->ne[3];
}

bool ggml_are_same_stride(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
    return t0->nb[0] == t1->nb[0] && t0->nb[1] == t1->nb[1] && t0->nb[2] == t1->nb[2] && t0->nb[3] == t1->nb[3];
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t) {
    return GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t, const struct ggml_tensor * tensor) {
    return ggml_nbytes(tensor);
}

[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    throw std::runtime_error(std::string("ggml-cuda: ") + msg + " in " + func + " at " + file + ":" + std::to_string(line) + ": " + stmt);
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

void ggml_cuda_set_device(int device) {
    int current;
    CUDA_CHECK(cudaGetDevice(&current));
    if (current != device) {
        CUDA_CHECK(cudaSetDevice(device));
    }
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static const ggml_cuda_device_info info = [] {
        ggml_cuda_device_info in = {};
        CUDA_CHECK(cudaGetDeviceCount(&in.physical_device_count));
        GGML_ASSERT(in.physical_device_count <= GGML_CUDA_MAX_DEVICES);
        in.device_count = in.physical_device_count;
        for (int id = 0; id < in.device_count; ++id) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
            auto & d = in.devices[id];
#ifdef GGML_USE_HIP
            GGML_ASSERT(std::string(prop.gcnArchName).starts_with("gfx"));
            d.cc = GGML_CUDA_CC_OFFSET_AMD + static_cast<int>(std::strtol(prop.gcnArchName + 3, nullptr, 16));
#else
            d.cc = 100 * prop.major + 10 * prop.minor;
#endif
            d.nsm = prop.multiProcessorCount;
            d.smpb = prop.sharedMemPerBlock;
            d.smpbo = prop.sharedMemPerBlockOptin;
            d.warp_size = prop.warpSize;
            d.total_vram = prop.totalGlobalMem;
            d.physical_device = id;
            d.physical_share_count = 1;
            int coop = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, id));
            d.supports_cooperative_launch = coop != 0;
        }
        return in;
    }();
    return info;
}

// The streams belong to the caller, so there is nothing to release.
ggml_backend_cuda_context::~ggml_backend_cuda_context() {}

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int device, int) {
    return std::make_unique<EnvPool>(device);
}
