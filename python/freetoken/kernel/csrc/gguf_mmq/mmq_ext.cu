// Upstream llama.cpp int8-tensor-core MMQ (mul_mat_q) for Q4_K/Q6_K.
//
// The sibling csrc/gguf tree carries llama.cpp b2899 DP4A kernels (via
// vLLM/sgl-kernel); upstream has since rewritten MMQ around int8 MMA tiles
// (turing_mma, sm_75+), which is ~13x faster at prefill row counts on sm_120
// and also beats transient dequant+cuBLAS at every batch size. Files in this
// directory other than this one are vendored VERBATIM from llama.cpp master
// eab8ee41f889ef7823af517e8098fb8a9b3cf601 (ggml/src/ggml-cuda + the ggml
// headers they include); this file supplies the small backend shims that
// ggml-cuda.cu would normally provide (device info, pool, error/abort) plus
// the torch bindings. Only Q4_K/Q6_K mul_mat_q cases are instantiated to keep
// JIT compile time down; the GEMV (batch<=6) and MoE paths stay on the
// existing csrc/gguf kernels.

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdarg>
#include <cstdio>

// ---------------------------------------------------------------------------
// ggml backend shims
// ---------------------------------------------------------------------------

extern "C" size_t ggml_type_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K: return sizeof(block_q4_K);
        case GGML_TYPE_Q6_K: return sizeof(block_q6_K);
        case GGML_TYPE_F32:  return sizeof(float);
        default: GGML_ABORT("ggml_type_size: unsupported type %d", (int) type);
    }
}

extern "C" int64_t ggml_blck_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K: return QK_K;
        case GGML_TYPE_Q6_K: return QK_K;
        case GGML_TYPE_F32:  return 1;
        default: GGML_ABORT("ggml_blck_size: unsupported type %d", (int) type);
    }
}

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    char msg[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);
    TORCH_CHECK(false, "ggml abort at ", file, ":", line, ": ", msg);
}

void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    TORCH_CHECK(false, "CUDA error in ", func, " at ", file, ":", line, ": ", stmt, ": ", msg);
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

void ggml_cuda_set_device(int device) {
    CUDA_CHECK(cudaSetDevice(device));
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = []() {
        ggml_cuda_device_info inf = {};
        CUDA_CHECK(cudaGetDeviceCount(&inf.device_count));
        inf.physical_device_count = inf.device_count;
        for (int id = 0; id < inf.device_count; ++id) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
            auto & dev = inf.devices[id];
            dev.cc = 100 * prop.major + 10 * prop.minor;
            dev.nsm = prop.multiProcessorCount;
            dev.smpb = prop.sharedMemPerBlock;
            dev.smpbo = prop.sharedMemPerBlockOptin;
            dev.integrated = prop.integrated;
            dev.warp_size = prop.warpSize;
            dev.total_vram = prop.totalGlobalMem;
            dev.physical_device = id;
            dev.physical_share_count = 1;
            dev.virtual_index = 0;
        }
        return inf;
    }();
    return info;
}

// Pool backed by torch's caching allocator so transient MMQ scratch (stream-k
// fixup buffers) shares the framework's memory accounting.
struct ggml_cuda_pool_torch : public ggml_cuda_pool {
    void * alloc(size_t size, size_t * actual_size) override {
        void * ptr = c10::cuda::CUDACachingAllocator::raw_alloc(size);
        *actual_size = size;
        return ptr;
    }
    void free(void * ptr, size_t /*size*/) override {
        c10::cuda::CUDACachingAllocator::raw_delete(ptr);
    }
};

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int /*device*/, int /*stream_no*/) {
    return std::make_unique<ggml_cuda_pool_torch>();
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() = default;

// ---------------------------------------------------------------------------
// MMQ instantiations (Q4_K / Q6_K only)
// ---------------------------------------------------------------------------

DECL_MMQ_CASE(GGML_TYPE_Q4_K);
DECL_MMQ_CASE(GGML_TYPE_Q6_K);

// ---------------------------------------------------------------------------
// Torch entry: y = x @ dequant(W).T, W packed ggml rows [nrows, row_bytes]
// ---------------------------------------------------------------------------

static torch::Tensor ggml_mul_mat_a8_mma(torch::Tensor W, torch::Tensor X, int64_t type, int64_t nrows) {
    TORCH_CHECK(W.is_cuda() && X.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(W.dtype() == torch::kUInt8 && W.dim() == 2 && W.is_contiguous());
    TORCH_CHECK(X.dim() == 2 && X.is_contiguous());
    TORCH_CHECK(type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K, "only Q4_K/Q6_K instantiated");
    TORCH_CHECK(W.size(0) == nrows);

    const ggml_type type_x = (ggml_type) type;
    const c10::cuda::CUDAGuard guard(W.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Upstream's activation quantize kernel reads fp32.
    torch::Tensor Xf = X.scalar_type() == torch::kFloat32 ? X : X.to(torch::kFloat32);

    const int64_t ne01 = nrows;      // weight rows (output features)
    const int64_t ne11 = X.size(0);  // tokens
    const int64_t ne10 = X.size(1);  // input features
    const size_t ts = ggml_type_size(type_x);
    const int64_t qk = ggml_blck_size(type_x);
    TORCH_CHECK((int64_t) W.size(1) * qk == ne10 * (int64_t) ts, "row_bytes mismatch");
    const int64_t s01 = W.size(1) / ts;  // row stride in blocks

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const bool fallback = ne01 % 128 != 0;

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t nbytes_src1_q8_1 = ne11 * ne10_padded * sizeof(block_q8_1_mmq) / QK8_1_MMQ +
        ggml_cuda_mmq_get_J_max(type_x, fallback, cc, ne11) * sizeof(block_q8_1_mmq);

    auto opts = torch::TensorOptions().device(W.device());
    torch::Tensor y_q = torch::empty({(int64_t) nbytes_src1_q8_1}, opts.dtype(torch::kUInt8));
    torch::Tensor dst = torch::empty({ne11, ne01}, opts.dtype(torch::kFloat32));

    quantize_mmq_q8_1_cuda(
        Xf.data_ptr<float>(), nullptr, y_q.data_ptr(), type_x,
        ne10, /*s01=*/ne10, /*s02=*/ne10 * ne11, /*s03=*/ne10 * ne11,
        ne10_padded, ne11, /*ne2=*/1, /*ne3=*/1, stream);
    CUDA_CHECK(cudaGetLastError());

    const int64_t s12 = ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const mmq_args args = {
        (const char *) W.data_ptr(), type_x, (const int *) y_q.data_ptr(), nullptr, nullptr,
        dst.data_ptr<float>(), nullptr,
        /*ncols_x=*/ne10, /*nrows_x=*/ne01, /*ncols_dst=*/ne11, /*stride_row_x=*/s01,
        /*ncols_y=*/ne11, /*nrows_dst=*/ne01,
        /*nchannels_x=*/1, /*nchannels_y=*/1, /*stride_channel_x=*/0, /*stride_channel_y=*/s12, /*stride_channel_dst=*/0,
        /*nsamples_x=*/1, /*nsamples_y=*/1, /*stride_sample_x=*/0, /*stride_sample_y=*/s12, /*stride_sample_dst=*/0,
        /*ncols_max=*/ne11};

    static ggml_backend_cuda_context ctx(id);
    switch (type_x) {
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("unsupported type");
    }
    CUDA_CHECK(cudaGetLastError());
    return dst;
}

// ---------------------------------------------------------------------------
// Torch entry: grouped MoE matmul over flat padded expert slots.
//
// Mirrors upstream's ggml_cuda_mul_mat_q ids branch (mmq.cu): build the
// expert-sorted row maps with mm_ids_helper, quantize each token once and
// scatter to its expert slots (broadcast=true: gate/up, X shared by the
// token's top_k experts) or gather-quantize per routed row (broadcast=false:
// down, X row t*top_k+k belongs to (token t, slot k)), then one mul_mat_q
// launch with expert_bounds. Returns fp32 [tokens*top_k, rows]; row
// t*top_k+k belongs to topk_ids[t][k].
// ---------------------------------------------------------------------------

static torch::Tensor ggml_moe_a8_mma(
        torch::Tensor X, torch::Tensor W, torch::Tensor topk_ids, int64_t top_k,
        int64_t type, int64_t nrows, int64_t tokens, int64_t expert_stride_bytes,
        bool broadcast) {
    TORCH_CHECK(W.is_cuda() && X.is_cuda() && topk_ids.is_cuda());
    TORCH_CHECK(W.dtype() == torch::kUInt8 && W.dim() == 2 && W.is_contiguous());
    TORCH_CHECK(X.dim() == 2 && X.is_contiguous());
    TORCH_CHECK(topk_ids.dtype() == torch::kInt32 && topk_ids.is_contiguous());
    TORCH_CHECK(topk_ids.dim() == 2 && topk_ids.size(0) == tokens && topk_ids.size(1) == top_k);
    TORCH_CHECK(type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K, "only Q4_K/Q6_K instantiated");

    const ggml_type type_x = (ggml_type) type;
    const c10::cuda::CUDAGuard guard(W.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor Xf = X.scalar_type() == torch::kFloat32 ? X : X.to(torch::kFloat32);

    const int64_t ne02 = W.size(0);   // experts (slots)
    const int64_t ne01 = nrows;       // output features per expert
    const int64_t ne10 = X.size(1);   // input features
    const int64_t ne12 = tokens;
    const int64_t n_expert_used = top_k;
    TORCH_CHECK(X.size(0) == (broadcast ? tokens : tokens * top_k), "bad activation row count");
    const size_t ts = ggml_type_size(type_x);
    const int64_t qk = ggml_blck_size(type_x);
    TORCH_CHECK(ne10 % qk == 0);
    const int64_t s01 = ne10 / qk;    // row stride in blocks
    if (expert_stride_bytes == 0) {
        expert_stride_bytes = W.size(1);
    }
    TORCH_CHECK(expert_stride_bytes % (int64_t) ts == 0,
                "expert slot stride must be a multiple of the block size");
    TORCH_CHECK(ne01 * s01 * (int64_t) ts <= expert_stride_bytes, "slot smaller than payload");
    const int64_t s02 = expert_stride_bytes / ts;  // expert stride in blocks

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const bool fallback = ne01 % 128 != 0;

    auto opts = torch::TensorOptions().device(W.device());
    const int64_t ne_get_rows = ne12 * n_expert_used;
    torch::Tensor ids_src1 = torch::empty({ne_get_rows}, opts.dtype(torch::kInt32));
    torch::Tensor ids_dst = torch::empty({ne_get_rows}, opts.dtype(torch::kInt32));
    torch::Tensor expert_bounds = torch::empty({ne02 + 1}, opts.dtype(torch::kInt32));

    // Broadcast activations (gate/up): each token row is shared by its top_k
    // experts -- quantize once and scatter via the inverse map. Per-slot
    // activations (down): ids_src1 holds the forward map it*top_k + slot,
    // which is exactly the flattened X row -- gather-quantize.
    const bool dedup_bcast = broadcast && n_expert_used > 1;
    const int nchannels_y = broadcast ? 1 : (int) top_k;
    const int sis1 = broadcast ? 1 : (int) top_k;

    ggml_cuda_launch_mm_ids_helper(
        (const int32_t *) topk_ids.data_ptr(), (int32_t *) ids_src1.data_ptr(),
        (int32_t *) ids_dst.data_ptr(), (int32_t *) expert_bounds.data_ptr(),
        ne02, ne12, n_expert_used, nchannels_y,
        /*si1=*/(int) top_k, sis1, /*write_inverse=*/dedup_bcast, stream);
    CUDA_CHECK(cudaGetLastError());

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t nbytes_src1_q8_1 = ne_get_rows * ne10_padded * sizeof(block_q8_1_mmq) / QK8_1_MMQ +
        ggml_cuda_mmq_get_J_max(type_x, fallback, cc, /*ne11=*/1) * sizeof(block_q8_1_mmq);
    torch::Tensor y_q = torch::empty({(int64_t) nbytes_src1_q8_1}, opts.dtype(torch::kUInt8));
    torch::Tensor dst = torch::empty({ne_get_rows, ne01}, opts.dtype(torch::kFloat32));

    if (dedup_bcast) {
        quantize_scatter_mmq_q8_1_cuda(
            Xf.data_ptr<float>(), (const int32_t *) ids_src1.data_ptr(), y_q.data_ptr(), type_x,
            ne10, /*stride_token=*/ne10, ne10_padded, ne12, ne_get_rows, n_expert_used, stream);
    } else {
        // ids_src1[compact] indexes rows of X (stride ne10); ne2 == ne3 == 1.
        quantize_mmq_q8_1_cuda(
            Xf.data_ptr<float>(), (const int32_t *) ids_src1.data_ptr(), y_q.data_ptr(), type_x,
            ne10, /*s01=*/ne10, /*s02=*/0, /*s03=*/0,
            ne10_padded, ne_get_rows, /*ne2=*/1, /*ne3=*/1, stream);
    }
    CUDA_CHECK(cudaGetLastError());

    // Per-channel strides in the quantized-activation buffer (ne11 == 1).
    const int64_t s12 = 1 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12 * s12;
    const int64_t s1 = ne01;                 // dst row stride
    const int64_t s2 = n_expert_used * s1;   // dst channel (token) stride
    const int64_t s3 = ne12 * s2;

    const mmq_args args = {
        (const char *) W.data_ptr(), type_x, (const int *) y_q.data_ptr(),
        (const int32_t *) ids_dst.data_ptr(), (const int32_t *) expert_bounds.data_ptr(),
        dst.data_ptr<float>(), nullptr,
        /*ncols_x=*/ne10, /*nrows_x=*/ne01, /*ncols_dst=*/ne_get_rows, /*stride_row_x=*/s01,
        /*ncols_y=*/ne_get_rows, /*nrows_dst=*/s1,
        /*nchannels_x=*/ne02, /*nchannels_y=*/ne02, /*stride_channel_x=*/s02, /*stride_channel_y=*/s12, /*stride_channel_dst=*/s2,
        /*nsamples_x=*/1, /*nsamples_y=*/1, /*stride_sample_x=*/0, /*stride_sample_y=*/s13, /*stride_sample_dst=*/s3,
        /*ncols_max=*/ne12};

    static ggml_backend_cuda_context ctx(id);
    switch (type_x) {
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("unsupported type");
    }
    CUDA_CHECK(cudaGetLastError());
    return dst;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ggml_mul_mat_a8_mma", &ggml_mul_mat_a8_mma,
          "y = x @ dequant(W).T via upstream int8-MMA MMQ (Q4_K/Q6_K)");
    m.def("ggml_moe_a8_mma", &ggml_moe_a8_mma,
          "grouped MoE matmul over flat padded expert slots (Q4_K/Q6_K)");
}
