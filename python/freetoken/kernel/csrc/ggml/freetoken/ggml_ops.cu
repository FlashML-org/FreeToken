// tvm-ffi ops over the vendored ggml-cuda kernels. Weights stay in their packed ggml blocks.
// Activations go through the same q8_1 quantization and kernel choice as llama.cpp's mul_mat / mul_mat_id.
// Activations are fp32, bf16 or fp16 with contiguous rows. Results have the activation dtype and are accumulated in fp32; the caller allocates every output.

#include <freetoken/utils.h>

#include "common.cuh"
#include "convert.cuh"
#include "getrows.cuh"
#include "mmq.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
// after the ggml headers, so their HIP macros win over hip_compat.h
#include <freetoken/pinned.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/string.h>

#include <algorithm>
#include <array>
#include <memory>
#include <string>

using tvm::ffi::Optional;
using tvm::ffi::String;
using tvm::ffi::TensorView;

namespace {

using host::RuntimeCheck;

struct DeviceGuard {
    int prev, device;

    explicit DeviceGuard(DLDevice d) : prev(ggml_cuda_get_device()), device(d.device_id) {
#ifdef GGML_USE_HIP
        RuntimeCheck(d.device_type == kDLROCM, "ggml ops take ROCm tensors");
#else
        RuntimeCheck(d.device_type == kDLCUDA, "ggml ops take CUDA tensors");
#endif
        ggml_cuda_set_device(device);
    }

    ~DeviceGuard() {
        if (prev != device) {
            cudaSetDevice(prev);
        }
    }
};

// ggml treats a null stream as unset and would create its own; name the default stream explicitly
cudaStream_t current_stream(DLDevice d) {
    auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(d.device_type, d.device_id));
#ifdef GGML_USE_HIP
    return stream == nullptr ? hipStreamLegacy : stream;
#else
    return stream == nullptr ? cudaStreamLegacy : stream;
#endif
}

ggml_backend_cuda_context & context(DLDevice d) {
    static std::unique_ptr<ggml_backend_cuda_context> ctxs[GGML_CUDA_MAX_DEVICES];
    const int dev = d.device_id;
    if (!ctxs[dev]) {
        ctxs[dev] = std::make_unique<ggml_backend_cuda_context>(dev);
    }
    ctxs[dev]->curr_stream_no = 0;
    ctxs[dev]->streams[dev][0] = current_stream(d);
    return *ctxs[dev];
}

ggml_tensor describe(ggml_type type, const void * data, std::array<int64_t, 4> ne, std::array<size_t, 4> nb) {
    ggml_tensor t{};
    t.type = type;
    t.data = const_cast<void *>(data);
    for (int i = 0; i < 4; ++i) {
        t.ne[i] = ne[i];
        t.nb[i] = nb[i];
    }
    return t;
}

int device_cc() { return ggml_cuda_info().devices[ggml_cuda_get_device()].cc; }

bool is_dtype(DLDataType dt, DLDataTypeCode code, int bits) { return dt.code == code && dt.bits == bits && dt.lanes == 1; }

ggml_type float_type(TensorView t) {
    const DLDataType dt = t.dtype();
    if (is_dtype(dt, kDLFloat, 32)) {
        return GGML_TYPE_F32;
    }
    if (is_dtype(dt, kDLBfloat, 16)) {
        return GGML_TYPE_BF16;
    }
    RuntimeCheck(is_dtype(dt, kDLFloat, 16), "ggml tensors must be fp32, bf16 or fp16");
    return GGML_TYPE_F16;
}

// the activation dtype, which the result must share
ggml_type io_type(TensorView x, TensorView out) {
    const ggml_type type = float_type(x);
    RuntimeCheck(float_type(out) == type, "the result must have the activation dtype");
    return type;
}

int64_t check_weight(TensorView w, ggml_type type, int64_t k) {
    RuntimeCheck(is_dtype(w.dtype(), kDLUInt, 8), "ggml weight must be a uint8 tensor");
    RuntimeCheck(w.stride(w.ndim() - 1) == 1, "ggml weight rows must be contiguous");
    const int64_t bs = ggml_blck_size(type);
    RuntimeCheck(k % bs == 0, "K=", k, " is not a multiple of the ", ggml_type_name(type), " block size ", bs);
    const int64_t row_bytes = k / bs * (int64_t) ggml_type_size(type);
    RuntimeCheck(w.size(w.ndim() - 1) == row_bytes, "ggml weight row is ", w.size(w.ndim() - 1), " bytes, ", ggml_type_name(type),
                 " K=", k, " needs ", row_bytes);
    return row_bytes;
}

// one fp32 scale per expert, e.g. a column of a [E, n] scale bank; the MMVQ epilogue multiplies NVFP4 results by it
ggml_tensor describe_scale(TensorView s, ggml_type type, int64_t e) {
    RuntimeCheck(type == GGML_TYPE_NVFP4, "per-expert scales apply to NVFP4 weights, not ", ggml_type_name(type));
    RuntimeCheck(is_dtype(s.dtype(), kDLFloat, 32) && s.ndim() == 1 && s.size(0) == e && s.stride(0) > 0, "a per-expert scale must be an fp32 [E] view");
    const size_t nb0 = sizeof(float) * s.stride(0);
    return describe(GGML_TYPE_F32, s.data_ptr(), {e, 1, 1, 1}, {nb0, nb0 * e, nb0 * e, nb0 * e});
}

int64_t q8_1_row_bytes(int64_t k) { return GGML_PAD(k, MATRIX_ROW_PADDING) / QK8_1 * (int64_t) sizeof(block_q8_1); }

// MMQ tiles read up to MATRIX_ROW_PADDING values past the last row; the caller keeps this many bytes after it
int64_t mmq_tail_bytes(int64_t type_id, int64_t k) {
    const int64_t rem = k % MATRIX_ROW_PADDING;
    return rem == 0 ? 0 : (int64_t) ggml_row_size((ggml_type) type_id, MATRIX_ROW_PADDING - rem);
}

// MMQ / MMVQ index weight blocks from the data pointer with 32-bit ints, MMQ tail reads included
void check_block_offsets(TensorView w, ggml_type type, int64_t k) {
    if (w.numel() == 0) {
        return;
    }
    int64_t bytes = 1 + mmq_tail_bytes(type, k);
    for (int d = 0; d < w.ndim(); ++d) {
        bytes += (w.size(d) - 1) * w.stride(d);
    }
    const int64_t blocks = bytes / (int64_t) ggml_type_size(type);
    RuntimeCheck(blocks <= INT32_MAX, "ggml weight spans ", blocks, " ", ggml_type_name(type), " blocks; MMQ / MMVQ address at most ", INT32_MAX);
}

struct Activation {
    const void * data;
    ggml_type type;
    size_t esz;
};

// The native fp4 MMQ path quantizes fp32 activations only; it is built only with GGML_CUDA_FP4_MMA.
bool mmq_reads_16bit(ggml_type type) {
#ifdef GGML_CUDA_FP4_MMA
    return type != GGML_TYPE_MXFP4 && type != GGML_TYPE_NVFP4;
#else
    GGML_UNUSED(type);
    return true;
#endif
}

// x as the kernel reads it: x itself, or an fp32 copy in ``tmp`` for a kernel that reads fp32 only
Activation activation(TensorView x, bool f16_ok, ggml_cuda_pool_alloc<float> & tmp, cudaStream_t stream) {
    RuntimeCheck(x.IsContiguous(), "ggml activations must be contiguous");
    const ggml_type type = float_type(x);
    if (type == GGML_TYPE_F32) {
        return {x.data_ptr(), type, 4};
    }
    if (f16_ok) {
        return {x.data_ptr(), type, 2};
    }
    float * f32 = tmp.alloc(x.numel());
    ggml_get_to_fp32_cuda(type)(x.data_ptr(), f32, x.numel(), stream);
    return {f32, GGML_TYPE_F32, 4};
}

}  // namespace

// out[M, q8_1_row_bytes(K)] = x[M, K] as MMVQ's q8_1 blocks, for GEMVs that share one input
void op_quantize_q8_1(TensorView x, TensorView out) {
    RuntimeCheck(x.ndim() == 2 && x.stride(1) == 1, "quantize_q8_1 takes x[M, K] with contiguous rows");
    const DeviceGuard guard(x.device());
    const int64_t m = x.size(0), k = x.size(1), k_padded = GGML_PAD(k, MATRIX_ROW_PADDING);
    RuntimeCheck(is_dtype(out.dtype(), kDLUInt, 8) && out.IsContiguous() && out.ndim() == 2 && out.size(0) == m &&
                     out.size(1) == q8_1_row_bytes(k),
                 "out must be a contiguous uint8 [M, q8_1_row_bytes(K)] tensor");
    if (m == 0) {
        return;
    }
    const int64_t s1 = x.stride(0);
    const cudaStream_t stream = current_stream(x.device());
    const ggml_type type = float_type(x);
    if (type == GGML_TYPE_BF16) {
        quantize_row_q8_1_cuda((const nv_bfloat16 *) x.data_ptr(), nullptr, out.data_ptr(), GGML_TYPE_Q8_0, k, s1, s1 * m, s1 * m, k_padded, m, 1, 1, stream);
    } else if (type == GGML_TYPE_F16) {
        quantize_row_q8_1_cuda((const half *) x.data_ptr(), nullptr, out.data_ptr(), GGML_TYPE_Q8_0, k, s1, s1 * m, s1 * m, k_padded, m, 1, 1, stream);
    } else {
        quantize_row_q8_1_cuda((const float *) x.data_ptr(), nullptr, out.data_ptr(), GGML_TYPE_Q8_0, k, s1, s1 * m, s1 * m, k_padded, m, 1, 1, stream);
    }
}

// out[M, N] = x[M, K] @ W[N, K]^T; out may be a view with strided rows; MMVQ reads ``xq`` (op_quantize_q8_1 of x) instead of quantizing x again
void op_mul_mat(TensorView w, int64_t type_id, TensorView x, TensorView out, Optional<TensorView> xq) {
    RuntimeCheck(x.ndim() == 2 && w.ndim() == 2, "mul_mat takes w[N, row_bytes] and x[M, K]");
    const DeviceGuard guard(x.device());
    const ggml_type type = (ggml_type) type_id;
    const int64_t m = x.size(0), k = x.size(1), n = w.size(0);
    check_weight(w, type, k);
    check_block_offsets(w, type, k);
    RuntimeCheck(out.ndim() == 2 && out.size(0) == m && out.size(1) == n && out.stride(1) == 1, "mul_mat out must be a [M, N] view with contiguous rows");
    const ggml_type otype = io_type(x, out);
    if (m == 0) {
        return;
    }
    const size_t s1 = w.stride(0);
    ggml_tensor src0 = describe(type, w.data_ptr(), {k, n, 1, 1}, {ggml_type_size(type), s1, s1 * n, s1 * n});
    const size_t oe = ggml_type_size(otype), os = out.stride(0) * oe;
    ggml_tensor dst = describe(otype, out.data_ptr(), {n, m, 1, 1}, {oe, os, os * m, os * m});
    auto & ctx = context(x.device());
    const cudaStream_t stream = ctx.stream();
    const int cc = device_cc();
    ggml_cuda_pool_alloc<float> x32(ctx.pool());
    if (ggml_cuda_should_use_mmvq(type, cc, m)) {
        const Activation a = activation(x, true, x32, stream);
        ggml_tensor src1 = describe(a.type, a.data, {k, m, 1, 1}, {a.esz, a.esz * k, a.esz * k * m, a.esz * k * m});
        const void * q8 = nullptr;
        if (xq.has_value()) {
            const TensorView q = xq.value();
            RuntimeCheck(is_dtype(q.dtype(), kDLUInt, 8) && q.IsContiguous() && q.ndim() == 2 && q.size(0) == m && q.size(1) == q8_1_row_bytes(k),
                         "xq must be quantize_q8_1 of x");
            q8 = q.data_ptr();
        }
        ggml_cuda_mul_mat_vec_q(ctx, &src0, &src1, nullptr, &dst, nullptr, q8);
    } else if (ggml_cuda_should_use_mmq(type, cc, m, 0)) {
        const Activation a = activation(x, mmq_reads_16bit(type), x32, stream);
        ggml_tensor src1 = describe(a.type, a.data, {k, m, 1, 1}, {a.esz, a.esz * k, a.esz * k * m, a.esz * k * m});
        ggml_cuda_mul_mat_q(ctx, &src0, &src1, nullptr, &dst);
    } else {
        RuntimeCheck(false, "no ggml kernel for ", ggml_type_name(type), " with M=", m);
    }
}

// out[T, k, N]: row (t, j) multiplies x[t, j or 0] by expert ids[t, j] of W[E, N, K]; the matrix kernel runs a flat grid over the tiles that hold routes
// scale[ids[t, j]] multiplies row (t, j); only the vector kernel takes it
void op_mul_mat_id(TensorView w, int64_t type_id, TensorView x, TensorView ids, TensorView out, Optional<TensorView> scale) {
    RuntimeCheck(w.ndim() == 3, "mul_mat_id takes w[E, N, row_bytes]");
    RuntimeCheck(is_dtype(ids.dtype(), kDLInt, 32) && ids.ndim() == 2 && ids.IsContiguous(), "ids must be a contiguous int32 [T, k] tensor");
    const DeviceGuard guard(x.device());
    const ggml_type type = (ggml_type) type_id;
    const int64_t t = ids.size(0), top_k = ids.size(1);
    RuntimeCheck(x.ndim() == 3 && x.size(0) == t && (x.size(1) == 1 || x.size(1) == top_k), "x must be [T, 1, K] or [T, k, K]");
    const int64_t ky = x.size(1), k = x.size(2), e = w.size(0), n = w.size(1);
    check_weight(w, type, k);
    check_block_offsets(w, type, k);
    RuntimeCheck(out.ndim() == 3 && out.size(0) == t && out.size(1) == top_k && out.size(2) == n && out.IsContiguous(),
                 "out must be a contiguous [T, k, N] tensor");
    const ggml_type otype = io_type(x, out);
    if (t == 0) {
        return;
    }
    const size_t s1 = w.stride(1), s2 = w.stride(0);
    ggml_tensor src0 = describe(type, w.data_ptr(), {k, n, e, 1}, {ggml_type_size(type), s1, s2, s2 * e});
    const size_t ids_stride = sizeof(int32_t) * top_k, ids_bytes = ids_stride * t;
    ggml_tensor tids = describe(GGML_TYPE_I32, ids.data_ptr(), {top_k, t, 1, 1}, {sizeof(int32_t), ids_stride, ids_bytes, ids_bytes});
    auto & ctx = context(x.device());
    const cudaStream_t stream = ctx.stream();
    const int cc = device_cc();
    const bool mmvq = t <= MMVQ_MAX_BATCH_SIZE && t <= get_mmvq_mmid_max_batch(type, cc);
    RuntimeCheck(mmvq || ggml_cuda_should_use_mmq(type, cc, t, e), "no ggml MoE kernel for ", ggml_type_name(type), " with T=", t);
    ggml_cuda_pool_alloc<float> x32(ctx.pool());
    const Activation a = activation(x, mmvq || mmq_reads_16bit(type), x32, stream);
    ggml_tensor src1 = describe(a.type, a.data, {k, ky, t, 1}, {a.esz, a.esz * k, a.esz * k * ky, a.esz * k * ky * t});
    const size_t oe = ggml_type_size(otype);
    ggml_tensor dst = describe(otype, out.data_ptr(), {n, top_k, t, 1}, {oe, oe * n, oe * n * top_k, oe * n * top_k * t});
    RuntimeCheck(mmvq || !scale.has_value(), "mul_mat_id takes a per-expert scale only on the vector kernel, T=", t);
    if (mmvq) {
        ggml_cuda_mm_fusion_args_host fusion{};
        ggml_tensor tscale;
        if (scale.has_value()) {
            tscale = describe_scale(scale.value(), type, e);
            fusion.x_scale = &tscale;
        }
        ggml_cuda_mul_mat_vec_q(ctx, &src0, &src1, &tids, &dst, scale.has_value() ? &fusion : nullptr);
    } else {
        ggml_cuda_mul_mat_q(ctx, &src0, &src1, &tids, &dst, 0);
    }
}

namespace {

// The GLUs the MMVQ epilogue computes; any other ggml_glu_op falls through to a plain gate * up there.
ggml_glu_op fused_glu_op(const String & name) {
    if (name == "swiglu") {
        return GGML_GLU_OP_SWIGLU;
    }
    if (name == "geglu") {
        return GGML_GLU_OP_GEGLU;
    }
    if (name == "swiglu_oai") {
        return GGML_GLU_OP_SWIGLU_OAI;
    }
    RuntimeCheck(name == "swiglu_clamp", "MMVQ fuses swiglu, geglu, swiglu_oai or swiglu_clamp, not ", std::string(name));
    return GGML_GLU_OP_SWIGLU_CLAMP;
}

}  // namespace

// out[T, k, I] = glu(x @ gate^T, x @ up^T) for small T in one MMVQ pass; w[E, 2I, row_bytes] holds each expert's gate rows then its up rows; limit is swiglu_clamp's
// gate_scale / up_scale multiply the expert's gate / up results before the glu
void op_mul_mat_id_glu(TensorView w, int64_t type_id, TensorView x, TensorView ids, TensorView out, String glu, double limit,
                       Optional<TensorView> gate_scale, Optional<TensorView> up_scale) {
    const ggml_glu_op glu_op = fused_glu_op(glu);
    RuntimeCheck(w.ndim() == 3 && w.size(1) % 2 == 0, "mul_mat_id_glu takes w[E, 2I, row_bytes]");
    RuntimeCheck(is_dtype(ids.dtype(), kDLInt, 32) && ids.ndim() == 2 && ids.IsContiguous(), "ids must be a contiguous int32 [T, k] tensor");
    RuntimeCheck(x.ndim() == 2 && x.size(0) == ids.size(0), "x must be [T, K]");
    const DeviceGuard guard(x.device());
    const ggml_type type = (ggml_type) type_id;
    const int64_t t = ids.size(0), top_k = ids.size(1), k = x.size(1), e = w.size(0), n = w.size(1) / 2;
    check_weight(w, type, k);
    check_block_offsets(w, type, k);
    RuntimeCheck(out.IsContiguous() && out.ndim() == 3 && out.size(0) == t && out.size(1) == top_k && out.size(2) == n,
                 "out must be a contiguous [T, k, I] tensor");
    const ggml_type otype = io_type(x, out);
    const int cc = device_cc();
    RuntimeCheck(t <= MMVQ_MAX_BATCH_SIZE && t <= get_mmvq_mmid_max_batch(type, cc), "mul_mat_id_glu is the small-batch path, T=", t);
    if (t == 0) {
        return;
    }
    auto & ctx = context(x.device());
    ggml_cuda_pool_alloc<float> x32(ctx.pool());
    const Activation a = activation(x, true, x32, ctx.stream());
    const size_t s1 = w.stride(1), s2 = w.stride(0);
    const char * base = (const char *) w.data_ptr();
    ggml_tensor up = describe(type, base + n * s1, {k, n, e, 1}, {ggml_type_size(type), s1, s2, s2 * e});
    ggml_tensor gate = describe(type, base, {k, n, e, 1}, {ggml_type_size(type), s1, s2, s2 * e});
    ggml_tensor src1 = describe(a.type, a.data, {k, 1, t, 1}, {a.esz, a.esz * k, a.esz * k, a.esz * k * t});
    const size_t ids_stride = sizeof(int32_t) * top_k, ids_bytes = ids_stride * t;
    ggml_tensor tids = describe(GGML_TYPE_I32, ids.data_ptr(), {top_k, t, 1, 1}, {sizeof(int32_t), ids_stride, ids_bytes, ids_bytes});
    const size_t oe = ggml_type_size(otype);
    ggml_tensor dst = describe(otype, out.data_ptr(), {n, top_k, t, 1}, {oe, oe * n, oe * n * top_k, oe * n * top_k * t});
    ggml_cuda_mm_fusion_args_host fusion{};
    fusion.gate = &gate;
    fusion.glu_op = glu_op;
    fusion.glu_limit = (float) limit;
    ggml_tensor tgate_scale, tup_scale;
    if (gate_scale.has_value()) {
        tgate_scale = describe_scale(gate_scale.value(), type, e);
        fusion.gate_scale = &tgate_scale;
    }
    if (up_scale.has_value()) {
        tup_scale = describe_scale(up_scale.value(), type, e);
        fusion.x_scale = &tup_scale;
    }
    ggml_cuda_mul_mat_vec_q(ctx, &up, &src1, &tids, &dst, &fusion);
}

namespace {

template <typename T>
__global__ void moe_weighted_sum_kernel(const T * __restrict__ y, const float * __restrict__ w, T * __restrict__ out,
                                        const int64_t n, const int top_k) {
    const int64_t token = blockIdx.x;
    const int64_t col = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n) {
        return;
    }
    const int64_t row0 = token * top_k;
    float sum = 0.0f;
    for (int j = 0; j < top_k; ++j) {
        sum += ggml_cuda_cast<float>(y[(row0 + j) * n + col]) * w[row0 + j];
    }
    out[token * n + col] = ggml_cuda_cast<T>(sum);
}

}  // namespace

// out[T, N] = sum_j weights[t, j] * y[t, j, :], summed in fp32
void op_moe_weighted_sum(TensorView y, TensorView weights, TensorView out) {
    RuntimeCheck(y.ndim() == 3 && y.IsContiguous(), "y must be a contiguous [T, k, N] tensor");
    const int64_t t = y.size(0), top_k = y.size(1), n = y.size(2);
    RuntimeCheck(float_type(weights) == GGML_TYPE_F32 && weights.IsContiguous() && weights.numel() == t * top_k, "weights must be a contiguous fp32 [T, k] tensor");
    RuntimeCheck(out.ndim() == 2 && out.size(0) == t && out.size(1) == n && out.IsContiguous(), "out must be a contiguous [T, N] tensor");
    const ggml_type otype = io_type(y, out);
    const DeviceGuard guard(y.device());
    if (t == 0) {
        return;
    }
    constexpr int threads = 256;
    const dim3 blocks(t, (n + threads - 1) / threads);
    const cudaStream_t stream = current_stream(y.device());
    const float * wp = (const float *) weights.data_ptr();
    if (otype == GGML_TYPE_BF16) {
        moe_weighted_sum_kernel<<<blocks, threads, 0, stream>>>((const nv_bfloat16 *) y.data_ptr(), wp, (nv_bfloat16 *) out.data_ptr(), n, top_k);
    } else if (otype == GGML_TYPE_F16) {
        moe_weighted_sum_kernel<<<blocks, threads, 0, stream>>>((const half *) y.data_ptr(), wp, (half *) out.data_ptr(), n, top_k);
    } else {
        moe_weighted_sum_kernel<<<blocks, threads, 0, stream>>>((const float *) y.data_ptr(), wp, (float *) out.data_ptr(), n, top_k);
    }
    CUDA_CHECK(cudaGetLastError());
}

// out[rows, K] = W[rows, row_bytes] dequantized to out's dtype
void op_dequantize(TensorView w, int64_t type_id, int64_t k, TensorView out) {
    const DeviceGuard guard(w.device());
    const ggml_type type = (ggml_type) type_id;
    check_weight(w, type, k);
    RuntimeCheck(w.IsContiguous(), "dequantize needs a contiguous weight");
    const int64_t rows = w.numel() / w.size(w.ndim() - 1);
    RuntimeCheck(out.IsContiguous() && out.numel() == rows * k, "out must be a contiguous [rows, K] tensor");
    const ggml_type otype = float_type(out);
    if (rows == 0) {
        return;
    }
    const cudaStream_t stream = current_stream(w.device());
    if (type == otype) {
        CUDA_CHECK(cudaMemcpyAsync(out.data_ptr(), w.data_ptr(), rows * k * ggml_type_size(type), cudaMemcpyDeviceToDevice, stream));
        return;
    }
    if (otype == GGML_TYPE_BF16) {
        const auto convert = ggml_get_to_bf16_cuda(type);
        RuntimeCheck(convert != nullptr, "no ggml bf16 conversion for ", ggml_type_name(type));
        convert(w.data_ptr(), (nv_bfloat16 *) out.data_ptr(), rows * k, stream);
    } else if (otype == GGML_TYPE_F16) {
        const auto convert = ggml_get_to_fp16_cuda(type);
        RuntimeCheck(convert != nullptr, "no ggml fp16 conversion for ", ggml_type_name(type));
        convert(w.data_ptr(), (half *) out.data_ptr(), rows * k, stream);
    } else {
        const auto convert = ggml_get_to_fp32_cuda(type);
        RuntimeCheck(convert != nullptr, "no ggml fp32 conversion for ", ggml_type_name(type));
        convert(w.data_ptr(), (float *) out.data_ptr(), rows * k, stream);
    }
}

// out[i] = dequant(W[ids[i]]) for a table W[rows, row_bytes]
void op_get_rows(TensorView w, int64_t type_id, int64_t k, TensorView ids, TensorView out) {
    RuntimeCheck(is_dtype(ids.dtype(), kDLInt, 32) && ids.ndim() == 1 && ids.IsContiguous(), "ids must be a contiguous int32 vector");
    RuntimeCheck(w.ndim() == 2, "get_rows takes w[rows, row_bytes]");
    // w may be a pinned host table read zero-copy, so the launch device comes from out
    const DeviceGuard guard(out.device());
    const ggml_type type = (ggml_type) type_id;
    check_weight(w, type, k);
    const int64_t n = ids.size(0);
    RuntimeCheck(out.IsContiguous() && out.ndim() == 2 && out.size(0) == n && out.size(1) == k, "out must be a contiguous [n, K] tensor");
    const ggml_type otype = float_type(out);
    if (n == 0) {
        return;
    }
    const size_t nb01 = w.stride(0), esz = otype == GGML_TYPE_F32 ? 4 : 2;
    get_rows_cuda(host::device_alias(w.data_ptr(), w.device()), type, (const int32_t *) ids.data_ptr(), out.data_ptr(), otype, k, nb01, nb01 * w.size(0), nb01 * w.size(0),
                  n, 1, 1, 4, 4 * n, 4 * n, k * esz, k * esz * n, k * esz * n, current_stream(out.device()));
}

bool op_uses_mmvq(int64_t type_id, int64_t m) { return ggml_cuda_should_use_mmvq((ggml_type) type_id, device_cc(), m); }

int64_t op_mmvq_max_tokens(int64_t type_id) {
    return std::min<int64_t>(MMVQ_MAX_BATCH_SIZE, get_mmvq_mmid_max_batch((ggml_type) type_id, device_cc()));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_q8_1, op_quantize_q8_1);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(q8_1_row_bytes, q8_1_row_bytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mmq_tail_bytes, mmq_tail_bytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mul_mat, op_mul_mat);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mul_mat_id, op_mul_mat_id);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mul_mat_id_glu, op_mul_mat_id_glu);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_weighted_sum, op_moe_weighted_sum);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dequantize, op_dequantize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_rows, op_get_rows);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(uses_mmvq, op_uses_mmvq);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mmvq_max_tokens, op_mmvq_max_tokens);
