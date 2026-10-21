// The ggml-base symbols and lookup tables that the vendored ggml-cpu quant kernels reference, and the row kernel of each weight type.

#include "cpu_kernels.h"

#include "ggml-impl.h"
#include "ggml-cpu/ggml-cpu-impl.h"
#include "ggml-cpu/quants.h"
#include "ggml-cpu/simd-mappings.h"
#include "ggml-cpu/vec.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float ggml_table_f32_f16[1 << 16];
float ggml_table_f32_e8m0_half[1 << 8];
float ggml_table_f32_ue4m3[1 << 8];

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "ggml: %s:%d: ", file, line);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    fflush(stderr);
    abort();
}

struct type_info {
    const char * name;
    int64_t blck_size;
    size_t type_size;
};

static const struct type_info k_types[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32] = {"f32", 1, sizeof(float)},
    [GGML_TYPE_F16] = {"f16", 1, sizeof(ggml_fp16_t)},
    [GGML_TYPE_BF16] = {"bf16", 1, sizeof(ggml_bf16_t)},
    [GGML_TYPE_Q1_0] = {"q1_0", QK1_0, sizeof(block_q1_0)},
    [GGML_TYPE_Q2_0] = {"q2_0", QK2_0, sizeof(block_q2_0)},
    [GGML_TYPE_Q4_0] = {"q4_0", QK4_0, sizeof(block_q4_0)},
    [GGML_TYPE_Q4_1] = {"q4_1", QK4_1, sizeof(block_q4_1)},
    [GGML_TYPE_Q5_0] = {"q5_0", QK5_0, sizeof(block_q5_0)},
    [GGML_TYPE_Q5_1] = {"q5_1", QK5_1, sizeof(block_q5_1)},
    [GGML_TYPE_Q8_0] = {"q8_0", QK8_0, sizeof(block_q8_0)},
    [GGML_TYPE_Q8_1] = {"q8_1", QK8_1, sizeof(block_q8_1)},
    [GGML_TYPE_MXFP4] = {"mxfp4", QK_MXFP4, sizeof(block_mxfp4)},
    [GGML_TYPE_NVFP4] = {"nvfp4", QK_NVFP4, sizeof(block_nvfp4)},
    [GGML_TYPE_Q2_K] = {"q2_K", QK_K, sizeof(block_q2_K)},
    [GGML_TYPE_Q3_K] = {"q3_K", QK_K, sizeof(block_q3_K)},
    [GGML_TYPE_Q4_K] = {"q4_K", QK_K, sizeof(block_q4_K)},
    [GGML_TYPE_Q5_K] = {"q5_K", QK_K, sizeof(block_q5_K)},
    [GGML_TYPE_Q6_K] = {"q6_K", QK_K, sizeof(block_q6_K)},
    [GGML_TYPE_Q8_K] = {"q8_K", QK_K, sizeof(block_q8_K)},
    [GGML_TYPE_TQ1_0] = {"tq1_0", QK_K, sizeof(block_tq1_0)},
    [GGML_TYPE_TQ2_0] = {"tq2_0", QK_K, sizeof(block_tq2_0)},
    [GGML_TYPE_IQ2_XXS] = {"iq2_xxs", QK_K, sizeof(block_iq2_xxs)},
    [GGML_TYPE_IQ2_XS] = {"iq2_xs", QK_K, sizeof(block_iq2_xs)},
    [GGML_TYPE_IQ2_S] = {"iq2_s", QK_K, sizeof(block_iq2_s)},
    [GGML_TYPE_IQ3_XXS] = {"iq3_xxs", QK_K, sizeof(block_iq3_xxs)},
    [GGML_TYPE_IQ3_S] = {"iq3_s", QK_K, sizeof(block_iq3_s)},
    [GGML_TYPE_IQ1_S] = {"iq1_s", QK_K, sizeof(block_iq1_s)},
    [GGML_TYPE_IQ1_M] = {"iq1_m", QK_K, sizeof(block_iq1_m)},
    [GGML_TYPE_IQ4_NL] = {"iq4_nl", QK4_NL, sizeof(block_iq4_nl)},
    [GGML_TYPE_IQ4_XS] = {"iq4_xs", QK_K, sizeof(block_iq4_xs)},
};

static int known_type(int type) { return type >= 0 && type < GGML_TYPE_COUNT && k_types[type].name != NULL; }

int64_t ggml_blck_size(enum ggml_type type) { return k_types[type].blck_size; }

size_t ggml_type_size(enum ggml_type type) { return k_types[type].type_size; }

size_t ggml_row_size(enum ggml_type type, int64_t ne) { return k_types[type].type_size * ne / k_types[type].blck_size; }

const char * ggml_type_name(enum ggml_type type) { return known_type(type) ? k_types[type].name : "unknown"; }

const char * ft_ggml_type_name(int type) { return ggml_type_name((enum ggml_type) type); }

struct dot_info {
    ft_ggml_vec_dot_t vec_dot;
    int act_type;
};

// The weight types of type_traits_cpu in ggml-cpu.c, with their vec_dot and vec_dot_type.
static const struct dot_info k_dots[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32] = {(ft_ggml_vec_dot_t) ggml_vec_dot_f32, GGML_TYPE_F32},
    [GGML_TYPE_F16] = {(ft_ggml_vec_dot_t) ggml_vec_dot_f16, GGML_TYPE_F16},
    [GGML_TYPE_BF16] = {(ft_ggml_vec_dot_t) ggml_vec_dot_bf16, GGML_TYPE_BF16},
    [GGML_TYPE_Q1_0] = {ggml_vec_dot_q1_0_q8_0, GGML_TYPE_Q8_0},
    [GGML_TYPE_Q2_0] = {ggml_vec_dot_q2_0_q8_0, GGML_TYPE_Q8_0},
    [GGML_TYPE_Q4_0] = {ggml_vec_dot_q4_0_q8_0, GGML_TYPE_Q8_0},
    [GGML_TYPE_Q4_1] = {ggml_vec_dot_q4_1_q8_1, GGML_TYPE_Q8_1},
    [GGML_TYPE_Q5_0] = {ggml_vec_dot_q5_0_q8_0, GGML_TYPE_Q8_0},
    [GGML_TYPE_Q5_1] = {ggml_vec_dot_q5_1_q8_1, GGML_TYPE_Q8_1},
    [GGML_TYPE_Q8_0] = {ggml_vec_dot_q8_0_q8_0, GGML_TYPE_Q8_0},
    [GGML_TYPE_MXFP4] = {ggml_vec_dot_mxfp4_q8_0, GGML_TYPE_Q8_0},
    [GGML_TYPE_NVFP4] = {ggml_vec_dot_nvfp4_q8_0, GGML_TYPE_Q8_0},
    [GGML_TYPE_Q2_K] = {ggml_vec_dot_q2_K_q8_K, GGML_TYPE_Q8_K},
    [GGML_TYPE_Q3_K] = {ggml_vec_dot_q3_K_q8_K, GGML_TYPE_Q8_K},
    [GGML_TYPE_Q4_K] = {ggml_vec_dot_q4_K_q8_K, GGML_TYPE_Q8_K},
    [GGML_TYPE_Q5_K] = {ggml_vec_dot_q5_K_q8_K, GGML_TYPE_Q8_K},
    [GGML_TYPE_Q6_K] = {ggml_vec_dot_q6_K_q8_K, GGML_TYPE_Q8_K},
    [GGML_TYPE_TQ1_0] = {ggml_vec_dot_tq1_0_q8_K, GGML_TYPE_Q8_K},
    [GGML_TYPE_TQ2_0] = {ggml_vec_dot_tq2_0_q8_K, GGML_TYPE_Q8_K},
    [GGML_TYPE_IQ2_XXS] = {ggml_vec_dot_iq2_xxs_q8_K, GGML_TYPE_Q8_K},
    [GGML_TYPE_IQ2_XS] = {ggml_vec_dot_iq2_xs_q8_K, GGML_TYPE_Q8_K},
    [GGML_TYPE_IQ2_S] = {ggml_vec_dot_iq2_s_q8_K, GGML_TYPE_Q8_K},
    [GGML_TYPE_IQ3_XXS] = {ggml_vec_dot_iq3_xxs_q8_K, GGML_TYPE_Q8_K},
    [GGML_TYPE_IQ3_S] = {ggml_vec_dot_iq3_s_q8_K, GGML_TYPE_Q8_K},
    [GGML_TYPE_IQ1_S] = {ggml_vec_dot_iq1_s_q8_K, GGML_TYPE_Q8_K},
    [GGML_TYPE_IQ1_M] = {ggml_vec_dot_iq1_m_q8_K, GGML_TYPE_Q8_K},
    [GGML_TYPE_IQ4_NL] = {ggml_vec_dot_iq4_nl_q8_0, GGML_TYPE_Q8_0},
    [GGML_TYPE_IQ4_XS] = {ggml_vec_dot_iq4_xs_q8_K, GGML_TYPE_Q8_K},
};

static void fp32_to_fp32_row(const float * x, void * y, int64_t k) { memcpy(y, x, (size_t) k * sizeof(float)); }

static void fp32_to_fp16_row(const float * x, void * y, int64_t k) {
    ggml_fp16_t * out = y;
    for (int64_t i = 0; i < k; ++i) {
        out[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}

static void fp32_to_bf16_row(const float * x, void * y, int64_t k) {
    ggml_bf16_t * out = y;
    for (int64_t i = 0; i < k; ++i) {
        out[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

static ft_ggml_from_float_t act_quantizer(int act_type) {
    switch (act_type) {
        case GGML_TYPE_F32: return fp32_to_fp32_row;
        case GGML_TYPE_F16: return fp32_to_fp16_row;
        case GGML_TYPE_BF16: return fp32_to_bf16_row;
        case GGML_TYPE_Q8_0: return quantize_row_q8_0;
        case GGML_TYPE_Q8_1: return quantize_row_q8_1;
        case GGML_TYPE_Q8_K: return quantize_row_q8_K;
        default: return NULL;
    }
}

// The tables ggml_cpu_init fills for the quant kernels.
void ft_ggml_cpu_init(void) {
    for (int i = 0; i < (1 << 16); ++i) {
        ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32((ggml_fp16_t) i);
    }
    for (int i = 0; i < (1 << 8); ++i) {
        ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i);
        ggml_table_f32_ue4m3[i] = ggml_ue4m3_to_fp32((uint8_t) i);
    }
}

int ft_ggml_cpu_row_kernel(int type, int64_t k, struct ft_ggml_row_kernel * out) {
    if (!known_type(type) || k_dots[type].vec_dot == NULL) {
        return FT_GGML_NO_KERNEL;
    }
    const int act = k_dots[type].act_type;
    if (k % k_types[type].blck_size != 0 || k % k_types[act].blck_size != 0) {
        return FT_GGML_PARTIAL_BLOCK;
    }
    out->vec_dot = k_dots[type].vec_dot;
    out->from_float = act_quantizer(act);
    out->act_type = act;
    out->row_bytes = ggml_row_size((enum ggml_type) type, k);
    out->act_row_bytes = ggml_row_size((enum ggml_type) act, k);
    return FT_GGML_OK;
}
