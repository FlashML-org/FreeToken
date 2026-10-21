// ggml-cpu row kernels for the CPU MoE executor. setup.py builds them once per x86 variant into freetoken.kernel.ggml_cpu._<variant>; call them only through the table of a variant whose score() is not 0.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*ft_ggml_vec_dot_t)(int n, float * s, size_t bs, const void * x, size_t bx, const void * y, size_t by, int nrc);
typedef void (*ft_ggml_from_float_t)(const float * x, void * y, int64_t k);

struct ft_ggml_row_kernel {
    ft_ggml_vec_dot_t vec_dot;        // one weight row . one activation row in act_type
    ft_ggml_from_float_t from_float;  // fp32 activation row -> act_type
    int act_type;
    size_t row_bytes;                 // one weight row of k values
    size_t act_row_bytes;             // one activation row of k values in act_type
};

enum { FT_GGML_OK = 0, FT_GGML_NO_KERNEL = 1, FT_GGML_PARTIAL_BLOCK = 2 };

void ft_ggml_cpu_init(void);

// The kernels for weight rows of ggml type `type` over k values.
int ft_ggml_cpu_row_kernel(int type, int64_t k, struct ft_ggml_row_kernel * out);

const char * ft_ggml_type_name(int type);

// One variant's functions, as its module's api() hands them to _cpu_moe.set_ggml_cpu.
#define FT_GGML_CPU_ABI 1
struct ft_ggml_cpu_api {
    int abi;           // FT_GGML_CPU_ABI of the module that built it
    const char * name; // the variant, e.g. "haswell"
    void (*init)(void);
    int (*row_kernel)(int type, int64_t k, struct ft_ggml_row_kernel * out);
    const char * (*type_name)(int type);
};

#ifdef __cplusplus
}
#endif
