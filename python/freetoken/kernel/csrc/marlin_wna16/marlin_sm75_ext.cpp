/*
 * marlin_sm75_ext.cpp — pybind11 wrapper for the Marlin WNA16 sm_75 MoE kernel.
 *
 * Exposes a single Python function:
 *
 *   marlin_mm(A, B, s, sorted_token_ids, expert_ids, num_tokens_post_padded,
 *             topk_weights, top_k, mul_topk_weights) -> torch.Tensor
 *
 * which calls the generated marlin_mm() from ops.cu with the sm_75-specific
 * (stages=2) kernel instantiations.
 *
 * The kernel files (ops.cu, marlin_template.h, sm75_kernel_*.cu,
 * kernel_selector.h) are fetched from weicj/vLLM-2080Ti-Definitive at build
 * time by scripts/fetch_marlin_sm75.py, which is called by setup.py before
 * the CUDAExtension is compiled.
 */

#include <torch/extension.h>

// Forward declaration — implemented in ops.cu (pulled in via the CUDAExtension
// sources list). The actual symbol lives in namespace marlin_moe_wna16.
torch::Tensor marlin_sm75_mm(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& s,
    const torch::Tensor& sorted_token_ids,
    const torch::Tensor& expert_ids,
    const torch::Tensor& num_tokens_post_padded,
    const torch::Tensor& topk_weights,
    int64_t              top_k,
    bool                 mul_topk_weights
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FreeToken Marlin WNA16 sm_75 MoE extension (stages=2, FP16/INT8 activation)";
    m.def(
        "marlin_mm",
        &marlin_sm75_mm,
        "Marlin WNA16 fused MoE GEMM for sm_75 (Turing, stages=2)",
        py::arg("A"),
        py::arg("B"),
        py::arg("s"),
        py::arg("sorted_token_ids"),
        py::arg("expert_ids"),
        py::arg("num_tokens_post_padded"),
        py::arg("topk_weights"),
        py::arg("top_k"),
        py::arg("mul_topk_weights") = false
    );
}
