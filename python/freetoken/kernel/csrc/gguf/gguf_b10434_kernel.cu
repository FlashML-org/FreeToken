// Narrow caller-owned workspace ABI for pinned llama.cpp b10434 single-token work.
// The candidate source supplies the quantized GEMV implementation; this translation unit
// owns ABI validation, stable Q8_1 scratch slicing, and the public binding. Building both
// translation units into one extension keeps one pybind module and one stream contract.
#include <torch/extension.h>
#include <cstdint>
#include <string>

void bind_gguf_moe_gfx1100(pybind11::module_&);

torch::Tensor ggml_moe_mmvq_id(
    torch::Tensor X, torch::Tensor W, torch::Tensor topk_ids,
    int64_t top_k, int64_t type, int64_t row, int64_t tokens,
    int64_t expert_stride_bytes, int64_t row_stride_bytes,
    const std::string& id_space, torch::Tensor output, torch::Tensor quant_X_input);

namespace {
constexpr int64_t kAlign = 256;
constexpr int64_t align256(int64_t value) { return (value + kAlign - 1) / kAlign * kAlign; }
}

int64_t mmvq_bs1_workspace_bytes(int64_t hidden, int64_t rows, int64_t channels) {
  TORCH_CHECK(hidden > 0 && rows > 0 && channels > 0, "MMVQ shape must be positive");
  TORCH_CHECK(hidden % 32 == 0, "MMVQ b10434 ABI requires hidden dimension divisible by 32");
  const int64_t activation = align256((hidden / 32) * 36);
  const int64_t output = align256(channels * rows * static_cast<int64_t>(sizeof(float)));
  return activation + output;
}

torch::Tensor mmvq_bs1(
    torch::Tensor x, torch::Tensor weight, torch::Tensor output, torch::Tensor workspace,
    int64_t quant_type, int64_t rows, int64_t channels,
    torch::Tensor route_ids = torch::Tensor()) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda() && output.is_cuda() && workspace.is_cuda(),
              "MMVQ b10434 ABI requires CUDA/HIP tensors");
  TORCH_CHECK(x.device() == weight.device() && output.device() == weight.device() &&
              workspace.device() == weight.device(),
              "MMVQ b10434 tensors must share device");
  TORCH_CHECK(x.is_contiguous() && weight.stride(2) == 1 && output.is_contiguous() &&
              workspace.is_contiguous(),
              "MMVQ b10434 ABI requires contiguous caller-owned buffers");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "MMVQ b10434 accepts [1,H] activation");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32 && output.sizes() ==
              torch::IntArrayRef({channels, rows}), "MMVQ output must be FP32 [channels, rows]");
  TORCH_CHECK(workspace.numel() * workspace.element_size() >=
              mmvq_bs1_workspace_bytes(x.size(1), rows, channels), "MMVQ workspace too small");
  TORCH_CHECK(workspace.scalar_type() == torch::kUInt8 && workspace.dim() == 1,
              "MMVQ workspace must be a contiguous uint8 byte buffer");
  TORCH_CHECK(quant_type == 2 || quant_type == 8 || quant_type == 12 ||
              quant_type == 13 || quant_type == 14, "unsupported GGUF quant type");
  TORCH_CHECK(weight.dim() == 3 && weight.size(1) == rows && weight.size(0) >= channels,
              "MMVQ weight must be [experts, rows, row_bytes]");
  // The linked b10434 candidate currently uses the upstream 512-column Q8_1 tile.
  // Qwen H=2048 is exact; rejecting other geometry avoids hidden padding allocation.
  TORCH_CHECK(x.size(1) % 512 == 0,
              "MMVQ b10434 ABI requires hidden dimension divisible by 512");
  torch::Tensor ids = route_ids;
  if (!ids.defined()) {
    // Convenience path for eager ABI smoke only. Captured callers pass route_ids so
    // no device allocation occurs after graph capture begins.
    ids = torch::arange(channels, torch::TensorOptions().dtype(torch::kInt32).device(x.device()))
             .view({1, channels});
  }
  TORCH_CHECK(ids.is_cuda() && ids.device() == weight.device() && ids.is_contiguous() &&
              ids.scalar_type() == torch::kInt32 && ids.sizes() == torch::IntArrayRef({1, channels}),
              "MMVQ route_ids must be contiguous CUDA int32 [1, channels]");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16,
              "MMVQ b10434 ABI currently accepts BF16 activation");
  const int64_t activation_bytes = (x.size(1) / 32) * 36;
  const int64_t activation_region = align256(activation_bytes);
  auto quant_x = workspace.narrow(0, 0, activation_bytes).view(torch::kInt32)
      .view({1, x.size(1) / 32 * 9});
  // Candidate GEMV preserves activation dtype. Use the caller-owned aligned output
  // region as BF16 scratch, then cast into the ABI's FP32 output without allocation.
  auto candidate_output = workspace.narrow(0, activation_region, channels * rows * 2)
      .view(torch::kBFloat16).view({channels, rows});
  ggml_moe_mmvq_id(
      x, weight, ids, channels, quant_type, rows, 1,
      weight.stride(0), weight.stride(1), "raw", candidate_output, quant_x);
  output.copy_(candidate_output);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  bind_gguf_moe_gfx1100(module);
  module.def("mmvq_bs1_workspace_bytes", &mmvq_bs1_workspace_bytes);
  module.def("mmvq_bs1", &mmvq_bs1);
}
