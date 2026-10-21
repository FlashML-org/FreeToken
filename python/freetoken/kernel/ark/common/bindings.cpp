#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

extern "C" {
int radeon_dense_gemv_fp8(const void*, const void*, const void*, void*, int, int, int, void*);
int radeon_moe_decode_gemv_fp8(const void*, const void*, const void*, const void*, const void*, const void*, const void*, void*, void*, int, int, int, int, int, int, void*);
}

using at::Tensor;

namespace {
void check(const Tensor& t, const Tensor& x, at::ScalarType dtype) {
  TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous() && t.scalar_type() == dtype,
              "ARK requires contiguous tensors of the declared dtype on one GPU");
}

void checked(int rc) { TORCH_CHECK(rc == 0, "ARK native HIP return code ", rc); }
void* stream() { return (void*)at::cuda::getCurrentCUDAStream().stream(); }

Tensor dense_decode(const Tensor& x, const Tensor& w, const Tensor& scale) {
  c10::cuda::CUDAGuard guard(x.device());
  check(x, x, at::kBFloat16);
  check(w, x, at::kFloat8_e4m3fn);
  check(scale, x, at::kBFloat16);
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1 && w.dim() == 2 && scale.dim() == 2,
              "ARK dense decode requires M=1 and 2D weight/scales");
  int K = x.size(1), N = w.size(0);
  TORCH_CHECK(K > 0 && N > 0 && K % 128 == 0 && w.size(1) == K,
              "ARK dense decode dimensions/configuration");
  TORCH_CHECK(scale.size(0) >= (N + 127) / 128 && scale.size(1) >= K / 128,
              "ARK dense scale storage too small");
  auto out = at::empty({1, N}, x.options());
  checked(radeon_dense_gemv_fp8(x.data_ptr(), w.data_ptr(), scale.data_ptr(), out.data_ptr(),
                               N, K, scale.stride(0), stream()));
  return out;
}

Tensor moe_decode(const Tensor& x, const Tensor& w1, const Tensor& w2, const Tensor& weights,
                  const Tensor& ids, const Tensor& s1, const Tensor& s2) {
  c10::cuda::CUDAGuard guard(x.device());
  check(x, x, at::kBFloat16);
  for (const auto& w : {w1, w2}) check(w, x, at::kFloat8_e4m3fn);
  for (const auto& s : {s1, s2}) check(s, x, at::kBFloat16);
  check(ids, x, at::kInt);
  check(weights, x, at::kFloat);
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1 && w1.dim() == 3 && w2.dim() == 3 &&
              s1.dim() == 3 && s2.dim() == 3 && ids.dim() == 2 && weights.sizes() == ids.sizes(),
              "ARK MoE decode requires M=1 and packed expert banks");
  int H = x.size(1), I = w2.size(2), K = ids.size(1), E = w1.size(0);
  TORCH_CHECK(H > 0 && I > 0 && H % 128 == 0 && I % 128 == 0 && E > 0 && K > 0 && K <= 16 &&
              ids.size(0) == 1 && w1.size(1) == 2 * I && w1.size(2) == H &&
              w2.size(0) == E && w2.size(1) == H, "ARK MoE decode dimensions");
  TORCH_CHECK(s1.size(0) == E && s1.size(1) == 2 * I / 128 && s1.size(2) >= H / 128 &&
              s2.size(0) == E && s2.size(1) == H / 128 && s2.size(2) >= I / 128,
              "ARK MoE scale storage too small");
  auto intermediate = at::empty({1, K, I}, x.options().dtype(at::kFloat));
  auto out = at::empty({1, H}, x.options());
  checked(radeon_moe_decode_gemv_fp8(x.data_ptr(), w1.data_ptr(), s1.data_ptr(), w2.data_ptr(),
    s2.data_ptr(), ids.data_ptr(), weights.data_ptr(), intermediate.data_ptr(), out.data_ptr(),
    1, K, H, I, s1.stride(1), s2.stride(1), stream()));
  return out;
}
}  // namespace

TORCH_LIBRARY(freetoken_ark, m) {
  m.def("dense_decode(Tensor x, Tensor w, Tensor scale) -> Tensor");
  m.def("moe_decode(Tensor x, Tensor w1, Tensor w2, Tensor weights, Tensor ids, Tensor s1, Tensor s2) -> Tensor");
}
TORCH_LIBRARY_IMPL(freetoken_ark, CUDA, m) {
  m.impl("dense_decode", TORCH_FN(dense_decode));
  m.impl("moe_decode", TORCH_FN(moe_decode));
}
