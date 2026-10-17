#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "../format.h"
#include "cpu_kernels.h"

namespace freetoken::cpu_moe {
namespace {

std::atomic<const ft_ggml_cpu_api*> g_ggml{nullptr};

const ft_ggml_cpu_api& ggml_api() {
  const ft_ggml_cpu_api* api = g_ggml.load();
  if (api == nullptr) throw std::logic_error("CPU MoE gguf experts need a ggml-cpu variant; call _cpu_moe.set_ggml_cpu first");
  return *api;
}

}  // namespace

void set_ggml_cpu(uintptr_t addr) {
  const auto* api = reinterpret_cast<const ft_ggml_cpu_api*>(addr);
  if (api == nullptr || api->abi != FT_GGML_CPU_ABI) throw std::invalid_argument("set_ggml_cpu: not a ggml-cpu kernel table of this build");
  const ft_ggml_cpu_api* prev = nullptr;
  if (!g_ggml.compare_exchange_strong(prev, api) && prev != api)
    throw std::invalid_argument(std::string("set_ggml_cpu: this process already uses the ") + prev->name + " ggml-cpu variant");
}

namespace {

// Packed ggml rows of the layers whose experts have this gate_up type and this down type, dotted by ggml-cpu against the activation converted to each type's vec_dot input (q8_0, q8_1 or q8_K, or the float type itself).
// With scales, each expert's (gate, up, down, 1) NVFP4 weight_scale_2 multiplies its gate and up rows before the glu and its down rows before the routing weight, as llama.cpp's build_moe_ffn does.
class GgufFormat final : public MoeFormat {
 public:
  GgufFormat(const MoeParams& p, const FormatArgs& a, bool scaled)
      : MoeFormat(p), ggml_(ggml_api()), gate_up_(a.bank("gate_up")), down_(a.bank("down")), scales_(scaled ? a.bank("global") : nullptr) {
    static std::once_flag tables;
    std::call_once(tables, ggml_.init);
    gu_ = kernel(static_cast<int>(a.option("gate_up_type")), p.H, "gate_up");
    dn_ = kernel(static_cast<int>(a.option("down_type")), p.I, "down");
    xf_.resize(p.H);
    xq_.resize(static_cast<size_t>(p.max_tokens) * gu_.act_row_bytes);
    gq_.resize(static_cast<size_t>(p.max_tokens) * p.top_k * dn_.act_row_bytes);
  }

  std::string isa_name() const override { return std::string(ggml_.name) + "(ggml-cpu)"; }

  void prep_input(int tok, const bf16_t* x) override {
    for (int i = 0; i < p.H; ++i) xf_[i] = bf16_to_f32(x[i]);
    gu_.from_float(xf_.data(), xq_.data() + static_cast<size_t>(tok) * gu_.act_row_bytes, p.H);
  }

  bool preps_inter() const override { return true; }

  void prep_inter(int64_t r, bf16_t* g) override {
    thread_local std::vector<float> gf;
    gf.resize(p.I);
    for (int i = 0; i < p.I; ++i) gf[i] = bf16_to_f32(g[i]);
    dn_.from_float(gf.data(), gq_.data() + static_cast<size_t>(r) * dn_.act_row_bytes, p.I);
  }

  void gate_up(int layer, int e, int tok, int i0, int i1, const bf16_t* /*x*/, float w_in, bf16_t* g) const override {
    const uint8_t* w = static_cast<const uint8_t*>(tbl_at(gate_up_, layer)) + static_cast<size_t>(e) * 2 * p.I * gu_.row_bytes;
    const uint8_t* xq = xq_.data() + static_cast<size_t>(tok) * gu_.act_row_bytes;
    const float* sc = expert_scales(layer, e);
    gated_rows(p, i0, i1, w_in, g, [&](int row) {
      float s;
      gu_.vec_dot(p.H, &s, 0, w + static_cast<size_t>(row) * gu_.row_bytes, 0, xq, 0, 1);
      return sc == nullptr ? s : s * sc[row < p.I ? 0 : 1];
    });
  }

  void down(int layer, int h0, int h1, const DownRoute* routes, int n, float* out) const override {
    const uint8_t* w = static_cast<const uint8_t*>(tbl_at(down_, layer));
    for (int h = h0; h < h1; ++h)
      for (int j = 0; j < n; ++j)
        dn_.vec_dot(p.I, &out[j * kRowBlock + h - h0], 0, w + (static_cast<size_t>(routes[j].e) * p.H + h) * dn_.row_bytes, 0,
                    gq_.data() + static_cast<size_t>(routes[j].r) * dn_.act_row_bytes, 0, 1);
    if (scales_ == nullptr) return;
    for (int j = 0; j < n; ++j) {
      const float s = expert_scales(layer, routes[j].e)[2];
      for (int h = h0; h < h1; ++h) out[j * kRowBlock + h - h0] *= s;
    }
  }

 private:
  // the expert's (gate, up, down, 1) row of the layer's [E, 4] fp32 scale bank, or nullptr without scales
  const float* expert_scales(int layer, int e) const {
    return scales_ == nullptr ? nullptr : static_cast<const float*>(tbl_at(scales_, layer)) + static_cast<size_t>(e) * 4;
  }

  ft_ggml_row_kernel kernel(int type, int k, const char* role) const {
    ft_ggml_row_kernel out{};
    const int rc = ggml_.row_kernel(type, k, &out);
    const std::string name = ggml_.type_name(type);
    if (rc == FT_GGML_NO_KERNEL) throw std::invalid_argument("ggml-cpu has no dot product for " + name + " experts");
    if (rc != FT_GGML_OK)
      throw std::invalid_argument("CPU MoE " + name + " " + role + " rows of " + std::to_string(k) + " values are not whole blocks");
    return out;
  }

  const ft_ggml_cpu_api& ggml_;
  const uint64_t* gate_up_;
  const uint64_t* down_;
  const uint64_t* scales_;
  ft_ggml_row_kernel gu_{}, dn_{};
  std::vector<float> xf_;
  // the activation in the vec_dot input type: x per token, g per route
  AlignedVector<uint8_t> xq_, gq_;
};

}  // namespace

std::unique_ptr<MoeFormat> make_gguf_format(const MoeParams& p, const FormatArgs& a) {
  return std::make_unique<GgufFormat>(p, a, false);
}

std::unique_ptr<MoeFormat> make_gguf_scaled_format(const MoeParams& p, const FormatArgs& a) {
  return std::make_unique<GgufFormat>(p, a, true);
}

}  // namespace freetoken::cpu_moe
