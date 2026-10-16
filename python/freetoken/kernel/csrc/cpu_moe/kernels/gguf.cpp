#ifdef FT_GGML_CPU

#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

#include "../format.h"
#include "cpu_kernels.h"

namespace freetoken::cpu_moe {
namespace {

#if CPU_MOE_X86
void cpuid(unsigned leaf, unsigned sub, unsigned r[4]) {
#if defined(_MSC_VER)
  int v[4];
  __cpuidex(v, static_cast<int>(leaf), static_cast<int>(sub));
  for (int i = 0; i < 4; ++i) r[i] = static_cast<unsigned>(v[i]);
#else
  __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}

unsigned long long xcr0() {
#if defined(_MSC_VER)
  return _xgetbv(0);
#else
  unsigned lo, hi;
  __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  return (static_cast<unsigned long long>(hi) << 32) | lo;
#endif
}
#endif  // CPU_MOE_X86

}  // namespace

std::string gguf_cpu_unavailable() {
#if CPU_MOE_X86
  unsigned r[4];
  cpuid(0, 0, r);
  const unsigned max_leaf = r[0];
  cpuid(1, 0, r);
  const unsigned ecx1 = r[2];
  unsigned ebx7 = 0;
  if (max_leaf >= 7) {
    cpuid(7, 0, r);
    ebx7 = r[1];
  }
  // xgetbv faults unless the OS enabled XSAVE (OSXSAVE)
  const bool os_ymm = (ecx1 >> 27 & 1) && (xcr0() & 6) == 6;
  std::string missing;
  const auto need = [&](bool ok, const char* name) {
    if (!ok) missing += (missing.empty() ? "" : ", ") + std::string(name);
  };
  need((ecx1 >> 28 & 1) && os_ymm, "AVX");
  need((ebx7 >> 5 & 1) && os_ymm, "AVX2");
  need(ecx1 >> 12 & 1, "FMA");
  need(ecx1 >> 29 & 1, "F16C");
  need(ebx7 >> 8 & 1, "BMI2");
  return missing.empty() ? "" : "the ggml-cpu kernels are built for AVX2 and this CPU lacks " + missing;
#else
  return "the ggml-cpu kernels are built for x86-64 only";
#endif
}

namespace {

// Packed ggml rows of the layers whose experts have this gate_up type and this down type, dotted by ggml-cpu against the activation quantized to each type's vec_dot input (q8_0, q8_1 or q8_K).
class GgufFormat final : public MoeFormat {
 public:
  GgufFormat(const MoeParams& p, const FormatArgs& a)
      : MoeFormat(p), gate_up_(a.bank("gate_up")), down_(a.bank("down")) {
    if (const std::string why = gguf_cpu_unavailable(); !why.empty()) throw std::runtime_error(why);
    static std::once_flag tables;
    std::call_once(tables, ft_ggml_cpu_init);
    gu_ = kernel(static_cast<int>(a.option("gate_up_type")), p.H, "gate_up");
    dn_ = kernel(static_cast<int>(a.option("down_type")), p.I, "down");
    xf_.resize(p.H);
    xq_.resize(static_cast<size_t>(p.max_tokens) * gu_.act_row_bytes);
    gq_.resize(static_cast<size_t>(p.max_tokens) * p.top_k * dn_.act_row_bytes);
  }

  std::string isa_name() const override { return "avx2(ggml-cpu)"; }

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
    gated_rows(p, i0, i1, w_in, g, [&](int row) {
      float s;
      gu_.vec_dot(p.H, &s, 0, w + static_cast<size_t>(row) * gu_.row_bytes, 0, xq, 0, 1);
      return s;
    });
  }

  void down(int layer, int h0, int h1, const DownRoute* routes, int n, float* out) const override {
    const uint8_t* w = static_cast<const uint8_t*>(tbl_at(down_, layer));
    for (int h = h0; h < h1; ++h)
      for (int j = 0; j < n; ++j)
        dn_.vec_dot(p.I, &out[j * kRowBlock + h - h0], 0, w + (static_cast<size_t>(routes[j].e) * p.H + h) * dn_.row_bytes, 0,
                    gq_.data() + static_cast<size_t>(routes[j].r) * dn_.act_row_bytes, 0, 1);
  }

 private:
  static ft_ggml_row_kernel kernel(int type, int k, const char* role) {
    ft_ggml_row_kernel out{};
    const int rc = ft_ggml_cpu_row_kernel(type, k, &out);
    const std::string name = ft_ggml_type_name(type);
    if (rc == FT_GGML_NO_KERNEL) throw std::invalid_argument("ggml-cpu has no dot product for " + name + " experts");
    if (rc != FT_GGML_OK)
      throw std::invalid_argument("CPU MoE " + name + " " + role + " rows of " + std::to_string(k) + " values are not whole blocks");
    return out;
  }

  const uint64_t* gate_up_;
  const uint64_t* down_;
  ft_ggml_row_kernel gu_{}, dn_{};
  std::vector<float> xf_;
  // the activation in the vec_dot input type: x per token, g per route
  AlignedVector<uint8_t> xq_, gq_;
};

}  // namespace

std::unique_ptr<MoeFormat> make_gguf_format(const MoeParams& p, const FormatArgs& a) {
  return std::make_unique<GgufFormat>(p, a);
}

}  // namespace freetoken::cpu_moe

#endif  // FT_GGML_CPU
