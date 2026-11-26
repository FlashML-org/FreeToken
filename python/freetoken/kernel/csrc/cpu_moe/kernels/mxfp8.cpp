#include <string>
#include <vector>

#include "../format.h"

namespace freetoken::cpu_moe {
namespace {

// ------------------------ MXFP8 (W8A16) dequant ------------------------
// fp8-e4m3 weights row-major [rows, K] plus one e8m0 code per 32 elements of a row: w = e4m3 * 2^(code - 127). K is a multiple of 32.
// The activations are read as fp32 even/odd halves (xe[m] = x[2m], xo[m] = x[2m+1]).
// 0x7F/0xFF (NaN in torch float8_e4m3fn) decode to +-480 in every kernel, as e4m3_decode does.
using mxfp8_dot_fn = float (*)(const uint8_t*, const uint8_t*, const float*, const float*, int,
                               const float*, const float*);

constexpr int MX_GROUP = 32;

float dot_mxfp8_scalar(const uint8_t* w, const uint8_t* s, const float* xe, const float* xo, int K,
                       const float* e4m3, const float* e8m0) {
  float acc = 0.0f;
  for (int g = 0; g < K / MX_GROUP; ++g) {
    // 8 independent sums: a single running sum is bound by the add latency
    float a[8] = {};
    const uint8_t* wb = w + g * MX_GROUP;
    const float* pe = xe + g * (MX_GROUP / 2);
    const float* po = xo + g * (MX_GROUP / 2);
    for (int j = 0; j < MX_GROUP / 2; j += 4)
      for (int l = 0; l < 4; ++l) {
        a[2 * l] += e4m3[wb[2 * (j + l)]] * pe[j + l];
        a[2 * l + 1] += e4m3[wb[2 * (j + l) + 1]] * po[j + l];
      }
    acc += (((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))) * e8m0[s[g]];
  }
  return acc;
}

#if CPU_MOE_X86
// A byte b read as fp16 bits (b & 0x80) << 8 | (b & 0x7F) << 7 is its e4m3 value times 2^-8, subnormals included, so the dot is scaled by 256 at the end.
// In a 16-bit lane holding bytes (2m, 2m+1), srai 1 then & 0xBF80 builds that for byte 2m+1; a left shift by 8 first does byte 2m. Every AVX2 CPU has F16C.
CPU_MOE_TARGET("avx2,fma,f16c")
inline __m256 mxfp8_group_avx2(const uint8_t* w, const float* pe, const float* po) {
  const __m256i keep = _mm256_set1_epi16((short)0xBF80);
  const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w));
  const __m256i odd = _mm256_and_si256(_mm256_srai_epi16(v, 1), keep);
  const __m256i even = _mm256_and_si256(_mm256_srai_epi16(_mm256_slli_epi16(v, 8), 1), keep);
  __m256 a = _mm256_mul_ps(_mm256_cvtph_ps(_mm256_castsi256_si128(even)), _mm256_loadu_ps(pe));
  __m256 b = _mm256_mul_ps(_mm256_cvtph_ps(_mm256_castsi256_si128(odd)), _mm256_loadu_ps(po));
  a = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm256_extracti128_si256(even, 1)), _mm256_loadu_ps(pe + 8), a);
  b = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm256_extracti128_si256(odd, 1)), _mm256_loadu_ps(po + 8), b);
  return _mm256_add_ps(a, b);
}

CPU_MOE_TARGET("avx2,fma,f16c")
float dot_mxfp8_avx2(const uint8_t* w, const uint8_t* s, const float* xe, const float* xo, int K,
                     const float*, const float* e8m0) {
  const int G = K / MX_GROUP;
  __m256 t0 = _mm256_setzero_ps(), t1 = _mm256_setzero_ps();
  int g = 0;
  for (; g + 2 <= G; g += 2) {
    if ((g & 3) == 0) _mm_prefetch(reinterpret_cast<const char*>(w + g * MX_GROUP) + PF_AHEAD, _MM_HINT_T0);
    t0 = _mm256_fmadd_ps(mxfp8_group_avx2(w + g * MX_GROUP, xe + g * 16, xo + g * 16),
                         _mm256_set1_ps(e8m0[s[g]]), t0);
    t1 = _mm256_fmadd_ps(mxfp8_group_avx2(w + (g + 1) * MX_GROUP, xe + (g + 1) * 16, xo + (g + 1) * 16),
                         _mm256_set1_ps(e8m0[s[g + 1]]), t1);
  }
  if (g < G)
    t0 = _mm256_fmadd_ps(mxfp8_group_avx2(w + g * MX_GROUP, xe + g * 16, xo + g * 16),
                         _mm256_set1_ps(e8m0[s[g]]), t0);
  return 256.0f * hsum256(_mm256_add_ps(t0, t1));
}

CPU_MOE_TARGET("avx512f,avx2,fma,f16c")
inline __m512 mxfp8_group_avx512(const uint8_t* w, const float* pe, const float* po) {
  const __m256i keep = _mm256_set1_epi16((short)0xBF80);
  const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w));
  const __m256i odd = _mm256_and_si256(_mm256_srai_epi16(v, 1), keep);
  const __m256i even = _mm256_and_si256(_mm256_srai_epi16(_mm256_slli_epi16(v, 8), 1), keep);
  return _mm512_fmadd_ps(_mm512_cvtph_ps(even), _mm512_loadu_ps(pe),
                         _mm512_mul_ps(_mm512_cvtph_ps(odd), _mm512_loadu_ps(po)));
}

CPU_MOE_TARGET("avx512f,avx2,fma,f16c")
float dot_mxfp8_avx512f(const uint8_t* w, const uint8_t* s, const float* xe, const float* xo, int K,
                        const float*, const float* e8m0) {
  const int G = K / MX_GROUP;
  // 4 running sums: one sum per row would be bound by the fma latency
  __m512 t0 = _mm512_setzero_ps(), t1 = _mm512_setzero_ps();
  __m512 t2 = _mm512_setzero_ps(), t3 = _mm512_setzero_ps();
  int g = 0;
  for (; g + 4 <= G; g += 4) {
    _mm_prefetch(reinterpret_cast<const char*>(w + g * MX_GROUP) + PF_AHEAD, _MM_HINT_T0);
    const uint8_t* wg = w + g * MX_GROUP;
    const float* pe = xe + g * 16;
    const float* po = xo + g * 16;
    t0 = _mm512_fmadd_ps(mxfp8_group_avx512(wg, pe, po), _mm512_set1_ps(e8m0[s[g]]), t0);
    t1 = _mm512_fmadd_ps(mxfp8_group_avx512(wg + 32, pe + 16, po + 16), _mm512_set1_ps(e8m0[s[g + 1]]), t1);
    t2 = _mm512_fmadd_ps(mxfp8_group_avx512(wg + 64, pe + 32, po + 32), _mm512_set1_ps(e8m0[s[g + 2]]), t2);
    t3 = _mm512_fmadd_ps(mxfp8_group_avx512(wg + 96, pe + 48, po + 48), _mm512_set1_ps(e8m0[s[g + 3]]), t3);
  }
  for (; g < G; ++g)
    t0 = _mm512_fmadd_ps(mxfp8_group_avx512(w + g * MX_GROUP, xe + g * 16, xo + g * 16),
                         _mm512_set1_ps(e8m0[s[g]]), t0);
  return 256.0f * _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(t0, t1), _mm512_add_ps(t2, t3)));
}
#endif  // CPU_MOE_X86

mxfp8_dot_fn select_mxfp8_dot() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
  if (t >= ISA_AVX512) return dot_mxfp8_avx512f;
  if (t >= ISA_AVX2) return dot_mxfp8_avx2;
#endif
  (void)t;
  return dot_mxfp8_scalar;
}

class Mxfp8Format final : public MoeFormat {
 public:
  Mxfp8Format(const MoeParams& p, const FormatArgs& a) : MoeFormat(p), dot_(select_mxfp8_dot()) {
    if (p.H % MX_GROUP || p.I % MX_GROUP)
      throw std::invalid_argument("CPU MoE mxfp8 experts need hidden and intermediate sizes divisible by 32");
    gate_up_ = a.bank("gate_up");
    gu_scale_ = a.bank("gate_up_scale");
    down_ = a.bank("down");
    dn_scale_ = a.bank("down_scale");
    const size_t tokens = p.max_tokens, routes = (size_t)p.max_tokens * p.top_k;
    xe_.resize(tokens * (p.H / 2));
    xo_.resize(tokens * (p.H / 2));
    ge_.resize(routes * (p.I / 2));
    go_.resize(routes * (p.I / 2));
    for (int i = 0; i < 256; ++i) e4m3_lut_[i] = e4m3_decode((uint8_t)i);
    // code 255 gives inf, as in the GPU kernels
    for (int i = 0; i < 256; ++i) e8m0_lut_[i] = std::ldexp(1.0f, i - 127);
  }

  std::string isa_name() const override { return tier_name(pick_isa()); }

  void prep_input(int tok, const bf16_t* x) override {
    deinterleave_bf16_f32(x, xe_.data() + (size_t)tok * (p.H / 2), xo_.data() + (size_t)tok * (p.H / 2), p.H);
  }

  bool preps_inter() const override { return true; }

  void prep_inter(int64_t r, bf16_t* g) override {
    deinterleave_bf16_f32(g, ge_.data() + (size_t)r * (p.I / 2), go_.data() + (size_t)r * (p.I / 2), p.I);
  }

  void gate_up(int layer, int e, int tok, int i0, int i1, const bf16_t*, float w_in, bf16_t* g) const override {
    const int H = p.H, I = p.I, Hs = H / MX_GROUP;
    const uint8_t* packed = static_cast<const uint8_t*>(tbl_at(gate_up_, layer));
    const uint8_t* scale = static_cast<const uint8_t*>(tbl_at(gu_scale_, layer));
    const float* xe = xe_.data() + (size_t)tok * (H / 2);
    const float* xo = xo_.data() + (size_t)tok * (H / 2);
    gated_rows(p, i0, i1, w_in, g, [&](int row) {
      const size_t r = (size_t)e * (2 * I) + row;
      return dot_(packed + r * H, scale + r * Hs, xe, xo, H, e4m3_lut_, e8m0_lut_);
    });
  }

  void down(int layer, int h0, int h1, const DownRoute* routes, int n, float* out) const override {
    const int H = p.H, I = p.I, Is = I / MX_GROUP;
    const uint8_t* packed = static_cast<const uint8_t*>(tbl_at(down_, layer));
    const uint8_t* scale = static_cast<const uint8_t*>(tbl_at(dn_scale_, layer));
    for (int h = h0; h < h1; ++h) {
      for (int j = 0; j < n; ++j) {
        const size_t row = (size_t)routes[j].e * H + h, r = (size_t)routes[j].r;
        out[j * kRowBlock + h - h0] = dot_(packed + row * I, scale + row * Is, ge_.data() + r * (I / 2),
                                           go_.data() + r * (I / 2), I, e4m3_lut_, e8m0_lut_);
      }
    }
  }

 private:
  mxfp8_dot_fn dot_;
  // e4m3 rows [E,2I,H] and [E,H,I] with e8m0 codes [E,2I,H/32] and [E,H,I/32].
  const uint64_t* gate_up_ = nullptr;
  const uint64_t* gu_scale_ = nullptr;
  const uint64_t* down_ = nullptr;
  const uint64_t* dn_scale_ = nullptr;
  float e4m3_lut_[256];
  float e8m0_lut_[256];
  AlignedVector<float> xe_, xo_, ge_, go_;
};

}  // namespace

std::unique_ptr<MoeFormat> make_mxfp8_format(const MoeParams& p, const FormatArgs& a) {
  return std::make_unique<Mxfp8Format>(p, a);
}

}  // namespace freetoken::cpu_moe
