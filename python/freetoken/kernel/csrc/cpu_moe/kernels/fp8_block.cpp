#include <string>
#include <vector>

#include "../format.h"

namespace freetoken::cpu_moe {
namespace {

// ------------------------ block-FP8 (W8A16) dequant ------------------------
// fp8-e4m3 weights row-major [rows, K] plus one bf16 scale per 128x128 block. The scale tensor is named weight_scale_inv but multiplies, as in kernel/triton/fp8_blockscale_moe.py.
// Whole 128-blocks read the activations as fp32 even/odd halves (xe[m] = x[2m], xo[m] = x[2m+1]); a ragged tail reads the bf16 row x.
// 0x7F/0xFF (NaN in torch float8_e4m3fn) decode to +-480 in every kernel, as e4m3_decode does.
using fp8_block_dot_fn = float (*)(const uint8_t*, const float*, const float*, const bf16_t*,
                                   const bf16_t*, int, const float*);

constexpr int FP8_BLK = 128;  // K elements covered by one weight scale

inline float fp8_block_tail(const uint8_t* w, const bf16_t* x, int k0, int K, const float* e4m3) {
  float blk = 0.0f;
  for (int k = k0; k < K; ++k) blk += e4m3[w[k]] * bf16_to_f32(x[k]);
  return blk;
}

float dot_fp8_block_scalar(const uint8_t* w, const float* xe, const float* xo, const bf16_t* x,
                           const bf16_t* s, int K, const float* e4m3) {
  float acc = 0.0f;
  int b = 0, k0 = 0;
  for (; k0 + FP8_BLK <= K; k0 += FP8_BLK, ++b) {
    // 8 independent sums: a single running sum is bound by the add latency
    float a[8] = {};
    const uint8_t* wb = w + k0;
    const float* pe = xe + k0 / 2;
    const float* po = xo + k0 / 2;
    for (int j = 0; j < FP8_BLK / 2; j += 4)
      for (int l = 0; l < 4; ++l) {
        a[2 * l] += e4m3[wb[2 * (j + l)]] * pe[j + l];
        a[2 * l + 1] += e4m3[wb[2 * (j + l) + 1]] * po[j + l];
      }
    acc += (((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))) *
           bf16_to_f32(s[b]);
  }
  if (k0 < K) acc += fp8_block_tail(w, x, k0, K, e4m3) * bf16_to_f32(s[b]);
  return acc;
}

#if CPU_MOE_X86
// A byte b read as fp16 bits (b & 0x80) << 8 | (b & 0x7F) << 7 is its e4m3 value times 2^-8, subnormals included, so the block scale carries the 256.
// In a 16-bit lane holding bytes (2m, 2m+1), srai 1 then & 0xBF80 builds that for byte 2m+1; a left shift by 8 first does byte 2m. Every AVX2 CPU has F16C.
CPU_MOE_TARGET("avx2,fma,f16c")
float dot_fp8_block_avx2(const uint8_t* w, const float* xe, const float* xo, const bf16_t* x,
                         const bf16_t* s, int K, const float* e4m3) {
  const __m256i keep = _mm256_set1_epi16((short)0xBF80);
  __m256 total = _mm256_setzero_ps();
  int b = 0, k0 = 0;
  for (; k0 + FP8_BLK <= K; k0 += FP8_BLK, ++b) {
    _mm_prefetch(reinterpret_cast<const char*>(w + k0) + PF_AHEAD, _MM_HINT_T0);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    for (int j = 0; j < FP8_BLK; j += 32) {
      const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + k0 + j));
      const __m256i odd = _mm256_and_si256(_mm256_srai_epi16(v, 1), keep);
      const __m256i even = _mm256_and_si256(_mm256_srai_epi16(_mm256_slli_epi16(v, 8), 1), keep);
      const float* pe = xe + (k0 + j) / 2;
      const float* po = xo + (k0 + j) / 2;
      a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm256_castsi256_si128(even)), _mm256_loadu_ps(pe), a0);
      a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm256_extracti128_si256(even, 1)), _mm256_loadu_ps(pe + 8), a1);
      a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm256_castsi256_si128(odd)), _mm256_loadu_ps(po), a2);
      a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm256_extracti128_si256(odd, 1)), _mm256_loadu_ps(po + 8), a3);
    }
    const __m256 blk = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    total = _mm256_fmadd_ps(blk, _mm256_set1_ps(bf16_to_f32(s[b]) * 256.0f), total);
  }
  float acc = hsum256(total);
  if (k0 < K) acc += fp8_block_tail(w, x, k0, K, e4m3) * bf16_to_f32(s[b]);
  return acc;
}

CPU_MOE_TARGET("avx512f,avx2,fma,f16c")
float dot_fp8_block_avx512f(const uint8_t* w, const float* xe, const float* xo, const bf16_t* x,
                            const bf16_t* s, int K, const float* e4m3) {
  const __m256i keep = _mm256_set1_epi16((short)0xBF80);
  __m512 total = _mm512_setzero_ps();
  int b = 0, k0 = 0;
  for (; k0 + FP8_BLK <= K; k0 += FP8_BLK, ++b) {
    _mm_prefetch(reinterpret_cast<const char*>(w + k0) + PF_AHEAD, _MM_HINT_T0);
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    for (int j = 0; j < FP8_BLK; j += 64) {
      const __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + k0 + j));
      const __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + k0 + j + 32));
      const __m256i o0 = _mm256_and_si256(_mm256_srai_epi16(v0, 1), keep);
      const __m256i e0 = _mm256_and_si256(_mm256_srai_epi16(_mm256_slli_epi16(v0, 8), 1), keep);
      const __m256i o1 = _mm256_and_si256(_mm256_srai_epi16(v1, 1), keep);
      const __m256i e1 = _mm256_and_si256(_mm256_srai_epi16(_mm256_slli_epi16(v1, 8), 1), keep);
      const float* pe = xe + (k0 + j) / 2;
      const float* po = xo + (k0 + j) / 2;
      a0 = _mm512_fmadd_ps(_mm512_cvtph_ps(e0), _mm512_loadu_ps(pe), a0);
      a1 = _mm512_fmadd_ps(_mm512_cvtph_ps(o0), _mm512_loadu_ps(po), a1);
      a2 = _mm512_fmadd_ps(_mm512_cvtph_ps(e1), _mm512_loadu_ps(pe + 16), a2);
      a3 = _mm512_fmadd_ps(_mm512_cvtph_ps(o1), _mm512_loadu_ps(po + 16), a3);
    }
    const __m512 blk = _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3));
    total = _mm512_fmadd_ps(blk, _mm512_set1_ps(bf16_to_f32(s[b]) * 256.0f), total);
  }
  float acc = _mm512_reduce_add_ps(total);
  if (k0 < K) acc += fp8_block_tail(w, x, k0, K, e4m3) * bf16_to_f32(s[b]);
  return acc;
}
#endif  // CPU_MOE_X86

fp8_block_dot_fn select_fp8_block_dot() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
  // the avx512bf16 tier keeps the fp32 kernel: on Sapphire Rapids it is faster than widening to bf16 for dpbf16
  if (t >= ISA_AVX512) return dot_fp8_block_avx512f;
  if (t >= ISA_AVX2) return dot_fp8_block_avx2;
#endif
  (void)t;
  return dot_fp8_block_scalar;
}

class Fp8BlockFormat final : public MoeFormat {
 public:
  Fp8BlockFormat(const MoeParams& p, const FormatArgs& a) : MoeFormat(p), dot_(select_fp8_block_dot()) {
    gate_up_ = a.bank("gate_up");
    gu_scale_ = a.bank("gate_up_scale");
    down_ = a.bank("down");
    dn_scale_ = a.bank("down_scale");
    gu_scale_stride_ = a.option("gu_scale_stride");
    dn_scale_stride_ = a.option("dn_scale_stride");
    const size_t tokens = p.max_tokens, routes = (size_t)p.max_tokens * p.top_k;
    xe_.resize(tokens * (p.H / 2));
    xo_.resize(tokens * (p.H / 2));
    ge_.resize(routes * (p.I / 2));
    go_.resize(routes * (p.I / 2));
    for (int i = 0; i < 256; ++i) e4m3_lut_[i] = e4m3_decode((uint8_t)i);
  }

  std::string isa_name() const override { return tier_name(pick_isa()); }


  void prep_input(int tok, const bf16_t* x) override {
    deinterleave_bf16_f32(x, xe_.data() + (size_t)tok * (p.H / 2), xo_.data() + (size_t)tok * (p.H / 2), p.H);
  }

  bool preps_inter() const override { return true; }

  void prep_inter(int64_t r, bf16_t* g) override {
    deinterleave_bf16_f32(g, ge_.data() + (size_t)r * (p.I / 2), go_.data() + (size_t)r * (p.I / 2), p.I);
  }

  // The halves cover whole 128-blocks; the bf16 row covers a ragged tail.
  void gate_up(int layer, int e, int tok, int i0, int i1, const bf16_t* x, float w_in,
               bf16_t* g) const override {
    const int H = p.H, I = p.I;
    const uint8_t* packed = static_cast<const uint8_t*>(tbl_at(gate_up_, layer));
    const bf16_t* scale = static_cast<const bf16_t*>(tbl_at(gu_scale_, layer));
    const float* xe = xe_.data() + (size_t)tok * (H / 2);
    const float* xo = xo_.data() + (size_t)tok * (H / 2);
    gated_rows(p, i0, i1, w_in, g, [&](int row) {
      const uint8_t* w = packed + ((size_t)e * (2 * I) + row) * (size_t)H;
      const bf16_t* sc = scale + ((size_t)e * ((2 * I + FP8_BLK - 1) / FP8_BLK) + row / FP8_BLK) *
                                     (size_t)gu_scale_stride_;
      return dot_(w, xe, xo, x, sc, H, e4m3_lut_);
    });
  }

  void down(int layer, int h0, int h1, const DownRoute* routes, int n, float* out) const override {
    const int H = p.H, I = p.I;
    const uint8_t* packed = static_cast<const uint8_t*>(tbl_at(down_, layer));
    const bf16_t* scale = static_cast<const bf16_t*>(tbl_at(dn_scale_, layer));
    for (int h = h0; h < h1; ++h) {
      for (int j = 0; j < n; ++j) {
        const size_t e = (size_t)routes[j].e, r = (size_t)routes[j].r;
        const uint8_t* w = packed + (e * H + h) * (size_t)I;
        const bf16_t* sc = scale + (e * ((H + FP8_BLK - 1) / FP8_BLK) + h / FP8_BLK) *
                                       (size_t)dn_scale_stride_;
        out[j * kRowBlock + h - h0] =
            dot_(w, ge_.data() + r * (I / 2), go_.data() + r * (I / 2), routes[j].g, sc, I, e4m3_lut_);
      }
    }
  }

 private:
  fp8_block_dot_fn dot_;
  // e4m3 rows [E,2I,H] and [E,H,I] with one bf16 scale per 128x128 block.
  const uint64_t* gate_up_ = nullptr;
  const uint64_t* gu_scale_ = nullptr;
  const uint64_t* down_ = nullptr;
  const uint64_t* dn_scale_ = nullptr;
  // Row strides of the scale tensors, including any padding of the host bank layout.
  int64_t gu_scale_stride_ = 0, dn_scale_stride_ = 0;
  float e4m3_lut_[256];
  AlignedVector<float> xe_, xo_, ge_, go_;
};

}  // namespace

std::unique_ptr<MoeFormat> make_fp8_block_format(const MoeParams& p, const FormatArgs& a) {
  return std::make_unique<Fp8BlockFormat>(p, a);
}

}  // namespace freetoken::cpu_moe
