#include <vector>

#include "../format.h"

namespace freetoken::cpu_moe {
namespace {

// ----------------------- DeepSeek-V4 ds_fp4 (W4A8) ---------------------------
// Row-major e2m1 (2/byte, low nibble first) + e8m0 per-32 block scale, no global
// (w = E2M1[code] * 2^(e8m0-127)); activations are FP8-e4m3 round-tripped (per-128
// block, ue8m0 scale) before each GEMM. Matches kernel/triton/dsv4 (fused_moe +
// fp8_linear): silu(clamp(gate,max=lim)) * clamp(up,-lim,lim), router weight on the
// down output.

// Activations are pre-deinterleaved to fp32 (xe[m]=x[2m], xo[m]=x[2m+1]) once per
// token/route and reused across all output rows. This drops the hot dot to one
// vpmovzxbd + two vpermps per 32 weights (1.5 shuffle ops / 16 vs 4 for a bf16,
// dup-permute, per-element gather), so the row-major fp4 GEMV stops being port-5
// bound and approaches the bf16 memory-bandwidth ceiling.
using dsdot_fn = float (*)(const uint8_t*, const uint8_t*, const float*, const float*, int,
                           const float*, const float*);

float dot_dsfp4_scalar(const uint8_t* packed, const uint8_t* scale, const float* xe,
                       const float* xo, int K, const float* e2m1, const float* e8m0) {
  float acc = 0.0f;
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const float sc = e8m0[scale[b]];
    const uint8_t* pk = packed + (size_t)b * 16;  // 16 bytes = 32 codes
    const float* xeb = xe + (size_t)b * 16;
    const float* xob = xo + (size_t)b * 16;
    float bsum = 0.0f;
    for (int j = 0; j < 16; ++j) {
      const uint8_t byte = pk[j];
      bsum += e2m1[byte & 0xF] * xeb[j];   // low nibble  -> even-K activation
      bsum += e2m1[byte >> 4] * xob[j];    // high nibble -> odd-K activation
    }
    acc += sc * bsum;
  }
  return acc;
}

#if CPU_MOE_X86
// One 32-block: 16 bytes -> 16 low + 16 high nibble values via two vpermps, times
// the pre-split even/odd fp32 activations, folded by the per-32 e8m0 scale.
CPU_MOE_TARGET("avx512f")
inline __m512 dsfp4_blk(const uint8_t* pk, const float* xeb, const float* xob, __m512 lut,
                        __m512i loma, float sc) {
  __m512i wi = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pk)));
  __m512 vlo = _mm512_permutexvar_ps(_mm512_and_si512(wi, loma), lut);
  __m512 vhi = _mm512_permutexvar_ps(_mm512_and_si512(_mm512_srli_epi32(wi, 4), loma), lut);
  __m512 prod = _mm512_fmadd_ps(vlo, _mm512_loadu_ps(xeb), _mm512_mul_ps(vhi, _mm512_loadu_ps(xob)));
  return _mm512_mul_ps(prod, _mm512_set1_ps(sc));
}

CPU_MOE_TARGET("avx512f")
float dot_dsfp4_avx512(const uint8_t* packed, const uint8_t* scale, const float* xe,
                       const float* xo, int K, const float* e2m1, const float* e8m0) {
  const __m512 lut = _mm512_loadu_ps(e2m1);
  const __m512i loma = _mm512_set1_epi32(0xF);
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  const int nb = K / 32;  // 16 packed bytes + one e8m0 scale per block
  int b = 0;
  for (; b + 2 <= nb; b += 2) {  // two independent accumulators hide FMA latency
    acc0 = _mm512_add_ps(acc0, dsfp4_blk(packed + (size_t)b * 16, xe + (size_t)b * 16,
                                         xo + (size_t)b * 16, lut, loma, e8m0[scale[b]]));
    acc1 = _mm512_add_ps(acc1, dsfp4_blk(packed + (size_t)(b + 1) * 16, xe + (size_t)(b + 1) * 16,
                                         xo + (size_t)(b + 1) * 16, lut, loma, e8m0[scale[b + 1]]));
  }
  for (; b < nb; ++b)
    acc0 = _mm512_add_ps(acc0, dsfp4_blk(packed + (size_t)b * 16, xe + (size_t)b * 16,
                                         xo + (size_t)b * 16, lut, loma, e8m0[scale[b]]));
  return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
}

// AVX2: a 32-K block is 16 bytes -> two 8-lane halves (8 even + 8 odd each).
CPU_MOE_TARGET("avx2,fma")
inline __m256 dsfp4_half_avx2(const uint8_t* pk, const float* xeb, const float* xob, __m256 mag8) {
  __m256i wi = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(pk)));
  __m256 vlo = e2m1_decode8(_mm256_and_si256(wi, _mm256_set1_epi32(0xF)), mag8);
  __m256 vhi = e2m1_decode8(_mm256_srli_epi32(wi, 4), mag8);
  return _mm256_fmadd_ps(vlo, _mm256_loadu_ps(xeb), _mm256_mul_ps(vhi, _mm256_loadu_ps(xob)));
}

CPU_MOE_TARGET("avx2,fma")
float dot_dsfp4_avx2(const uint8_t* packed, const uint8_t* scale, const float* xe,
                     const float* xo, int K, const float* e2m1, const float* e8m0) {
  const __m256 mag8 = _mm256_loadu_ps(e2m1);
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* pk = packed + (size_t)b * 16;
    const float* xeb = xe + (size_t)b * 16;
    const float* xob = xo + (size_t)b * 16;
    const __m256 sc = _mm256_set1_ps(e8m0[scale[b]]);
    acc0 = _mm256_fmadd_ps(dsfp4_half_avx2(pk, xeb, xob, mag8), sc, acc0);
    acc1 = _mm256_fmadd_ps(dsfp4_half_avx2(pk + 8, xeb + 8, xob + 8, mag8), sc, acc1);
  }
  return hsum256(_mm256_add_ps(acc0, acc1));
}
#endif

dsdot_fn select_dsdot() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
  if (t >= ISA_AVX512) return dot_dsfp4_avx512;
  if (t >= ISA_AVX2) return dot_dsfp4_avx2;
#endif
  (void)t;
  return dot_dsfp4_scalar;
}

// Round a clamped |x|<=448 to nearest float8-e4m3 (RNE), back to fp32. Matches
// torch.float8_e4m3fn / triton .to(float8e4nv).
inline float e4m3_round(float x) {
  const float sign = x < 0.0f ? -1.0f : 1.0f;
  const float a = std::fabs(x);
  if (a == 0.0f) return 0.0f;
  if (a >= 448.0f) return sign * 448.0f;
  int e;
  std::frexp(a, &e);  // a in [2^(e-1), 2^e)
  float step = std::ldexp(1.0f, e - 4);
  const float min_step = std::ldexp(1.0f, -9);  // e4m3 subnormal step (2^-9)
  if (step < min_step) step = min_step;
  float r = std::nearbyint(a / step) * step;
  if (r > 448.0f) r = 448.0f;
  return sign * r;
}

// IEEE ceil(log2(v)) for v>0 (matches dsv4 _log2_ceil / fast_round_scale).
inline int ceil_log2_pos(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  const int exp = (int)((bits >> 23) & 0xFF);
  const int man = (int)(bits & 0x7FFFFF);
  return exp - 127 + (man != 0 ? 1 : 0);
}

// DeepSeek-V4 / V4.1 activation FP8 round-trip (bf16 in/out): per `block` (the checkpoint's
// fp8 block: 128 on V4, 32 on V4.1), s = 2^ceil(log2(max(|x|,1e-4)/448));
// y = round_e4m3(clamp(x/s,+-448)) * s.
void fp8_roundtrip_bf16(const bf16_t* src, bf16_t* dst, int K, int block) {
  for (int b0 = 0; b0 < K; b0 += block) {
    const int b1 = std::min(K, b0 + block);
    float amax = 1e-4f;
    for (int i = b0; i < b1; ++i) amax = std::max(amax, std::fabs(bf16_to_f32(src[i])));
    const float s = std::ldexp(1.0f, ceil_log2_pos(amax * (1.0f / 448.0f)));
    const float inv_s = 1.0f / s;
    for (int i = b0; i < b1; ++i) {
      float q = bf16_to_f32(src[i]) * inv_s;
      q = std::min(448.0f, std::max(-448.0f, q));
      dst[i] = f32_to_bf16(e4m3_round(q) * s);
    }
  }
}

class DsFp4Format final : public MoeFormat {
 public:
  DsFp4Format(const MoeParams& p, const FormatArgs& a) : MoeFormat(p), dot_(select_dsdot()) {
    gate_up_ = a.bank("gate_up");
    gu_scale_ = a.bank("gate_up_scale");
    down_ = a.bank("down");
    dn_scale_ = a.bank("down_scale");
    act_block_ = static_cast<int>(a.option("act_block"));
    input_prequant_ = a.option("input_prequant", 0) != 0;
    const size_t tokens = p.max_tokens, routes = (size_t)p.max_tokens * p.top_k;
    xq_.resize(tokens * p.H);
    xe_.resize(tokens * (p.H / 2));
    xo_.resize(tokens * (p.H / 2));
    ge_.resize(routes * (p.I / 2));
    go_.resize(routes * (p.I / 2));
    for (int i = 0; i < 16; ++i) e2m1_lut_[i] = kE2M1[i];
    for (int i = 0; i < 256; ++i) e8m0_lut_[i] = std::ldexp(1.0f, std::min(i, 254) - 127);
  }

  std::string isa_name() const override { return tier_name(pick_isa()); }


  // DSV4 FP8-round-trips the input before the gate_up GEMV, unless the GPU already did.
  void prep_input(int tok, const bf16_t* x) override {
    const bf16_t* src = x;
    if (!input_prequant_) {
      bf16_t* xq = xq_.data() + (size_t)tok * p.H;
      fp8_roundtrip_bf16(x, xq, p.H, act_block_);
      src = xq;
    }
    deinterleave_bf16_f32(src, xe_.data() + (size_t)tok * (p.H / 2), xo_.data() + (size_t)tok * (p.H / 2), p.H);
  }

  bool preps_inter() const override { return true; }

  void prep_inter(int64_t r, bf16_t* g) override {
    fp8_roundtrip_bf16(g, g, p.I, act_block_);
    deinterleave_bf16_f32(g, ge_.data() + (size_t)r * (p.I / 2), go_.data() + (size_t)r * (p.I / 2), p.I);
  }

  void gate_up(int layer, int e, int tok, int i0, int i1, const bf16_t* /*x*/, float /*w_in*/,
               bf16_t* g) const override {
    const int H = p.H, I = p.I;
    const uint8_t* gu_packed_l = static_cast<const uint8_t*>(tbl_at(gate_up_, layer));
    const uint8_t* gu_scale_l = static_cast<const uint8_t*>(tbl_at(gu_scale_, layer));
    const int N2 = 2 * I, Hh = H / 2, Hs = H / 32;
    const float* xe = xe_.data() + (size_t)tok * (H / 2);
    const float* xo = xo_.data() + (size_t)tok * (H / 2);
    const uint8_t* gp = gu_packed_l + (size_t)e * N2 * Hh;
    const uint8_t* gs = gu_scale_l + (size_t)e * N2 * Hs;
    const float lim = p.swiglu_limit;
    for (int i = i0; i < i1; ++i) {
      // gate_up is stored bf16 by the reference GEMV before swiglu; round to match.
      float gate = bf16_to_f32(f32_to_bf16(
          dot_(gp + (size_t)i * Hh, gs + (size_t)i * Hs, xe, xo, H, e2m1_lut_, e8m0_lut_)));
      float up = bf16_to_f32(f32_to_bf16(dot_(
          gp + (size_t)(I + i) * Hh, gs + (size_t)(I + i) * Hs, xe, xo, H, e2m1_lut_, e8m0_lut_)));
      if (lim > 0.0f) {
        if (gate > lim) gate = lim;
        if (up > lim) up = lim;
        else if (up < -lim) up = -lim;
      }
      const float glu = gate / (1.0f + std::exp(-gate));  // silu(gate)
      g[i] = f32_to_bf16(glu * up);
    }
  }

  void down(int layer, int h0, int h1, const DownRoute* routes, int n, float* out) const override {
    const int H = p.H, I = p.I, Ih = I / 2, Is = I / 32;
    const uint8_t* dn_packed_l = static_cast<const uint8_t*>(tbl_at(down_, layer));
    const uint8_t* dn_scale_l = static_cast<const uint8_t*>(tbl_at(dn_scale_, layer));
    for (int h = h0; h < h1; ++h) {
      for (int j = 0; j < n; ++j) {
        const size_t e = (size_t)routes[j].e, r = (size_t)routes[j].r;
        const uint8_t* dp = dn_packed_l + e * (size_t)H * Ih + (size_t)h * Ih;
        const uint8_t* ds = dn_scale_l + e * (size_t)H * Is + (size_t)h * Is;
        out[j * kRowBlock + h - h0] = dot_(dp, ds, ge_.data() + r * Ih, go_.data() + r * Ih, I, e2m1_lut_, e8m0_lut_);
      }
    }
  }

  // The reference rounds each route's weighted down output to bf16 before the fp32 sum over routes.
  bool rounds_routes() const override { return true; }

 private:
  dsdot_fn dot_;
  // e2m1 rows [E,2I,H/2] with e8m0 scales per 32 along K [E,2I,H/32]; down likewise with K = I.
  const uint64_t* gate_up_ = nullptr;
  const uint64_t* gu_scale_ = nullptr;
  const uint64_t* down_ = nullptr;
  const uint64_t* dn_scale_ = nullptr;
  // Activation FP8 round-trip block, the checkpoint's fp8 block.
  int act_block_ = 128;
  // The caller already FP8-round-tripped the input activations on the GPU
  // (same reference grid), so prep_input must not repeat it on the host-callback
  // thread. That scalar per-element pass is single-threaded ON THE DECODE CRITICAL
  // PATH (~0.3ms/layer at H=4096, every worker and the GPU waiting on it); moving
  // it to a captured GPU elementwise kernel removes it while keeping the official
  // W4A8 numerics bit-exact (see cpu_executor.py).
  bool input_prequant_ = false;
  float e2m1_lut_[16];
  float e8m0_lut_[256];
  AlignedVector<bf16_t> xq_;
  AlignedVector<float> xe_, xo_, ge_, go_;
};

}  // namespace

std::unique_ptr<MoeFormat> make_ds_fp4_format(const MoeParams& p, const FormatArgs& a) {
  return std::make_unique<DsFp4Format>(p, a);
}

}  // namespace freetoken::cpu_moe
