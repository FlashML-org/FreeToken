#include "../format.h"

namespace freetoken::cpu_moe {
namespace {

// ------------------------- mxfp4 (gpt-oss) GEMV -----------------------------
// Transposed split-K layout: blk[Kpairs, N2] (N innermost), scl[Kpairs/16, N2]
// e8m0 per 32-K. Computes out[c] = sum_kb (E2M1[lo]*x[2kb] + E2M1[hi]*x[2kb+1])
// * 2^(e8m0-127) for a contiguous column tile (blk/scl already offset to col 0 of
// the tile). Vectorized over N (16 columns / __m512), K stays the outer (cache-
// sequential) loop. Used by both gate_up (K=H) and down (K=I).
using mxgemv_fn = void (*)(float*, const uint8_t*, const uint8_t*, const bf16_t*, int, int,
                           int, const float*, const float*);

void mxfp4_gemv_scalar(float* out, const uint8_t* blk, const uint8_t* scl, const bf16_t* x,
                       int Kpairs, int N2, int ncol, const float* e2m1, const float* e8m0) {
  for (int c = 0; c < ncol; ++c) out[c] = 0.0f;
  for (int kb = 0; kb < Kpairs; ++kb) {
    const uint8_t* w = blk + (size_t)kb * N2;
    const uint8_t* s = scl + (size_t)(kb >> 4) * N2;
    const float xl = bf16_to_f32(x[2 * kb]);
    const float xh = bf16_to_f32(x[2 * kb + 1]);
    for (int c = 0; c < ncol; ++c) {
      const uint8_t byte = w[c];
      out[c] += (e2m1[byte & 0xF] * xl + e2m1[byte >> 4] * xh) * e8m0[s[c]];
    }
  }
}

#if CPU_MOE_X86
CPU_MOE_TARGET("avx512f")
void mxfp4_gemv_avx512(float* out, const uint8_t* blk, const uint8_t* scl, const bf16_t* x,
                       int Kpairs, int N2, int ncol, const float* e2m1, const float* e8m0) {
  (void)e8m0;  // e8m0[c]=2^(c-127) computed via bit construction (no gather)
  const __m512 lut = _mm512_loadu_ps(e2m1);
  const __m512i loma = _mm512_set1_epi32(0xF);
  // K-outer / N-inner: each kb cache line is read once and all live column chunks
  // (up to 4 -> 64 cols) accumulate from registers, so DRAM/L2 stream the tile once.
  int c0 = 0;
  for (; c0 + 16 <= ncol; c0 += 64) {
    const int nchunk = std::min(4, (ncol - c0) / 16);
    __m512 acc[4];
    for (int ci = 0; ci < nchunk; ++ci) acc[ci] = _mm512_setzero_ps();
    for (int kblk = 0; kblk < Kpairs; kblk += 16) {  // 16 K-pairs = 32 K = one scale row
      __m512 sc[4];
      for (int ci = 0; ci < nchunk; ++ci) {
        __m128i sraw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
            scl + (size_t)(kblk >> 4) * N2 + c0 + ci * 16));
        sc[ci] = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu8_epi32(sraw), 23));
      }
      __m512 blk_acc[4];
      for (int ci = 0; ci < nchunk; ++ci) blk_acc[ci] = _mm512_setzero_ps();
      for (int kk = 0; kk < 16; ++kk) {
        const int kb = kblk + kk;
        const uint8_t* wbase = blk + (size_t)kb * N2 + c0;
        // The transposed layout strides K by N2 bytes; prefetch ahead so the strided
        // reads are not exposed to DRAM latency (the HW streamer misses big strides).
        constexpr int PFD = 8;
        if (kb + PFD < Kpairs)
          _mm_prefetch(reinterpret_cast<const char*>(blk + (size_t)(kb + PFD) * N2 + c0),
                       _MM_HINT_T0);
        const __m512 xl = _mm512_set1_ps(bf16_to_f32(x[2 * kb]));
        const __m512 xh = _mm512_set1_ps(bf16_to_f32(x[2 * kb + 1]));
        for (int ci = 0; ci < nchunk; ++ci) {
          __m512i wi = _mm512_cvtepu8_epi32(
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(wbase + ci * 16)));
          __m512 vlo = _mm512_permutexvar_ps(_mm512_and_si512(wi, loma), lut);
          __m512 vhi = _mm512_permutexvar_ps(_mm512_and_si512(_mm512_srli_epi32(wi, 4), loma), lut);
          blk_acc[ci] = _mm512_fmadd_ps(vlo, xl, blk_acc[ci]);
          blk_acc[ci] = _mm512_fmadd_ps(vhi, xh, blk_acc[ci]);
        }
      }
      for (int ci = 0; ci < nchunk; ++ci) acc[ci] = _mm512_fmadd_ps(blk_acc[ci], sc[ci], acc[ci]);
    }
    for (int ci = 0; ci < nchunk; ++ci) _mm512_storeu_ps(out + c0 + ci * 16, acc[ci]);
  }
  for (int c = c0; c < ncol; ++c) {  // tail columns (< 16)
    float o = 0.0f;
    for (int kb = 0; kb < Kpairs; ++kb) {
      const uint8_t byte = blk[(size_t)kb * N2 + c];
      uint32_t bits = (uint32_t)scl[(size_t)(kb >> 4) * N2 + c] << 23;
      float sc;
      std::memcpy(&sc, &bits, 4);
      o += (e2m1[byte & 0xF] * bf16_to_f32(x[2 * kb]) +
            e2m1[byte >> 4] * bf16_to_f32(x[2 * kb + 1])) * sc;
    }
    out[c] = o;
  }
}

CPU_MOE_TARGET("avx2,fma")
void mxfp4_gemv_avx2(float* out, const uint8_t* blk, const uint8_t* scl, const bf16_t* x,
                     int Kpairs, int N2, int ncol, const float* e2m1, const float* e8m0) {
  (void)e8m0;  // e8m0[s]=2^(s-127) built via s<<23 (no gather)
  const __m256 mag8 = _mm256_loadu_ps(e2m1);
  int c0 = 0;
  for (; c0 + 8 <= ncol; c0 += 32) {  // up to 4 chunks of 8 = 32 cols
    const int nchunk = std::min(4, (ncol - c0) / 8);
    __m256 acc[4];
    for (int ci = 0; ci < nchunk; ++ci) acc[ci] = _mm256_setzero_ps();
    for (int kblk = 0; kblk < Kpairs; kblk += 16) {  // 16 K-pairs = one scale row
      __m256 sc[4];
      for (int ci = 0; ci < nchunk; ++ci) {
        __m128i sraw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
            scl + (size_t)(kblk >> 4) * N2 + c0 + ci * 8));
        sc[ci] = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu8_epi32(sraw), 23));
      }
      __m256 blk_acc[4];
      for (int ci = 0; ci < nchunk; ++ci) blk_acc[ci] = _mm256_setzero_ps();
      for (int kk = 0; kk < 16; ++kk) {
        const int kb = kblk + kk;
        const uint8_t* wbase = blk + (size_t)kb * N2 + c0;
        constexpr int PFD = 8;
        if (kb + PFD < Kpairs)
          _mm_prefetch(reinterpret_cast<const char*>(blk + (size_t)(kb + PFD) * N2 + c0),
                       _MM_HINT_T0);
        const __m256 xl = _mm256_set1_ps(bf16_to_f32(x[2 * kb]));
        const __m256 xh = _mm256_set1_ps(bf16_to_f32(x[2 * kb + 1]));
        for (int ci = 0; ci < nchunk; ++ci) {
          __m256i wi = _mm256_cvtepu8_epi32(
              _mm_loadl_epi64(reinterpret_cast<const __m128i*>(wbase + ci * 8)));
          __m256 vlo = e2m1_decode8(_mm256_and_si256(wi, _mm256_set1_epi32(0xF)), mag8);
          __m256 vhi = e2m1_decode8(_mm256_srli_epi32(wi, 4), mag8);
          blk_acc[ci] = _mm256_fmadd_ps(vlo, xl, blk_acc[ci]);
          blk_acc[ci] = _mm256_fmadd_ps(vhi, xh, blk_acc[ci]);
        }
      }
      for (int ci = 0; ci < nchunk; ++ci) acc[ci] = _mm256_fmadd_ps(blk_acc[ci], sc[ci], acc[ci]);
    }
    for (int ci = 0; ci < nchunk; ++ci) _mm256_storeu_ps(out + c0 + ci * 8, acc[ci]);
  }
  for (int c = c0; c < ncol; ++c) {  // tail columns (< 8); none when ncol%8==0
    float o = 0.0f;
    for (int kb = 0; kb < Kpairs; ++kb) {
      const uint8_t byte = blk[(size_t)kb * N2 + c];
      uint32_t bits = (uint32_t)scl[(size_t)(kb >> 4) * N2 + c] << 23;
      float sc;
      std::memcpy(&sc, &bits, 4);
      o += (e2m1[byte & 0xF] * bf16_to_f32(x[2 * kb]) +
            e2m1[byte >> 4] * bf16_to_f32(x[2 * kb + 1])) * sc;
    }
    out[c] = o;
  }
}
#endif

mxgemv_fn select_mxgemv() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
  if (t >= ISA_AVX512) return mxfp4_gemv_avx512;
  if (t >= ISA_AVX2) return mxfp4_gemv_avx2;
#endif
  (void)t;
  return mxfp4_gemv_scalar;
}

// ----------------------------- mxfp4 (gpt-oss) -----------------------------
// Transposed split-K layout (N innermost), so the GEMV streams K and accumulates
// a contiguous block of N output columns -> cache-line-efficient, no repack, no
// extra host memory. Pass 1 fuses gate_up + clamped-swiglu(+bias); pass 2 adds the
// down bias, and the executor applies the router weight and sums the routes.
//
// Dequant: w = E2M1[code] * 2^(e8m0_scale - 127); two codes per byte (low nibble
// first), one e8m0 scale per 32 contiguous K. Matches kernel/triton/mxfp4_moe.py.
class Mxfp4Format final : public MoeFormat {
 public:
  Mxfp4Format(const MoeParams& p, const FormatArgs& a) : MoeFormat(p), gemv_(select_mxgemv()) {
    gate_up_ = a.bank("gate_up");
    gu_scale_ = a.bank("gate_up_scale");
    gu_bias_ = a.bank("gate_up_bias");
    down_ = a.bank("down");
    dn_scale_ = a.bank("down_scale");
    dn_bias_ = a.bank("down_bias");
    for (int i = 0; i < 16; ++i) e2m1_lut_[i] = kE2M1[i];
    // e8m0 (mxfp4 block scale) = 2^(s-127); the GPU GEMV clamps s to [0,254].
    for (int i = 0; i < 256; ++i) e8m0_lut_[i] = std::ldexp(1.0f, std::min(i, 254) - 127);
  }

  std::string isa_name() const override { return tier_name(pick_isa()); }

  void gate_up(int layer, int e, int /*tok*/, int i0, int i1, const bf16_t* x, float /*w_in*/,
               bf16_t* g) const override {
    const int H = p.H, I = p.I;
    const uint8_t* gu_packed_l = static_cast<const uint8_t*>(tbl_at(gate_up_, layer));
    const uint8_t* gu_scale_l = static_cast<const uint8_t*>(tbl_at(gu_scale_, layer));
    const bf16_t* gu_bias_l = static_cast<const bf16_t*>(tbl_at(gu_bias_, layer));
    const int N2 = 2 * I;            // gate_up output width (gate/up interleaved)
    const int Hh = H / 2;            // packed-K rows
    const int nunit = i1 - i0;       // intermediate units owned by this tile
    const int col0 = 2 * i0;         // first gate_up column
    const int ncol = 2 * nunit;      // gate_up columns owned by this tile
    const uint8_t* blk_e = gu_packed_l + (size_t)e * Hh * N2;
    const uint8_t* scl_e = gu_scale_l + (size_t)e * (size_t)(H / 32) * N2;
    float gu[2 * kRowBlock];
    gemv_(gu, blk_e + col0, scl_e + col0, x, Hh, N2, ncol, e2m1_lut_, e8m0_lut_);
    const bf16_t* bias_e = gu_bias_l + (size_t)e * N2 + col0;
    const float lim = p.swiglu_limit, alpha = p.swiglu_alpha;
    for (int j = 0; j < nunit; ++j) {
      float gate = gu[2 * j] + bf16_to_f32(bias_e[2 * j]);
      float up = gu[2 * j + 1] + bf16_to_f32(bias_e[2 * j + 1]);
      if (gate > lim) gate = lim;
      if (up > lim) up = lim;
      else if (up < -lim) up = -lim;
      const float glu = gate / (1.0f + std::exp(-gate * alpha));  // gate * sigmoid(alpha*gate)
      g[i0 + j] = f32_to_bf16(glu * (up + 1.0f));
    }
  }

  // The transposed layout already streams a whole column block per call, so the routes go one after another.
  void down(int layer, int h0, int h1, const DownRoute* routes, int n, float* out) const override {
    const int H = p.H, I = p.I;
    const int Ih = I / 2;            // packed-K rows
    const int nh = h1 - h0;
    const uint8_t* dn_packed_l = static_cast<const uint8_t*>(tbl_at(down_, layer));
    const uint8_t* dn_scale_l = static_cast<const uint8_t*>(tbl_at(dn_scale_, layer));
    const bf16_t* dn_bias_l = static_cast<const bf16_t*>(tbl_at(dn_bias_, layer));
    for (int j = 0; j < n; ++j) {
      const size_t e = (size_t)routes[j].e;
      const uint8_t* blk_e = dn_packed_l + e * Ih * H;
      const uint8_t* scl_e = dn_scale_l + e * (size_t)(I / 32) * H;
      float* part = out + j * kRowBlock;
      gemv_(part, blk_e + h0, scl_e + h0, routes[j].g, Ih, H, nh, e2m1_lut_, e8m0_lut_);
      const bf16_t* bias_e = dn_bias_l + e * H + h0;
      for (int c = 0; c < nh; ++c) part[c] += bf16_to_f32(bias_e[c]);
    }
  }

 private:
  mxgemv_fn gemv_;
  // gpt-oss split-K layout, N innermost: blocks [E,H/2,2I], e8m0 scales [E,H/32,2I], bf16 bias [E,2I]; down likewise with K = I, N = H.
  const uint64_t* gate_up_ = nullptr;
  const uint64_t* gu_scale_ = nullptr;
  const uint64_t* gu_bias_ = nullptr;
  const uint64_t* down_ = nullptr;
  const uint64_t* dn_scale_ = nullptr;
  const uint64_t* dn_bias_ = nullptr;
  float e2m1_lut_[16];
  float e8m0_lut_[256];
};

}  // namespace

std::unique_ptr<MoeFormat> make_mxfp4_format(const MoeParams& p, const FormatArgs& a) {
  return std::make_unique<Mxfp4Format>(p, a);
}

}  // namespace freetoken::cpu_moe
