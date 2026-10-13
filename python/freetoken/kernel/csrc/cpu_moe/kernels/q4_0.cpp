#include <stdexcept>
#include <vector>

#include "../format.h"

namespace freetoken::cpu_moe {
namespace {

// -------------------------------- Q4_0 (W4A8) --------------------------------
// Native GGUF Q4_0 experts (gemma4 GGUF): per-32 block = fp16 scale d + 16 packed
// bytes; byte j holds element j in its low nibble and j+16 in its high nibble, so a
// block's storage order is [lo0..lo15, hi0..hi15] and w = (nibble - 8) * d. Matches
// the reference dequant (models/gguf/dequant.py) and the packed banks the GPU offload
// path streams.
//
// llama.cpp ggml_vec_dot_q4_0_q8_0: W4A8. The activation is pre-quantized to Q8_0
// (per-32-block int8 ``aq`` + fp32 scale ``asb``); each block unpacks its 16 bytes to
// 32 int8 weights in [-8,7] (bytes_from_nibbles_32: low nibbles -> elems 0..15, high
// -> 16..31) and runs an integer block dot -- VPDPBUSD (AVX-VNNI) or VPMADDUBSW+VPMADDWD
// (AVX2) with the ggml sign trick |w|*(sign(w)*a)=w*a, or a scalar int loop -- then
// scales the block sum by wd*xd in fp32. No fp weight dequant / shuffle chain. The GPU
// offload path (ggml_moe_a8_vec / MMVQ) is also W4A8, so cpu and hybrid stay close.
using q4dot_fn = float (*)(const uint8_t*, const int8_t*, const float*, int);

float q4_0_dot_i8_scalar(const uint8_t* w, const int8_t* aq, const float* asb, int K) {
  float acc = 0.0f;
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* blk = w + (size_t)b * 18;
    uint16_t dh;
    std::memcpy(&dh, blk, sizeof(dh));
    const uint8_t* q = blk + 2;  // 16 nibble bytes
    const int8_t* a = aq + (size_t)b * 32;
    int isum = 0;
    for (int j = 0; j < 16; ++j) {
      isum += ((int)(q[j] & 0x0F) - 8) * (int)a[j];       // elem j
      isum += ((int)(q[j] >> 4) - 8) * (int)a[16 + j];    // elem 16+j
    }
    acc += fp16_to_f32(dh) * asb[b] * (float)isum;
  }
  return acc;
}

#if CPU_MOE_X86
// fp16 block scale -> fp32 via HW F16C (single value in lane 0).
CPU_MOE_TARGET("f16c")
static inline float q4_scale(uint16_t h) {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
}

// Unpack one Q4_0 block's 16 bytes -> 32 int8 weights in [-8,7] (elems 0..15 = low
// nibbles, 16..31 = high nibbles). ``eight`` = _mm256_set1_epi8(8).
CPU_MOE_TARGET("avx2")
static inline __m256i q4_unpack32(const uint8_t* blk, __m128i mask, __m256i eight) {
  const __m128i qb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2));
  const __m128i lo = _mm_and_si128(qb, mask);
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask);
  return _mm256_sub_epi8(_mm256_set_m128i(hi, lo), eight);
}

// AVX2 W4A8 (llama.cpp non-VNNI mul_sum_i8_pairs): integer block dot via VPMADDUBSW +
// VPMADDWD (sign trick), scaled by wd*xd. |aw*sa| pair sums <= 8*127*2 < 32767 -> no
// int16 saturation. This is the fast path on AVX2 CPUs without AVX-VNNI (and the
// avx512-tier fallback, since the block dot is 256-bit either way).
CPU_MOE_TARGET("avx2,fma,f16c")
float q4_0_dot_i8_avx2(const uint8_t* w, const int8_t* aq, const float* asb, int K) {
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m256i eight = _mm256_set1_epi8(8);
  const __m256i ones16 = _mm256_set1_epi16(1);
  __m256 accF = _mm256_setzero_ps();
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* blk = w + (size_t)b * 18;
    _mm_prefetch(reinterpret_cast<const char*>(blk) + 512, _MM_HINT_T0);
    uint16_t dh;
    std::memcpy(&dh, blk, sizeof(dh));
    __m256i wq = q4_unpack32(blk, mask, eight);
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq + (size_t)b * 32));
    __m256i aw = _mm256_sign_epi8(wq, wq);              // |wq|          (unsigned operand)
    __m256i sa = _mm256_sign_epi8(a, wq);               // sign(wq) * a  (signed operand)
    __m256i d32 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sa), ones16);  // 8 int32
    accF = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d32), _mm256_set1_ps(q4_scale(dh) * asb[b]), accF);
  }
  return hsum256(accF);
}

// AVX-VNNI W4A8: one VPDPBUSD per block (the fast path on modern CPUs).
CPU_MOE_TARGET("avx2,avxvnni,fma,f16c")
float q4_0_dot_i8_vnni(const uint8_t* w, const int8_t* aq, const float* asb, int K) {
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m256i eight = _mm256_set1_epi8(8);
  __m256 accF = _mm256_setzero_ps();
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* blk = w + (size_t)b * 18;
    _mm_prefetch(reinterpret_cast<const char*>(blk) + 512, _MM_HINT_T0);
    uint16_t dh;
    std::memcpy(&dh, blk, sizeof(dh));
    __m256i wq = q4_unpack32(blk, mask, eight);
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq + (size_t)b * 32));
    __m256i aw = _mm256_sign_epi8(wq, wq);   // |wq|            (unsigned operand)
    __m256i sa = _mm256_sign_epi8(a, wq);    // sign(wq) * a    (signed operand)
    __m256i di = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), aw, sa);
    // All 32 elems of the block share wd*xd; distribute over di's 8 partial sums and
    // reduce at the end (equivalent to scale * block_total).
    accF = _mm256_fmadd_ps(_mm256_cvtepi32_ps(di), _mm256_set1_ps(q4_scale(dh) * asb[b]), accF);
  }
  return hsum256(accF);
}
#endif  // CPU_MOE_X86

// All tiers are W4A8 (int8 activations pre-quantized to Q8_0). AVX-VNNI is orthogonal to
// the ISA tier (gated by cpu_has_avxvnni() / FREETOKEN_CPU_MOE_NO_VNNI), so it wins when
// present; otherwise the 256-bit VPMADDUBSW kernel covers both the avx2 and avx512 tiers.
q4dot_fn select_q4dot() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
  if (cpu_has_avxvnni()) return q4_0_dot_i8_vnni;
  if (t >= ISA_AVX2) return q4_0_dot_i8_avx2;
#endif
  (void)t;
  return q4_0_dot_i8_scalar;
}

// Quantize a bf16 activation row to Q8_0 (llama.cpp): per-32-block symmetric int8 in
// natural order + one fp32 scale (amax/127) per block. Done once per token/route,
// amortized over every output row that the q4_0 W4A8 GEMV reads. K % 32 == 0.
void quant_q8_0(const bf16_t* x, int K, int8_t* aq, float* asb) {
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const bf16_t* xb = x + (size_t)b * 32;
    float xf[32], amax = 0.0f;
    for (int j = 0; j < 32; ++j) {
      xf[j] = bf16_to_f32(xb[j]);
      amax = std::max(amax, std::fabs(xf[j]));
    }
    const float d = amax > 0.0f ? amax / 127.0f : 1.0f;
    asb[b] = d;
    const float inv = amax > 0.0f ? 1.0f / d : 0.0f;
    int8_t* o = aq + (size_t)b * 32;
    for (int j = 0; j < 32; ++j)
      o[j] = (int8_t)std::max(-127, std::min(127, (int)std::lround(xf[j] * inv)));
  }
}

class Q4_0Format final : public MoeFormat {
 public:
  Q4_0Format(const MoeParams& p, const FormatArgs& a) : MoeFormat(p), dot_(select_q4dot()) {
    gate_up_ = a.bank("gate_up");
    down_ = a.bank("down");
    const size_t tokens = p.max_tokens, routes = (size_t)p.max_tokens * p.top_k;
    xi8_.resize(tokens * p.H);
    xas_.resize(tokens * (p.H / 32));
    gi8_.resize(routes * p.I);
    gas_.resize(routes * (p.I / 32));
    if (p.H % 32 != 0 || p.I % 32 != 0)
      throw std::runtime_error("Q4_0 CPU MoE requires H and I to be multiples of 32");
    gu_row_bytes_ = (p.H / 32) * 18;  // K = H (gate_up rows)
    dn_row_bytes_ = (p.I / 32) * 18;  // K = I (down rows)
    // select_q4dot picks VPDPBUSD / VPMADDUBSW / scalar for the tier, so the tag reflects which of those it resolved to.
    isa_ = std::string(tier_name(pick_isa())) + (cpu_has_avxvnni() ? "+vnni(q4_0-w4a8)" : "+q4_0-w4a8");
  }

  std::string isa_name() const override { return isa_; }


  void prep_input(int tok, const bf16_t* x) override {
    quant_q8_0(x, p.H, xi8_.data() + (size_t)tok * p.H, xas_.data() + (size_t)tok * (p.H / 32));
  }

  bool preps_inter() const override { return true; }

  void prep_inter(int64_t r, bf16_t* g) override {
    quant_q8_0(g, p.I, gi8_.data() + (size_t)r * p.I, gas_.data() + (size_t)r * (p.I / 32));
  }

  void gate_up(int layer, int e, int tok, int i0, int i1, const bf16_t* /*x*/, float w_in,
               bf16_t* g) const override {
    const uint8_t* packed = static_cast<const uint8_t*>(tbl_at(gate_up_, layer));
    const int8_t* xi8 = xi8_.data() + (size_t)tok * p.H;
    const float* xas = xas_.data() + (size_t)tok * (p.H / 32);
    gated_rows(p, i0, i1, w_in, g, [&](int row) {
      return dot_(packed + ((size_t)e * (2 * p.I) + row) * (size_t)gu_row_bytes_, xi8, xas, p.H);
    });
  }

  void down(int layer, int h0, int h1, const DownRoute* routes, int n, float* out) const override {
    const uint8_t* packed = static_cast<const uint8_t*>(tbl_at(down_, layer));
    for (int h = h0; h < h1; ++h) {
      for (int j = 0; j < n; ++j) {
        const size_t r = (size_t)routes[j].r;
        out[j * kRowBlock + h - h0] = dot_(packed + ((size_t)routes[j].e * p.H + h) * (size_t)dn_row_bytes_,
                                          gi8_.data() + r * p.I, gas_.data() + r * (p.I / 32), p.I);
      }
    }
  }

 private:
  q4dot_fn dot_;
  // GGUF Q4_0 rows of 32-element blocks (fp16 scale + 16 nibble bytes): gate_up [E,2I,H/32*18], down [E,H,I/32*18].
  const uint64_t* gate_up_ = nullptr;
  const uint64_t* down_ = nullptr;
  int gu_row_bytes_ = 0, dn_row_bytes_ = 0;
  std::string isa_;
  // Q8_0 activations: per-32-block int8 + fp32 scale, for the input and the intermediate.
  AlignedVector<int8_t> xi8_, gi8_;
  AlignedVector<float> xas_, gas_;
};

}  // namespace

std::unique_ptr<MoeFormat> make_q4_0_format(const MoeParams& p, const FormatArgs& a) {
  return std::make_unique<Q4_0Format>(p, a);
}

}  // namespace freetoken::cpu_moe
