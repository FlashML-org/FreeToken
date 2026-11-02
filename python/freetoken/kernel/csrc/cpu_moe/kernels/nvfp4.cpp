#include <vector>

#include "../format.h"

namespace freetoken::cpu_moe {
namespace {

// --------------------------- NVFP4 (W4A16) dequant ---------------------------
// Weights: e2m1 4-bit codes (2/byte, low nibble first), per-16 block scale in
// fp8-e4m3, per-output-row global scale in fp32. Dequant matches the GPU kernels
// (freetoken/kernel/triton/nvfp4_dequant.py): w = E2M1[code] * e4m3(scale) * global.
// Activations stay bf16 (W4A16); the GEMV dequantizes weights inside the K-loop.


// e2m1 * 2 as exact int8 (all e2m1 values are multiples of 0.5); the W4A8 dots undo the *2 with a 0.5 in the final scale. Mirrors ggml's kvalues_mxfp4.
alignas(16) const int8_t kE2M1x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
// kE2M1x2 + kE2M1Bias, the PSHUFB LUT of the VNNI dots: VPDPBUSD takes the weight as the unsigned operand, and W4a8Act::corr carries the -kE2M1Bias * sum(a) correction.
alignas(16) const uint8_t kE2M1x2u[16] = {12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0};
constexpr int kE2M1Bias = 12;

// Activations pre-deinterleaved to fp32 (xe[m]=x[2m], xo[m]=x[2m+1]); see the
// ds_fp4 dot in ds_fp4.cpp for why (drops the hot loop to ~1.5 shuffle ops / 16 weights).
using nvdot_fn = float (*)(const uint8_t*, const uint8_t*, float, const float*, const float*,
                           int, const float*, const float*);

float dot_nvfp4_scalar(const uint8_t* packed, const uint8_t* scale, float global,
                       const float* xe, const float* xo, int K, const float* e2m1,
                       const float* e4m3) {
  float acc = 0.0f;
  const int nb = K / 16;
  for (int b = 0; b < nb; ++b) {
    const float bs = e4m3[scale[b]];
    const uint8_t* pk = packed + (size_t)b * 8;
    const float* xeb = xe + (size_t)b * 8;  // 16 K -> 8 even + 8 odd
    const float* xob = xo + (size_t)b * 8;
    float bsum = 0.0f;
    for (int j = 0; j < 8; ++j) {
      const uint8_t byte = pk[j];
      bsum += e2m1[byte & 0xF] * xeb[j];
      bsum += e2m1[byte >> 4] * xob[j];
    }
    acc += bs * bsum;
  }
  return acc * global;
}

// ---- NVFP4 W4A8 (int8 activations) dot: nibble->int8 via LUT, per-16 act scale ----
// Activations are grouped in chunks of 8 16-K blocks: the even-K int8 of the chunk's blocks, then its odd-K int8.
// A 64-byte load of packed weights then splits into low (even-K) and high (odd-K) nibbles that line up with the activations, and int32 lane l of a dot holds block l / 2.
struct W4a8Act {
  const int8_t* q;      // [K]
  const float* scale;   // [K/16], absmax / 127 of each block
  const int32_t* corr;  // [K/8], one per int32 lane: -kE2M1Bias * the sum of the 8 activations it multiplies
};

using nvi8dot_fn = float (*)(const uint8_t*, const uint8_t*, float, const W4a8Act&, int);
using nvi8quant_fn = void (*)(const bf16_t*, int, int8_t*, float*, int32_t*);

#if CPU_MOE_X86
// Two 16-K blocks (16 packed bytes) per iter: lo nibbles -> even-K, hi -> odd-K,
// gathered via two vpermps. The per-16 e4m3 scale differs across the two blocks, so
// it is applied per lane (low 8 lanes = block b, high 8 = block b+1).
CPU_MOE_TARGET("avx512f")
inline __m512 nvfp4_blk2(const uint8_t* pk, const float* xeb, const float* xob, __m512 lut,
                         __m512i loma, float s0, float s1) {
  __m512i wi = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pk)));
  __m512 vlo = _mm512_permutexvar_ps(_mm512_and_si512(wi, loma), lut);
  __m512 vhi = _mm512_permutexvar_ps(_mm512_and_si512(_mm512_srli_epi32(wi, 4), loma), lut);
  __m512 prod = _mm512_fmadd_ps(vlo, _mm512_loadu_ps(xeb), _mm512_mul_ps(vhi, _mm512_loadu_ps(xob)));
  // lanes 0-7 (block b) -> s0, lanes 8-15 (block b+1) -> s1  (pure AVX512F mask move)
  __m512 scv = _mm512_mask_mov_ps(_mm512_set1_ps(s0), 0xFF00, _mm512_set1_ps(s1));
  return _mm512_mul_ps(prod, scv);
}

CPU_MOE_TARGET("avx512f")
float dot_nvfp4_avx512(const uint8_t* packed, const uint8_t* scale, float global,
                       const float* xe, const float* xo, int K, const float* e2m1,
                       const float* e4m3) {
  const __m512 lut = _mm512_loadu_ps(e2m1);  // 16 e2m1 values
  const __m512i loma = _mm512_set1_epi32(0xF);
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  const int nb = K / 16;  // 8 packed bytes + one e4m3 scale per block
  int b = 0;
  for (; b + 4 <= nb; b += 4) {  // two blk2 calls -> 4 blocks, 2 accumulators
    acc0 = _mm512_add_ps(acc0, nvfp4_blk2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                          xo + (size_t)b * 8, lut, loma, e4m3[scale[b]], e4m3[scale[b + 1]]));
    acc1 = _mm512_add_ps(acc1, nvfp4_blk2(packed + (size_t)(b + 2) * 8, xe + (size_t)(b + 2) * 8,
                                          xo + (size_t)(b + 2) * 8, lut, loma, e4m3[scale[b + 2]], e4m3[scale[b + 3]]));
  }
  for (; b + 2 <= nb; b += 2)
    acc0 = _mm512_add_ps(acc0, nvfp4_blk2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                          xo + (size_t)b * 8, lut, loma, e4m3[scale[b]], e4m3[scale[b + 1]]));
  float s = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
  for (; b < nb; ++b) {  // odd final 16-K block
    const uint8_t* pk = packed + (size_t)b * 8;
    const float* xeb = xe + (size_t)b * 8;
    const float* xob = xo + (size_t)b * 8;
    float bsum = 0.0f;
    for (int j = 0; j < 8; ++j) {
      bsum += e2m1[pk[j] & 0xF] * xeb[j] + e2m1[pk[j] >> 4] * xob[j];
    }
    s += e4m3[scale[b]] * bsum;
  }
  return s * global;
}

// AVX2: one 16-K block (8 packed bytes) per call, 8 even + 8 odd lanes.
CPU_MOE_TARGET("avx2,fma")
inline __m256 nvfp4_blk_avx2(const uint8_t* pk, const float* xeb, const float* xob,
                             __m256 mag8, float sc) {
  __m256i wi = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(pk)));
  __m256 vlo = e2m1_decode8(_mm256_and_si256(wi, _mm256_set1_epi32(0xF)), mag8);
  __m256 vhi = e2m1_decode8(_mm256_srli_epi32(wi, 4), mag8);
  __m256 prod = _mm256_fmadd_ps(vlo, _mm256_loadu_ps(xeb), _mm256_mul_ps(vhi, _mm256_loadu_ps(xob)));
  return _mm256_mul_ps(prod, _mm256_set1_ps(sc));
}

CPU_MOE_TARGET("avx2,fma")
float dot_nvfp4_avx2(const uint8_t* packed, const uint8_t* scale, float global,
                     const float* xe, const float* xo, int K, const float* e2m1,
                     const float* e4m3) {
  const __m256 mag8 = _mm256_loadu_ps(e2m1);  // e2m1[0..7] magnitudes
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  const int nb = K / 16;
  int b = 0;
  for (; b + 2 <= nb; b += 2) {
    acc0 = _mm256_add_ps(acc0, nvfp4_blk_avx2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                              xo + (size_t)b * 8, mag8, e4m3[scale[b]]));
    acc1 = _mm256_add_ps(acc1, nvfp4_blk_avx2(packed + (size_t)(b + 1) * 8, xe + (size_t)(b + 1) * 8,
                                              xo + (size_t)(b + 1) * 8, mag8, e4m3[scale[b + 1]]));
  }
  for (; b < nb; ++b)
    acc0 = _mm256_add_ps(acc0, nvfp4_blk_avx2(packed + (size_t)b * 8, xe + (size_t)b * 8,
                                              xo + (size_t)b * 8, mag8, e4m3[scale[b]]));
  return hsum256(_mm256_add_ps(acc0, acc1)) * global;
}

// Software-prefetch distance for the W4A8 weight stream, in 16-K blocks (8 packed
// bytes each). Returns -1 when FREETOKEN_CPU_MOE_PF_BLOCKS is unset: the kernel then
// uses the built-in default min(512 blocks = 4 KB, 2 rows) -- 4 KB is the empirical
// optimum on large-row machines (Emerald Rapids sweep), while the 2-row cap keeps a small-row
// model's overshoot bounded (the executor works in 32-row tiles, so a fixed byte
// distance otherwise prefetches another worker's tile: duplicated DRAM traffic that
// regresses at the bandwidth ceiling). An EXPLICIT env value is honored verbatim
// (no clamp; 0 disables): the per-machine optimum can sit past the safe default (+20%
// at 4 KB on a 24-thread Ice Lake with 256B rows), so the escape hatch must reach it.
// Prefetch never faults, so overshooting a row/bank tail is safe.
static int nvfp4_pf_blocks() {
  static const int v = [] {
    const char* s = getenv("FREETOKEN_CPU_MOE_PF_BLOCKS");
    return (s && s[0]) ? atoi(s) : -1;
  }();
  return v;
}

// e4m3 -> fp16 is a shift: exponent and mantissa move up 7 bits, and the bias gap (15 - 7) leaves a 2^-8 that the kernels undo in the final scale.
// The byte is sign-extended first, so its sign also lands in bit 15 and & 0xBF80 keeps it.
constexpr float kE4m3ViaF16 = 256.0f;
constexpr int16_t kE4m3F16Mask = (int16_t)0xBF80;

int w4a8_pf(int nb) {
  const int pfb = nvfp4_pf_blocks();
  return (pfb < 0) ? std::min(512, 2 * nb) : pfb;
}

// Weight scale * act scale of 8 blocks, in 2^-8 units.
CPU_MOE_TARGET("avx2,fma,f16c")
inline __m256 w4a8_scales256(__m128i s8, const float* asc) {
  __m128i h = _mm_cvtepi8_epi16(s8);
  h = _mm_and_si128(_mm_slli_epi16(h, 7), _mm_set1_epi16(kE4m3F16Mask));
  return _mm256_mul_ps(_mm256_cvtph_ps(h), _mm256_loadu_ps(asc));
}

CPU_MOE_TARGET("avx2,fma,f16c")
inline void w4a8_split256(__m256i v, __m256i lut, __m256i& lo, __m256i& hi) {
  const __m256i m4 = _mm256_set1_epi8(0x0F);
  lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(v, m4));
  hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v, 4), m4));
}

CPU_MOE_TARGET("avx2,fma,f16c,avxvnni")
inline __m256i w4a8_dot256_vnni(__m256i corr, __m256i lo, __m256i qe, __m256i hi, __m256i qo) {
  return _mm256_dpbusd_avx_epi32(_mm256_dpbusd_avx_epi32(corr, lo, qe), hi, qo);
}

// sum of e4m3(scale) * act scale * int dot over blocks [b0, nb) in 2^-8 units, before the final 0.5 * global
CPU_MOE_TARGET("avx2,fma,f16c")
float w4a8_blocks_scalar(const uint8_t* packed, const uint8_t* scale, const W4a8Act& a, int b0, int nb) {
  float acc = 0.0f;
  for (int b = b0; b < nb; ++b) {
    const int c0 = b & ~7, nbc = std::min(8, nb - c0);
    const int8_t* qe = a.q + (size_t)c0 * 16 + (size_t)(b - c0) * 8;
    const int8_t* qo = qe + nbc * 8;
    const uint8_t* pk = packed + (size_t)b * 8;
    int isum = 0;
    for (int j = 0; j < 8; ++j) isum += kE2M1x2[pk[j] & 0xF] * qe[j] + kE2M1x2[pk[j] >> 4] * qo[j];
    acc += (_cvtsh_ss((uint16_t)(((int16_t)(int8_t)scale[b] << 7) & kE4m3F16Mask)) * a.scale[b]) * (float)isum;
  }
  return acc;
}

CPU_MOE_TARGET("avx2,fma,f16c,avxvnni")
float dot_nvfp4_i8_vnni(const uint8_t* packed, const uint8_t* scale, float global, const W4a8Act& a, int K) {
  const __m256i lut = _mm256_broadcastsi128_si256(_mm_load_si128(reinterpret_cast<const __m128i*>(kE2M1x2u)));
  const __m256i lanes_lo = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
  const __m256i lanes_hi = _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4);
  const int nb = K / 16, pf = w4a8_pf(nb);
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  __m256i lo, hi;
  int b = 0;
  for (; b + 8 <= nb; b += 8) {
    const uint8_t* pk = packed + (size_t)b * 8;
    if (pf > 0) _mm_prefetch(reinterpret_cast<const char*>(pk + (size_t)pf * 8), _MM_HINT_T0);
    const int8_t* q = a.q + (size_t)b * 16;
    const int32_t* c = a.corr + (size_t)b * 2;
    const __m256 sc = w4a8_scales256(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(scale + b)), a.scale + b);
    w4a8_split256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pk)), lut, lo, hi);
    __m256i d = w4a8_dot256_vnni(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(c)), lo,
                                 _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q)), hi,
                                 _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 64)));
    acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), _mm256_permutevar8x32_ps(sc, lanes_lo), acc0);
    w4a8_split256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pk + 32)), lut, lo, hi);
    d = w4a8_dot256_vnni(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(c + 8)), lo,
                         _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 32)), hi,
                         _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 96)));
    acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), _mm256_permutevar8x32_ps(sc, lanes_hi), acc1);
  }
  // A last chunk of 4..7 blocks runs its first half here (its odd-K bytes start 8 * n after the even-K ones); fewer than 4 left go scalar.
  if (b + 4 <= nb) {
    const int n = nb - b;
    const int8_t* q = a.q + (size_t)b * 16;
    int s4;
    std::memcpy(&s4, scale + b, 4);
    float as8[8] = {a.scale[b], a.scale[b + 1], a.scale[b + 2], a.scale[b + 3]};
    const __m256 sc = w4a8_scales256(_mm_cvtsi32_si128(s4), as8);
    w4a8_split256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + (size_t)b * 8)), lut, lo, hi);
    const __m256i d = w4a8_dot256_vnni(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.corr + (size_t)b * 2)), lo,
                                       _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q)), hi,
                                       _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 8 * n)));
    acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), _mm256_permutevar8x32_ps(sc, lanes_lo), acc0);
    b += 4;
  }
  const float tail = w4a8_blocks_scalar(packed, scale, a, b, nb);
  return (hsum256(_mm256_add_ps(acc0, acc1)) + tail) * (0.5f * kE4m3ViaF16 * global);
}

CPU_MOE_TARGET("avx2")
inline int hsum_epi32_256(__m256i v) {
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
  return _mm_cvtsi128_si32(s);
}

CPU_MOE_TARGET("avx2,fma")
inline __m256i w4a8_round256(__m256 y) {
  const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
  const __m256 t = _mm256_round_ps(y, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  const __m256 away = _mm256_cmp_ps(_mm256_and_ps(_mm256_sub_ps(y, t), absm), _mm256_set1_ps(0.5f), _CMP_GE_OQ);
  const __m256 step = _mm256_or_ps(_mm256_and_ps(y, _mm256_set1_ps(-0.0f)), _mm256_set1_ps(1.0f));
  const __m256 r = _mm256_add_ps(t, _mm256_and_ps(away, step));
  return _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(r, _mm256_set1_ps(-127.0f)), _mm256_set1_ps(127.0f)));
}

// The quants match a scalar absmax / 127 per-block quant that rounds with lround (half away from zero), bit for bit.
// 1 / scale overflows for absmax below ~3.7e-37, so such a block quantizes to zeros.
// Dot lanes 2b and 2b + 1 multiply the first and the last 8 activations of block b, so corr is two plain sums.
CPU_MOE_TARGET("avx2,fma")
void quant_i8_avx2(const bf16_t* x, int K, int8_t* q, float* scale, int32_t* corr) {
  const __m128i even_odd = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
  const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
  const int nb = K / 16;
  for (int b = 0; b < nb; ++b) {
    const int c0 = b & ~7, nbc = std::min(8, nb - c0);
    int8_t* qe = q + (size_t)c0 * 16 + (size_t)(b - c0) * 8;
    const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + (size_t)b * 16));
    const __m256 v0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(raw)), 16));
    const __m256 v1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(raw, 1)), 16));
    const __m256 m8 = _mm256_max_ps(_mm256_and_ps(v0, absm), _mm256_and_ps(v1, absm));
    __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m8), _mm256_extractf128_ps(m8, 1));
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 0x55));
    const float amax = _mm_cvtss_f32(m4);
    const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
    scale[b] = s;
    const float inv_s = 1.0f / s;
    const __m256 inv = _mm256_set1_ps(inv_s < INFINITY ? inv_s : 0.0f);
    const __m256i i0 = w4a8_round256(_mm256_mul_ps(v0, inv)), i1 = w4a8_round256(_mm256_mul_ps(v1, inv));
    const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i0, i1), 0xD8);
    const __m128i bytes = _mm_shuffle_epi8(_mm_packs_epi16(_mm256_castsi256_si128(p16), _mm256_extracti128_si256(p16, 1)), even_odd);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(qe), bytes);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(qe + nbc * 8), _mm_unpackhi_epi64(bytes, bytes));
    corr[2 * b] = -kE2M1Bias * hsum_epi32_256(i0);
    corr[2 * b + 1] = -kE2M1Bias * hsum_epi32_256(i1);
  }
}

#ifdef CPU_MOE_HAS_AVX512VNNI

// Weight scale * act scale of n <= 16 blocks, in 2^-8 units.
CPU_MOE_TARGET("avx512f,avx512bw,avx512vl")
inline __m512 w4a8_scales512(const uint8_t* scale, const float* asc, int n) {
  const __mmask16 m = (__mmask16)((1u << n) - 1);
  __m256i h = _mm256_cvtepi8_epi16(_mm_maskz_loadu_epi8(m, scale));
  h = _mm256_and_si256(_mm256_slli_epi16(h, 7), _mm256_set1_epi16(kE4m3F16Mask));
  return _mm512_mul_ps(_mm512_cvtph_ps(h), _mm512_maskz_loadu_ps(m, asc));
}

CPU_MOE_TARGET("avx512f,avx512bw,avx512vl")
inline void w4a8_split512(__m512i v, __m512i lut, __m512i& lo, __m512i& hi) {
  const __m512i m4 = _mm512_set1_epi8(0x0F);
  lo = _mm512_shuffle_epi8(lut, _mm512_and_si512(v, m4));
  hi = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(v, 4), m4));
}

CPU_MOE_TARGET("avx512f,avx512bw,avx512vl,avx512vnni")
inline __m512i w4a8_dot512_vnni(__m512i corr, __m512i lo, __m512i qe, __m512i hi, __m512i qo) {
  return _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(corr, lo, qe), hi, qo);
}

CPU_MOE_TARGET("avx512f,avx512bw,avx512vl,avx512vnni")
float dot_nvfp4_i8_avx512vnni(const uint8_t* packed, const uint8_t* scale, float global, const W4a8Act& a, int K) {
  const __m512i lut = _mm512_broadcast_i32x4(_mm_load_si128(reinterpret_cast<const __m128i*>(kE2M1x2u)));
  const __m512i lanes_lo = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
  const __m512i lanes_hi = _mm512_add_epi32(lanes_lo, _mm512_set1_epi32(8));
  const int nb = K / 16, pf = w4a8_pf(nb);
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  __m512i lo, hi;
  int b = 0;
  for (; b + 16 <= nb; b += 16) {
    const uint8_t* pk = packed + (size_t)b * 8;
    // The HW prefetcher alone falls short here: with pf = 0 a GLM-5.3-Flash executor step is 1.27x slower on Emerald Rapids.
    if (pf > 0) {
      _mm_prefetch(reinterpret_cast<const char*>(pk + (size_t)pf * 8), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(pk + (size_t)pf * 8 + 64), _MM_HINT_T0);
    }
    const int8_t* q = a.q + (size_t)b * 16;
    const int32_t* c = a.corr + (size_t)b * 2;
    const __m512 sc = w4a8_scales512(scale + b, a.scale + b, 16);
    w4a8_split512(_mm512_loadu_si512(pk), lut, lo, hi);
    __m512i d = w4a8_dot512_vnni(_mm512_loadu_si512(c), lo, _mm512_loadu_si512(q), hi, _mm512_loadu_si512(q + 64));
    acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d), _mm512_permutexvar_ps(lanes_lo, sc), acc0);
    w4a8_split512(_mm512_loadu_si512(pk + 64), lut, lo, hi);
    d = w4a8_dot512_vnni(_mm512_loadu_si512(c + 16), lo, _mm512_loadu_si512(q + 128), hi, _mm512_loadu_si512(q + 192));
    acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d), _mm512_permutexvar_ps(lanes_hi, sc), acc1);
  }
  // The tail runs chunks of n <= 8 blocks with masked loads: masked-off activations are zero, so those lanes add nothing.
  for (; b < nb; b += 8) {
    const int n = std::min(8, nb - b);
    const __mmask64 mb = n == 8 ? ~0ULL : (1ULL << (8 * n)) - 1;
    const int8_t* q = a.q + (size_t)b * 16;
    const __m512 sc = w4a8_scales512(scale + b, a.scale + b, n);
    w4a8_split512(_mm512_maskz_loadu_epi8(mb, packed + (size_t)b * 8), lut, lo, hi);
    const __m512i c = _mm512_maskz_loadu_epi32((__mmask16)((1u << (2 * n)) - 1), a.corr + (size_t)b * 2);
    const __m512i d = w4a8_dot512_vnni(c, lo, _mm512_maskz_loadu_epi8(mb, q), hi, _mm512_maskz_loadu_epi8(mb, q + 8 * n));
    acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d), _mm512_permutexvar_ps(lanes_lo, sc), acc0);
  }
  return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1)) * (0.5f * kE4m3ViaF16 * global);
}

CPU_MOE_TARGET("avx512f,avx512bw,avx512vl")
void quant_i8_avx512(const bf16_t* x, int K, int8_t* q, float* scale, int32_t* corr) {
  const __m128i even_odd = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
  const __m512 half = _mm512_set1_ps(0.5f);
  const __m512i one = _mm512_castps_si512(_mm512_set1_ps(1.0f)), sign = _mm512_set1_epi32(INT32_MIN);
  const __m512 lim = _mm512_set1_ps(127.0f), nlim = _mm512_set1_ps(-127.0f);
  const int nb = K / 16;
  for (int b = 0; b < nb; ++b) {
    const int c0 = b & ~7, nbc = std::min(8, nb - c0);
    int8_t* qe = q + (size_t)c0 * 16 + (size_t)(b - c0) * 8;
    const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + (size_t)b * 16));
    const __m512 v = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
    const float amax = _mm512_reduce_max_ps(_mm512_abs_ps(v));
    const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
    scale[b] = s;
    const float inv = 1.0f / s;
    const __m512 y = _mm512_mul_ps(v, _mm512_set1_ps(inv < INFINITY ? inv : 0.0f));
    const __m512 t = _mm512_roundscale_ps(y, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    const __mmask16 away = _mm512_cmp_ps_mask(_mm512_abs_ps(_mm512_sub_ps(y, t)), half, _CMP_GE_OQ);
    const __m512 step = _mm512_castsi512_ps(_mm512_or_si512(_mm512_and_si512(_mm512_castps_si512(y), sign), one));
    const __m512 r = _mm512_mask_add_ps(t, away, t, step);
    const __m512i ri = _mm512_cvtps_epi32(_mm512_min_ps(_mm512_max_ps(r, nlim), lim));
    const __m128i bytes = _mm_shuffle_epi8(_mm512_cvtepi32_epi8(ri), even_odd);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(qe), bytes);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(qe + nbc * 8), _mm_unpackhi_epi64(bytes, bytes));
    corr[2 * b] = -kE2M1Bias * hsum_epi32_256(_mm512_castsi512_si256(ri));
    corr[2 * b + 1] = -kE2M1Bias * hsum_epi32_256(_mm512_extracti64x4_epi64(ri, 1));
  }
}
#endif  // avx512vnni available
#endif

nvdot_fn select_nvdot() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
  if (t >= ISA_AVX512) return dot_nvfp4_avx512;
  if (t >= ISA_AVX2) return dot_nvfp4_avx2;
#endif
  (void)t;
  return dot_nvfp4_scalar;
}

struct W4a8Kernel {
  nvi8dot_fn dot;
  nvi8quant_fn quant;
  const char* isa;
};

// Best W4A8 (int8-activation) nvfp4 dot for the picked ISA tier, or a null dot if that tier has no VNNI (caller keeps the faithful fp32 nvdot path).
W4a8Kernel select_nvi8dot() {
#if CPU_MOE_X86
  const IsaTier t = pick_isa();
#if defined(CPU_MOE_HAS_AVX512VNNI)
  if (t >= ISA_AVX512 && cpu_has_avx512vnni()) return {dot_nvfp4_i8_avx512vnni, quant_i8_avx512, "+avx512vnni(nvfp4-w4a8)"};
#endif
  if (t >= ISA_AVX2 && cpu_has_avxvnni()) return {dot_nvfp4_i8_vnni, quant_i8_avx2, "+vnni(nvfp4-w4a8)"};
#endif
  return {};
}

struct W4a8Rows {
  AlignedVector<int8_t> q;
  AlignedVector<float> scale;
  AlignedVector<int32_t> corr;
  int K = 0;

  void resize(size_t rows, int k) {
    K = k;
    q.resize(rows * k);
    scale.resize(rows * (k / 16));
    corr.resize(rows * (k / 8));
  }
  void quant(nvi8quant_fn fn, size_t row, const bf16_t* x) {
    fn(x, K, q.data() + row * K, scale.data() + row * (K / 16), corr.data() + row * (K / 8));
  }
  W4a8Act at(size_t row) const {
    return {q.data() + row * K, scale.data() + row * (K / 16), corr.data() + row * (K / 8)};
  }
};

class Nvfp4Format final : public MoeFormat {
 public:
  Nvfp4Format(const MoeParams& p, const FormatArgs& a)
      : MoeFormat(p), nvdot_(select_nvdot()), w4a8_(select_nvi8dot()) {
    gate_up_ = a.bank("gate_up");
    gu_scale_ = a.bank("gate_up_scale");
    gu_global_ = a.bank("gate_up_global");
    down_ = a.bank("down");
    dn_scale_ = a.bank("down_scale");
    dn_global_ = a.bank("down_global");
    const size_t tokens = p.max_tokens, routes = (size_t)p.max_tokens * p.top_k;
    if (w4a8_.dot != nullptr) {
      xq_.resize(tokens, p.H);
      gq_.resize(routes, p.I);
    } else {
      xe_.resize(tokens * (p.H / 2));
      xo_.resize(tokens * (p.H / 2));
      ge_.resize(routes * (p.I / 2));
      go_.resize(routes * (p.I / 2));
    }
    isa_ = tier_name(pick_isa());
    if (w4a8_.dot != nullptr) isa_ += w4a8_.isa;
    for (int i = 0; i < 16; ++i) e2m1_lut_[i] = kE2M1[i];
    for (int i = 0; i < 256; ++i) e4m3_lut_[i] = e4m3_decode((uint8_t)i);
  }

  std::string isa_name() const override { return isa_; }


  void prep_input(int tok, const bf16_t* x) override {
    if (w4a8_.dot != nullptr) {
      xq_.quant(w4a8_.quant, tok, x);
      return;
    }
    float* xe = xe_.data() + (size_t)tok * (p.H / 2);
    float* xo = xo_.data() + (size_t)tok * (p.H / 2);
    deinterleave_bf16_f32(x, xe, xo, p.H);
  }

  bool preps_inter() const override { return true; }

  void prep_inter(int64_t r, bf16_t* g) override {
    if (w4a8_.dot != nullptr) {
      gq_.quant(w4a8_.quant, r, g);
      return;
    }
    float* ge = ge_.data() + (size_t)r * (p.I / 2);
    float* go = go_.data() + (size_t)r * (p.I / 2);
    deinterleave_bf16_f32(g, ge, go, p.I);
  }

  void gate_up(int layer, int e, int tok, int i0, int i1, const bf16_t* /*x*/, float w_in,
               bf16_t* g) const override {
    const int H = p.H;
    const uint8_t* packed = static_cast<const uint8_t*>(tbl_at(gate_up_, layer));
    const uint8_t* scale = static_cast<const uint8_t*>(tbl_at(gu_scale_, layer));
    const float* global = static_cast<const float*>(tbl_at(gu_global_, layer));
    if (w4a8_.dot != nullptr) {
      const W4a8Act act = xq_.at(tok);
      gated_rows(p, i0, i1, w_in, g, [&](int row) {
        const size_t r = (size_t)e * (2 * p.I) + row;
        return w4a8_.dot(packed + r * (size_t)(H / 2), scale + r * (size_t)(H / 16), global[r], act, H);
      });
      return;
    }
    const float* xe = xe_.data() + (size_t)tok * (H / 2);
    const float* xo = xo_.data() + (size_t)tok * (H / 2);
    gated_rows(p, i0, i1, w_in, g, [&](int row) {
      const size_t r = (size_t)e * (2 * p.I) + row;
      return nvdot_(packed + r * (size_t)(H / 2), scale + r * (size_t)(H / 16),
                    global[r], xe, xo, H, e2m1_lut_, e4m3_lut_);
    });
  }

  void down(int layer, int h0, int h1, const DownRoute* routes, int n, float* out) const override {
    const int I = p.I;
    const uint8_t* packed = static_cast<const uint8_t*>(tbl_at(down_, layer));
    const uint8_t* scale = static_cast<const uint8_t*>(tbl_at(dn_scale_, layer));
    const float* global = static_cast<const float*>(tbl_at(dn_global_, layer));
    for (int h = h0; h < h1; ++h) {
      for (int j = 0; j < n; ++j) {
        const size_t r = (size_t)routes[j].r;
        const size_t row = (size_t)routes[j].e * p.H + h;
        out[j * kRowBlock + h - h0] = w4a8_.dot != nullptr
            ? w4a8_.dot(packed + row * (size_t)(I / 2), scale + row * (size_t)(I / 16), global[row], gq_.at(r), I)
            : nvdot_(packed + row * (size_t)(I / 2), scale + row * (size_t)(I / 16), global[row],
                     ge_.data() + r * (I / 2), go_.data() + r * (I / 2), I, e2m1_lut_, e4m3_lut_);
      }
    }
  }

 private:
  nvdot_fn nvdot_;
  const W4a8Kernel w4a8_;  // null dot -> use fp32 nvdot
  // ModelOpt rows: packed e2m1 [E,2I,H/2], e4m3 scales per 16 along K [E,2I,H/16], fp32 row globals [E,2I]; down likewise with K = I.
  const uint64_t* gate_up_ = nullptr;
  const uint64_t* gu_scale_ = nullptr;
  const uint64_t* gu_global_ = nullptr;
  const uint64_t* down_ = nullptr;
  const uint64_t* dn_scale_ = nullptr;
  const uint64_t* dn_global_ = nullptr;
  std::string isa_;
  float e2m1_lut_[16];
  float e4m3_lut_[256];
  // fp32 even/odd activations for the fp32 dot, or the W4A8 int8 rows.
  AlignedVector<float> xe_, xo_, ge_, go_;
  W4a8Rows xq_, gq_;
};

}  // namespace

std::unique_ptr<MoeFormat> make_nvfp4_format(const MoeParams& p, const FormatArgs& a) {
  return std::make_unique<Nvfp4Format>(p, a);
}

}  // namespace freetoken::cpu_moe
